import os
import sys

# CRITICAL: Add path BEFORE importing detectron2
sys.path.insert(1, os.path.join(sys.path[0], '..'))
print("Python path:", sys.path)
import torch
import numpy as np
import json
from tabulate import tabulate
from collections import defaultdict
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from detectron2.structures import ImageList
import torch.nn.functional as F

from mask2former import add_maskformer2_config
from mask2former_mars.modeling.meta_arch.mars_mask_former_head import add_mars_config


class EigenCAMGenerator:
    """
    Generates EigenCAM visualizations from model attention/feature maps
    
    EigenCAM uses the principal component (eigenvector) of the feature maps
    to create class-agnostic activation maps
    """
    
    def __init__(self, model):
        self.model = model
        self.feature_maps = []
        self.attention_maps = []
        self.hooks = []
        
    def register_hooks(self):
        """Register hooks to capture intermediate feature/attention maps"""
        
        def feature_hook(module, input, output):
            """Capture feature maps from backbone"""
            if isinstance(output, dict):
                # Take the highest resolution feature map
                for key in sorted(output.keys()):
                    self.feature_maps.append(output[key].detach())
                    break
            elif isinstance(output, torch.Tensor):
                self.feature_maps.append(output.detach())
        
        def attention_hook(module, input, output):
            """Capture attention maps from transformer"""
            if isinstance(output, tuple) and len(output) >= 2:
                attention = output[1]
                if attention is not None:
                    self.attention_maps.append(attention.detach())
        
        # Hook into backbone (last layer)
        if hasattr(self.model, 'backbone'):
            # Find the last convolutional layer
            for name, module in self.model.backbone.named_modules():
                if 'res5' in name or 'layer4' in name:
                    hook = module.register_forward_hook(feature_hook)
                    self.hooks.append(hook)
                    break
        
        # Hook into transformer attention
        if hasattr(self.model, 'sem_seg_head'):
            predictor = self.model.sem_seg_head.predictor
            if hasattr(predictor, 'transformer_cross_attention_layers'):
                layers = predictor.transformer_cross_attention_layers
                for layer in layers:
                    if hasattr(layer, 'multihead_attn'):
                        hook = layer.multihead_attn.register_forward_hook(attention_hook)
                        self.hooks.append(hook)
        
        print(f"✅ Registered {len(self.hooks)} hooks for EigenCAM")
    
    def clear_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.feature_maps = []
        self.attention_maps = []
    
    def compute_eigencam(self, image_tensor, target_size=None):
        """
        Compute EigenCAM for an image
        
        Args:
            image_tensor: Input image tensor [C, H, W]
            target_size: (H, W) to resize the CAM to
            
        Returns:
            cam: Activation map as numpy array [H, W]
        """
        self.feature_maps = []
        self.attention_maps = []
        
        with torch.no_grad():
            # Preprocess
            image = image_tensor.to(next(self.model.parameters()).device)
            image_normalized = (image - self.model.pixel_mean) / self.model.pixel_std
            images = ImageList.from_tensors([image_normalized], self.model.size_divisibility)
            
            # Forward pass
            features = self.model.backbone(images.tensor)
            outputs = self.model.sem_seg_head(features)
        
        # Get feature maps
        if len(self.feature_maps) == 0:
            print("⚠️ No feature maps captured")
            return None
        
        # Use the captured feature map
        feature_map = self.feature_maps[0]  # [1, C, H, W]
        
        if feature_map.dim() == 4:
            feature_map = feature_map[0]  # [C, H, W]
        
        # Compute EigenCAM using PCA
        cam = self._compute_eigencam_from_features(feature_map)
        
        # Resize to target size if specified
        if target_size is not None:
            cam = cv2.resize(cam, (target_size[1], target_size[0]))
        
        return cam
    
    def _compute_eigencam_from_features(self, feature_map):
        """
        Compute EigenCAM from feature maps using SVD (more stable than eigendecomposition)
        
        Args:
            feature_map: Feature tensor [C, H, W]
            
        Returns:
            cam: Activation map [H, W]
        """
        C, H, W = feature_map.shape
        
        # Reshape to [C, H*W]
        features = feature_map.reshape(C, -1)
        
        # Center the features
        features = features - features.mean(dim=1, keepdim=True)
        
        # Normalize features for numerical stability
        feature_norm = torch.norm(features, dim=0, keepdim=True)
        features = features / (feature_norm + 1e-8)
        
        # Use SVD instead of eigendecomposition (more stable)
        # features: [C, H*W]
        # U: [C, C], S: [min(C, H*W)], V: [H*W, H*W]
        try:
            U, S, V = torch.svd(features)
            # Principal component is the first column of U
            principal_component = U[:, 0]  # [C]
        except RuntimeError:
            # Fallback: use simpler method if SVD fails
            print("   ⚠️ SVD failed, using alternative method")
            # Just use mean activation across channels
            cam = feature_map.mean(dim=0)  # [H, W]
            cam = cam.cpu().numpy()
            cam = np.maximum(cam, 0)
            if cam.max() > 0:
                cam = cam / cam.max()
            return cam
        
        # Project features onto principal component
        cam = torch.matmul(principal_component, features)  # [H*W]
        
        # Reshape to spatial dimensions
        cam = cam.reshape(H, W)
        
        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)  # ReLU
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def visualize_eigencam(self, image_np, cam, alpha=0.5):
        """
        Create a heatmap overlay visualization
        
        Args:
            image_np: Original image as numpy array [H, W, 3] in range [0, 255]
            cam: Activation map [H, W] in range [0, 1]
            alpha: Blending factor
            
        Returns:
            overlay: Blended image with heatmap
        """
        # Ensure image is uint8
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        # Resize CAM to image size if needed
        if cam.shape[:2] != image_np.shape[:2]:
            cam = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
        
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
        
        return overlay


class RotatedDatasetMapper:
    """Dataset mapper that applies rotation to BOTH images AND annotations"""
    
    def __init__(self, cfg, rotation_angle=0):
        self.base_mapper = DatasetMapper(cfg, is_train=False, augmentations=[])
        self.rotation_angle = rotation_angle
    
    def __call__(self, dataset_dict):
        """Apply rotation after base mapper processing"""
        result = self.base_mapper(dataset_dict)
        
        # Store original image before rotation
        result["original_image"] = result["image"].clone()
        
        # Rotate the image tensor if needed
        if self.rotation_angle != 0:
            k = self.rotation_angle // 90
            result["image"] = torch.rot90(result["image"], k=k, dims=[1, 2])
            
            if self.rotation_angle in [90, 270]:
                if "height" in result and "width" in result:
                    result["height"], result["width"] = result["width"], result["height"]
            
            if "instances" in result:
                result["instances"] = self._rotate_instances(
                    result["instances"],
                    k,
                    result.get("height", result["image"].shape[1]),
                    result.get("width", result["image"].shape[2])
                )
        
        return result
    
    def _rotate_instances(self, instances, k, orig_h, orig_w):
        """Rotate instance annotations"""
        from detectron2.structures import Instances, Boxes, BitMasks
        import copy
        
        if k == 0:
            return copy.deepcopy(instances)
        
        new_h = orig_w if k % 2 == 1 else orig_h
        new_w = orig_h if k % 2 == 1 else orig_w
        rotated = Instances((new_h, new_w))
        
        if instances.has("gt_boxes"):
            boxes = instances.gt_boxes.tensor.clone()
            rotated_boxes = self._rotate_boxes(boxes, k, orig_h, orig_w)
            rotated.gt_boxes = Boxes(rotated_boxes)
        
        if instances.has("gt_masks"):
            masks = instances.gt_masks
            if isinstance(masks, torch.Tensor):
                rotated_masks = torch.rot90(masks, k=k, dims=[1, 2])
                rotated.gt_masks = rotated_masks
            elif hasattr(masks, 'tensor'):
                mask_tensor = masks.tensor
                rotated_masks = torch.rot90(mask_tensor, k=k, dims=[1, 2])
                rotated.gt_masks = BitMasks(rotated_masks)
        
        for field in instances.get_fields():
            if field not in ["gt_boxes", "gt_masks"]:
                rotated.set(field, copy.deepcopy(instances.get(field)))
        
        return rotated
    
    def _rotate_boxes(self, boxes, k, h, w):
        """Rotate bounding boxes"""
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        
        if k == 1:
            return torch.stack([y1, w - x2, y2, w - x1], dim=-1)
        elif k == 2:
            return torch.stack([w - x2, h - y2, w - x1, h - y1], dim=-1)
        elif k == 3:
            return torch.stack([h - y2, x1, h - y1, x2], dim=-1)
        else:
            return boxes


def save_eigencam_visualization(model, image_tensor, original_image_tensor, 
                                rotation_angle, output_dir, image_id):
    """
    Generate and save EigenCAM visualizations for original and rotated images
    
    Args:
        model: The detection model
        image_tensor: Rotated image tensor [C, H, W]
        original_image_tensor: Original (non-rotated) image tensor [C, H, W]
        rotation_angle: Rotation angle in degrees
        output_dir: Directory to save visualizations
        image_id: Identifier for the image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create EigenCAM generator
    cam_gen = EigenCAMGenerator(model)
    cam_gen.register_hooks()
    
    # Convert tensors to numpy for visualization
    def tensor_to_numpy(tensor):
        """Convert [C, H, W] tensor to [H, W, C] numpy array"""
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        # Denormalize if needed (assuming ImageNet normalization)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img = img * std + mean
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    # Get numpy images
    original_np = tensor_to_numpy(original_image_tensor)
    rotated_np = tensor_to_numpy(image_tensor)
    
    # Compute EigenCAM for original image
    print(f"  Computing EigenCAM for original image {image_id}...")
    cam_original = cam_gen.compute_eigencam(
        original_image_tensor, 
        target_size=(original_np.shape[0], original_np.shape[1])
    )
    
    # Compute EigenCAM for rotated image
    print(f"  Computing EigenCAM for rotated image ({rotation_angle}°) {image_id}...")
    cam_rotated = cam_gen.compute_eigencam(
        image_tensor,
        target_size=(rotated_np.shape[0], rotated_np.shape[1])
    )
    
    # Create visualizations
    if cam_original is not None:
        overlay_original = cam_gen.visualize_eigencam(original_np, cam_original)
    else:
        overlay_original = original_np
        print("⚠️ Could not generate EigenCAM for original image")
    
    if cam_rotated is not None:
        overlay_rotated = cam_gen.visualize_eigencam(rotated_np, cam_rotated)
    else:
        overlay_rotated = rotated_np
        print("⚠️ Could not generate EigenCAM for rotated image")
    
    # Create a combined figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'EigenCAM Visualization - Image {image_id} - Rotation: {rotation_angle}°', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Original image
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    if cam_original is not None:
        axes[0, 1].imshow(cam_original, cmap='jet')
        axes[0, 1].set_title('EigenCAM (Original)', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(overlay_original)
        axes[0, 2].set_title('Overlay (Original)', fontweight='bold')
        axes[0, 2].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, 'CAM not available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].axis('off')
        axes[0, 2].axis('off')
    
    # Row 2: Rotated image
    axes[1, 0].imshow(rotated_np)
    axes[1, 0].set_title(f'Rotated Image ({rotation_angle}°)', fontweight='bold')
    axes[1, 0].axis('off')
    
    if cam_rotated is not None:
        axes[1, 1].imshow(cam_rotated, cmap='jet')
        axes[1, 1].set_title(f'EigenCAM ({rotation_angle}°)', fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(overlay_rotated)
        axes[1, 2].set_title(f'Overlay ({rotation_angle}°)', fontweight='bold')
        axes[1, 2].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, 'CAM not available',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(output_dir, f'eigencam_{image_id}_rot{rotation_angle}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved EigenCAM visualization to: {save_path}")
    
    # Also save individual images
    cv2.imwrite(os.path.join(output_dir, f'original_{image_id}.png'), 
                cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f'rotated_{image_id}_rot{rotation_angle}.png'),
                cv2.cvtColor(rotated_np, cv2.COLOR_RGB2BGR))
    
    if cam_original is not None:
        cv2.imwrite(os.path.join(output_dir, f'eigencam_original_{image_id}.png'),
                    (cam_original * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(output_dir, f'overlay_original_{image_id}.png'),
                    cv2.cvtColor(overlay_original, cv2.COLOR_RGB2BGR))
    
    if cam_rotated is not None:
        cv2.imwrite(os.path.join(output_dir, f'eigencam_rotated_{image_id}_rot{rotation_angle}.png'),
                    (cam_rotated * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(output_dir, f'overlay_rotated_{image_id}_rot{rotation_angle}.png'),
                    cv2.cvtColor(overlay_rotated, cv2.COLOR_RGB2BGR))
    
    # Clean up
    cam_gen.clear_hooks()


def evaluate_with_eigencam(cfg, model_path, dataset_name, 
                          rotation_angles=[0, 90, 180, 270],
                          num_viz_samples=5):
    """
    Evaluate model and generate EigenCAM visualizations
    
    Args:
        cfg: Detectron2 config
        model_path: Path to model checkpoint
        dataset_name: Name of dataset to evaluate
        rotation_angles: List of rotation angles to test
        num_viz_samples: Number of samples to visualize with EigenCAM
    """
    
    # Build MARS model (same as training)
    from mask2former_mars.modeling.meta_arch.mars_mask_former_head import MaskFormerWithMARS
    model = MaskFormerWithMARS(cfg)
    model.eval()
    
    # Load checkpoint
    print(f"\n📂 Loading model from: {model_path}")
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(model_path)
    print("✅ Model loaded successfully\n")
    
    # Create EigenCAM output directory
    eigencam_dir = os.path.join(cfg.OUTPUT_DIR, "eigencam_visualizations")
    os.makedirs(eigencam_dir, exist_ok=True)
    
    all_results = {}
    
    for rotation_angle in rotation_angles:
        print(f"\n{'='*70}")
        print(f"EVALUATING WITH {rotation_angle}° ROTATION")
        print(f"{'='*70}\n")
        
        # Create data loader with rotation
        mapper = RotatedDatasetMapper(cfg, rotation_angle=rotation_angle)
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        
        # Generate EigenCAM visualizations for first N samples
        print(f"\n🎨 Generating EigenCAM visualizations for {num_viz_samples} samples...")
        
        sample_count = 0
        for idx, batch in enumerate(tqdm(data_loader, desc="Generating EigenCAMs")):
            if sample_count >= num_viz_samples:
                break
            
            for item in batch:
                if sample_count >= num_viz_samples:
                    break
                
                image_id = item.get('image_id', f'image_{idx}_{sample_count}')
                
                # Get original and rotated images
                rotated_image = item['image']
                original_image = item.get('original_image', rotated_image)
                
                # Generate and save EigenCAM
                save_eigencam_visualization(
                    model=model,
                    image_tensor=rotated_image,
                    original_image_tensor=original_image,
                    rotation_angle=rotation_angle,
                    output_dir=eigencam_dir,
                    image_id=image_id
                )
                
                sample_count += 1
        
        print(f"✅ Generated {sample_count} EigenCAM visualizations\n")
        
        # Run standard COCO evaluation
        print("Running COCO evaluation...")
        evaluator = COCOEvaluator(dataset_name, tasks=("segm",), output_dir=cfg.OUTPUT_DIR)
        results = inference_on_dataset(model, data_loader, evaluator)
        
        all_results[f"{rotation_angle}deg"] = {
            "rotation": rotation_angle,
            "ap_metrics": results
        }
        
        # Print results
        if "segm" in results:
            print(f"\nResults for {rotation_angle}° rotation:")
            print(f"  AP:    {results['segm']['AP']:.2f}")
            print(f"  AP50:  {results['segm']['AP50']:.2f}")
            print(f"  AP75:  {results['segm']['AP75']:.2f}")
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    table_data = []
    for angle in rotation_angles:
        if "segm" in all_results[f"{angle}deg"]["ap_metrics"]:
            segm = all_results[f"{angle}deg"]["ap_metrics"]["segm"]
            table_data.append([
                f"{angle}°",
                f"{segm['AP']:.2f}",
                f"{segm['AP50']:.2f}",
                f"{segm['AP75']:.2f}"
            ])
    
    print("\n" + tabulate(table_data, 
                          headers=["Rotation", "AP", "AP50", "AP75"],
                          tablefmt="grid"))
    
    # Save results
    output_file = os.path.join(cfg.OUTPUT_DIR, "evaluation_with_eigencam.json")
    with open(output_file, 'w') as f:
        json_results = {}
        for key, val in all_results.items():
            json_results[key] = {
                'rotation': val['rotation'],
                'ap_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                              for k, v in val['ap_metrics'].get('segm', {}).items()}
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"✅ EigenCAM visualizations saved to: {eigencam_dir}")
    print("="*70 + "\n")
    
    return all_results


def main():
    """Main evaluation script with EigenCAM generation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate model with EigenCAM visualizations"
    )
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", default="coco_2017_val_subset", help="Dataset name")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--num-viz-samples", type=int, default=5, 
                       help="Number of samples to visualize with EigenCAM")
    parser.add_argument("--rotations", type=int, nargs='+', default=[0, 90, 180, 270],
                       help="Rotation angles to evaluate")
    parser.add_argument(
        "--opts",
        help="Modify config options",
        default=[],
        nargs=argparse.REMAINDER,
    )
    
    args = parser.parse_args()
    
    # Setup logger
    setup_logger()
    
    # Load config
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_mars_config(cfg)
    
    from detectron2.config import CfgNode as CN
    if not hasattr(cfg.SOLVER, 'GRADIENT_ACCUMULATION_STEPS'):
        cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 16
    if not hasattr(cfg.SOLVER, 'AMP'):
        cfg.SOLVER.AMP = CN()
        cfg.SOLVER.AMP.ENABLED = True
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    
    cfg.freeze()
    
    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Register datasets
    from train_mars import register_coco_subset
    register_coco_subset(train_images=5000, val_images=1000)
    
    print("\n" + "="*70)
    print("MARS MODEL EVALUATION WITH EIGENCAM VISUALIZATION")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Rotations: {args.rotations}")
    print(f"Visualizing: {args.num_viz_samples} samples per rotation")
    print("="*70 + "\n")
    
    # Run evaluation
    results = evaluate_with_eigencam(
        cfg=cfg,
        model_path=args.model_path,
        dataset_name=args.dataset,
        rotation_angles=args.rotations,
        num_viz_samples=args.num_viz_samples
    )
    
    print("\n✅ Evaluation complete!")
    
    return results


if __name__ == "__main__":
    main()
