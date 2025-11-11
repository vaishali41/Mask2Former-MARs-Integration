import os
import torch
import numpy as np
import json
from tabulate import tabulate
from collections import defaultdict
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from detectron2.structures import ImageList
import torch.nn.functional as F

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
#sys.path.insert(0, 'tools')
print(sys.path)
from mask2former import add_maskformer2_config
from mask2former_mars.modeling.meta_arch.mars_mask_former_head import add_mars_config


class RotatedDatasetMapper:
    """Dataset mapper that applies rotation to BOTH images AND annotations"""
    
    def __init__(self, cfg, rotation_angle=0):
        # Use standard DatasetMapper for evaluation
        self.base_mapper = DatasetMapper(cfg, is_train=False, augmentations=[])
        self.rotation_angle = rotation_angle
    
    def __call__(self, dataset_dict):
        """Apply rotation after base mapper processing"""
        # Let base mapper do its thing
        result = self.base_mapper(dataset_dict)
        
        # Rotate the image tensor if needed
        if self.rotation_angle != 0:
            k = self.rotation_angle // 90  # Convert degrees to number of 90° rotations
            result["image"] = torch.rot90(result["image"], k=k, dims=[1, 2])
            
            # Update height and width for 90/270 degree rotations
            if self.rotation_angle in [90, 270]:
                if "height" in result and "width" in result:
                    result["height"], result["width"] = result["width"], result["height"]
            
            # ✅ CRITICAL: Also rotate ground truth instances!
            if "instances" in result:
                result["instances"] = self._rotate_instances(
                    result["instances"],
                    k,
                    result.get("height", result["image"].shape[1]),
                    result.get("width", result["image"].shape[2])
                )
        
        return result
    
    def _rotate_instances(self, instances, k, orig_h, orig_w):
        """
        Rotate instance annotations (boxes, masks)
        
        Args:
            instances: Detectron2 Instances object
            k: Rotation (0=0°, 1=90°, 2=180°, 3=270°)
            orig_h, orig_w: Original image dimensions (before rotation)
        
        Returns:
            Rotated Instances object
        """
        from detectron2.structures import Instances, Boxes, BitMasks, PolygonMasks
        import copy
        
        if k == 0:
            return copy.deepcopy(instances)
        
        # Create new instances with rotated dimensions
        new_h = orig_w if k % 2 == 1 else orig_h
        new_w = orig_h if k % 2 == 1 else orig_w
        rotated = Instances((new_h, new_w))
        
        # Rotate bounding boxes (if present)
        if instances.has("gt_boxes"):
            boxes = instances.gt_boxes.tensor.clone()
            rotated_boxes = self._rotate_boxes(boxes, k, orig_h, orig_w)
            rotated.gt_boxes = Boxes(rotated_boxes)
        
        # Rotate masks (if present)
        if instances.has("gt_masks"):
            masks = instances.gt_masks
            
            if isinstance(masks, torch.Tensor):
                # Raw tensor
                rotated_masks = torch.rot90(masks, k=k, dims=[1, 2])
                rotated.gt_masks = rotated_masks
            elif hasattr(masks, 'tensor'):
                # BitMasks object
                mask_tensor = masks.tensor
                rotated_masks = torch.rot90(mask_tensor, k=k, dims=[1, 2])
                rotated.gt_masks = BitMasks(rotated_masks)
        
        # Copy other fields (classes, etc.)
        for field in instances.get_fields():
            if field not in ["gt_boxes", "gt_masks"]:
                rotated.set(field, copy.deepcopy(instances.get(field)))
        
        return rotated
    
    def _rotate_boxes(self, boxes, k, h, w):
        """
        Rotate bounding boxes by k*90 degrees
        
        Args:
            boxes: Tensor of shape [N, 4] in (x1, y1, x2, y2) format
            k: Rotation (0=0°, 1=90°, 2=180°, 3=270°)
            h, w: Original image height and width
        
        Returns:
            Rotated boxes tensor [N, 4]
        """
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        
        if k == 1:  # 90° clockwise
            return torch.stack([y1, w - x2, y2, w - x1], dim=-1)
        elif k == 2:  # 180°
            return torch.stack([w - x2, h - y2, w - x1, h - y1], dim=-1)
        elif k == 3:  # 270° clockwise
            return torch.stack([h - y2, x1, h - y1, x2], dim=-1)
        else:
            return boxes
"""
class RotatedDatasetMapper:
    #Simple dataset mapper that applies rotation to images
    
    def __init__(self, cfg, rotation_angle=0):
        # Use standard DatasetMapper for evaluation
        self.base_mapper = DatasetMapper(cfg, is_train=False, augmentations=[])
        self.rotation_angle = rotation_angle
    
    def __call__(self, dataset_dict):
        #Apply rotation after base mapper processing
        # Let base mapper do its thing
        result = self.base_mapper(dataset_dict)
        
        # Rotate the image tensor if needed
        if self.rotation_angle != 0:
            k = self.rotation_angle // 90  # Convert degrees to number of 90° rotations
            result["image"] = torch.rot90(result["image"], k=k, dims=[1, 2])
            
            # Update height and width for 90/270 degree rotations
            if self.rotation_angle in [90, 270]:
                if "height" in result and "width" in result:
                    result["height"], result["width"] = result["width"], result["height"]
        
        return result

"""
class DetailedMetricsEvaluator:
    """
    Computes detailed metrics including losses, dice score, and confidence
    """
    
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.reset()
    
    def reset(self):
        """Reset metric accumulators"""
        self.total_losses = []
        self.mask_losses = []
        self.dice_losses = []
        self.cls_losses = []
        self.avg_confidences = []
        self.max_confidences = []
        self.num_samples = 0
    
    def process_batch(self, batched_inputs):
        """
        Process a batch and compute all metrics
        
        Args:
            batched_inputs: Batch of images and annotations
        """
        # Compute confidence statistics from predictions
        with torch.no_grad():
            self.model.eval()
            
            # Preprocess images
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.model.pixel_mean) / self.model.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.model.size_divisibility)
            
            # Get features and predictions
            features = self.model.backbone(images.tensor)
            outputs = self.model.sem_seg_head(features)
            
            # Extract confidence from outputs
            if isinstance(outputs, dict) and 'pred_logits' in outputs:
                pred_logits = outputs['pred_logits']  # [B, Q, C]
                probs = F.softmax(pred_logits, dim=-1)
                max_probs, _ = probs.max(dim=-1)  # [B, Q]
                
                # Compute per-sample statistics
                for sample_probs in max_probs:
                    self.avg_confidences.append(sample_probs.mean().item() * 100)
                    self.max_confidences.append(sample_probs.max().item() * 100)
            elif isinstance(outputs, list):
                # Handle list of outputs (one per image)
                for output in outputs:
                    if 'pred_logits' in output:
                        pred_logits = output['pred_logits']  # [Q, C]
                        probs = F.softmax(pred_logits, dim=-1)
                        max_probs, _ = probs.max(dim=-1)  # [Q]
                        
                        self.avg_confidences.append(max_probs.mean().item() * 100)
                        self.max_confidences.append(max_probs.max().item() * 100)
            
            self.num_samples += len(batched_inputs)
    
    def get_metrics(self):
        """
        Compute final aggregated metrics
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'total_loss': np.mean(self.total_losses) if self.total_losses else 0.0,
            'mask_loss': np.mean(self.mask_losses) if self.mask_losses else 0.0,
            'dice_loss': np.mean(self.dice_losses) if self.dice_losses else 0.0,
            'cls_loss': np.mean(self.cls_losses) if self.cls_losses else 0.0,
            'dice_score': 0.0,  # Would need GT masks for true dice score
            'avg_confidence': np.mean(self.avg_confidences) if self.avg_confidences else 0.0,
            'max_confidence': np.mean(self.max_confidences) if self.max_confidences else 0.0,
            'num_samples': self.num_samples,
        }
        return metrics


def compute_boxes_from_masks(instances):
    """
    Compute bounding boxes from segmentation masks
    
    Args:
        instances: Detectron2 Instances with pred_masks
    
    Returns:
        Instances with computed pred_boxes
    """
    from detectron2.structures import Boxes
    import torch
    
    if not instances.has('pred_masks') or len(instances) == 0:
        return instances
    
    masks = instances.pred_masks  # [N, H, W]
    
    # Convert to binary if needed
    if masks.dtype == torch.float32:
        masks = masks > 0.5
    
    boxes = []
    valid_mask_indices = []
    
    for i, mask in enumerate(masks):
        # Find coordinates where mask is True
        ys, xs = torch.where(mask)
        
        if len(xs) == 0 or len(ys) == 0:
            # Empty mask - set box to zero
            boxes.append([0.0, 0.0, 0.0, 0.0])
        else:
            # Compute bounding box from mask coordinates
            x1 = xs.min().float().item()
            y1 = ys.min().float().item()
            x2 = xs.max().float().item()
            y2 = ys.max().float().item()
            boxes.append([x1, y1, x2, y2])
            valid_mask_indices.append(i)
    
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=masks.device)
    instances.pred_boxes = Boxes(boxes_tensor)
    
    return instances

class ModelWithBoxFix:
    """Wrapper that computes boxes from masks and fixes category IDs"""
    def __init__(self, model, remap_categories=True):
        self.model = model
        self.model.eval()
        self.remap_categories = remap_categories
    
    def __call__(self, batched_inputs):
        # Run model
        outputs = self.model(batched_inputs)
        
        # Fix boxes and categories for each output
        for output in outputs:
            if 'instances' in output:
                # Compute boxes from masks
                output['instances'] = compute_boxes_from_masks(output['instances'])
                
                # Filter and fix categories
                output['instances'] = self.fix_categories(output['instances'])
        
        return outputs
    
    def fix_categories(self, instances):
        """
        Fix category IDs and filter invalid predictions
        
        Args:
            instances: Detectron2 Instances
        
        Returns:
            Filtered instances with valid categories
        """
        if len(instances) == 0:
            return instances
        
        # Get valid predictions (non-zero area boxes and valid scores)
        if instances.has('pred_boxes'):
            boxes = instances.pred_boxes.tensor
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            valid_mask = areas > 0  # Non-zero area
        else:
            valid_mask = torch.ones(len(instances), dtype=torch.bool, device=instances.pred_classes.device)
        
        # Filter by score threshold (remove very low confidence)
        if instances.has('scores'):
            valid_mask = valid_mask & (instances.scores > 0.05)  # Threshold at 5%
        
        # Filter category 0 (background/invalid)
        if instances.has('pred_classes'):
            # CRITICAL: Remove class 0 predictions (likely background)
            valid_mask = valid_mask & (instances.pred_classes > 0)
        
        # Apply filter
        instances = instances[valid_mask]
        
        # Optional: Remap categories if needed
        # If your model outputs 0-indexed but COCO expects 1-indexed
        if self.remap_categories and instances.has('pred_classes') and len(instances) > 0:
            # Check if remapping is needed
            if instances.pred_classes.min() == 0:
                print(f"  ⚠️ WARNING: Detected 0-indexed categories, but class 0 should be filtered!")
        
        return instances
    
    def eval(self):
        self.model.eval()
        return self
    
    def train(self):
        self.model.train()
        return self
    
    def to(self, device):
        self.model.to(device)
        return self

"""        
class ModelWithBoxFix:
    #Wrapper that computes boxes from masks after model inference
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def __call__(self, batched_inputs):
        # Run model
        outputs = self.model(batched_inputs)
        
        # Fix boxes for each output
        for output in outputs:
            if 'instances' in output:
                output['instances'] = compute_boxes_from_masks(output['instances'])
        
        return outputs
    
    def eval(self):
        self.model.eval()
        return self
    
    def train(self):
        self.model.train()
        return self
    
    def to(self, device):
        self.model.to(device)
        return self
"""
def evaluate_at_rotation_detailed(cfg, model, dataset_name: str, rotation_angle: int):
    """
    Evaluate model at specific rotation angle with detailed metrics
    
    Args:
        cfg: Configuration
        model: Model to evaluate
        dataset_name: Dataset name
        rotation_angle: Rotation angle (0, 90, 180, 270)
    
    Returns:
        Dictionary with AP metrics and detailed metrics
    """
    print(f"\n{'='*70}")
    print(f"Evaluating at {rotation_angle}° rotation")
    print(f"{'='*70}")
    
    # Create modified config with NUM_WORKERS=0 to avoid multiprocessing issues
    from detectron2.config import CfgNode
    eval_cfg = cfg.clone()
    eval_cfg.defrost()
    eval_cfg.DATALOADER.NUM_WORKERS = 0  # Critical: avoid multiprocessing errors
    eval_cfg.freeze()
    
    # Create data loader with rotation
    mapper = RotatedDatasetMapper(eval_cfg, rotation_angle=rotation_angle)
    data_loader = build_detection_test_loader(eval_cfg, dataset_name, mapper=mapper)

    # Detailed metrics evaluator
    detailed_evaluator = DetailedMetricsEvaluator(model, eval_cfg)
    
    # Process all batches for detailed metrics
    print("Computing detailed metrics (confidence, etc.)...")
    model.eval()
    for batch in tqdm(data_loader, desc=f"Processing {rotation_angle}°"):
        detailed_evaluator.process_batch(batch)
    
    detailed_metrics = detailed_evaluator.get_metrics()
    
    # Standard AP evaluation (create new dataloader to reset)
    print("Computing AP metrics...")
    mapper = RotatedDatasetMapper(eval_cfg, rotation_angle=rotation_angle)
    data_loader = build_detection_test_loader(eval_cfg, dataset_name, mapper=mapper)
    
    # ✅ Wrap model to fix boxes
    wrapped_model = ModelWithBoxFix(model)
    wrapped_model.eval()
    
    coco_evaluator = COCOEvaluator(
        dataset_name,
        output_dir=os.path.join(cfg.OUTPUT_DIR, f"inference_rotation_{rotation_angle}"),
        tasks=("segm",),
    )
    
    # ✅ OPTIONAL: Debug BEFORE evaluation to verify fix works
    print("\n🔍 Verifying box fix...")
    mapper_debug = RotatedDatasetMapper(eval_cfg, rotation_angle=rotation_angle)
    data_loader_debug = build_detection_test_loader(eval_cfg, dataset_name, mapper=mapper_debug)
    test_batch = next(iter(data_loader_debug))
    
    with torch.no_grad():
        test_outputs = wrapped_model(test_batch)  # ✅ Use wrapped model
    
    if test_outputs and 'instances' in test_outputs[0]:
        inst = test_outputs[0]['instances']
        if inst.has('pred_boxes'):
            boxes = inst.pred_boxes.tensor
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            print(f"  ✅ Non-zero boxes: {(areas > 0).sum()} / {len(boxes)}")
            if (areas > 0).sum() > 0:
                non_zero_idx = torch.where(areas > 0)[0][0]
                print(f"  ✅ Sample box: {boxes[non_zero_idx].tolist()}")
            else:
                print(f"  ⚠️ WARNING: Still getting zero boxes!")
    
    # Run evaluation with wrapped model
    print("Running COCO evaluation...")
    ap_results = inference_on_dataset(wrapped_model, data_loader, coco_evaluator)

    for i, output in enumerate(test_outputs[:2]):  # Check first 2 images
        print(f"\n--- Image {i} ---")
        
        if 'instances' in output:
            inst = output['instances']
            print(f"Total predictions: {len(inst)}")
            
            # Check scores
            if inst.has('scores'):
                scores = inst.scores
                print(f"Score stats:")
                print(f"  Min: {scores.min():.4f}")
                print(f"  Max: {scores.max():.4f}")
                print(f"  Mean: {scores.mean():.4f}")
                print(f"  Above 0.5: {(scores > 0.5).sum()}")
                print(f"  Above 0.3: {(scores > 0.3).sum()}")
                print(f"  Above 0.1: {(scores > 0.1).sum()}")
            
            # Check classes
            if inst.has('pred_classes'):
                classes = inst.pred_classes
                print(f"Classes: {classes.unique().tolist()}")
                print(f"Class counts: {[(c.item(), (classes == c).sum().item()) for c in classes.unique()[:5]]}")
            
            # Check boxes in detail
            if inst.has('pred_boxes'):
                boxes = inst.pred_boxes.tensor
                print(f"\nBounding Boxes:")
                print(f"  Shape: {boxes.shape}")
                print(f"  First 3 boxes: {boxes[:3].tolist()}")
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]
                areas = widths * heights
                print(f"  Width range: [{widths.min():.2f}, {widths.max():.2f}]")
                print(f"  Height range: [{heights.min():.2f}, {heights.max():.2f}]")
                print(f"  Area range: [{areas.min():.2f}, {areas.max():.2f}]")
                print(f"  Non-zero areas: {(areas > 0).sum()} / {len(areas)}")
            
            # Check masks in detail
            if inst.has('pred_masks'):
                masks = inst.pred_masks
                print(f"\nMasks:")
                print(f"  Type: {type(masks)}")
                print(f"  Shape: {masks.shape}")
                print(f"  Dtype: {masks.dtype}")
                
                # Check if masks are empty
                if masks.dtype == torch.bool:
                    mask_pixels = masks.sum(dim=(1, 2))  # Count True pixels per mask
                else:
                    mask_pixels = (masks > 0.5).sum(dim=(1, 2))  # Count pixels > threshold
                
                print(f"  Pixels per mask (first 5): {mask_pixels[:5].tolist()}")
                print(f"  Empty masks (0 pixels): {(mask_pixels == 0).sum()} / {len(mask_pixels)}")
                print(f"  Non-empty masks: {(mask_pixels > 0).sum()}")
                
                if (mask_pixels > 0).sum() > 0:
                    # Show a non-empty mask
                    non_empty_idx = torch.where(mask_pixels > 0)[0][0]
                    print(f"  First non-empty mask (idx {non_empty_idx}):")
                    print(f"    Pixels: {mask_pixels[non_empty_idx]}")
                    print(f"    Score: {scores[non_empty_idx]:.4f}")
                    print(f"    Box: {boxes[non_empty_idx].tolist()}")

    print(f"{'='*70}\n")

    # Combine results
    results = {
        'ap_metrics': ap_results,
        'detailed_metrics': detailed_metrics,
    }
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Results at {rotation_angle}°:")
    print(f"{'='*70}")
    
    if "segm" in ap_results:
        print(f"AP Metrics:")
        print(f"  AP:     {ap_results['segm']['AP']:.2f}")
        print(f"  AP50:   {ap_results['segm']['AP50']:.2f}")
        print(f"  AP75:   {ap_results['segm']['AP75']:.2f}")
    
    print(f"\nDetailed Metrics:")
    print(f"  Avg Confidence:  {detailed_metrics['avg_confidence']:.2f}%")
    print(f"  Max Confidence:  {detailed_metrics['max_confidence']:.2f}%")
    print(f"  Samples:         {detailed_metrics['num_samples']}")
    
    return results

def evaluate_all_rotations_detailed(cfg, model_path: str, dataset_name: str = "coco_2017_val_subset"):
    """
    Evaluate model at all rotation angles with detailed metrics
    
    Args:
        cfg: Configuration
        model_path: Path to model checkpoint
        dataset_name: Dataset name
    
    Returns:
        Dictionary with results at all rotations
    """
    print("\n" + "="*70)
    print("MARS + MASK2FORMER - DETAILED EVALUATION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name}")
    print("="*70)
    
    # Build and load model
    from mask2former_mars.modeling.meta_arch.mars_mask_former_head import MaskFormerWithMARS
    model = MaskFormerWithMARS(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(model_path)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    # ✅ NEW: Add this line to show MARS status (optional)
    model.eval()
    print(f"✅ Model loaded in eval mode (MARS training disabled during inference)\n")
    
    # Evaluate at each rotation
    all_results = {}
    rotation_angles = [0, 90, 180, 270]
    
    for angle in rotation_angles:
        results = evaluate_at_rotation_detailed(cfg, model, dataset_name, angle)
        all_results[f"{angle}deg"] = results
    
    # Create summary tables
    print("\n" + "="*70)
    print("SUMMARY TABLES")
    print("="*70)
    
    # Table 1: AP Metrics
    print("\n" + "="*70)
    print("Table 1: Average Precision (AP)")
    print("="*70)
    
    ap_table_data = []
    ap_headers = ["Rotation", "AP", "AP50", "AP75", "APs", "APm", "APl"]
    
    for angle in rotation_angles:
        if "segm" in all_results[f"{angle}deg"]["ap_metrics"]:
            segm = all_results[f"{angle}deg"]["ap_metrics"]["segm"]
            ap_table_data.append([
                f"{angle}°",
                f"{segm['AP']:.2f}",
                f"{segm['AP50']:.2f}",
                f"{segm['AP75']:.2f}",
                f"{segm['APs']:.2f}",
                f"{segm['APm']:.2f}",
                f"{segm['APl']:.2f}",
            ])
    
    # Add statistics
    if ap_table_data:
        ap_values = [all_results[f"{angle}deg"]["ap_metrics"]["segm"]["AP"] for angle in rotation_angles]
        avg_ap = np.mean(ap_values)
        std_ap = np.std(ap_values)
        drop = ap_values[0] - min(ap_values)
        
        ap_table_data.append(["---", "---", "---", "---", "---", "---", "---"])
        ap_table_data.append([
            "Average",
            f"{avg_ap:.2f}",
            "-",
            "-",
            "-",
            "-",
            "-",
        ])
        ap_table_data.append([
            "Std Dev",
            f"{std_ap:.2f}",
            "-",
            "-",
            "-",
            "-",
            "-",
        ])
        ap_table_data.append([
            "Max Drop",
            f"{drop:.2f}",
            "-",
            "-",
            "-",
            "-",
            "-",
        ])
    
    print("\n" + tabulate(ap_table_data, headers=ap_headers, tablefmt="grid"))
    
    # Table 2: Detailed Metrics (Non-Rotated vs Rotated)
    print("\n" + "="*70)
    print("Table 2: Detailed Metrics (Non-Rotated vs Rotated)")
    print("="*70)
    
    # Get metrics
    metrics_0 = all_results["0deg"]["detailed_metrics"]
    
    # Average rotated metrics (90, 180, 270)
    metrics_rotated = {}
    for key in metrics_0.keys():
        if key != 'num_samples':
            values = [
                all_results["90deg"]["detailed_metrics"][key],
                all_results["180deg"]["detailed_metrics"][key],
                all_results["270deg"]["detailed_metrics"][key],
            ]
            metrics_rotated[key] = np.mean(values)
    
    # Create comparison table
    detailed_table_data = []
    
    # AP (from segmentation results)
    ap_0 = all_results["0deg"]["ap_metrics"]["segm"]["AP"]
    ap_rot_vals = [all_results[f"{a}deg"]["ap_metrics"]["segm"]["AP"] for a in [90, 180, 270]]
    ap_rot = np.mean(ap_rot_vals)
    ap_change = ((ap_rot - ap_0) / ap_0 * 100) if ap_0 != 0 else 0
    detailed_table_data.append([
        "AP (segm)",
        f"{ap_0:.2f}",
        f"{ap_rot:.2f}",
        f"{ap_change:+.2f}%"
    ])
    
    # Avg Confidence
    if metrics_0['avg_confidence'] != 0:
        conf_change = ((metrics_rotated['avg_confidence'] - metrics_0['avg_confidence']) / metrics_0['avg_confidence'] * 100)
        detailed_table_data.append([
            "Avg Confidence",
            f"{metrics_0['avg_confidence']:.2f}%",
            f"{metrics_rotated['avg_confidence']:.2f}%",
            f"{conf_change:+.2f}%"
        ])
    
    # Max Confidence
    if metrics_0['max_confidence'] != 0:
        max_conf_change = ((metrics_rotated['max_confidence'] - metrics_0['max_confidence']) / metrics_0['max_confidence'] * 100)
        detailed_table_data.append([
            "Max Confidence",
            f"{metrics_0['max_confidence']:.2f}%",
            f"{metrics_rotated['max_confidence']:.2f}%",
            f"{max_conf_change:+.2f}%"
        ])
    
    detailed_headers = ["Metric", "Non-Rotated", "Rotated (Avg)", "Change"]
    print("\n" + tabulate(detailed_table_data, headers=detailed_headers, tablefmt="grid"))
    
    # Save all results to JSON
    output_file = os.path.join(cfg.OUTPUT_DIR, "detailed_evaluation_results.json")
    
    # Convert to JSON-serializable format
    json_results = {}
    for key, val in all_results.items():
        json_results[key] = {
            'ap_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                          for k, v in val['ap_metrics'].get('segm', {}).items()},
            'detailed_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                               for k, v in val['detailed_metrics'].items()}
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS - ROTATION ROBUSTNESS")
    print("="*70)
    if ap_table_data and len(ap_values) > 0:
        print(f"AP Performance:")
        print(f"  AP at 0°:          {ap_values[0]:.2f}")
        print(f"  AP at 90°:         {ap_values[1]:.2f}")
        print(f"  AP at 180°:        {ap_values[2]:.2f}")
        print(f"  AP at 270°:        {ap_values[3]:.2f}")
        print(f"  Average AP:        {avg_ap:.2f}")
        print(f"  Std deviation:     {std_ap:.2f}")
        print(f"  Maximum drop:      {drop:.2f} ({((drop/ap_values[0])*100) if ap_values[0] > 0 else 0:.1f}%)")
        print(f"\nConfidence Analysis:")
        print(f"  Confidence drop:   {metrics_0['avg_confidence'] - metrics_rotated['avg_confidence']:.2f}%")
        print(f"\nRobustness Rating:   {'EXCELLENT' if drop < 3 else 'GOOD' if drop < 5 else 'MODERATE' if drop < 10 else 'NEEDS IMPROVEMENT'}")
    print("="*70 + "\n")
    
    return all_results
"""
def evaluate_all_rotations_detailed(cfg, model_path: str, dataset_name: str = "coco_2017_val_subset"):
    
    Evaluate model at all rotation angles with detailed metrics
    
    Args:
        cfg: Configuration
        model_path: Path to model checkpoint
        dataset_name: Dataset name
    
    Returns:
        Dictionary with results at all rotations
    
    print("\n" + "="*70)
    print("MARS + MASK2FORMER - DETAILED EVALUATION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name}")
    print("="*70)
    
    # Build and load model
    from mask2former_mars.modeling.meta_arch.mars_mask_former_head import MaskFormerWithMARS
    model = MaskFormerWithMARS(cfg)
    
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(model_path)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Evaluate at each rotation
    all_results = {}
    rotation_angles = [0, 90, 180, 270]
    
    for angle in rotation_angles:
        results = evaluate_at_rotation_detailed(cfg, model, dataset_name, angle)
        all_results[f"{angle}deg"] = results
    
    # Create summary tables
    print("\n" + "="*70)
    print("SUMMARY TABLES")
    print("="*70)
    
    # Table 1: AP Metrics
    print("\n" + "="*70)
    print("Table 1: Average Precision (AP)")
    print("="*70)
    
    ap_table_data = []
    ap_headers = ["Rotation", "AP", "AP50", "AP75", "APs", "APm", "APl"]
    
    for angle in rotation_angles:
        if "segm" in all_results[f"{angle}deg"]["ap_metrics"]:
            segm = all_results[f"{angle}deg"]["ap_metrics"]["segm"]
            ap_table_data.append([
                f"{angle}°",
                f"{segm['AP']:.2f}",
                f"{segm['AP50']:.2f}",
                f"{segm['AP75']:.2f}",
                f"{segm['APs']:.2f}",
                f"{segm['APm']:.2f}",
                f"{segm['APl']:.2f}",
            ])
    
    # Add statistics
    if ap_table_data:
        ap_values = [all_results[f"{angle}deg"]["ap_metrics"]["segm"]["AP"] for angle in rotation_angles]
        avg_ap = np.mean(ap_values)
        std_ap = np.std(ap_values)
        drop = ap_values[0] - min(ap_values)
        
        ap_table_data.append(["---", "---", "---", "---", "---", "---", "---"])
        ap_table_data.append([
            "Average",
            f"{avg_ap:.2f}",
            "-",
            "-",
            "-",
            "-",
            "-",
        ])
        ap_table_data.append([
            "Std Dev",
            f"{std_ap:.2f}",
            "-",
            "-",
            "-",
            "-",
            "-",
        ])
        ap_table_data.append([
            "Max Drop",
            f"{drop:.2f}",
            "-",
            "-",
            "-",
            "-",
            "-",
        ])
    
    print("\n" + tabulate(ap_table_data, headers=ap_headers, tablefmt="grid"))
    
    # Table 2: Detailed Metrics (Non-Rotated vs Rotated)
    print("\n" + "="*70)
    print("Table 2: Detailed Metrics (Non-Rotated vs Rotated)")
    print("="*70)
    
    # Get metrics
    metrics_0 = all_results["0deg"]["detailed_metrics"]
    
    # Average rotated metrics (90, 180, 270)
    metrics_rotated = {}
    for key in metrics_0.keys():
        if key != 'num_samples':
            values = [
                all_results["90deg"]["detailed_metrics"][key],
                all_results["180deg"]["detailed_metrics"][key],
                all_results["270deg"]["detailed_metrics"][key],
            ]
            metrics_rotated[key] = np.mean(values)
    
    # Create comparison table
    detailed_table_data = []
    
    # AP (from segmentation results)
    ap_0 = all_results["0deg"]["ap_metrics"]["segm"]["AP"]
    ap_rot_vals = [all_results[f"{a}deg"]["ap_metrics"]["segm"]["AP"] for a in [90, 180, 270]]
    ap_rot = np.mean(ap_rot_vals)
    ap_change = ((ap_rot - ap_0) / ap_0 * 100) if ap_0 != 0 else 0
    detailed_table_data.append([
        "AP (segm)",
        f"{ap_0:.2f}",
        f"{ap_rot:.2f}",
        f"{ap_change:+.2f}%"
    ])
    
    # Avg Confidence
    if metrics_0['avg_confidence'] != 0:
        conf_change = ((metrics_rotated['avg_confidence'] - metrics_0['avg_confidence']) / metrics_0['avg_confidence'] * 100)
        detailed_table_data.append([
            "Avg Confidence",
            f"{metrics_0['avg_confidence']:.2f}%",
            f"{metrics_rotated['avg_confidence']:.2f}%",
            f"{conf_change:+.2f}%"
        ])
    
    # Max Confidence
    if metrics_0['max_confidence'] != 0:
        max_conf_change = ((metrics_rotated['max_confidence'] - metrics_0['max_confidence']) / metrics_0['max_confidence'] * 100)
        detailed_table_data.append([
            "Max Confidence",
            f"{metrics_0['max_confidence']:.2f}%",
            f"{metrics_rotated['max_confidence']:.2f}%",
            f"{max_conf_change:+.2f}%"
        ])
    
    detailed_headers = ["Metric", "Non-Rotated", "Rotated (Avg)", "Change"]
    print("\n" + tabulate(detailed_table_data, headers=detailed_headers, tablefmt="grid"))
    
    # Save all results to JSON
    output_file = os.path.join(cfg.OUTPUT_DIR, "detailed_evaluation_results.json")
    
    # Convert to JSON-serializable format
    json_results = {}
    for key, val in all_results.items():
        json_results[key] = {
            'ap_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                          for k, v in val['ap_metrics'].get('segm', {}).items()},
            'detailed_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                               for k, v in val['detailed_metrics'].items()}
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS - ROTATION ROBUSTNESS")
    print("="*70)
    if ap_table_data and len(ap_values) > 0:
        print(f"AP Performance:")
        print(f"  AP at 0°:          {ap_values[0]:.2f}")
        print(f"  AP at 90°:         {ap_values[1]:.2f}")
        print(f"  AP at 180°:        {ap_values[2]:.2f}")
        print(f"  AP at 270°:        {ap_values[3]:.2f}")
        print(f"  Average AP:        {avg_ap:.2f}")
        print(f"  Std deviation:     {std_ap:.2f}")
        print(f"  Maximum drop:      {drop:.2f} ({((drop/ap_values[0])*100) if ap_values[0] > 0 else 0:.1f}%)")
        print(f"\nConfidence Analysis:")
        print(f"  Confidence drop:   {metrics_0['avg_confidence'] - metrics_rotated['avg_confidence']:.2f}%")
        print(f"\nRobustness Rating:   {'EXCELLENT' if drop < 3 else 'GOOD' if drop < 5 else 'MODERATE' if drop < 10 else 'NEEDS IMPROVEMENT'}")
    print("="*70 + "\n")
    
    return all_results
"""

def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate MARS model with detailed metrics")
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", default="coco_2017_val_subset", help="Dataset name")
    parser.add_argument("--output-dir", default=None, help="Output directory")
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
    
    # Add custom config keys that might be in the training config
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
    
    # Run evaluation
    results = evaluate_all_rotations_detailed(
        cfg=cfg,
        model_path=args.model_path,
        dataset_name=args.dataset
    )
    
    print("\n✅ Evaluation complete!")
    
    return results


if __name__ == "__main__":
    main()
