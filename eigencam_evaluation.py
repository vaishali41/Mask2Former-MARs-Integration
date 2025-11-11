#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EigenCAM Evaluation for Mask2Former + MARs
Generates attention visualizations using EigenCAM method for interpretability analysis.

Usage:
    python eigencam_evaluation.py --config mars_10class_200ep_gem.yaml \
                                  --weights output_mars_10class_200ep_gem/model_final.pth \
                                  --input datasets/coco/val2017 \
                                  --output eigencam_results \
                                  --num-images 50
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor
from mask2former import add_maskformer2_config
from mask2former_mars.modeling.meta_arch.mars_mask_former_head_gem import MaskFormerWithMARS


class EigenCAMExtractor:
    """
    EigenCAM: Uses principal component analysis (PCA) on attention maps
    to extract the most significant attention patterns.
    """
    
    def __init__(self, model, layer_names=None):
        """
        Args:
            model: The neural network model
            layer_names: List of layer names to extract attention from
        """
        self.model = model
        self.layer_names = layer_names or []
        self.attention_maps = {}
        self.hooks = []
        
        # Register hooks to capture attention weights
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights"""
        
        def get_attention_hook(name):
            def hook(module, input, output):
                # Store attention weights if available
                if isinstance(output, tuple) and len(output) > 1:
                    # Transformer attention: (output, attention_weights)
                    self.attention_maps[name] = output[1].detach()
                elif hasattr(module, 'attention_weights'):
                    self.attention_maps[name] = module.attention_weights.detach()
            return hook
        
        # Hook into transformer decoder layers
        if hasattr(self.model, 'sem_seg_head'):
            predictor = self.model.sem_seg_head.predictor
            if hasattr(predictor, 'transformer_cross_attention_layers'):
                for i, layer in enumerate(predictor.transformer_cross_attention_layers):
                    name = f"decoder_layer_{i}"
                    self.hooks.append(
                        layer.register_forward_hook(get_attention_hook(name))
                    )
    
    def compute_eigencam(self, attention_maps, top_k=3):
        """
        Compute EigenCAM from attention maps using PCA.
        
        Args:
            attention_maps: Tensor of shape (B, H, W, W) - attention weights
            top_k: Number of principal components to use
            
        Returns:
            eigencam: Tensor of shape (B, H, W) - weighted attention map
        """
        B, num_heads, H, W = attention_maps.shape
        
        # Reshape for PCA: (B, num_heads, H*W)
        attn_flat = attention_maps.reshape(B, num_heads, -1)
        
        # Compute covariance matrix
        attn_mean = attn_flat.mean(dim=2, keepdim=True)
        attn_centered = attn_flat - attn_mean
        
        # SVD for PCA
        U, S, V = torch.svd(attn_centered)
        
        # Take top-k principal components
        principal_components = V[:, :, :top_k]  # (B, H*W, top_k)
        
        # Project attention onto principal components
        eigencam = torch.matmul(attn_flat.transpose(1, 2), principal_components[:, :top_k, :])
        eigencam = eigencam.abs().sum(dim=-1)  # (B, H*W)
        
        # Reshape back to spatial dimensions
        eigencam = eigencam.reshape(B, int(np.sqrt(H)), int(np.sqrt(H)))
        
        # Normalize to [0, 1]
        eigencam = eigencam - eigencam.min()
        eigencam = eigencam / (eigencam.max() + 1e-8)
        
        return eigencam
    
    def __call__(self, image):
        """
        Generate EigenCAM for an image.
        
        Args:
            image: Input image tensor
            
        Returns:
            eigencam_maps: Dictionary of EigenCAM maps per layer
        """
        self.attention_maps = {}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model([{"image": image}])
        
        # Compute EigenCAM for each layer
        eigencam_maps = {}
        for layer_name, attn in self.attention_maps.items():
            if attn.dim() == 4:  # (B, num_heads, H, W)
                eigencam = self.compute_eigencam(attn)
                eigencam_maps[layer_name] = eigencam
        
        return eigencam_maps, outputs
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()


def setup_cfg(args):
    """Setup config and add MARs configuration"""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Add MARs config
    from detectron2.config import CfgNode as CN
    cfg.MODEL.MARS = CN()
    cfg.MODEL.MARS.ENABLED = True
    cfg.MODEL.MARS.WEIGHT = 1.0
    cfg.MODEL.MARS.LOSS_TYPE = "cosine"
    cfg.MODEL.MARS.USE_GEM = True
    cfg.MODEL.MARS.GEM_INIT_P = 3.0
    cfg.MODEL.MARS.GEM_MIN_P = 1.0
    cfg.MODEL.MARS.GEM_MAX_P = 6.0
    cfg.MODEL.MARS.WARMUP_ITERS = 1000
    
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.freeze()
    
    return cfg


def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on image.
    
    Args:
        image: Original image (H, W, 3)
        heatmap: Attention heatmap (H, W)
        alpha: Blending factor
        colormap: OpenCV colormap
    
    Returns:
        overlayed: Image with heatmap overlay
    """
    # Resize heatmap to image size
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Normalize heatmap
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlayed = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    
    return overlayed


def visualize_eigencam(image, eigencam_maps, output_path, class_names):
    """
    Create comprehensive EigenCAM visualization.
    
    Args:
        image: Original image (H, W, 3)
        eigencam_maps: Dictionary of EigenCAM maps
        output_path: Path to save visualization
        class_names: List of class names
    """
    num_layers = len(eigencam_maps)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, (num_layers + 1) // 2 + 1, figsize=(20, 10))
    axes = axes.flatten()
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Show EigenCAM for each layer
    for idx, (layer_name, eigencam) in enumerate(eigencam_maps.items(), 1):
        eigencam_np = eigencam[0].cpu().numpy()
        overlay = overlay_heatmap(image, eigencam_np)
        
        axes[idx].imshow(overlay)
        axes[idx].set_title(f"EigenCAM: {layer_name}")
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(eigencam_maps) + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_eigencam(args):
    """Main evaluation function"""
    
    # Setup
    cfg = setup_cfg(args)
    
    # Load model
    print("Loading model...")
    model = MaskFormerWithMARS(cfg)
    model.eval()
    
    # Load checkpoint
    checkpoint = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Initialize EigenCAM extractor
    eigencam_extractor = EigenCAMExtractor(model)
    
    # Get image files
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = sorted(input_path.glob("*.jpg"))[:args.num_images]
    
    print(f"Processing {len(image_files)} images...")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    
    for img_path in tqdm(image_files):
        # Read image
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare input
        height, width = image.shape[:2]
        image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float()
        
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        # Get EigenCAM
        eigencam_maps, predictions = eigencam_extractor(image_tensor)
        
        # Visualize
        output_path = output_dir / f"{img_path.stem}_eigencam.png"
        visualize_eigencam(image_rgb, eigencam_maps, output_path, class_names)
        
        # Save predictions visualization
        v = Visualizer(image_rgb, 
                      MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                      instance_mode=ColorMode.IMAGE)
        
        if predictions:
            pred = predictions[0]
            if "instances" in pred:
                instances = pred["instances"].to("cpu")
                vis = v.draw_instance_predictions(instances)
                vis_output = output_dir / f"{img_path.stem}_predictions.png"
                vis.save(str(vis_output))
    
    # Cleanup
    eigencam_extractor.remove_hooks()
    
    print(f"\n✅ EigenCAM evaluation complete!")
    print(f"   Results saved to: {output_dir}")
    print(f"   Processed {len(image_files)} images")


def main():
    parser = argparse.ArgumentParser(description="EigenCAM Evaluation for Mask2Former + MARs")
    
    parser.add_argument("--config", required=True, 
                       help="Path to config file")
    parser.add_argument("--weights", required=True,
                       help="Path to model weights")
    parser.add_argument("--input", required=True,
                       help="Path to input image or directory")
    parser.add_argument("--output", default="eigencam_results",
                       help="Output directory for visualizations")
    parser.add_argument("--num-images", type=int, default=50,
                       help="Number of images to process (if input is directory)")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                       help="Confidence threshold for predictions")
    
    args = parser.parse_args()
    
    evaluate_eigencam(args)


if __name__ == "__main__":
    main()
