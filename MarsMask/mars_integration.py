"""
Proper MARS Integration into Mask2Former
Adds multi-view attention regularization to improve rotation robustness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import copy

from detectron2.modeling import META_ARCH_REGISTRY, build_model
from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead
from mask2former import MaskFormer


class MultiViewAttentionRegularization(nn.Module):
    """
    MARS-style attention regularization
    Encourages consistent attention patterns across augmented views
    """
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Simple consistency regularization
        self.attention_proj = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, features_view1: torch.Tensor, features_view2: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss between two views
        
        Args:
            features_view1: Features from view 1 [B, C, H, W]
            features_view2: Features from view 2 [B, C, H, W]
        
        Returns:
            consistency_loss: Scalar loss encouraging similar representations
        """
        # Flatten spatial dimensions
        B, C, H, W = features_view1.shape
        f1_flat = features_view1.view(B, C, -1).transpose(1, 2)  # [B, HW, C]
        f2_flat = features_view2.view(B, C, -1).transpose(1, 2)  # [B, HW, C]
        
        # Project features
        f1_proj = self.attention_proj(f1_flat)
        f2_proj = self.attention_proj(f2_flat)
        
        # Compute self-attention for each view
        attn1 = torch.bmm(f1_proj, f1_proj.transpose(1, 2)) / (C ** 0.5)  # [B, HW, HW]
        attn2 = torch.bmm(f2_proj, f2_proj.transpose(1, 2)) / (C ** 0.5)  # [B, HW, HW]
        
        # Normalize attention maps
        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)
        
        # Consistency loss: encourage similar attention patterns
        consistency_loss = F.mse_loss(attn1, attn2)
        
        return consistency_loss


@META_ARCH_REGISTRY.register()
class MaskFormerWithMARS(MaskFormer):
    """
    MaskFormer with MARS regularization
    Extends standard MaskFormer to use multi-view training
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # MARS configuration
        self.mars_enabled = cfg.MODEL.MARS.ENABLED if hasattr(cfg.MODEL, 'MARS') else True
        self.mars_weight = cfg.MODEL.MARS.WEIGHT if hasattr(cfg.MODEL, 'MARS') else 0.1
        self.num_views = cfg.MODEL.MARS.NUM_VIEWS if hasattr(cfg.MODEL, 'MARS') else 2
        
        # Add MARS regularization modules
        if self.mars_enabled:
            # Get feature dimension from backbone
            feature_dim = 256  # Standard for Mask2Former
            
            # Add attention regularizers for each backbone level
            self.mars_regularizers = nn.ModuleDict({
                'res3': MultiViewAttentionRegularization(feature_dim),
                'res4': MultiViewAttentionRegularization(feature_dim),
                'res5': MultiViewAttentionRegularization(feature_dim),
            })
    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Forward pass with optional MARS regularization
        
        During training, if MARS is enabled:
        1. Process original images
        2. Generate augmented view (rotation)
        3. Process augmented images
        4. Compute MARS consistency loss
        5. Add to total loss
        """
        if not self.training or not self.mars_enabled:
            # Standard forward pass for inference or non-MARS training
            return super().forward(batched_inputs)
        
        # MARS Training Mode
        device = batched_inputs[0]["image"].device
        
        # Process original view
        losses_view1 = super().forward(batched_inputs)
        
        # Extract features from view 1 (store for comparison)
        with torch.no_grad():
            images_view1 = self.preprocess_image(batched_inputs)
            features_view1 = self.backbone(images_view1.tensor)
        
        # Generate augmented view (random rotation)
        batched_inputs_view2 = self._generate_augmented_view(batched_inputs)
        
        # Process augmented view
        images_view2 = self.preprocess_image(batched_inputs_view2)
        features_view2 = self.backbone(images_view2.tensor)
        
        # Compute MARS regularization loss
        mars_loss = 0.0
        for level in ['res3', 'res4', 'res5']:
            if level in features_view1 and level in features_view2:
                consistency_loss = self.mars_regularizers[level](
                    features_view1[level].detach(),  # Don't backprop through view 1
                    features_view2[level]
                )
                mars_loss += consistency_loss
        
        # Average MARS loss
        mars_loss = mars_loss / len(self.mars_regularizers)
        
        # Add MARS loss to total
        losses_view1["loss_mars"] = mars_loss * self.mars_weight
        
        return losses_view1
    
    def _generate_augmented_view(self, batched_inputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        Generate augmented view by applying random rotation
        
        Args:
            batched_inputs: Original batch
        
        Returns:
            Augmented batch with rotated images
        """
        augmented_inputs = []
        
        for inputs in batched_inputs:
            aug_input = {}
            
            # Random rotation: 0, 90, 180, or 270 degrees
            rotation = torch.randint(0, 4, (1,)).item()
            
            # Rotate image
            if rotation > 0:
                aug_input["image"] = torch.rot90(inputs["image"], k=rotation, dims=[1, 2])
            else:
                aug_input["image"] = inputs["image"].clone()
            
            # Copy other fields (don't rotate GT for view 2, we only care about features)
            for key in ["height", "width", "image_id"]:
                if key in inputs:
                    aug_input[key] = inputs[key]
            
            # For MARS, we don't need GT for view 2, only for view 1
            # But include dummy targets to avoid errors
            if "instances" in inputs:
                aug_input["instances"] = inputs["instances"]
            
            augmented_inputs.append(aug_input)
        
        return augmented_inputs


def add_mars_config(cfg):
    """
    Add MARS-specific configuration options
    
    Usage:
        from mars_integration import add_mars_config
        cfg = get_cfg()
        add_mars_config(cfg)
    """
    from detectron2.config import CfgNode as CN
    
    cfg.MODEL.MARS = CN()
    cfg.MODEL.MARS.ENABLED = True
    cfg.MODEL.MARS.WEIGHT = 0.1  # Weight for MARS regularization loss
    cfg.MODEL.MARS.NUM_VIEWS = 2  # Number of views (1 original + 1 augmented)
