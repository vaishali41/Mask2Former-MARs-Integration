"""
MARs Adaptation for Mask2Former
Adapts the MARs philosophy to instance segmentation

Core Principle (from MARs paper):
- Attention patterns should be consistent across different views
- This improves robustness to viewpoint changes

Adaptation:
- Original MARs: SE/CA attention in ResNeXt
- This version: Transformer cross-attention in Mask2Former
- Task: Instance segmentation instead of metric learning

File: mask2former_mars/modeling/meta_arch/mars_mask_former_head.py
"""
from detectron2.structures import ImageList
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from collections import defaultdict

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from mask2former import MaskFormer


class AttentionMapExtractor:
    """
    Extracts attention maps from Mask2Former transformer layers
    
    This replaces SE/CA attention from original MARs with transformer attention
    """
    
    def __init__(self):
        self.attention_maps = []
        self.hooks = []
    
    def hook_attention(self, module, input, output):
        """
        Hook function to capture attention weights
        
        Args:
            module: The attention module
            input: Input to the module
            output: Output from the module
        """
        if isinstance(output, tuple) and len(output) >= 2:
            attention = output[1]  # Attention weights
            if attention is not None:
                self.attention_maps.append(attention.detach())
    """
    def hook_attention(self, module, input, output):
        #Hook function to capture attention weights
        print(f"Hook called! Output type: {type(output)}, Is tuple: {isinstance(output, tuple)}")
        
        if isinstance(output, tuple):
            print(f"   Tuple length: {len(output)}")
            if len(output) >= 2:
                attention = output[1]
                if attention is not None:
                    print(f"   Captured attention shape: {attention.shape}")
                    self.attention_maps.append(attention.detach())
                else:
                    print(f"   Attention is None")
            else:
                print(f"    Tuple too short, no attention weights")
        else:
            print(f"   Output is not a tuple")
    """
    """
    def hook_attention(self, module, input, output):
        
        #Hook function to capture attention weights, Args:module: The attention module,input: Input to the module,output: Output from the module
        
        # Transformer attention usually returns (output, attention_weights)
        if isinstance(output, tuple):
            if len(output) >= 2:
                attention = output[1]  # Attention weights
                if attention is not None:
                    # Store detached copy
                    self.attention_maps.append(attention.detach().clone())
        
        # Some implementations store attention in module
        elif hasattr(module, 'attention_weights'):
            self.attention_maps.append(module.attention_weights.detach().clone())
    """

    def register(self, model):
        """
        Register hooks on all cross-attention layers
        
        Args:
            model: Mask2Former model
        """
        if not hasattr(model, 'sem_seg_head'):
            print(" No sem_seg_head found")
            return
        
        predictor = model.sem_seg_head.predictor
        
        # Mask2Former structure: predictor.transformer_cross_attention_layers
        if hasattr(predictor, 'transformer_cross_attention_layers'):
            transformer_layers = predictor.transformer_cross_attention_layers
            
            # Register hooks on each cross-attention layer
            for i, layer in enumerate(transformer_layers):
                if hasattr(layer, 'multihead_attn'):
                    hook = layer.multihead_attn.register_forward_hook(self.hook_attention)
                    self.hooks.append(hook)
            
            print(f" Registered MARs hooks on {len(self.hooks)} transformer layers")
        else:
            print(" Could not find transformer_cross_attention_layers")

    """
    def register(self, model):
        if not hasattr(model, 'sem_seg_head'):
            print("  No sem_seg_head found")
            return
        
        predictor = model.sem_seg_head.predictor
        
        # Mask2Former structure: transformer_cross_attention_layers
        if hasattr(predictor, 'transformer_cross_attention_layers'):
            transformer_layers = predictor.transformer_cross_attention_layers
            
            # Register hooks on each cross-attention layer
            for i, layer in enumerate(transformer_layers):
                if hasattr(layer, 'multihead_attn'):
                    hook = layer.multihead_attn.register_forward_hook(self.hook_attention)
                    self.hooks.append(hook)
            
            print(f" Registered MARs hooks on {len(self.hooks)} transformer layers")
        else:
            print(" Could not find transformer_cross_attention_layers")
            print(f"   Available: {[a for a in dir(predictor) if 'transformer' in a.lower()]}")
    """
    """
    def register(self, model):
        
        #Register hooks on all cross-attention layers
        
        # Find transformer decoder layers
        if not hasattr(model, 'sem_seg_head'):
            print(" No sem_seg_head found")
            return
        
        predictor = model.sem_seg_head.predictor
        
        # Try different possible locations for transformer layers
        transformer_layers = None
        
        if hasattr(predictor, 'transformer_predictor'):
            if hasattr(predictor.transformer_predictor, 'layers'):
                transformer_layers = predictor.transformer_predictor.layers
            elif hasattr(predictor.transformer_predictor, 'decoder'):
                if hasattr(predictor.transformer_predictor.decoder, 'layers'):
                    transformer_layers = predictor.transformer_predictor.decoder.layers
        elif hasattr(predictor, 'layers'):
            transformer_layers = predictor.layers
        
        if transformer_layers is None:
            print(" Could not find transformer layers")
            return
        
        # Register hooks on cross-attention in each layer
        for layer in transformer_layers:
            if hasattr(layer, 'cross_attn') or hasattr(layer, 'multihead_attn'):
                attn_module = getattr(layer, 'cross_attn', None) or getattr(layer, 'multihead_attn', None)
                hook = attn_module.register_forward_hook(self.hook_attention)
                self.hooks.append(hook)
        
        print(f" Registered MARs hooks on {len(self.hooks)} transformer layers")
    """
    def clear(self):
        """Clear stored attention maps"""
        self.attention_maps = []
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_maps = []


class MARsLoss(nn.Module):
    """
    MARs Loss: Multi-view Attention Regularization
    
    Adapts MARs from the paper to work with transformer attention:
    - Original: Regularizes SE/CA channel/spatial attention
    - Adapted: Regularizes transformer cross-attention patterns
    
    Loss encourages attention to be consistent across augmented views
    """
    
    def __init__(self, loss_type: str = "kl"):
        """
        Args:
            loss_type: Type of consistency loss
                - "kl": KL divergence (distribution similarity)
                - "l2": L2 distance (direct similarity)
                - "cosine": Cosine similarity
        """
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, attention_v1: List[torch.Tensor], 
            attention_v2: List[torch.Tensor]) -> torch.Tensor:
        """Compute attention consistency loss"""
        
        if len(attention_v1) == 0 or len(attention_v2) == 0:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        num_valid = 0
        
        for attn1, attn2 in zip(attention_v1, attention_v2):
            if attn1.shape != attn2.shape:
                continue
            
            # Use RAW attention - no softmax or normalization
            attn1_flat = attn1.flatten(start_dim=1)
            attn2_flat = attn2.flatten(start_dim=1)
            
            # Cosine similarity (inherently normalized by magnitude)
            cosine_sim = F.cosine_similarity(attn1_flat, attn2_flat, dim=1).mean()
            loss = 1 - cosine_sim
            
            total_loss += loss
            num_valid += 1
        
        if num_valid > 0:
            total_loss = total_loss / num_valid
        
        return total_loss

    """
    def forward(self, attention_v1: List[torch.Tensor], 
                attention_v2: List[torch.Tensor]) -> torch.Tensor:
        
        Compute attention consistency loss (original MARS approach)
        
        Args:
            attention_v1: Attention maps from view 1
            attention_v2: Attention maps from view 2
        
        Returns:
            Scalar loss value
        
        if len(attention_v1) == 0 or len(attention_v2) == 0:
            return torch.tensor(0.0, device=attention_v1[0].device if len(attention_v1) > 0 else None)
        
        total_loss = 0.0
        num_valid = 0
        
        # Compute loss for each layer (like original MARS)
        for attn1, attn2 in zip(attention_v1, attention_v2):
            # Skip if shapes don't match
            if attn1.shape != attn2.shape:
                continue
            
            # Original MARS approach: Direct L2 distance
            if self.loss_type == "l2":
                # MSE (mean squared error) = L2 distance
                loss = F.mse_loss(attn1, attn2)
                
            elif self.loss_type == "cosine":
                # Alternative: Cosine similarity
                attn1_flat = attn1.flatten(start_dim=1)
                attn2_flat = attn2.flatten(start_dim=1)
                cosine_sim = F.cosine_similarity(attn1_flat, attn2_flat, dim=1).mean()
                loss = 1 - cosine_sim
            else:
                loss = F.mse_loss(attn1, attn2)
            
            total_loss += loss
            num_valid += 1
        
        # Average across layers
        if num_valid > 0:
            total_loss = total_loss / num_valid
        
        return total_loss
    """
    """
    def forward(self, attention_v1: List[torch.Tensor], 
                attention_v2: List[torch.Tensor]) -> torch.Tensor:
        
        Compute attention consistency loss
        
        Args:
            attention_v1: Attention maps from view 1
            attention_v2: Attention maps from view 2
        
        Returns:
            Scalar loss value
        
        if len(attention_v1) == 0 or len(attention_v2) == 0:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        num_valid = 0
        
        # Compute loss for each layer
        for attn1, attn2 in zip(attention_v1, attention_v2):
            # Skip if shapes don't match
            if attn1.shape != attn2.shape:
                continue
            
            # Normalize attention maps
            attn1 = self._normalize_attention(attn1)
            attn2 = self._normalize_attention(attn2)
            
            # Compute loss based on type
            if self.loss_type == "kl":
                # Symmetric KL divergence
                loss = 0.5 * (
                    F.kl_div(attn1.log(), attn2, reduction='batchmean') +
                    F.kl_div(attn2.log(), attn1, reduction='batchmean')
                )
            elif self.loss_type == "l2":
                # L2 distance
                loss = F.mse_loss(attn1, attn2)
            elif self.loss_type == "cosine":
                # Cosine similarity loss
                attn1_flat = attn1.flatten(start_dim=1)
                attn2_flat = attn2.flatten(start_dim=1)
                cosine_sim = F.cosine_similarity(attn1_flat, attn2_flat, dim=1).mean()
                loss = 1 - cosine_sim
            else:
                loss = F.mse_loss(attn1, attn2)
            
            total_loss += loss
            num_valid += 1
        
        # Average across layers
        if num_valid > 0:
            total_loss = total_loss / num_valid
        
        return total_loss
    """
    """
    def forward(self, attention_v1: List[torch.Tensor], 
                attention_v2: List[torch.Tensor]) -> torch.Tensor:
        
        Compute attention consistency loss
        
        Args:
            attention_v1: Attention maps from view 1
                Each tensor: [B, num_heads, num_queries, HW] or [B, num_queries, HW]
            attention_v2: Attention maps from view 2
        
        Returns:
            Scalar loss value
        
        if len(attention_v1) == 0 or len(attention_v2) == 0:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        num_valid = 0
        
        # Compute loss for each layer
        for attn1, attn2 in zip(attention_v1, attention_v2):
            # Skip if shapes don't match
            if attn1.shape != attn2.shape:
                continue
            
            # Normalize attention maps
            attn1 = self._normalize_attention(attn1)
            attn2 = self._normalize_attention(attn2)
            
            # Compute loss based on type
            if self.loss_type == "kl":
                # Symmetric KL divergence
                loss = 0.5 * (
                    F.kl_div(attn1.log(), attn2, reduction='batchmean') +
                    F.kl_div(attn2.log(), attn1, reduction='batchmean')
                )
            elif self.loss_type == "l2":
                # L2 distance
                loss = F.mse_loss(attn1, attn2)
            elif self.loss_type == "cosine":
                # Cosine similarity loss
                attn1_flat = attn1.flatten(start_dim=1)
                attn2_flat = attn2.flatten(start_dim=1)
                cosine_sim = F.cosine_similarity(attn1_flat, attn2_flat, dim=1).mean()
                loss = 1 - cosine_sim
            else:
                loss = F.mse_loss(attn1, attn2)
            
            total_loss += loss
            num_valid += 1
        
        # Average across layers
        if num_valid > 0:
            total_loss = total_loss / num_valid
        
        return total_loss
    """
    def _normalize_attention(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Normalize attention to valid probability distribution
        
        Args:
            attention: Raw attention tensor
        
        Returns:
            Normalized attention
        """
        # Add small epsilon for numerical stability
        eps = 1e-8
        
        # Softmax over last dimension (spatial locations)
        attention = F.softmax(attention, dim=-1)
        
        # Ensure non-negative and sum to 1
        attention = attention + eps
        attention = attention / attention.sum(dim=-1, keepdim=True)
        
        return attention


@META_ARCH_REGISTRY.register()
class MaskFormerWithMARS(MaskFormer):
    """
    Mask2Former + MARs Adaptation
    
    Integrates MARs multi-view attention regularization into Mask2Former:
    1. Creates two views (original + rotated)
    2. Captures transformer attention from both views
    3. Regularizes attention to be consistent
    
    Training Loss:
        L_total = L_seg + λ * L_MARs
        
    where:
        L_seg = standard Mask2Former losses (mask + class + dice)
        L_MARs = attention consistency loss
        λ = mars_weight hyperparameter
    """

    @configurable
    def __init__(
        self,
        *,
        backbone,
        sem_seg_head,
        criterion,
        num_queries,
        object_mask_threshold,
        overlap_threshold,
        metadata,
        size_divisibility,
        sem_seg_postprocess_before_inference,
        pixel_mean,
        pixel_std,
        # MARs parameters (separate from base class params)
        mars_enabled: bool = True,
        mars_weight: float = 0.1,
        mars_loss_type: str = "kl",
        # Accept any additional base class parameters
        **kwargs
    ):
        """
        Initialize MaskFormerWithMARS
        
        CRITICAL: MARS parameters are extracted here and NOT passed to parent.
        All other kwargs (like semantic_on, panoptic_on, etc.) are passed to parent.
        """
        # Call parent WITH all base class parameters
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            **kwargs  #  Pass remaining base class params (semantic_on, etc.)
        )
        
        # NOW initialize MARS-specific components AFTER parent init
        self.mars_enabled = mars_enabled
        self.mars_weight = mars_weight
        
        if self.mars_enabled:
            # Initialize attention extractor
            self.attention_extractor = AttentionMapExtractor()
            self.attention_extractor.register(self)
            
            # Initialize MARs loss
            self.mars_loss_fn = MARsLoss(loss_type=mars_loss_type)
            
            print(f"\n{'='*70}")
            print("MARs Adaptation for Mask2Former")
            print(f"{'='*70}")
            print(f"Status:      Enabled")
            print(f"Weight (λ):  {self.mars_weight}")
            print(f"Loss Type:   {mars_loss_type}")
            print(f"Hooks:       {len(self.attention_extractor.hooks)} layers")
            print(f"Philosophy:  Attention consistency across rotations")
            print(f"{'='*70}\n")

    """
    @configurable
    def __init__(
        self,
        *,
        backbone,
        sem_seg_head,
        criterion,
        num_queries,
        object_mask_threshold,
        overlap_threshold,
        metadata,
        size_divisibility,
        sem_seg_postprocess_before_inference,
        pixel_mean,
        pixel_std,
        # MARs parameters
        mars_enabled: bool = True,
        mars_weight: float = 0.1,
        mars_loss_type: str = "kl",
        # Accept any other parameters from base class
        **kwargs
    ):
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            **kwargs  # Pass any extra parameters to base class
        )
        
        self.mars_enabled = mars_enabled
        self.mars_weight = mars_weight
        
        if self.mars_enabled:
            # Initialize attention extractor
            self.attention_extractor = AttentionMapExtractor()
            self.attention_extractor.register(self)
            
            # Initialize MARs loss
            self.mars_loss_fn = MARsLoss(loss_type=mars_loss_type)
            
            print(f"\n{'='*70}")
            print("MARs Adaptation for Mask2Former")
            print(f"{'='*70}")
            print(f"Status:      Enabled")
            print(f"Weight (λ):  {self.mars_weight}")
            print(f"Loss Type:   {mars_loss_type}")
            print(f"Hooks:       {len(self.attention_extractor.hooks)} layers")
            print(f"Philosophy:  Attention consistency across rotations")
            print(f"{'='*70}\n")
    """    
    """
    @configurable
    def __init__(
        self,
        *,
        backbone,
        sem_seg_head,
        criterion,
        num_queries,
        object_mask_threshold,
        overlap_threshold,
        metadata,
        size_divisibility,
        sem_seg_postprocess_before_inference,
        pixel_mean,
        pixel_std,
        # MARs parameters
        mars_enabled: bool = True,
        mars_weight: float = 0.1,
        mars_loss_type: str = "kl",
    ):
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        
        self.mars_enabled = mars_enabled
        self.mars_weight = mars_weight
        
        if self.mars_enabled:
            # Initialize attention extractor
            self.attention_extractor = AttentionMapExtractor()
            self.attention_extractor.register(self)
            
            # Initialize MARs loss
            self.mars_loss_fn = MARsLoss(loss_type=mars_loss_type)
            
            print(f"\n{'='*70}")
            print("MARs Adaptation for Mask2Former")
            print(f"{'='*70}")
            print(f"Status:      Enabled")
            print(f"Weight (λ):  {self.mars_weight}")
            print(f"Loss Type:   {mars_loss_type}")
            print(f"Hooks:       {len(self.attention_extractor.hooks)} layers")
            print(f"Philosophy:  Attention consistency across rotations")
            print(f"{'='*70}\n")
    """

    @classmethod
    def from_config(cls, cfg):
        """
        Create an instance from config
        
        This extracts MARS-specific parameters and passes everything else to parent
        """
        # Get base class config
        ret = super().from_config(cfg)
        
        # Add MARS-specific parameters (these won't be in parent's from_config)
        ret["mars_enabled"] = cfg.MODEL.MARS.ENABLED
        ret["mars_weight"] = cfg.MODEL.MARS.WEIGHT
        ret["mars_loss_type"] = cfg.MODEL.MARS.LOSS_TYPE
        
        return ret
    """
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["mars_enabled"] = cfg.MODEL.MARS.ENABLED
        ret["mars_weight"] = cfg.MODEL.MARS.WEIGHT
        ret["mars_loss_type"] = cfg.MODEL.MARS.LOSS_TYPE
        return ret
    """

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """Forward pass with MARs regularization (ULTRA MEMORY EFFICIENT)"""
        
        # Inference or MARs disabled
        if not self.training or not self.mars_enabled:
            return super().forward(batched_inputs)
        
        # === MARs Training (Ultra Memory Efficient) ===
        
        # Clear previous attention maps
        self.attention_extractor.clear()
        
        # View 1: Process with base class forward (compute all losses)
        losses = super().forward(batched_inputs)
        
        # Store attention from view 1
        attention_v1 = [attn.clone() for attn in self.attention_extractor.attention_maps]
        
        # === View 2: Lightweight feature extraction ONLY ===
        # DO NOT call super().forward() - it's too heavy!
        # Instead, manually extract just the features we need
        
        self.attention_extractor.clear()
        
        with torch.no_grad():
            # Process each image individually to minimize memory
            for item in batched_inputs:
                # Random rotation
                k = torch.randint(0, 4, (1,)).item()
                
                # Rotate image tensor
                if k > 0:
                    rotated_image = torch.rot90(item["image"], k=k, dims=[1, 2])
                else:
                    rotated_image = item["image"].clone()
                
                # Preprocess (normalize) - SINGLE image
                image = rotated_image.to(self.device)
                image = (image - self.pixel_mean) / self.pixel_std
                
                # Pad to size divisibility
                h, w = image.shape[-2:]
                pad_h = (self.size_divisibility - h % self.size_divisibility) % self.size_divisibility
                pad_w = (self.size_divisibility - w % self.size_divisibility) % self.size_divisibility
                
                if pad_h > 0 or pad_w > 0:
                    image = F.pad(image, (0, pad_w, 0, pad_h), value=0)
                
                # Add batch dimension
                image = image.unsqueeze(0)
                
                # Extract features (this triggers attention hooks)
                features = self.backbone(image)
                _ = self.sem_seg_head(features)
                
                # Aggressive memory cleanup
                del image, features, rotated_image
                
            # Small final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        attention_v2 = self.attention_extractor.attention_maps
        
        # Compute MARs loss
        if len(attention_v1) > 0 and len(attention_v2) > 0:
            mars_loss = self.mars_loss_fn(attention_v1, attention_v2)
            losses["loss_mars"] = mars_loss * self.mars_weight
        else:
            device = next(iter(losses.values())).device
            losses["loss_mars"] = torch.tensor(0.0, device=device)
        
        return losses

    """
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        Forward pass with MARs regularization
        
        # Inference or MARs disabled
        if not self.training or not self.mars_enabled:
            return super().forward(batched_inputs)
        
        # === MARs Training ===
        
        # Clear previous attention maps
        self.attention_extractor.clear()
        
        # View 1: Process with base class forward
        losses = super().forward(batched_inputs)
        
        # Store attention from view 1
        attention_v1 = [attn.clone() for attn in self.attention_extractor.attention_maps]
        
        # View 2: Rotated images with rotated instances
        self.attention_extractor.clear()
        
        # Create rotated inputs
        batched_inputs_v2 = []
        for item in batched_inputs:
            k = torch.randint(0, 4, (1,)).item()
            
            rotated_item = {
                "image": torch.rot90(item["image"], k=k, dims=[1, 2]) if k > 0 else item["image"].clone(),
                "height": item["width"] if k % 2 == 1 else item["height"],
                "width": item["height"] if k % 2 == 1 else item["width"],
            }
            
            if "image_id" in item:
                rotated_item["image_id"] = item["image_id"]
            
            if "file_name" in item:
                rotated_item["file_name"] = item["file_name"]
            
            # Rotate instances
            if "instances" in item:
                rotated_item["instances"] = self._rotate_instances(
                    item["instances"],
                    k,
                    item["height"],
                    item["width"]
                )
            
            batched_inputs_v2.append(rotated_item)
        
        # Process View 2
        with torch.no_grad():
            _ = super().forward(batched_inputs_v2)
        
        attention_v2 = self.attention_extractor.attention_maps
        
        # Compute MARs loss
        if len(attention_v1) > 0 and len(attention_v2) > 0:
            mars_loss = self.mars_loss_fn(attention_v1, attention_v2)
            losses["loss_mars"] = mars_loss * self.mars_weight
        else:
            device = next(iter(losses.values())).device
            losses["loss_mars"] = torch.tensor(0.0, device=device)
        
        return losses
    """
    """
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        
        Forward pass with MARs regularization
        
        Training:
            1. Process view 1 (original) → get segmentation losses
            2. Extract attention maps from view 1
            3. Process view 2 (rotated) → extract attention only
            4. Compute MARs consistency loss
            5. Return L_total = L_seg + λ * L_MARs
        
        Inference:
            Standard Mask2Former inference (no MARs)
        
        # Inference or MARs disabled
        if not self.training or not self.mars_enabled:
            return super().forward(batched_inputs)
        
        # === MARs Training ===
        
        # Clear previous attention maps
        self.attention_extractor.clear()
        
        # View 1: Process with base class forward (captures losses and attention)
        losses = super().forward(batched_inputs)
        
        # Store attention from view 1
        attention_v1 = [attn.clone() for attn in self.attention_extractor.attention_maps]
        
        # View 2: Extract attention from rotated images (no loss computation)
        self.attention_extractor.clear()

        # Apply rotation to both images and annotations
        batched_inputs_v2 = self._apply_rotation(batched_inputs)
        
        # Extract attention from view 2 (no gradients for efficiency)
        with torch.no_grad():
            # Process rotated images with rotated ground truth
            # This ensures attention is computed with proper target guidance
            _ = super().forward(batched_inputs_v2)

        attention_v2 = self.attention_extractor.attention_maps
        
        with torch.no_grad():
            # Rotate images
            batched_inputs_v2 = self._apply_rotation(batched_inputs)
            
            # Extract features (mimicking what forward does, but without targets)
            images_v2 = [x["image"].to(self.device) for x in batched_inputs_v2]
            images_v2 = [(x - self.pixel_mean) / self.pixel_std for x in images_v2]
            images_v2 = ImageList.from_tensors(images_v2, self.size_divisibility)
            
            # Run through backbone and segmentation head (attention hooks will capture)
            features_v2 = self.backbone(images_v2.tensor)
            _ = self.sem_seg_head(features_v2)
        
        attention_v2 = self.attention_extractor.attention_maps
        
        # Compute MARs loss
        if len(attention_v1) > 0 and len(attention_v2) > 0:
            mars_loss = self.mars_loss_fn(attention_v1, attention_v2)
            losses["loss_mars"] = mars_loss * self.mars_weight
        else:
            # Get device from existing losses
            device = next(iter(losses.values())).device
            losses["loss_mars"] = torch.tensor(0.0, device=device)
        
        return losses
    """
    """
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):

        # Inference or MARs disabled
        if not self.training or not self.mars_enabled:
            return super().forward(batched_inputs)
        
        # === MARs Training ===
        
        # Clear previous attention maps
        self.attention_extractor.clear()
        
        # View 1: Process with base class forward (captures losses and attention)
        losses = super().forward(batched_inputs)
        
        # Store attention from view 1
        attention_v1 = [attn.clone() for attn in self.attention_extractor.attention_maps]
        
        # View 2: Rotated images
        self.attention_extractor.clear()
        
        batched_inputs_v2 = self._apply_rotation(batched_inputs)
        
        # Extract attention from view 2 (no gradients needed for segmentation losses)
        with torch.no_grad():
            _ = super().forward(batched_inputs_v2)
        
        attention_v2 = self.attention_extractor.attention_maps
        
        # Compute MARs loss
        if len(attention_v1) > 0 and len(attention_v2) > 0:
            mars_loss = self.mars_loss_fn(attention_v1, attention_v2)
            losses["loss_mars"] = mars_loss * self.mars_weight
        else:
            # Get device from existing losses
            device = next(iter(losses.values())).device
            losses["loss_mars"] = torch.tensor(0.0, device=device)
        
        return losses
    """
    """
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):

        # Inference or MARs disabled
        if not self.training or not self.mars_enabled:
            return super().forward(batched_inputs)
        
        # === MARs Training ===
        
        # Clear previous attention maps
        self.attention_extractor.clear()
        
        # View 1: Original images (with standard augmentations from dataloader)
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        # Forward through segmentation head (hooks capture attention)
        outputs = self.sem_seg_head(features)
        
        # Compute standard segmentation losses
        targets = self.prepare_targets(batched_inputs, images)
        losses = self.criterion(outputs, targets)
        
        # Apply loss weights
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)
        
        # Store attention from view 1
        attention_v1 = [attn.clone() for attn in self.attention_extractor.attention_maps]
        
        # View 2: Rotated images
        self.attention_extractor.clear()
        
        batched_inputs_v2 = self._apply_rotation(batched_inputs)
        images_v2 = self.preprocess_image(batched_inputs_v2)
        
        # Extract attention from view 2 (no gradients needed for view 1)
        with torch.no_grad():
            features_v2 = self.backbone(images_v2.tensor)
            _ = self.sem_seg_head(features_v2)
        
        attention_v2 = self.attention_extractor.attention_maps
        
        # Compute MARs loss
        if len(attention_v1) > 0 and len(attention_v2) > 0:
            mars_loss = self.mars_loss_fn(attention_v1, attention_v2)
            losses["loss_mars"] = mars_loss * self.mars_weight
        else:
            losses["loss_mars"] = torch.tensor(0.0, device=images.tensor.device)
        
        return losses
    """

    def _apply_rotation(self, batched_inputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        Apply random 90-degree rotation to BOTH images and annotations
        
        This creates the second view for MARs regularization with proper ground truth
        """
        augmented = []
        
        for i, item in enumerate(batched_inputs):
            aug_item = {}
            
            # DEBUG: Check what's in the original item
            print(f"\n{'='*60}")
            print(f"DEBUG: Item {i} keys: {item.keys()}")
            
            if "instances" in item:
                print(f"  Has 'instances' field")
                print(f"  Instance fields: {item['instances'].get_fields()}")
                print(f"  Has gt_masks: {item['instances'].has('gt_masks')}")
                if item['instances'].has('gt_masks'):
                    print(f"  gt_masks type: {type(item['instances'].gt_masks)}")
            
            if "annotations" in item:
                print(f"  Has 'annotations' field")
                print(f"  Num annotations: {len(item['annotations'])}")
            print(f"{'='*60}\n")
            
            # Random rotation (0°, 90°, 180°, 270°)
            k = torch.randint(0, 4, (1,)).item()
            
            # Rotate image
            if k > 0:
                aug_item["image"] = torch.rot90(item["image"], k=k, dims=[1, 2])
            else:
                aug_item["image"] = item["image"].clone()
            
            # Update dimensions (they swap for 90° and 270°)
            if k % 2 == 1:  # 90° or 270° rotation
                aug_item["height"] = item["width"]
                aug_item["width"] = item["height"]
            else:  # 0° or 180° rotation
                aug_item["height"] = item["height"]
                aug_item["width"] = item["width"]
            
            # Copy image_id
            if "image_id" in item:
                aug_item["image_id"] = item["image_id"]
            
            # Rotate annotations (if present)
            if "annotations" in item:
                print("  → Rotating annotations")
                aug_item["annotations"] = self._rotate_annotations(
                    item["annotations"],
                    k,
                    item["height"],
                    item["width"]
                )
            # Handle case where instances are already created
            elif "instances" in item:
                print("  → Rotating instances")
                aug_item["instances"] = self._rotate_instances(
                    item["instances"],
                    k,
                    item["height"],
                    item["width"]
                )
                print(f"  → Rotated instance fields: {aug_item['instances'].get_fields()}")
                print(f"  → Rotated has gt_masks: {aug_item['instances'].has('gt_masks')}")
            
            augmented.append(aug_item)
        
        return augmented
    """
    def _apply_rotation(self, batched_inputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        
        #Apply random 90-degree rotation to BOTH images and annotations
        
        #This creates the second view for MARs regularization with proper ground truth
        
        augmented = []
        
        for item in batched_inputs:
            aug_item = {}
            
            # Random rotation (0°, 90°, 180°, 270°)
            k = torch.randint(0, 4, (1,)).item()
            
            # Rotate image
            if k > 0:
                aug_item["image"] = torch.rot90(item["image"], k=k, dims=[1, 2])
            else:
                aug_item["image"] = item["image"].clone()
            
            # Update dimensions (they swap for 90° and 270°)
            if k % 2 == 1:  # 90° or 270° rotation
                aug_item["height"] = item["width"]
                aug_item["width"] = item["height"]
            else:  # 0° or 180° rotation
                aug_item["height"] = item["height"]
                aug_item["width"] = item["width"]
            
            # Copy image_id
            if "image_id" in item:
                aug_item["image_id"] = item["image_id"]
            
            # Rotate annotations (if present)
            if "annotations" in item:
                aug_item["annotations"] = self._rotate_annotations(
                    item["annotations"],
                    k,
                    item["height"],
                    item["width"]
                )
            # Handle case where instances are already created
            elif "instances" in item:
                aug_item["instances"] = self._rotate_instances(
                    item["instances"],
                    k,
                    item["height"],
                    item["width"]
                )
            
            augmented.append(aug_item)
        
        return augmented
    """
    def _rotate_annotations(self, annotations, k, orig_h, orig_w):
        """
        Rotate annotation dictionaries (format from dataloader)
        
        Args:
            annotations: List of annotation dicts with 'bbox', 'segmentation', 'category_id'
            k: Rotation (0=0°, 1=90°, 2=180°, 3=270°)
            orig_h, orig_w: Original image dimensions
        
        Returns:
            List of rotated annotation dicts
        """
        import copy
        from detectron2.structures import BoxMode
        import numpy as np
        
        if k == 0:
            return copy.deepcopy(annotations)
        
        rotated_annotations = []
        
        for ann in annotations:
            rotated_ann = copy.deepcopy(ann)
            
            # Rotate bounding box
            if "bbox" in ann:
                bbox = ann["bbox"]
                bbox_mode = ann.get("bbox_mode", BoxMode.XYWH_ABS)
                
                # Convert to XYXY format for rotation
                if bbox_mode == BoxMode.XYWH_ABS:
                    x, y, w, h = bbox
                    x1, y1, x2, y2 = x, y, x + w, y + h
                else:  # Assume XYXY_ABS
                    x1, y1, x2, y2 = bbox
                
                # Rotate corners
                if k == 1:  # 90° clockwise
                    new_x1, new_y1 = y1, orig_w - x2
                    new_x2, new_y2 = y2, orig_w - x1
                elif k == 2:  # 180°
                    new_x1, new_y1 = orig_w - x2, orig_h - y2
                    new_x2, new_y2 = orig_w - x1, orig_h - y1
                elif k == 3:  # 270° clockwise
                    new_x1, new_y1 = orig_h - y2, x1
                    new_x2, new_y2 = orig_h - y1, x2
                
                # Ensure correct order (x1 < x2, y1 < y2)
                new_x1, new_x2 = min(new_x1, new_x2), max(new_x1, new_x2)
                new_y1, new_y2 = min(new_y1, new_y2), max(new_y1, new_y2)
                
                # Convert back to original format
                if bbox_mode == BoxMode.XYWH_ABS:
                    rotated_ann["bbox"] = [new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1]
                else:
                    rotated_ann["bbox"] = [new_x1, new_y1, new_x2, new_y2]
            
            # Rotate segmentation
            if "segmentation" in ann:
                seg = ann["segmentation"]
                
                # Handle different segmentation formats
                if isinstance(seg, list):  # Polygon format
                    rotated_seg = []
                    for poly in seg:
                        if isinstance(poly, list):
                            rotated_poly = self._rotate_polygon_coords(poly, k, orig_h, orig_w)
                            rotated_seg.append(rotated_poly)
                        else:
                            rotated_seg.append(poly)
                    rotated_ann["segmentation"] = rotated_seg
                
                elif isinstance(seg, dict):  # RLE format
                    # For RLE, we'd need to decode, rotate, and re-encode
                    # This is complex, so for now we'll skip RLE rotation
                    # Most COCO data uses polygon format anyway
                    print(" RLE segmentation rotation not implemented, skipping")
                    rotated_ann["segmentation"] = seg
            
            rotated_annotations.append(rotated_ann)
        
        return rotated_annotations

    def _rotate_polygon_coords(self, polygon, k, h, w):
        """
        Rotate polygon coordinates (flat list format: [x1, y1, x2, y2, ...])
        
        Args:
            polygon: List of coordinates [x1, y1, x2, y2, ...]
            k: Rotation (0=0°, 1=90°, 2=180°, 3=270°)
            h, w: Original image height and width
        
        Returns:
            Rotated polygon as list
        """
        import numpy as np
        
        if k == 0:
            return polygon
        
        # Convert to numpy array
        coords = np.array(polygon).reshape(-1, 2)
        x, y = coords[:, 0], coords[:, 1]
        
        # Rotate coordinates
        if k == 1:  # 90° clockwise
            new_x, new_y = y, w - x
        elif k == 2:  # 180°
            new_x, new_y = w - x, h - y
        elif k == 3:  # 270° clockwise
            new_x, new_y = h - y, x
        
        # Flatten back to list
        rotated = np.stack([new_x, new_y], axis=-1).flatten()
        
        return rotated.tolist()

    """
    def _apply_rotation(self, batched_inputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        
        #Apply random 90-degree rotation to BOTH images and annotations
        
        #This creates the second view for MARs regularization with proper ground truth
        
        augmented = []
        
        for item in batched_inputs:
            aug_item = {}
            
            # Random rotation (0°, 90°, 180°, 270°)
            k = torch.randint(0, 4, (1,)).item()
            
            # Rotate image
            if k > 0:
                aug_item["image"] = torch.rot90(item["image"], k=k, dims=[1, 2])
            else:
                aug_item["image"] = item["image"].clone()
            
            # Update dimensions (they swap for 90° and 270°)
            if k % 2 == 1:  # 90° or 270° rotation
                aug_item["height"] = item["width"]
                aug_item["width"] = item["height"]
            else:  # 0° or 180° rotation
                aug_item["height"] = item["height"]
                aug_item["width"] = item["width"]
            
            # Copy image_id
            if "image_id" in item:
                aug_item["image_id"] = item["image_id"]
            
            # Rotate instances (ground truth annotations)
            if "instances" in item:
                aug_item["instances"] = self._rotate_instances(
                    item["instances"], 
                    k, 
                    item["height"], 
                    item["width"]
                )
            
            augmented.append(aug_item)
        
        return augmented
    """

    def _rotate_instances(self, instances, k, orig_h, orig_w):
        """
        Rotate instance annotations (boxes, masks)
        
        Args:
            instances: Detectron2 Instances object
            k: Rotation (0=0°, 1=90°, 2=180°, 3=270°)
            orig_h, orig_w: Original image dimensions
        
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
            
            # Check if it's a raw tensor or a wrapped object
            if isinstance(masks, torch.Tensor):
                # Raw tensor - keep it as raw tensor after rotation
                rotated_masks = torch.rot90(masks, k=k, dims=[1, 2])
                rotated.gt_masks = rotated_masks  # Store as raw tensor
            elif isinstance(masks, BitMasks):
                # BitMasks object
                mask_tensor = masks.tensor  # [N, H, W]
                rotated_masks = torch.rot90(mask_tensor, k=k, dims=[1, 2])
                rotated.gt_masks = BitMasks(rotated_masks)
            elif isinstance(masks, PolygonMasks):
                # PolygonMasks object
                rotated_polygons = []
                for poly_per_instance in masks.polygons:
                    rotated_poly = [
                        self._rotate_polygon(poly, k, orig_h, orig_w) 
                        for poly in poly_per_instance
                    ]
                    rotated_polygons.append(rotated_poly)
                rotated.gt_masks = PolygonMasks(rotated_polygons)
        
        # Copy other fields (classes, etc.)
        for field in instances.get_fields():
            if field not in ["gt_boxes", "gt_masks"]:
                rotated.set(field, copy.deepcopy(instances.get(field)))
        
        return rotated

    """
    def _rotate_instances(self, instances, k, orig_h, orig_w):
        
        Rotate instance annotations (boxes, masks)
        
        Args:
            instances: Detectron2 Instances object
            k: Rotation (0=0°, 1=90°, 2=180°, 3=270°)
            orig_h, orig_w: Original image dimensions
        
        Returns:
            Rotated Instances object
        
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
            if isinstance(masks, BitMasks):
                # Rotate bit masks
                mask_tensor = masks.tensor  # [N, H, W]
                rotated_masks = torch.rot90(mask_tensor, k=k, dims=[1, 2])
                rotated.gt_masks = BitMasks(rotated_masks)
            elif isinstance(masks, PolygonMasks):
                # Rotate polygon masks
                rotated_polygons = []
                for poly_per_instance in masks.polygons:
                    rotated_poly = [
                        self._rotate_polygon(poly, k, orig_h, orig_w) 
                        for poly in poly_per_instance
                    ]
                    rotated_polygons.append(rotated_poly)
                rotated.gt_masks = PolygonMasks(rotated_polygons)
        
        # Copy other fields (classes, etc.)
        for field in instances.get_fields():
            if field not in ["gt_boxes", "gt_masks"]:
                rotated.set(field, copy.deepcopy(instances.get(field)))
        
        return rotated
    """
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
        elif k == 3:  # 270° clockwise (90° counter-clockwise)
            return torch.stack([h - y2, x1, h - y1, x2], dim=-1)
        else:
            return boxes

    def _rotate_polygon(self, polygon, k, h, w):
        """
        Rotate polygon coordinates by k*90 degrees
        
        Args:
            polygon: List or tensor of [x1, y1, x2, y2, ...] coordinates
            k: Rotation (0=0°, 1=90°, 2=180°, 3=270°)
            h, w: Original image height and width
        
        Returns:
            Rotated polygon as list
        """
        import numpy as np
        
        # Convert to numpy for easier manipulation
        if isinstance(polygon, torch.Tensor):
            polygon = polygon.cpu().numpy()
        elif isinstance(polygon, list):
            polygon = np.array(polygon)
        
        # Reshape to [N, 2]
        coords = polygon.reshape(-1, 2)
        x, y = coords[:, 0], coords[:, 1]
        
        if k == 1:  # 90° clockwise
            new_x, new_y = y, w - x
        elif k == 2:  # 180°
            new_x, new_y = w - x, h - y
        elif k == 3:  # 270° clockwise
            new_x, new_y = h - y, x
        else:
            new_x, new_y = x, y
        
        # Flatten back to [x1, y1, x2, y2, ...]
        rotated = np.stack([new_x, new_y], axis=-1).flatten()
        
        return rotated.tolist()

    """
    def _apply_rotation(self, batched_inputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        
        #Apply random 90-degree rotation (0°, 90°, 180°, 270°)
        
        #This creates the second view for MARs regularization
        
        augmented = []
        
        for item in batched_inputs:
            aug_item = {}
            
            # Random rotation
            k = torch.randint(0, 4, (1,)).item()
            
            if k > 0:
                aug_item["image"] = torch.rot90(item["image"], k=k, dims=[1, 2])
            else:
                aug_item["image"] = item["image"].clone()
            
            # Copy metadata
            for key in ["height", "width", "image_id"]:
                if key in item:
                    aug_item[key] = item[key]
            
            augmented.append(aug_item)
        
        return augmented
    """

def add_mars_config(cfg):
    """
    Add MARs configuration to Detectron2 config
    
    Usage:
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        add_mars_config(cfg)  # Add this
        cfg.merge_from_file(config_file)
    """
    from detectron2.config import CfgNode as CN
    
    cfg.MODEL.MARS = CN()
    cfg.MODEL.MARS.ENABLED = True
    cfg.MODEL.MARS.WEIGHT = 0.1  # λ in the loss equation
    cfg.MODEL.MARS.LOSS_TYPE = "kl"  # "kl", "l2", or "cosine"
