"""
Mask2Former + MARs integration 
- Reads MODEL.MARS.* from cfg
- MARs loss supports {"kl", "cosine", "l2"}
- Optional GeM pooling (learnable p) before MARs loss
- Lambda (λ) warm-up
- Batched rotated view with size_divisibility to keep attention shapes aligned
- Lightweight decoder cross-attn hook (if available) to collect attention maps
"""

from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.structures import ImageList

# Base Mask2Former
from mask2former.maskformer_model import MaskFormer as BaseMaskFormer


# ------------------ GeM ------------------
class GeM(nn.Module):
    """Generalized Mean Pooling with learnable p (per module)."""
    
    def __init__(self, p_init: float = 3.0, eps: float = 1e-6, p_min: float = 1.0, p_max: float = 6.0):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p_init))
        self.eps = eps
        self.p_min = float(p_min)
        self.p_max = float(p_max)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # === DEBUG: Input ===
        # Only print occasionally to avoid spam
        if torch.rand(1).item() < 0.01:  # 1% of calls
            print(f"        [GeM] Input shape: {x.shape}")
            print(f"        [GeM] Input range: [{x.min().item():.6f}, {x.max().item():.6f}]")
            print(f"        [GeM] p value: {self.p.item():.4f}")
        
        # Constrain p for stability (non-in-place)
        p = self.p.clamp(self.p_min, self.p_max)
        
        # Apply GeM pooling
        x = x.clamp(min=self.eps)
        x = x.pow(p).mean(dim=-1, keepdim=True).pow(1.0 / p)
        
        # === DEBUG: Output ===
        if torch.rand(1).item() < 0.01:  # 1% of calls
            print(f"        [GeM] Output shape: {x.shape}")
            print(f"        [GeM] Output range: [{x.min().item():.6f}, {x.max().item():.6f}]")
        
        return x
"""
class GeM(nn.Module):
    #Generalized Mean Pooling with learnable p (per module).
    def __init__(self, p_init: float = 3.0, eps: float = 1e-6, p_min: float = 1.0, p_max: float = 6.0):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p_init))
        self.eps = eps
        self.p_min = float(p_min)
        self.p_max = float(p_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # constrain p for stability (non-in-place)
        p = self.p.clamp(self.p_min, self.p_max)
        x = x.clamp(min=self.eps)
        x = x.pow(p).mean(dim=-1, keepdim=True).pow(1.0 / p)
        return x
"""

# ------------------ MARs Loss ------------------
class MARsLoss(nn.Module):
    """
    Multi-view Attention Regularization across rotations.
    loss_type ∈ {"kl", "cosine", "l2"}. Optional GeM reduces noise.
    """
    def __init__(
        self,
        loss_type: str = "kl",
        use_gem: bool = False,
        gem_init_p: float = 3.0,
        gem_min_p: float = 1.0,
        gem_max_p: float = 6.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.loss_type = str(loss_type).lower()
        self.tau = float(temperature)
        self.gem = GeM(gem_init_p, p_min=gem_min_p, p_max=gem_max_p) if use_gem else None

        # Initialize GeM pooling
        if use_gem:
            self.gem_pool = GeM(
                p_init=gem_init_p,
                eps=1e-6,
                p_min=gem_min_p,
                p_max=gem_max_p
            )
        else:
            # Fallback to average pooling
            self.gem_pool = lambda x: x.mean(dim=-1, keepdim=True)
        
        print(f"[MARsLoss] Initialized with loss_type={loss_type}, use_gem={use_gem}, gem_p={gem_init_p}")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x / max(self.tau, 1e-8)
        x = torch.softmax(x, dim=-1).clamp(min=1e-8)  # Non-in-place
        return x
    """
    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        
        #Reduce spatial dimension via GeM pooling, keep query dimension
        #Input: [B, Q, S] where B=batch, Q=queries, S=spatial
        #Output: [B, Q] 
        
        # Apply GeM pooling over spatial dimension
        x = self.gem_pool(x)  # [B, Q, S] -> [B, Q, 1]
        x = x.squeeze(-1)     # [B, Q, 1] -> [B, Q]
        
        # Keep it as [B, Q] for cosine similarity computation
        
        return x  # Shape: [B, Q]
    """
    """
    def _reduce(self, t: torch.Tensor) -> torch.Tensor:
        # flatten (B, *) -> (B, L); optionally GeM -> (B,)
        t = t.flatten(start_dim=1)
        if self.gem is not None:
            t = self.gem(t).squeeze(-1)
        return t
    """
    
    def _reduce(self, x):
        x = self.gem_pool(x)   # [B, Q, S] -> [B, Q, 1]
        x = x.squeeze(-1)      # [B, Q, 1] -> [B, Q]
        x = x.reshape(-1)      # [B, Q] -> [B*Q] PRESERVES VARIATION!
        return x
    
    def forward(self, a_list: List[torch.Tensor], b_list: List[torch.Tensor]) -> torch.Tensor:
        # === DEBUG: Entry ===
        print(f"\n    [MARsLoss] Processing {len(a_list)} attention pairs")
        
        if not a_list or not b_list:
            print(f"    [MARsLoss] WARNING: Empty lists! Returning 0")
            return torch.tensor(0.0, device=a_list[0].device if a_list else "cpu")
        
        total, n = 0.0, 0
        
        for idx, (a, b) in enumerate(zip(a_list, b_list)):
            if a is None or b is None or a.shape != b.shape:
                print(f"    [MARsLoss] Layer {idx}: Skipping (None or shape mismatch)")
                continue
            
            # === DEBUG: Before reduction ===
            if idx == 0:  # Only print for first layer
                print(f"    [MARsLoss] Layer {idx} BEFORE reduction:")
                print(f"      a shape: {a.shape}, range: [{a.min().item():.6f}, {a.max().item():.6f}]")
                print(f"      b shape: {b.shape}, range: [{b.min().item():.6f}, {b.max().item():.6f}]")
            
            # Reduce via GeM pooling
            a = self._reduce(a)
            b = self._reduce(b)
            
            # === DEBUG: After reduction ===
            if idx == 0:
                print(f"    [MARsLoss] Layer {idx} AFTER reduction:")
                print(f"      a shape: {a.shape}, range: [{a.min().item():.6f}, {a.max().item():.6f}]")
                print(f"      b shape: {b.shape}, range: [{b.min().item():.6f}, {b.max().item():.6f}]")
            
            # Compute loss
            if self.loss_type == "kl":
                pa, pb = self._normalize(a), self._normalize(b)
                loss = 0.5 * (F.kl_div(pa.log(), pb, reduction="batchmean") +
                            F.kl_div(pb.log(), pa, reduction="batchmean"))
            elif self.loss_type == "cosine":
                # === DEBUG: Cosine similarity ===
                cos_sim = F.cosine_similarity(a, b, dim=-1)
                if idx == 0:
                    print(f"      Cosine similarity: range=[{cos_sim.min().item():.6f}, {cos_sim.max().item():.6f}], mean={cos_sim.mean().item():.6f}")
                loss = 1.0 - cos_sim.mean()
            elif self.loss_type == "l2":
                loss = F.mse_loss(a, b)
            else:
                loss = F.mse_loss(a, b)
            
            # === DEBUG: Loss per layer ===
            if idx == 0:
                print(f"      Loss value: {loss.item():.6f}")
            
            total += loss
            n += 1
        
        final_loss = total / max(n, 1)
        
        # === DEBUG: Final result ===
        print(f"    [MARsLoss] Processed {n} layers, Final loss: {final_loss.item():.6f}\n")
        
        return final_loss
    """
    def forward(self, a_list: List[torch.Tensor], b_list: List[torch.Tensor]) -> torch.Tensor:
        # DEBUG
        print(f"\n    [MARsLoss] Processing {len(attn_view1)} attention pairs")
        if not a_list or not b_list:
            return torch.tensor(0.0, device=a_list[0].device if a_list else "cpu")

        total, n = 0.0, 0
        for a, b in zip(a_list, b_list):
            if a is None or b is None or a.shape != b.shape:
                continue
            a = self._reduce(a)
            b = self._reduce(b)

            if self.loss_type == "kl":
                pa, pb = self._normalize(a), self._normalize(b)
                loss = 0.5 * (F.kl_div(pa.log(), pb, reduction="batchmean") +
                              F.kl_div(pb.log(), pa, reduction="batchmean"))
            elif self.loss_type == "cosine":
                loss = 1.0 - F.cosine_similarity(a, b, dim=-1).mean()
            elif self.loss_type == "l2":
                loss = F.mse_loss(a, b)
            else:
                loss = F.mse_loss(a, b)

            total += loss
            n += 1
        return total / max(n, 1)
    """

# ------------------ Model Wrapper ------------------
class MaskFormerWithMARS(BaseMaskFormer):
    """
    Adds MARs to Mask2Former during training:
      - view1: normal batch
      - view2: 90° CCW rotated batch (batched & size-divisible)
      - attention maps are collected from decoder cross-attn 
    """
    @configurable
    def __init__(
        self,
        *,
        mars_enabled: bool,
        mars_weight: float,
        mars_loss_type: str,
        mars_use_gem: bool,
        mars_gem_init_p: float,
        mars_gem_min_p: float,
        mars_gem_max_p: float,
        mars_warmup_iters: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mars_enabled = bool(mars_enabled)
        self.mars_weight = float(mars_weight)
        self.mars_warmup_iters = int(max(0, mars_warmup_iters))
        self._iter = 0

        self.mars_loss = MARsLoss(
            loss_type=mars_loss_type,
            use_gem=mars_use_gem,
            gem_init_p=mars_gem_init_p,
            gem_min_p=mars_gem_min_p,
            gem_max_p=mars_gem_max_p,
            temperature=1.0,
        )

        # buffers for attention maps from both views
        self._attn_view1: List[torch.Tensor] = []
        self._attn_view2: List[torch.Tensor] = []

        # optional: attach lightweight forward hooks to decoder cross-attn (if present)
        self._hooks = []
        self._attach_decoder_attn_hooks()

        print("\n" + "=" * 70)
        print("MARs Adaptation for Mask2Former")
        print("=" * 70)
        print(f"Status:      {'Enabled' if self.mars_enabled else 'Disabled'}")
        print(f"Weight (λ):  {self.mars_weight}")
        print(f"Loss Type:   {mars_loss_type}")
        print(f"GeM:         {'On' if mars_use_gem else 'Off'} (init p={mars_gem_init_p})")
        print(f"Warm-up iters: {self.mars_warmup_iters}")
        print("=" * 70 + "\n")

    # -------- configuration --------
    @classmethod
    def from_config(cls, cfg):
        base = super().from_config(cfg)
        m = cfg.MODEL.MARS
        base.update(dict(
            mars_enabled=bool(m.ENABLED),
            mars_weight=float(m.WEIGHT),
            mars_loss_type=str(m.LOSS_TYPE),
            mars_use_gem=bool(getattr(m, "USE_GEM", False)),
            mars_gem_init_p=float(getattr(m, "GEM_INIT_P", 3.0)),
            mars_gem_min_p=float(getattr(m, "GEM_MIN_P", 1.0)),
            mars_gem_max_p=float(getattr(m, "GEM_MAX_P", 6.0)),
            mars_warmup_iters=int(getattr(m, "WARMUP_ITERS", 3000)),
        ))
        return base

    # -------- attention hooks (FIXED for Mask2Former) --------
    def _attach_decoder_attn_hooks(self):
        """
        Hook Mask2Former decoder cross-attention to collect attention maps.
        Mask2Former structure: self.sem_seg_head.predictor.transformer_cross_attention_layers[i].multihead_attn
        """
        # Check for Mask2Former decoder
        if not hasattr(self, "sem_seg_head") or not hasattr(self.sem_seg_head, "predictor"):
            print("[WARN] No Mask2Former predictor found; MARs may be inactive.")
            return

        predictor = self.sem_seg_head.predictor
        
        # Check for cross-attention layers
        if not hasattr(predictor, "transformer_cross_attention_layers"):
            print("[WARN] No transformer_cross_attention_layers found; MARs may be inactive.")
            return

        def make_hook(buf: List[torch.Tensor]):
            def _hook(m, inp, out):
                # MultiheadAttention returns (attn_output, attn_weights)
                # We want attn_weights which is the second element
                if isinstance(out, (list, tuple)) and len(out) >= 2:
                    attn = out[1]  # attention weights
                    if isinstance(attn, torch.Tensor):
                        #with torch.no_grad():
                        buf.append(attn.detach())
            return _hook

        # Hook all cross-attention layers
        cross_attn_layers = predictor.transformer_cross_attention_layers
        count = 0
        for i, layer in enumerate(cross_attn_layers):
            if hasattr(layer, "multihead_attn"):
                h = layer.multihead_attn.register_forward_hook(make_hook(self._attn_view1))
                self._hooks.append(h)
                count += 1
        
        print(f"Attached MARs hooks to {count} cross-attention layers")

    def _clear_attn(self):
        self._attn_view1, self._attn_view2 = [], []

    def _swap_hook_buffer_to_view(self, view_id: int):
        """Swap hook buffer for view 2"""
        if not self._hooks:
            return
        
        target = self._attn_view1 if view_id == 1 else self._attn_view2
        
        # Remove old hooks
        for h in self._hooks:
            h.remove()
        self._hooks = []
        
        # Re-attach hooks to new buffer
        if not hasattr(self, "sem_seg_head") or not hasattr(self.sem_seg_head, "predictor"):
            return
        
        predictor = self.sem_seg_head.predictor
        if not hasattr(predictor, "transformer_cross_attention_layers"):
            return
        
        def make_hook(buf: List[torch.Tensor]):
            def _hook(m, inp, out):
                if isinstance(out, (list, tuple)) and len(out) >= 2:
                    attn = out[1]
                    if isinstance(attn, torch.Tensor):
                        #with torch.no_grad():
                        buf.append(attn.detach())
                            
            return _hook
        
        cross_attn_layers = predictor.transformer_cross_attention_layers
        for layer in cross_attn_layers:
            if hasattr(layer, "multihead_attn"):
                h = layer.multihead_attn.register_forward_hook(make_hook(target))
                self._hooks.append(h)

    # -------- utilities --------
    def _effective_lambda(self) -> float:
        if self.mars_warmup_iters <= 0:
            return self.mars_weight
        return self.mars_weight * min(1.0, float(self._iter) / float(self.mars_warmup_iters))

    def _rotate_instances_90(self, instances, orig_h: int, orig_w: int):
        """
        Rotate instances 90° CCW: (H, W) -> (W, H)
        - Bounding boxes: (x1, y1, x2, y2) -> (y1, W-x2, y2, W-x1)
        - Masks: rotate 90° CCW
        """
        from detectron2.structures import Instances, Boxes, BitMasks, PolygonMasks
        
        new_instances = Instances((orig_w, orig_h))  # Note: swapped dimensions
        
        # Rotate bounding boxes
        if instances.has("gt_boxes"):
            boxes = instances.gt_boxes.tensor  # (N, 4) as [x1, y1, x2, y2]
            # After 90° CCW rotation: x' = y, y' = W - x
            new_boxes = torch.stack([
                boxes[:, 1],           # x1' = y1
                orig_w - boxes[:, 2],  # y1' = W - x2
                boxes[:, 3],           # x2' = y2
                orig_w - boxes[:, 0],  # y2' = W - x1
            ], dim=1)
            new_instances.gt_boxes = Boxes(new_boxes)
        
        # Rotate masks
        if instances.has("gt_masks"):
            masks = instances.gt_masks
            if isinstance(masks, torch.Tensor):
                # Plain tensor: shape (N, H, W)
                rotated_masks = torch.rot90(masks, k=1, dims=[1, 2])  # CCW
                new_instances.gt_masks = rotated_masks
            elif isinstance(masks, BitMasks):
                # BitMasks: tensor of shape (N, H, W)
                mask_tensor = masks.tensor
                rotated_masks = torch.rot90(mask_tensor, k=1, dims=[1, 2])  # CCW
                new_instances.gt_masks = BitMasks(rotated_masks)
            elif isinstance(masks, PolygonMasks):
                # PolygonMasks: need to transform polygon coordinates
                # After 90° CCW: (x, y) -> (y, W - x)
                new_polygons = []
                for poly_list in masks.polygons:
                    new_poly_list = []
                    for poly in poly_list:
                        poly_arr = torch.as_tensor(poly).reshape(-1, 2)
                        x, y = poly_arr[:, 0], poly_arr[:, 1]
                        new_x = y
                        new_y = orig_w - x
                        new_poly = torch.stack([new_x, new_y], dim=1).flatten().tolist()
                        new_poly_list.append(new_poly)
                    new_polygons.append(new_poly_list)
                new_instances.gt_masks = PolygonMasks(new_polygons)
            else:
                raise NotImplementedError(f"Unsupported mask type: {type(masks)}")
        
        # Copy other fields without transformation
        for field in instances.get_fields():
            if field not in ["gt_boxes", "gt_masks"]:
                new_instances.set(field, instances.get(field))
        
        return new_instances

    # -------- forward --------
    def forward(self, batched_inputs: List[Dict[str, Any]]) -> Any:
        # === CRITICAL: Move inputs to GPU ===
        if self.training:
            # Move all input tensors to GPU
            for x in batched_inputs:
                if "image" in x and isinstance(x["image"], torch.Tensor):
                    x["image"] = x["image"].to(self.device)
                if "instances" in x:
                    x["instances"] = x["instances"].to(self.device)
        # === END GPU MOVE ===
        # === DEVICE CHECK (runs once) ===
        if self.training and self._iter == 0:
            print("\n" + "="*70)
            print("DEVICE DIAGNOSTIC")
            print("="*70)
            
            # Check model
            model_device = next(self.parameters()).device
            print(f"Model parameters: {model_device}")
            
            # Check input image
            input_img = batched_inputs[0]['image']
            print(f"Input image: {input_img.device}")
            
            # Check if instances exist
            if 'instances' in batched_inputs[0] and batched_inputs[0]['instances'].has('gt_boxes'):
                boxes = batched_inputs[0]['instances'].gt_boxes.tensor
                print(f"GT boxes: {boxes.device}")
            
            # Summary
            all_cuda = model_device.type == 'cuda' and input_img.device.type == 'cuda'
            if all_cuda:
                print("\nALL TENSORS ON GPU - TRAINING WILL BE FAST!")
            else:
                print("\n WARNING: SOME TENSORS ON CPU - TRAINING WILL BE SLOW!")
                
            print("="*70 + "\n")
        # === END DEVICE CHECK ===
        # EVAL = base forward
        if not self.training or not self.mars_enabled:
            return super().forward(batched_inputs)

        self._iter += 1
        self._clear_attn()

        # ----- View 1 -----
        self._swap_hook_buffer_to_view(view_id=1)
        out1 = super().forward(batched_inputs)  # expect dict of losses
        #self._attn_view1 = [a.detach() for a in self._attn_view1] 

        #print(f"[DEBUG] View 1: Captured {len(self._attn_view1)} attention tensors")
        #if self._attn_view1:
            #print(f"[DEBUG] First attention shape: {self._attn_view1[0].shape}")

        # ----- View 2 (90° CCW) -----
        rot_inputs, images = [], []
        for x in batched_inputs:
            xi = x.copy()
            img = xi["image"]  # (C,H,W)
            orig_h, orig_w = img.shape[1], img.shape[2]
            
            # Rotate image 90° CCW
            img_rot = torch.rot90(img, k=1, dims=[1, 2])  # (C, W, H) after rotation
            xi["image"] = img_rot
            
            # Update height/width metadata
            if "height" in xi and "width" in xi:
                xi["height"], xi["width"] = xi["width"], xi["height"]
            
            # Rotate ground truth instances (boxes and masks)
            if "instances" in xi:
                xi["instances"] = self._rotate_instances_90(xi["instances"], orig_h, orig_w)
            
            rot_inputs.append(xi)
            images.append(img_rot)

        # ========== FIXED DEBUG CODE ==========
        # Save first batch images for verification (iteration 1 only)
        
        """
        if self._iter == 1:
            import cv2
            import numpy as np
            from detectron2.utils.visualizer import Visualizer
            from detectron2.data import MetadataCatalog
            import os
            
            debug_dir = "debug_rotation"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Get metadata for visualization
            metadata = MetadataCatalog.get("coco_2017_train_subset")
            
            # Visualize first image from batch
            idx = 0
            
            # Original image + annotations
            img_orig = batched_inputs[idx]["image"].permute(1, 2, 0).cpu().numpy()
            img_orig = np.ascontiguousarray(img_orig)
            
            visualizer_orig = Visualizer(img_orig, metadata=metadata, scale=1.0)
            if "instances" in batched_inputs[idx]:
                vis_orig = visualizer_orig.draw_dataset_dict(batched_inputs[idx])
                cv2.imwrite(f"{debug_dir}/iter1_original.jpg", vis_orig.get_image())
            
            # Save plain original (BGR format for OpenCV)
            img_orig_bgr = cv2.cvtColor((img_orig * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{debug_dir}/iter1_original_with_annot.jpg", img_orig_bgr)
            
            # Rotated image + annotations
            img_rot_np = rot_inputs[idx]["image"].permute(1, 2, 0).cpu().numpy()
            img_rot_np = np.ascontiguousarray(img_rot_np)
            
            visualizer_rot = Visualizer(img_rot_np, metadata=metadata, scale=1.0)
            if "instances" in rot_inputs[idx]:
                vis_rot = visualizer_rot.draw_dataset_dict(rot_inputs[idx])
                cv2.imwrite(f"{debug_dir}/iter1_rotated.jpg", vis_rot.get_image())
            
            # Save plain rotated (BGR format for OpenCV)
            img_rot_bgr = cv2.cvtColor((img_rot_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{debug_dir}/iter1_rotated_with_annot.jpg", img_rot_bgr)
            
            # Print info (safely check for fields)
            print("\n" + "="*70)
            print("DEBUG: Saved rotation verification images to debug_rotation/")
            print("="*70)
            print(f"Original shape: {batched_inputs[idx]['image'].shape}")
            print(f"Rotated shape: {rot_inputs[idx]['image'].shape}")
            
            if "instances" in batched_inputs[idx]:
                orig_inst = batched_inputs[idx]['instances']
                rot_inst = rot_inputs[idx]['instances']
                print(f"Number of instances: {len(orig_inst)}")
                
                # Check what fields are available
                if orig_inst.has("gt_boxes"):
                    print(f"Original boxes (first 3): {orig_inst.gt_boxes.tensor[:3]}")
                    print(f"Rotated boxes (first 3): {rot_inst.gt_boxes.tensor[:3]}")
                
                if orig_inst.has("gt_masks"):
                    print(f"Has masks: True (type: {type(orig_inst.gt_masks)})")
                
                if orig_inst.has("gt_classes"):
                    print(f"Classes (first 5): {orig_inst.gt_classes[:5]}")
            
            print("="*70 + "\n")
        # ========== END DEBUG CODE ==========
        """
        
        # pack tensors with same size_divisibility as backbone/decoder
        size_div = getattr(self.backbone, "size_divisibility", 32)
        _ = ImageList.from_tensors(images, size_divisibility=size_div)  # ensures same padding policy (even if not used downstream)

        self._swap_hook_buffer_to_view(view_id=2)
        out2 = super().forward(rot_inputs)  # run the rotated batch through the same pipeline
        #self._attn_view2 = [a.detach() for a in self._attn_view2]
        #print(f"[DEBUG] View 2: Captured {len(self._attn_view2)} attention tensors")
        #if self._attn_view2:
            #print(f"[DEBUG] First attention shape: {self._attn_view2[0].shape}")

        # === ADD DEBUG PRINTS HERE (before MARs loss) ===
        if self._iter % 100 == 0:
            print(f"\n{'='*70}")
            print(f"[Iter {self._iter}] MARs Debug:")
            print(f"{'='*70}")
            print(f"  _attn_view1: {len(self._attn_view1) if self._attn_view1 else 'None/Empty'}")
            print(f"  _attn_view2: {len(self._attn_view2) if self._attn_view2 else 'None/Empty'}")
            if self._attn_view1 and len(self._attn_view1) > 0:
                print(f"  Sample attn shape: {self._attn_view1[0].shape}")
                print(f"  Sample attn device: {self._attn_view1[0].device}")
            else:
                print(f"    WARNING: No attention tensors captured!")
            print(f"{'='*70}\n")
        # === END DEBUG ===

        # ----- Collect base losses -----
        if isinstance(out1, dict) and "losses" in out1:
            losses = out1["losses"]
        elif isinstance(out1, dict):
            losses = out1
        else:
            raise RuntimeError("Unexpected base forward return; expected dict of losses in training.")

        # ----- MARs loss (only if both buffers captured something) -----
        # Get device from model parameters
        """
        device = next(self.parameters()).device
        mars_val = torch.tensor(0.0, device=device)
        #mars_val = torch.tensor(0.0, device=images[0].device) if images else torch.tensor(0.0)
        if self._attn_view1 and self._attn_view2:
            #print(f"[DEBUG] Computing MARs loss with {len(self._attn_view1)} tensors")
            try:
                mars_val = self.mars_loss(self._attn_view1, self._attn_view2)
                #print(f"[DEBUG] MARs loss value: {mars_val.item()}")
            except Exception as e:
                print(f"[WARN] MARs loss failed: {e}")
                import traceback
                traceback.print_exc()
                mars_val = torch.tensor(0.0, device=images[0].device)
        #else:
            #print(f"[DEBUG] Skipping MARs loss - view1: {len(self._attn_view1)}, view2: {len(self._attn_view2)}")
        losses["loss_mars"] = self._effective_lambda() * mars_val
        # (Optional) expose instrumentation
        from detectron2.utils.events import get_event_storage
        get_event_storage().put_scalar("mars_raw", float(mars_val.detach()))
        return losses
        """

        device = next(iter(losses.values())).device
        mars_val = torch.tensor(0.0, device=device)

        if self._attn_view1 and self._attn_view2:
            # === DEBUG: Check attention values ===
            if self._iter % 100 == 0:
                print(f"\n  Attention Statistics:")
                attn1_sample = self._attn_view1[0]
                attn2_sample = self._attn_view2[0]
                print(f"    View1 - min: {attn1_sample.min().item():.6f}, max: {attn1_sample.max().item():.6f}, mean: {attn1_sample.mean().item():.6f}")
                print(f"    View2 - min: {attn2_sample.min().item():.6f}, max: {attn2_sample.max().item():.6f}, mean: {attn2_sample.mean().item():.6f}")
                print(f"    Std dev: {attn1_sample.std().item():.6f}")
                print(f"    Are they identical? {torch.allclose(attn1_sample, attn2_sample, atol=1e-6)}")
            # === END DEBUG ===
            try:
                mars_val = self.mars_loss(self._attn_view1, self._attn_view2)
                # DEBUG: Print raw mars value
                if self._iter % 100 == 0:
                    print(f"  Raw mars_val: {mars_val.item():.6f}")
                    print(f"  Lambda: {self._effective_lambda():.6f}")
                    print(f"  Weighted loss_mars: {(self._effective_lambda() * mars_val).item():.9f}")
                # Ensure mars_val is on the same device
                if mars_val.device != device:
                    mars_val = mars_val.to(device)
            except Exception as e:
                print(f"[WARN] MARs loss computation failed: {e}")
                mars_val = torch.tensor(0.0, device=device)
        else:
            if self._iter % 100 == 0:
                print(f"   Attention tensors not captured!")
        losses["loss_mars"] = self._effective_lambda() * mars_val

        from detectron2.utils.events import get_event_storage
        get_event_storage().put_scalar("mars_raw", float(mars_val.detach()))

        return losses