import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.config import get_cfg, CfgNode as CN  # ← ADD CN HERE
from detectron2.projects.deeplab import add_deeplab_config  # ← ADD THIS
from mask2former import add_maskformer2_config
from mask2former_mars.modeling.meta_arch.mars_mask_former_head_gem import MaskFormerWithMARS
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import argparse
from detectron2.modeling import META_ARCH_REGISTRY
META_ARCH_REGISTRY.register(MaskFormerWithMARS)
from detectron2.data import DatasetCatalog, MetadataCatalog
# Import your MARs config - adjust path if needed
#from mask2former_mars.config import add_mars_config 

def register_coco_subset(train_images=5000, val_images=1000):
    """Register COCO subset datasets"""
    from detectron2.data.datasets import load_coco_json
    
    coco_root = "datasets/coco"
    train_json = os.path.join(coco_root, "annotations/instances_train2017.json")
    train_image_root = os.path.join(coco_root, "train2017")
    val_json = os.path.join(coco_root, "annotations/instances_val2017.json")
    val_image_root = os.path.join(coco_root, "val2017")
    
    # Load full datasets
    train_dicts = load_coco_json(train_json, train_image_root, "coco_2017_train")
    val_dicts = load_coco_json(val_json, val_image_root, "coco_2017_val")
    
    # Create subsets
    train_subset = train_dicts[:train_images]
    val_subset = val_dicts[:val_images]
    
    # Get metadata
    train_metadata = MetadataCatalog.get("coco_2017_train")
    val_metadata = MetadataCatalog.get("coco_2017_val")
    
    # Register training subset
    DatasetCatalog.register("coco_2017_train_subset", lambda: train_subset)
    train_subset_metadata = MetadataCatalog.get("coco_2017_train_subset")
    train_subset_metadata.set(thing_classes=train_metadata.thing_classes)
    if hasattr(train_metadata, "stuff_classes"):
        train_subset_metadata.set(stuff_classes=train_metadata.stuff_classes)
    
    # Register validation subset
    DatasetCatalog.register("coco_2017_val_subset", lambda: val_subset)
    val_subset_metadata = MetadataCatalog.get("coco_2017_val_subset")
    val_subset_metadata.set(thing_classes=val_metadata.thing_classes)
    if hasattr(val_metadata, "stuff_classes"):
        val_subset_metadata.set(stuff_classes=val_metadata.stuff_classes)
    
    print(f"✅ Registered datasets:")
    print(f"   coco_2017_train_subset: {len(train_subset)} images")
    print(f"   coco_2017_val_subset: {len(val_subset)} images")

def add_mars_config(cfg):
    from detectron2.config import CfgNode as CN
    cfg.MODEL.MARS = CN()
    cfg.MODEL.MARS.ENABLED = True
    cfg.MODEL.MARS.WEIGHT = 1.0
    cfg.MODEL.MARS.LOSS_TYPE = "cosine"
    cfg.MODEL.MARS.USE_GEM = True
    cfg.MODEL.MARS.GEM_INIT_P = 1.0   # start from 1.0 (mean) by default; sweep will override
    cfg.MODEL.MARS.GEM_MIN_P = 1.0
    cfg.MODEL.MARS.GEM_MAX_P = 6.0
    cfg.MODEL.MARS.WARMUP_ITERS = 3000

    # Optional SWEEP support
        # Optional SWEEP support
    cfg.SWEEP = CN()
    cfg.SWEEP.ENABLED = False
    cfg.SWEEP.SOLVER = CN()
    cfg.SWEEP.SOLVER.BASE_LR = [0.00005, 0.0001, 0.0002]
    cfg.SWEEP.MODEL = CN()
    cfg.SWEEP.MODEL.MARS_WEIGHT = [0.1, 0.5, 1.0]
    cfg.SWEEP.MODEL.LOSS_TYPE = ["kl", "cosine"]     # <-- ADD THIS
    cfg.SWEEP.MODEL.GEM_INIT_P = [1.0, 2.0, 3.0]     # <-- ADD THIS

def setup_cfg(args):
    """
    Config setup for visualization - must match training script setup
    """
    cfg = get_cfg()
    
    # Register base configs
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # --- Comprehensive MASK_FORMER config initialization ---
    if not hasattr(cfg.MODEL, "MASK_FORMER"):
        cfg.MODEL.MASK_FORMER = CN()
    
    # Set defaults for all MASK_FORMER keys
    mf = cfg.MODEL.MASK_FORMER
    if not hasattr(mf, "TRANSFORMER_DECODER_NAME"):
        mf.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"
    if not hasattr(mf, "TRANSFORMER_IN_FEATURE"):
        mf.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    if not hasattr(mf, "NUM_OBJECT_QUERIES"):
        mf.NUM_OBJECT_QUERIES = 100
    if not hasattr(mf, "DEEP_SUPERVISION"):
        mf.DEEP_SUPERVISION = True
    if not hasattr(mf, "NO_OBJECT_WEIGHT"):
        mf.NO_OBJECT_WEIGHT = 0.1
    if not hasattr(mf, "CLASS_WEIGHT"):
        mf.CLASS_WEIGHT = 2.0
    if not hasattr(mf, "MASK_WEIGHT"):
        mf.MASK_WEIGHT = 5.0
    if not hasattr(mf, "DICE_WEIGHT"):
        mf.DICE_WEIGHT = 5.0
    if not hasattr(mf, "HIDDEN_DIM"):
        mf.HIDDEN_DIM = 256
    if not hasattr(mf, "NUM_HEADS"):
        mf.NUM_HEADS = 8
    if not hasattr(mf, "DROPOUT"):
        mf.DROPOUT = 0.0
    if not hasattr(mf, "DIM_FEEDFORWARD"):
        mf.DIM_FEEDFORWARD = 2048
    if not hasattr(mf, "ENC_LAYERS"):
        mf.ENC_LAYERS = 0
    if not hasattr(mf, "DEC_LAYERS"):
        mf.DEC_LAYERS = 9
    if not hasattr(mf, "PRE_NORM"):
        mf.PRE_NORM = False
    if not hasattr(mf, "ENFORCE_INPUT_PROJ"):
        mf.ENFORCE_INPUT_PROJ = False
    if not hasattr(mf, "SIZE_DIVISIBILITY"):
        mf.SIZE_DIVISIBILITY = 32
    if not hasattr(mf, "TRAIN_NUM_POINTS"):
        mf.TRAIN_NUM_POINTS = 12544
    if not hasattr(mf, "OVERSAMPLE_RATIO"):
        mf.OVERSAMPLE_RATIO = 3.0
    if not hasattr(mf, "IMPORTANCE_SAMPLE_RATIO"):
        mf.IMPORTANCE_SAMPLE_RATIO = 0.75
    
    # TEST subconfig
    if not hasattr(mf, "TEST"):
        mf.TEST = CN()
    if not hasattr(mf.TEST, "SEMANTIC_ON"):
        mf.TEST.SEMANTIC_ON = True
    if not hasattr(mf.TEST, "INSTANCE_ON"):
        mf.TEST.INSTANCE_ON = True
    if not hasattr(mf.TEST, "PANOPTIC_ON"):
        mf.TEST.PANOPTIC_ON = False
    if not hasattr(mf.TEST, "OVERLAP_THRESHOLD"):
        mf.TEST.OVERLAP_THRESHOLD = 0.8
    if not hasattr(mf.TEST, "OBJECT_MASK_THRESHOLD"):
        mf.TEST.OBJECT_MASK_THRESHOLD = 0.8
    
    # SEM_SEG_HEAD config - critical for pixel decoder
    if not hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
        cfg.MODEL.SEM_SEG_HEAD = CN()
    
    ssh = cfg.MODEL.SEM_SEG_HEAD
    if not hasattr(ssh, "NAME"):
        ssh.NAME = "MaskFormerHead"
    if not hasattr(ssh, "PIXEL_DECODER_NAME"):
        ssh.PIXEL_DECODER_NAME = "MSDeformAttnPixelDecoder"
    if not hasattr(ssh, "IN_FEATURES"):
        ssh.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    if not hasattr(ssh, "DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES"):
        ssh.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    if not hasattr(ssh, "MASK_DIM"):
        ssh.MASK_DIM = 256
    if not hasattr(ssh, "NUM_CLASSES"):
        ssh.NUM_CLASSES = 80
    
    # Additional defaults
    if not hasattr(cfg.INPUT, "MIN_SIZE_TRAIN_SAMPLING"):
        cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    
    # DATALOADER settings
    if not hasattr(cfg.DATALOADER, "FILTER_EMPTY_ANNOTATIONS"):
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    
    # Add MARs config
    add_mars_config(cfg)
    # Register datasets
    register_coco_subset(train_images=5000, val_images=1000)
    # NOW load the YAML file (after all keys are registered)
    cfg.merge_from_file(args.config_file)
    
    # Set checkpoint
    cfg.MODEL.WEIGHTS = args.checkpoint
    
    return cfg


def compute_eigencam(attention_maps, image_size):
    """
    Compute EigenCAM from attention maps
    
    Args:
        attention_maps: List of attention tensors [num_layers x (batch, queries, spatial)]
        image_size: (H, W) of original image
    
    Returns:
        eigencam: (H, W) heatmap
    """
    # Average across all layers
    attn = torch.stack(attention_maps).mean(dim=0)  # (batch, queries, spatial)
    
    # Average across queries
    attn = attn.mean(dim=1)  # (batch, spatial)
    
    # Take first image in batch
    attn = attn[0]  # (spatial,)
    
    # Reshape to spatial dimensions (assuming square feature map)
    spatial_size = int(np.sqrt(attn.shape[0]))
    attn = attn.reshape(spatial_size, spatial_size)
    
    # Resize to original image size
    attn = F.interpolate(
        attn.unsqueeze(0).unsqueeze(0),
        size=image_size,
        mode='bilinear',
        align_corners=False
    )[0, 0]
    
    # Normalize to [0, 1]
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    
    return attn.cpu().numpy()


def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Overlay heatmap on image
    
    Args:
        image: (H, W, 3) RGB image in [0, 255]
        heatmap: (H, W) in [0, 1]
        alpha: transparency
    
    Returns:
        overlayed: (H, W, 3) RGB image
    """
    # Convert heatmap to color
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlayed = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
    
    return overlayed


class EigenCAMHook:
    """Hook to capture attention maps during forward pass"""
    
    def __init__(self):
        self.attention_maps = []
    
    def __call__(self, module, input, output):
        # Assuming output is attention weights
        if isinstance(output, torch.Tensor) and output.dim() == 3:
            self.attention_maps.append(output.detach())
    
    def clear(self):
        self.attention_maps = []


def generate_eigencam_grid(model, cfg, output_dir, num_samples=100):
    """
    Generate 10x10 grid of EigenCAM visualizations
    
    Args:
        model: Trained model
        cfg: Detectron2 config
        output_dir: Where to save the plot
        num_samples: Number of samples (should be 100 for 10x10 grid)
    """
    model.eval()
    
    # Print what dataset we're using
    print(f"\n{'='*70}")
    print(f"Dataset being used: {cfg.DATASETS.TEST[0]}")
    print(f"Number of samples to visualize: {num_samples}")
    print(f"{'='*70}\n")

    # Build validation dataloader
    mapper = DatasetMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)
    
    # Get all validation samples
    all_samples = list(data_loader)
    
    # Randomly sample 100
    random.seed(42)  # For reproducibility
    selected_samples = random.sample(all_samples, min(num_samples, len(all_samples)))
    
    # Register hooks to capture attention
    hook = EigenCAMHook()
    handles = []
    
    # Find attention modules (adjust based on your architecture)
    for name, module in model.named_modules():
        if 'attn' in name.lower() or 'attention' in name.lower():
            handle = module.register_forward_hook(hook)
            handles.append(handle)
    
    # Generate EigenCAMs
    eigencams = []
    images = []
    
    print(f"Generating EigenCAMs for {len(selected_samples)} samples...")
    
    with torch.no_grad():
        for idx, batched_input in enumerate(selected_samples):
            hook.clear()
            
            # Prepare input
            if not isinstance(batched_input, list):
                batched_input = [batched_input]
            
            # Forward pass
            _ = model(batched_input)
            
            # Get image
            img = batched_input[0]['image'].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            
            # Compute EigenCAM
            if len(hook.attention_maps) > 0:
                eigencam = compute_eigencam(hook.attention_maps, img.shape[:2])
                overlay = overlay_heatmap(img, eigencam, alpha=0.4)
            else:
                print(f"Warning: No attention captured for sample {idx}")
                overlay = img
            
            images.append(img)
            eigencams.append(overlay)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(selected_samples)}")
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Create 10x10 grid
    print("Creating visualization grid...")
    fig, axes = plt.subplots(10, 10, figsize=(30, 30))
    fig.suptitle('EigenCAM Visualizations (100 Random Validation Samples)', fontsize=24)
    
    for i in range(10):
        for j in range(10):
            idx = i * 10 + j
            if idx < len(eigencams):
                axes[i, j].imshow(eigencams[idx])
                axes[i, j].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / 'eigencam_grid.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved EigenCAM grid to: {output_path}")
    
    return output_path



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-samples', type=int, default=100)
    args = parser.parse_args()
    
    # Setup config - DO THIS ONLY ONCE
    cfg = setup_cfg(args)
    
    print(f"\n{'='*70}")
    print(f"Loading model from: {cfg.MODEL.WEIGHTS}")
    print(f"Config file: {args.config_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}\n")
    
    # Build model
    model = build_model(cfg)
    
    # Load checkpoint
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    
    print(f"✅ Model loaded successfully\n")
    
    # Generate visualizations
    generate_eigencam_grid(model, cfg, args.output_dir, args.num_samples)


if __name__ == '__main__':
    main()