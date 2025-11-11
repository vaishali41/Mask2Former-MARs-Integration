import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
from detectron2.config import get_cfg, CfgNode as CN
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog, DatasetCatalog
from mask2former import add_maskformer2_config
from detectron2.utils.visualizer import Visualizer
import cv2
import argparse

# Register model
import mask2former_mars.modeling.meta_arch.mars_mask_former_head_gem
from detectron2.modeling import META_ARCH_REGISTRY
from mask2former_mars.modeling.meta_arch.mars_mask_former_head_gem import MaskFormerWithMARS
META_ARCH_REGISTRY.register(MaskFormerWithMARS)

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
    cfg.MODEL.MARS = CN()
    cfg.MODEL.MARS.ENABLED = True
    cfg.MODEL.MARS.WEIGHT = 1.0
    cfg.MODEL.MARS.LOSS_TYPE = "cosine"
    cfg.MODEL.MARS.USE_GEM = True
    cfg.MODEL.MARS.GEM_INIT_P = 1.0
    cfg.MODEL.MARS.GEM_MIN_P = 1.0
    cfg.MODEL.MARS.GEM_MAX_P = 6.0
    cfg.MODEL.MARS.WARMUP_ITERS = 3000

    # Add SWEEP config
    cfg.SWEEP = CN()
    cfg.SWEEP.ENABLED = False
    cfg.SWEEP.SOLVER = CN()
    cfg.SWEEP.SOLVER.BASE_LR = [0.00005, 0.0001, 0.0002]
    cfg.SWEEP.MODEL = CN()
    cfg.SWEEP.MODEL.MARS_WEIGHT = [0.1, 0.5, 1.0]
    cfg.SWEEP.MODEL.LOSS_TYPE = ["kl", "cosine"]
    cfg.SWEEP.MODEL.GEM_INIT_P = [1.0, 2.0, 3.0]


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Add all MASK_FORMER defaults
    if not hasattr(cfg.MODEL, "MASK_FORMER"):
        cfg.MODEL.MASK_FORMER = CN()
    
    mf = cfg.MODEL.MASK_FORMER
    mf.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"
    mf.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    mf.NUM_OBJECT_QUERIES = 100
    mf.DEEP_SUPERVISION = True
    mf.NO_OBJECT_WEIGHT = 0.1
    mf.CLASS_WEIGHT = 2.0
    mf.MASK_WEIGHT = 5.0
    mf.DICE_WEIGHT = 5.0
    mf.HIDDEN_DIM = 256
    mf.NUM_HEADS = 8
    mf.DROPOUT = 0.0
    mf.DIM_FEEDFORWARD = 2048
    mf.ENC_LAYERS = 0
    mf.DEC_LAYERS = 9
    mf.PRE_NORM = False
    mf.ENFORCE_INPUT_PROJ = False
    mf.SIZE_DIVISIBILITY = 32
    mf.TRAIN_NUM_POINTS = 12544
    mf.OVERSAMPLE_RATIO = 3.0
    mf.IMPORTANCE_SAMPLE_RATIO = 0.75
    
    mf.TEST = CN()
    mf.TEST.SEMANTIC_ON = True
    mf.TEST.INSTANCE_ON = True
    mf.TEST.PANOPTIC_ON = False
    mf.TEST.OVERLAP_THRESHOLD = 0.8
    mf.TEST.OBJECT_MASK_THRESHOLD = 0.8
    mf.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    
    # SEM_SEG_HEAD
    if not hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
        cfg.MODEL.SEM_SEG_HEAD = CN()
    
    ssh = cfg.MODEL.SEM_SEG_HEAD
    ssh.NAME = "MaskFormerHead"
    ssh.PIXEL_DECODER_NAME = "MSDeformAttnPixelDecoder"
    ssh.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    ssh.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    ssh.MASK_DIM = 256
    ssh.NUM_CLASSES = 80
    
    if not hasattr(cfg.INPUT, "MIN_SIZE_TRAIN_SAMPLING"):
        cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    
    if not hasattr(cfg.DATALOADER, "FILTER_EMPTY_ANNOTATIONS"):
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    
    add_mars_config(cfg)
    # Register datasets
    register_coco_subset(train_images=5000, val_images=1000)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.checkpoint
    
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SIMPLE MODEL TEST")
    print("="*70)
    
    # Setup
    cfg = setup_cfg(args)
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()
    
    print("✅ Model loaded")
    
    # Get a test image
    from detectron2.data import build_detection_test_loader, DatasetMapper
    mapper = DatasetMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)
    
    # Get first image
    for batched_input in data_loader:
        if not isinstance(batched_input, list):
            batched_input = [batched_input]
        break
    
    print(f"✅ Got test image")
    
    # Run inference
    with torch.no_grad():
        outputs = model(batched_input)
    
    print(f"\n{'='*70}")
    print("MODEL OUTPUT ANALYSIS")
    print(f"{'='*70}")
    print(f"Output type: {type(outputs)}")
    print(f"Output is list: {isinstance(outputs, list)}")
    
    if isinstance(outputs, list):
        print(f"Number of outputs: {len(outputs)}")
        if len(outputs) > 0:
            print(f"First output type: {type(outputs[0])}")
            print(f"First output keys: {outputs[0].keys()}")
            
            if "instances" in outputs[0]:
                instances = outputs[0]["instances"]
                print(f"\n✅ Found 'instances' in output")
                print(f"   Number of instances: {len(instances)}")
                
                if len(instances) > 0:
                    print(f"   Fields: {instances.get_fields().keys()}")
                    
                    scores = instances.scores.cpu().numpy()
                    classes = instances.pred_classes.cpu().numpy()
                    
                    print(f"\n   Score statistics:")
                    print(f"     Min: {scores.min():.6f}")
                    print(f"     Max: {scores.max():.6f}")
                    print(f"     Mean: {scores.mean():.6f}")
                    print(f"     Median: {sorted(scores)[len(scores)//2]:.6f}")
                    
                    print(f"\n   Top 10 predictions:")
                    top_indices = scores.argsort()[-10:][::-1]
                    metadata = MetadataCatalog.get("coco_2017_val")
                    for i, idx in enumerate(top_indices):
                        class_name = metadata.thing_classes[classes[idx]]
                        print(f"     {i+1}. {class_name:20s} (score: {scores[idx]:.4f})")
                    
                    # Count predictions above various thresholds
                    print(f"\n   Predictions above threshold:")
                    for thresh in [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:
                        count = (scores >= thresh).sum()
                        print(f"     ≥ {thresh:.2f}: {count}")
                else:
                    print(f"   ❌ No instances detected!")
            else:
                print(f"   ❌ No 'instances' key in output")
                print(f"   Available keys: {list(outputs[0].keys())}")
    else:
        print(f"❌ Output is not a list!")
        print(f"   Output: {outputs}")
    
    print(f"{'='*70}\n")
    
    # Check ground truth
    if "instances" in batched_input[0]:
        gt_instances = batched_input[0]["instances"]
        print(f"Ground truth objects: {len(gt_instances)}")
        gt_classes = gt_instances.gt_classes.cpu().numpy()
        print(f"GT class distribution: {dict(zip(*np.unique(gt_classes, return_counts=True)))}")
    
    print("\nDONE!")


if __name__ == '__main__':
    import numpy as np
    main()