#!/usr/bin/env python3
"""
Quick Diagnostic Script - Run this to identify why AP=0

Usage:
    python diagnose_ap_zero.py --config your_config.yaml --weights your_weights.pth
"""
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
import argparse
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as utils
import numpy as np
from collections import defaultdict
import json

def add_mars_config(cfg):
    """Add MARs configuration"""
    from detectron2.config import CfgNode as CN
    cfg.MODEL.MARS = CN()
    cfg.MODEL.MARS.ENABLED = True
    cfg.MODEL.MARS.WEIGHT = 1.0
    cfg.MODEL.MARS.LOSS_TYPE = "cosine"
    cfg.MODEL.MARS.USE_GEM = True
    cfg.MODEL.MARS.GEM_INIT_P = 1.0
    cfg.MODEL.MARS.GEM_MIN_P = 1.0
    cfg.MODEL.MARS.GEM_MAX_P = 6.0
    cfg.MODEL.MARS.WARMUP_ITERS = 3000

def register_coco_10class_subset(samples_per_class=100):
    """Register COCO 10-class subset"""
    from detectron2.structures import BoxMode
    
    coco_root = "/home/vaishali/projects/Mask2Former/datasets/coco"
    train_json = os.path.join(coco_root, "annotations/instances_train2017.json")
    train_image_root = os.path.join(coco_root, "train2017")
    val_json = os.path.join(coco_root, "annotations/instances_val2017.json")
    val_image_root = os.path.join(coco_root, "val2017")
    
    selected_class_ids = [1, 3, 17, 18, 19, 21, 44, 62, 63, 67]
    class_names = ["person", "car", "cat", "dog", "horse", 
                   "cow", "bottle", "chair", "couch", "dining table"]
    
    coco_to_new_id = {coco_id: i for i, coco_id in enumerate(selected_class_ids)}
    
    with open(val_json, 'r') as f:
        val_coco = json.load(f)
    
    def create_balanced_subset(coco_data, image_root, samples_per_class):
        class_to_images = defaultdict(set)
        for ann in coco_data['annotations']:
            cat_id = ann['category_id']
            if cat_id in selected_class_ids:
                class_to_images[cat_id].add(ann['image_id'])
        
        selected_image_ids = set()
        for cat_id in selected_class_ids:
            available_images = list(class_to_images[cat_id])
            np.random.seed(42)
            np.random.shuffle(available_images)
            sampled = available_images[:min(samples_per_class, len(available_images))]
            selected_image_ids.update(sampled)
        
        image_id_to_info = {img['id']: img for img in coco_data['images']}
        dataset_dicts = []
        image_to_anns = defaultdict(list)
        
        for ann in coco_data['annotations']:
            if ann['image_id'] in selected_image_ids and ann['category_id'] in selected_class_ids:
                image_to_anns[ann['image_id']].append(ann)
        
        for img_id in selected_image_ids:
            if img_id not in image_id_to_info:
                continue
            img_info = image_id_to_info[img_id]
            anns = image_to_anns[img_id]
            if not anns:
                continue
            
            record = {}
            record["file_name"] = os.path.join(image_root, img_info["file_name"])
            record["image_id"] = img_id
            record["height"] = img_info["height"]
            record["width"] = img_info["width"]
            
            objs = []
            for ann in anns:
                if ann['category_id'] not in selected_class_ids:
                    continue
                obj = {
                    "bbox": ann["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": coco_to_new_id[ann['category_id']],
                    "iscrowd": ann.get("iscrowd", 0)
                }
                objs.append(obj)
            
            if objs:
                record["annotations"] = objs
                dataset_dicts.append(record)
        
        return dataset_dicts
    
    val_subset = create_balanced_subset(val_coco, val_image_root, samples_per_class)
    
    DatasetCatalog.register("coco_2017_val_10class", lambda: val_subset)
    MetadataCatalog.get("coco_2017_val_10class").set(
        thing_classes=class_names,
        thing_dataset_id_to_contiguous_id=coco_to_new_id
    )
    
    return len(val_subset)

def diagnose(args):
    print("\n" + "="*80)
    print("DIAGNOSTIC SCRIPT FOR AP=0 ISSUE")
    print("="*80)
    
    # ============================================================================
    # TEST 1: Check Config Loading
    # ============================================================================
    print("\n[TEST 1] Checking configuration...")
    try:
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        
        # Initialize MASK_FORMER config with defaults (same as evaluation script)
        from detectron2.config import CfgNode as CN
        if not hasattr(cfg.MODEL, "MASK_FORMER"):
            cfg.MODEL.MASK_FORMER = CN()
        
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
        
        # SEM_SEG_HEAD config
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
            ssh.NUM_CLASSES = 10
        
        # Add MARS config
        add_mars_config(cfg)
        
        # Register dataset
        print("  Registering 10-class dataset...")
        num_images = register_coco_10class_subset(samples_per_class=100)
        print(f"  Registered dataset with {num_images} images")
        
        # Now load the YAML config (will override defaults)
        cfg.merge_from_file(args.config)
        print("✓ Config loaded successfully")
        
        # Check key settings
        print(f"  - Num classes: {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
        print(f"  - Test dataset: {cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else 'NONE'}")
        print(f"  - Backbone: {cfg.MODEL.BACKBONE.NAME}")
        
        # Check threshold
        if hasattr(cfg.MODEL, 'MASK_FORMER') and hasattr(cfg.MODEL.MASK_FORMER, 'TEST'):
            threshold = cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD
            print(f"  - Object threshold: {threshold}")
            if threshold > 0.5:
                print(f"    ⚠️  WARNING: Threshold {threshold} is very high!")
                print(f"    ⚠️  This may filter out most predictions.")
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return
    
    # ============================================================================
    # TEST 2: Check Dataset Registration
    # ============================================================================
    print("\n[TEST 2] Checking dataset...")
    try:
        dataset_name = cfg.DATASETS.TEST[0]
        dataset_dicts = DatasetCatalog.get(dataset_name)
        metadata = MetadataCatalog.get(dataset_name)
        
        print(f"✓ Dataset '{dataset_name}' found")
        print(f"  - Images: {len(dataset_dicts)}")
        print(f"  - Classes: {metadata.thing_classes}")
        
        # Check class IDs
        if hasattr(metadata, 'thing_dataset_id_to_contiguous_id'):
            print(f"  - Class ID mapping: {metadata.thing_dataset_id_to_contiguous_id}")
        
        # Count ground truth boxes
        total_gt = sum(len(d.get('annotations', [])) for d in dataset_dicts)
        print(f"  - Total GT boxes: {total_gt}")
        
        if total_gt == 0:
            print("  ✗ ERROR: No ground truth boxes in dataset!")
            return
            
    except Exception as e:
        print(f"✗ Dataset check failed: {e}")
        print(f"  Make sure to register the dataset before running this script!")
        return
    
    # ============================================================================
    # TEST 3: Check Model and Weights
    # ============================================================================
    print("\n[TEST 3] Checking model and weights...")
    try:
        from mask2former_mars.modeling.meta_arch.mars_mask_former_head_gem import MaskFormerWithMARS
        
        model = MaskFormerWithMARS(cfg)
        print("✓ Model created")
        
        # Load weights
        checkpoint = torch.load(args.weights, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                print(f"  - Checkpoint structure: dict with 'model' key")
            else:
                state_dict = checkpoint
                print(f"  - Checkpoint structure: direct state_dict")
        else:
            state_dict = checkpoint
            print(f"  - Checkpoint structure: unknown")
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        print(f"  - Missing keys: {len(missing)}")
        print(f"  - Unexpected keys: {len(unexpected)}")
        
        if len(missing) > 0:
            print(f"    First few missing: {list(missing)[:3]}")
            print("    ⚠️  WARNING: Missing keys may indicate wrong checkpoint!")
        
        if len(unexpected) > 0:
            print(f"    First few unexpected: {list(unexpected)[:3]}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  - Total parameters: {total_params:,}")
        
        model.eval()
        print("✓ Weights loaded and model set to eval mode")
        
    except Exception as e:
        print(f"✗ Model/weights check failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================================
    # TEST 4: Run Inference on One Image
    # ============================================================================
    print("\n[TEST 4] Running inference on test image...")
    try:
        # Get first image
        img_dict = dataset_dicts[0]
        image = utils.read_image(img_dict["file_name"], format="RGB")
        
        print(f"✓ Image loaded: {img_dict['file_name']}")
        print(f"  - Image shape: {image.shape}")
        print(f"  - Image ID: {img_dict.get('image_id')}")
        print(f"  - GT boxes: {len(img_dict.get('annotations', []))}")
        
        if len(img_dict.get('annotations', [])) > 0:
            gt_classes = [ann['category_id'] for ann in img_dict['annotations']]
            print(f"  - GT class IDs: {gt_classes}")
        
        # Prepare input
        height, width = image.shape[:2]
        image_tensor = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        
        print(f"  - Tensor shape: {image_tensor.shape}")
        print(f"  - Value range: [{image_tensor.min():.1f}, {image_tensor.max():.1f}]")
        print(f"  - Mean value: {image_tensor.mean():.1f}")
        
        # Run inference
        with torch.no_grad():
            inputs = [{
                "image": image_tensor,
                "height": height,
                "width": width
            }]
            
            outputs = model(inputs)
        
        print("✓ Inference completed")
        
        # Check outputs
        if not outputs:
            print("  ✗ ERROR: Model returned empty output!")
            return
        
        if "instances" not in outputs[0]:
            print("  ✗ ERROR: No 'instances' key in output!")
            print(f"  Output keys: {outputs[0].keys()}")
            return
        
        instances = outputs[0]["instances"]
        print(f"  - Raw predictions: {len(instances)}")
        
        if len(instances) == 0:
            print("  ✗ ERROR: Model produced 0 predictions!")
            print("\n  Possible causes:")
            print("    1. Confidence threshold too high")
            print("    2. Model not trained properly")
            print("    3. Input preprocessing mismatch")
            return
        
        # Check prediction details
        scores = instances.scores
        classes = instances.pred_classes
        boxes = instances.pred_boxes.tensor
        
        print(f"  - Scores range: [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"  - Unique classes: {classes.unique().tolist()}")
        print(f"  - Boxes shape: {boxes.shape}")
        
        # Filter by threshold (simulate what happens in evaluation)
        threshold = cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD if hasattr(cfg.MODEL, 'MASK_FORMER') else 0.5
        keep = scores > threshold
        print(f"\n  After threshold {threshold}:")
        print(f"  - Kept predictions: {keep.sum()}")
        
        if keep.sum() == 0:
            print("  ✗ ERROR: All predictions filtered by threshold!")
            print(f"  ✗ All {len(scores)} predictions have score < {threshold}")
            print("\n  FIX: Lower the threshold in your config:")
            print("       cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.05")
            return
        
        kept_classes = classes[keep].unique().tolist()
        print(f"  - Predicted classes: {kept_classes}")
        
        # Compare with GT classes
        gt_classes = set([ann['category_id'] for ann in img_dict.get('annotations', [])])
        pred_classes = set(kept_classes)
        
        print(f"\n  Ground truth classes: {sorted(gt_classes)}")
        print(f"  Predicted classes: {sorted(pred_classes)}")
        
        overlap = gt_classes.intersection(pred_classes)
        if not overlap:
            print("  ✗ ERROR: No overlap between GT and predicted classes!")
            print("\n  This means class IDs don't match.")
            print("  Possible causes:")
            print("    1. Model trained on different class set")
            print("    2. Class ID mapping issue")
            print("    3. Wrong dataset used for evaluation")
        else:
            print(f"  ✓ Class overlap found: {sorted(overlap)}")
        
        print("\n✓ All basic checks passed!")
        
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    
    if keep.sum() > 0 and overlap:
        print("\n✓ Model seems to be working correctly!")
        print("\nIf you're still getting AP=0, check:")
        print("  1. Are you evaluating on the correct dataset?")
        print("  2. Is the evaluation script using the same class mapping?")
        print("  3. Try running evaluation on just a few images first")
    else:
        print("\n✗ Issues found! See messages above for details.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    args = parser.parse_args()
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    diagnose(args)