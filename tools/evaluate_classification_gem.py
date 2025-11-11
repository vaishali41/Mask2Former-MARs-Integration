import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import argparse
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.config import get_cfg, CfgNode as CN
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetMapper
import copy
from mask2former import add_maskformer2_config

# Import to register model
import mask2former_mars.modeling.meta_arch.mars_mask_former_head_gem
from detectron2.modeling import META_ARCH_REGISTRY

# Manual registration if needed
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

def evaluate_classification(model, cfg, conf_threshold=0.1):
    """
    Evaluate classification accuracy and generate confusion matrix
    
    Args:
        model: Trained model
        cfg: Config
        conf_threshold: Confidence threshold for predictions
    
    Returns:
        accuracy: Overall accuracy
        confusion_matrix: Confusion matrix
        per_class_accuracy: Per-class accuracy
    """
    model.eval()
    
    # Get dataset metadata
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    num_classes = len(metadata.thing_classes)
    class_names = metadata.thing_classes
    
    print(f"\n{'='*70}")
    print(f"Evaluating on: {cfg.DATASETS.TEST[0]}")
    print(f"Number of classes: {num_classes}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"{'='*70}\n")
    
    # Build dataloader
    #mapper = DatasetMapper(cfg, is_train=False)
    #data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)
    mapper_cfg = copy.deepcopy(cfg)
    mapper = DatasetMapper(mapper_cfg, is_train=True, augmentations=[])  # ← is_train=True to load GT
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    # Track predictions and ground truths
    all_pred_classes = []
    all_gt_classes = []
    
    # Statistics
    total_images = 0
    images_with_gt = 0
    images_with_preds = 0
    total_matches = 0
    
    print("Running inference on validation set...")
    
    with torch.no_grad():
        for idx, batched_input in enumerate(tqdm(data_loader)):
            total_images += 1
            
            # ===== BASIC DEBUG - ALWAYS PRINTS =====
            if idx < 5:
                print(f"\n{'='*50}")
                print(f"Image {idx}")
                print(f"Batched input type: {type(batched_input)}")
                print(f"Batched input length: {len(batched_input) if isinstance(batched_input, list) else 'not a list'}")
                if isinstance(batched_input, list) and len(batched_input) > 0:
                    print(f"First element type: {type(batched_input[0])}")
                    print(f"Has 'instances' key: {'instances' in batched_input[0]}")
            # ===== END BASIC DEBUG =====
            
            # Get ground truth
            if "instances" not in batched_input[0]:
                if idx < 5:
                    print("⚠️ No ground truth instances, skipping")
                continue
            
            images_with_gt += 1
            gt_instances = batched_input[0]["instances"]
            gt_classes = gt_instances.gt_classes.cpu().numpy()
            
            if idx < 5:
                print(f"Ground truth objects: {len(gt_classes)}")
                print(f"GT classes: {gt_classes[:10]}")  # Show first 10
            
            # Run inference (ONLY ONCE!)
            outputs = model(batched_input)
            
            if idx < 5:
                print(f"Output type: {type(outputs)}")
                print(f"Output length: {len(outputs)}")
                if len(outputs) > 0:
                    print(f"Output[0] type: {type(outputs[0])}")
                    print(f"Output[0] keys: {outputs[0].keys() if isinstance(outputs[0], dict) else 'not a dict'}")
            
            # Get predictions
            if "instances" not in outputs[0]:
                if idx < 5:
                    print("⚠️ No 'instances' in outputs[0], skipping")
                continue
            
            images_with_preds += 1
            pred_instances = outputs[0]["instances"]
            
            # Get prediction data
            scores = pred_instances.scores.cpu().numpy()
            pred_classes = pred_instances.pred_classes.cpu().numpy()
            pred_boxes = pred_instances.pred_boxes.tensor.cpu().numpy()
            
            # ===== DETAILED DEBUG =====
            if idx < 5:
                print(f"\nPredictions before filtering:")
                print(f"  Total predictions: {len(scores)}")
                if len(scores) > 0:
                    print(f"  Score range: {scores.min():.6f} to {scores.max():.6f}")
                    print(f"  Score mean: {scores.mean():.6f}")
                    print(f"  Score median: {np.median(scores):.6f}")
                    print(f"  Predictions >= 0.001: {(scores >= 0.001).sum()}")
                    print(f"  Predictions >= 0.01: {(scores >= 0.01).sum()}")
                    print(f"  Predictions >= 0.05: {(scores >= 0.05).sum()}")
                    print(f"  Predictions >= 0.1: {(scores >= 0.1).sum()}")
                    print(f"  Top 10 scores: {np.sort(scores)[-10:]}")
                else:
                    print("  No predictions at all!")
            # ===== END DETAILED DEBUG =====

            # Apply confidence threshold
            mask = scores >= conf_threshold
            pred_classes_filtered = pred_classes[mask]
            pred_boxes_filtered = pred_boxes[mask]
            scores_filtered = scores[mask]
            
            if idx < 5:
                print(f"\nAfter threshold {conf_threshold}:")
                print(f"  Remaining predictions: {len(pred_classes_filtered)}")
            
            # Match predictions to ground truth using IoU
            # Match predictions to ground truth using MASK IoU
            # Match predictions to ground truth using BOX IoU with lower threshold
            if len(pred_classes_filtered) > 0 and len(gt_classes) > 0:
                gt_boxes = gt_instances.gt_boxes.tensor.cpu().numpy()
                
                # ===== CRITICAL DEBUG - CHECK BOX COORDINATES =====
                if idx < 3:
                    print(f"\n=== DETAILED BOX DEBUG ===")
                    print(f"Image shape: {batched_input[0]['image'].shape}")  # (C, H, W)
                    
                    print(f"\nGround Truth Boxes (first 3):")
                    for i in range(min(3, len(gt_boxes))):
                        box = gt_boxes[i]
                        print(f"  GT {i}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}] - w={box[2]-box[0]:.1f}, h={box[3]-box[1]:.1f}")
                    
                    print(f"\nPredicted Boxes (first 3):")
                    for i in range(min(3, len(pred_boxes_filtered))):
                        box = pred_boxes_filtered[i]
                        print(f"  Pred {i}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}] - w={box[2]-box[0]:.1f}, h={box[3]-box[1]:.1f}")
                    
                    # Check if boxes are degenerate (zero area)
                    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
                    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
                    pred_widths = pred_boxes_filtered[:, 2] - pred_boxes_filtered[:, 0]
                    pred_heights = pred_boxes_filtered[:, 3] - pred_boxes_filtered[:, 1]
                    
                    print(f"\nBox Statistics:")
                    print(f"  GT boxes - width range: [{gt_widths.min():.1f}, {gt_widths.max():.1f}]")
                    print(f"  GT boxes - height range: [{gt_heights.min():.1f}, {gt_heights.max():.1f}]")
                    print(f"  Pred boxes - width range: [{pred_widths.min():.1f}, {pred_widths.max():.1f}]")
                    print(f"  Pred boxes - height range: [{pred_heights.min():.1f}, {pred_heights.max():.1f}]")
                    
                    # Check if boxes are off-image
                    h, w = batched_input[0]['image'].shape[1], batched_input[0]['image'].shape[2]
                    print(f"\nImage dimensions: {w}x{h}")
                    print(f"  GT boxes in bounds: {((gt_boxes >= 0) & (gt_boxes[:, [0,2]] <= w) & (gt_boxes[:, [1,3]] <= h)).all(axis=1).sum()}/{len(gt_boxes)}")
                    print(f"  Pred boxes in bounds: {((pred_boxes_filtered >= 0) & (pred_boxes_filtered[:, [0,2]] <= w) & (pred_boxes_filtered[:, [1,3]] <= h)).all(axis=1).sum()}/{len(pred_boxes_filtered)}")
                # ===== END CRITICAL DEBUG =====

                # Compute BOX IoU
                ious = compute_iou_matrix(pred_boxes_filtered, gt_boxes)
                
                if idx < 5:
                    print(f"\nBox IoU Analysis:")
                    print(f"  IoU matrix shape: {ious.shape}")
                    print(f"  Max IoU: {ious.max():.4f}")
                    print(f"  Mean IoU: {ious.mean():.4f}")
                    print(f"  Matches IoU > 0.5: {(ious > 0.5).sum()}")
                    print(f"  Matches IoU > 0.3: {(ious > 0.3).sum()}")
                    print(f"  Matches IoU > 0.1: {(ious > 0.1).sum()}")
                    print(f"  Matches IoU > 0.01: {(ious > 0.01).sum()}")
                    print(f"  Matches IoU > 0.001: {(ious > 0.001).sum()}")
                
                # Match with IoU > 0.01 (VERY LOW threshold for poorly trained model)
                matches_this_image = 0
                for gt_idx in range(len(gt_classes)):
                    best_pred_idx = np.argmax(ious[:, gt_idx])
                    best_iou = ious[best_pred_idx, gt_idx]
                    
                    if best_iou > 0.01:  # Lower threshold
                        gt_class = gt_classes[gt_idx]
                        pred_class = pred_classes_filtered[best_pred_idx]
                        
                        all_gt_classes.append(gt_class)
                        all_pred_classes.append(pred_class)
                        confusion_matrix[gt_class, pred_class] += 1
                        matches_this_image += 1
                
                total_matches += matches_this_image
                
                if idx < 5:
                    print(f"  Matched pairs (IoU > 0.01): {matches_this_image}")
            elif idx < 5:
                print(f"⚠️ Skipping matching: pred={len(pred_classes_filtered)}, gt={len(gt_classes)}")

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1} images, Total matches so far: {total_matches}")
            """
            if len(pred_classes_filtered) > 0 and len(gt_classes) > 0:
                if idx < 5:
                    print(f"\nAvailable GT fields: {gt_instances.get_fields().keys()}")
                    print(f"Available Pred fields: {pred_instances.get_fields().keys()}")
                # Get MASKS instead of boxes
                pred_masks = pred_instances.pred_masks.cpu().numpy()  # (N, H, W)
                pred_masks_filtered = pred_masks[mask]  # Filter by confidence threshold
                gt_masks = gt_instances.gt_masks.cpu().numpy()  # (M, H, W)
                
                # Compute MASK IoU
                ious = compute_mask_iou_matrix(pred_masks_filtered, gt_masks)
                
                if idx < 5:
                    print(f"\nMask IoU Analysis:")
                    print(f"  IoU matrix shape: {ious.shape}")
                    print(f"  Max IoU: {ious.max():.4f}")
                    print(f"  Mean IoU: {ious.mean():.4f}")
                    print(f"  Matches IoU > 0.5: {(ious > 0.5).sum()}")
                    print(f"  Matches IoU > 0.3: {(ious > 0.3).sum()}")
                    print(f"  Matches IoU > 0.1: {(ious > 0.1).sum()}")
                    print(f"  Matches IoU > 0.05: {(ious > 0.05).sum()}")
                
                # Match with IoU > 0.5
                matches_this_image = 0
                for gt_idx in range(len(gt_classes)):
                    best_pred_idx = np.argmax(ious[:, gt_idx])
                    best_iou = ious[best_pred_idx, gt_idx]
                    
                    if best_iou > 0.5:
                        gt_class = gt_classes[gt_idx]
                        pred_class = pred_classes_filtered[best_pred_idx]
                        
                        all_gt_classes.append(gt_class)
                        all_pred_classes.append(pred_class)
                        confusion_matrix[gt_class, pred_class] += 1
                        matches_this_image += 1
                
                total_matches += matches_this_image
                
                if idx < 5:
                    print(f"  Matched pairs (IoU > 0.5): {matches_this_image}")
            """
            
            """
            if len(pred_classes_filtered) > 0 and len(gt_classes) > 0:
                gt_boxes = gt_instances.gt_boxes.tensor.cpu().numpy()
                
                # Compute IoU
                ious = compute_iou_matrix(pred_boxes_filtered, gt_boxes)
                
                if idx < 5:
                    print(f"\nIoU Analysis:")
                    print(f"  IoU matrix shape: {ious.shape}")
                    print(f"  Max IoU: {ious.max():.4f}")
                    print(f"  Mean IoU: {ious.mean():.4f}")
                    print(f"  Matches IoU > 0.5: {(ious > 0.5).sum()}")
                    print(f"  Matches IoU > 0.3: {(ious > 0.3).sum()}")
                    print(f"  Matches IoU > 0.1: {(ious > 0.1).sum()}")
                    print(f"  Matches IoU > 0.05: {(ious > 0.05).sum()}")
                
                # Match with IoU > 0.5
                matches_this_image = 0
                for gt_idx in range(len(gt_classes)):
                    best_pred_idx = np.argmax(ious[:, gt_idx])
                    best_iou = ious[best_pred_idx, gt_idx]
                    
                    if best_iou > 0.5:
                        gt_class = gt_classes[gt_idx]
                        pred_class = pred_classes_filtered[best_pred_idx]
                        
                        all_gt_classes.append(gt_class)
                        all_pred_classes.append(pred_class)
                        confusion_matrix[gt_class, pred_class] += 1
                        matches_this_image += 1
                
                total_matches += matches_this_image
                
                if idx < 5:
                    print(f"  Matched pairs (IoU > 0.5): {matches_this_image}")
            
            elif idx < 5:
                print(f"⚠️ Skipping matching: pred={len(pred_classes_filtered)}, gt={len(gt_classes)}")
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1} images, Total matches so far: {total_matches}")
            """
    # Print final statistics
    print(f"\n{'='*70}")
    print(f"DATASET STATISTICS:")
    print(f"  Total images processed: {total_images}")
    print(f"  Images with ground truth: {images_with_gt}")
    print(f"  Images with predictions: {images_with_preds}")
    print(f"  Total matches (IoU > 0.5): {total_matches}")
    print(f"  Total GT objects matched: {len(all_gt_classes)}")
    print(f"{'='*70}\n")
    
    # Calculate metrics
    all_pred_classes = np.array(all_pred_classes)
    all_gt_classes = np.array(all_gt_classes)
    
    # Overall accuracy
    accuracy = np.sum(all_pred_classes == all_gt_classes) / len(all_gt_classes) if len(all_gt_classes) > 0 else 0
    
    # Per-class accuracy
    per_class_accuracy = {}
    for class_id in range(num_classes):
        mask = all_gt_classes == class_id
        if np.sum(mask) > 0:
            class_acc = np.sum((all_pred_classes == all_gt_classes) & mask) / np.sum(mask)
            per_class_accuracy[class_names[class_id]] = class_acc
        else:
            per_class_accuracy[class_names[class_id]] = 0.0
    
    return accuracy, confusion_matrix, per_class_accuracy, class_names

"""    
def evaluate_classification(model, cfg, conf_threshold=0.1):
    
    Evaluate classification accuracy and generate confusion matrix
    
    Args:
        model: Trained model
        cfg: Config
        conf_threshold: Confidence threshold for predictions
    
    Returns:
        accuracy: Overall accuracy
        confusion_matrix: Confusion matrix
        per_class_accuracy: Per-class accuracy
    
    model.eval()
    
    # Get dataset metadata
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    num_classes = len(metadata.thing_classes)
    class_names = metadata.thing_classes
    
    print(f"\n{'='*70}")
    print(f"Evaluating on: {cfg.DATASETS.TEST[0]}")
    print(f"Number of classes: {num_classes}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"{'='*70}\n")
    
    # Build dataloader
    mapper = DatasetMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    # Track predictions and ground truths
    all_pred_classes = []
    all_gt_classes = []
    
    print("Running inference on validation set...")
    
    with torch.no_grad():
        for idx, batched_input in enumerate(tqdm(data_loader)):
            #if not isinstance(batched_input, list):
                #batched_input = [batched_input]
            outputs = model(batched_input)
            # Get ground truth
            if "instances" in batched_input[0]:
                gt_instances = batched_input[0]["instances"]
                gt_classes = gt_instances.gt_classes.cpu().numpy()
            else:
                continue
            
            # Run inference
            outputs = model(batched_input)
            
            # Get predictions
            if "instances" in outputs[0]:
                pred_instances = outputs[0]["instances"]
                
                # Filter by confidence
                scores = pred_instances.scores.cpu().numpy()
                pred_classes = pred_instances.pred_classes.cpu().numpy()
                pred_boxes = pred_instances.pred_boxes.tensor.cpu().numpy()
                
                # ===== ADD DEBUGGING HERE =====
                if idx < 5:  # Debug first 5 images
                    print(f"\n=== Image {idx} ===")
                    print(f"Total predictions before threshold: {len(scores)}")
                    print(f"Score range: {scores.min():.4f} - {scores.max():.4f}")
                    print(f"Predictions with score >= 0.001: {(scores >= 0.001).sum()}")
                    print(f"Predictions with score >= 0.01: {(scores >= 0.01).sum()}")
                    print(f"Predictions with score >= 0.05: {(scores >= 0.05).sum()}")
                    print(f"Ground truth objects: {len(gt_classes)}")
                # ===== END DEBUGGING =====

                # Apply confidence threshold
                mask = scores >= conf_threshold
                pred_classes = pred_classes[mask]
                pred_boxes = pred_boxes[mask]
                scores = scores[mask]
                
                # ===== ADD MORE DEBUGGING =====
                if idx < 5:
                    print(f"Predictions after filtering: {len(pred_classes)}")
                    if len(pred_classes) > 0 and len(gt_classes) > 0:
                        gt_boxes = gt_instances.gt_boxes.tensor.cpu().numpy()
                        ious = compute_iou_matrix(pred_boxes, gt_boxes)
                        print(f"IoU matrix shape: {ious.shape}")
                        print(f"Max IoU: {ious.max():.4f}")
                        print(f"Mean IoU: {ious.mean():.4f}")
                        print(f"Matches with IoU > 0.5: {(ious > 0.5).sum()}")
                        print(f"Matches with IoU > 0.3: {(ious > 0.3).sum()}")
                        print(f"Matches with IoU > 0.1: {(ious > 0.1).sum()}")
                # ===== END DEBUGGING =====

                # Match predictions to ground truth using IoU
                if len(pred_classes) > 0 and len(gt_classes) > 0:
                    gt_boxes = gt_instances.gt_boxes.tensor.cpu().numpy()
                    
                    # Compute IoU between all pred and gt boxes
                    ious = compute_iou_matrix(pred_boxes, gt_boxes)
                    
                    # Match: for each GT, find best matching prediction
                    matched_preds = []
                    for gt_idx in range(len(gt_classes)):
                        best_pred_idx = np.argmax(ious[:, gt_idx])
                        best_iou = ious[best_pred_idx, gt_idx]
                        
                        if best_iou > 0.5:  # IoU threshold for matching
                            gt_class = gt_classes[gt_idx]
                            pred_class = pred_classes[best_pred_idx]
                            
                            all_gt_classes.append(gt_class)
                            all_pred_classes.append(pred_class)
                            
                            # Update confusion matrix
                            confusion_matrix[gt_class, pred_class] += 1
                            matched_preds.append(best_pred_idx)
                    
                    # Count unmatched GT as false negatives (predicted as background)
                    # This is optional - depends on how you want to handle it
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1} images")
    
    # Calculate metrics
    all_pred_classes = np.array(all_pred_classes)
    all_gt_classes = np.array(all_gt_classes)
    
    # Overall accuracy
    accuracy = np.sum(all_pred_classes == all_gt_classes) / len(all_gt_classes) if len(all_gt_classes) > 0 else 0
    
    # Per-class accuracy
    per_class_accuracy = {}
    for class_id in range(num_classes):
        mask = all_gt_classes == class_id
        if np.sum(mask) > 0:
            class_acc = np.sum((all_pred_classes == all_gt_classes) & mask) / np.sum(mask)
            per_class_accuracy[class_names[class_id]] = class_acc
        else:
            per_class_accuracy[class_names[class_id]] = 0.0
    
    return accuracy, confusion_matrix, per_class_accuracy, class_names

"""

def compute_mask_iou_matrix(masks1, masks2):
    """
    Compute IoU between two sets of binary masks
    
    Args:
        masks1: (N, H, W) boolean/binary array
        masks2: (M, H, W) boolean/binary array
    
    Returns:
        iou_matrix: (N, M) array of IoU values
    """
    N = masks1.shape[0]
    M = masks2.shape[0]
    
    iou_matrix = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            mask1 = masks1[i] > 0.5  # Binarize if needed
            mask2 = masks2[j] > 0.5
            
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0
    
    return iou_matrix

def compute_iou_matrix(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: (N, 4) array in (x1, y1, x2, y2) format
        boxes2: (M, 4) array in (x1, y1, x2, y2) format
    
    Returns:
        iou_matrix: (N, M) array of IoU values
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    iou_matrix = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            box1 = boxes1[i]
            box2 = boxes2[j]
            
            # Intersection
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
            else:
                intersection = 0
            
            # Union
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            iou_matrix[i, j] = intersection / union if union > 0 else 0
    
    return iou_matrix


def plot_confusion_matrix(confusion_matrix, class_names, output_path, normalize=False):
    """
    Plot and save confusion matrix
    
    Args:
        confusion_matrix: (num_classes, num_classes) array
        class_names: List of class names
        output_path: Where to save the plot
        normalize: Whether to normalize by row (show percentages)
    """
    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm = confusion_matrix.astype('float') / row_sums
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        cm = confusion_matrix
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage' if normalize else 'Count'})
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved confusion matrix to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--conf-threshold', type=float, default=0.1,
                       help='Confidence threshold for predictions')
    args = parser.parse_args()
    
    # Setup config
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
    
    # Evaluate
    accuracy, confusion_matrix, per_class_accuracy, class_names = evaluate_classification(
        model, cfg, conf_threshold=args.conf_threshold
    )
    
    # Print results
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"\nPer-Class Accuracy (Top 10):")
    sorted_classes = sorted(per_class_accuracy.items(), key=lambda x: x[1], reverse=True)
    for class_name, acc in sorted_classes[:10]:
        print(f"  {class_name:20s}: {acc*100:.2f}%")
    print(f"{'='*70}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save confusion matrices
    plot_confusion_matrix(confusion_matrix, class_names, 
                         os.path.join(args.output_dir, 'confusion_matrix.png'),
                         normalize=False)
    plot_confusion_matrix(confusion_matrix, class_names,
                         os.path.join(args.output_dir, 'confusion_matrix_normalized.png'),
                         normalize=True)
    
    # Save metrics to text file
    with open(os.path.join(args.output_dir, 'classification_metrics.txt'), 'w') as f:
        f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n\n")
        f.write("Per-Class Accuracy:\n")
        for class_name, acc in sorted(per_class_accuracy.items()):
            f.write(f"  {class_name:20s}: {acc*100:.2f}%\n")
    
    print(f"✅ Saved all results to: {args.output_dir}")


if __name__ == '__main__':
    main()