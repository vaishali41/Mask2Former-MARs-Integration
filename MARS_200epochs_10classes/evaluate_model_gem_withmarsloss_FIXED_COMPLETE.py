"""
Memory-Efficient Evaluation Script for Mask2Former + MARs
Evaluates model in small batches to avoid OOM (Out of Memory) issues.

Usage:
    python evaluate_model.py \
        --config mars_10class_200ep_gem.yaml \
        --weights output_mars_10class_200ep_gem/model_final.pth
"""
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from detectron2.evaluation import COCOEvaluator

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

"""
def setup_cfg(args):
    #Setup configuration
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_mars_config(cfg)
    
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.freeze()
    
    return cfg
"""
def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # --- Comprehensive MASK_FORMER config initialization ---
    if not hasattr(cfg.MODEL, "MASK_FORMER"):
        from detectron2.config import CfgNode as CN
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
        ssh.NUM_CLASSES = 10  # 10 classes for our subset
    
    # Additional defaults
    if not hasattr(cfg.INPUT, "MIN_SIZE_TRAIN_SAMPLING"):
        cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    
    # DATALOADER settings
    if not hasattr(cfg.DATALOADER, "FILTER_EMPTY_ANNOTATIONS"):
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

    add_mars_config(cfg)
    
    # Register 10-class dataset (100 samples per class)
    register_coco_10class_subset(samples_per_class=100)
    
    # Safety: gradient clip node
    if not hasattr(cfg.SOLVER, "CLIP_GRADIENTS"):
        cfg.SOLVER.CLIP_GRADIENTS = CN()
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    if not hasattr(cfg.SOLVER, "GRADIENT_ACCUMULATION_STEPS"):
        cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1

    # Load config file
    print(f"Loading config from: {args.config}")
    cfg.merge_from_file(args.config)
    
    # Set model weights
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights

    # TF32 for speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    from detectron2.engine import default_setup
    default_setup(cfg, args)
    return cfg

def register_coco_10class_subset(samples_per_class=100):
    """
    Register COCO 10-class subset with balanced samples per class.
    
    Selected classes (COCO IDs):
    1: person, 3: car, 17: cat, 18: dog, 19: horse,
    21: cow, 44: bottle, 62: chair, 63: couch, 67: dining table
    """
    from detectron2.data.datasets import load_coco_json
    from detectron2.structures import BoxMode
    import json
    
    coco_root = "/home/vaishali/projects/Mask2Former/datasets/coco"
    train_json = os.path.join(coco_root, "annotations/instances_train2017.json")
    train_image_root = os.path.join(coco_root, "train2017")
    val_json = os.path.join(coco_root, "annotations/instances_val2017.json")
    val_image_root = os.path.join(coco_root, "val2017")
    
    # Selected 10 classes - diverse set
    selected_class_ids = [1, 3, 17, 18, 19, 21, 44, 62, 63, 67]
    class_names = ["person", "car", "cat", "dog", "horse", 
                   "cow", "bottle", "chair", "couch", "dining table"]
    
    # Create mapping from COCO ID to our new class ID (0-9)
    coco_to_new_id = {coco_id: i for i, coco_id in enumerate(selected_class_ids)}
    
    print(f" Creating 10-class subset from COCO:")
    print(f"   Classes: {class_names}")
    print(f"   Target samples per class: {samples_per_class}")
    
    # Load annotations
    with open(train_json, 'r') as f:
        train_coco = json.load(f)
    with open(val_json, 'r') as f:
        val_coco = json.load(f)
    
    def create_balanced_subset(coco_data, image_root, samples_per_class):
        """Create balanced subset with exact samples per class"""
        # Group images by class
        class_to_images = defaultdict(set)
        
        for ann in coco_data['annotations']:
            cat_id = ann['category_id']
            if cat_id in selected_class_ids:
                class_to_images[cat_id].add(ann['image_id'])
        
        # Sample images for each class
        selected_image_ids = set()
        class_counts = {}
        
        for cat_id in selected_class_ids:
            available_images = list(class_to_images[cat_id])
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(available_images)
            
            # Take first samples_per_class images
            sampled = available_images[:min(samples_per_class, len(available_images))]
            selected_image_ids.update(sampled)
            class_counts[cat_id] = len(sampled)
        
        # Create image dict for quick lookup
        image_dict = {img['id']: img for img in coco_data['images']}
        
        # Build dataset dicts
        dataset_dicts = []
        for img_id in selected_image_ids:
            if img_id not in image_dict:
                continue
                
            img_info = image_dict[img_id]
            record = {
                "file_name": os.path.join(image_root, img_info['file_name']),
                "image_id": img_id,
                "height": img_info['height'],
                "width": img_info['width'],
            }
            
            # Get annotations for this image
            annotations = []
            for ann in coco_data['annotations']:
                if ann['image_id'] == img_id and ann['category_id'] in selected_class_ids:
                    # Convert COCO annotation to detectron2 format
                    obj = {
                        "bbox": ann['bbox'],
                        "bbox_mode": BoxMode.XYWH_ABS,  # COCO format is [x, y, width, height]
                        "category_id": coco_to_new_id[ann['category_id']],  # Remap to 0-9
                        "iscrowd": ann.get('iscrowd', 0),
                    }
                    
                    # Add segmentation if available
                    if 'segmentation' in ann:
                        obj['segmentation'] = ann['segmentation']
                    
                    annotations.append(obj)
            
            if annotations:  # Only add if has annotations
                record['annotations'] = annotations
                dataset_dicts.append(record)
        
        return dataset_dicts, class_counts
    
    # Create training subset
    train_subset, train_counts = create_balanced_subset(
        train_coco, train_image_root, samples_per_class
    )
    
    # Create validation subset (use fewer samples)
    val_subset, val_counts = create_balanced_subset(
        val_coco, val_image_root, samples_per_class // 5
    )
    
    # Register datasets
    DatasetCatalog.register("coco_10class_train", lambda: train_subset)
    train_metadata = MetadataCatalog.get("coco_10class_train")
    train_metadata.set(
        thing_classes=class_names,
        thing_dataset_id_to_contiguous_id=coco_to_new_id
    )
    
    DatasetCatalog.register("coco_10class_val", lambda: val_subset)
    val_metadata = MetadataCatalog.get("coco_10class_val")
    val_metadata.set(
        thing_classes=class_names,
        thing_dataset_id_to_contiguous_id=coco_to_new_id
    )
    
    print(f"\n Registered 10-class datasets:")
    print(f"   Training: {len(train_subset)} images")
    for i, cat_id in enumerate(selected_class_ids):
        print(f"      {class_names[i]}: {train_counts[cat_id]} samples")
    print(f"\n   Validation: {len(val_subset)} images")
    for i, cat_id in enumerate(selected_class_ids):
        print(f"      {class_names[i]}: {val_counts[cat_id]} samples")
    print()

def create_coco_format_gt(dataset_name, output_path):
    """Create COCO format ground truth JSON"""
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    # Create COCO format with all required fields
    coco_dict = {
        "info": {
            "description": f"{dataset_name} Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "Evaluation Script",
            "date_created": "2025/01/01"
        },
        "licenses": [{
            "id": 1,
            "name": "Unknown",
            "url": ""
        }],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    for i, name in enumerate(metadata.thing_classes):
        coco_dict["categories"].append({
            "id": i,
            "name": name,
            "supercategory": "object"
        })
    
    # Add images and annotations
    ann_id = 1
    for img_dict in dataset_dicts:
        image_id = img_dict.get("image_id", 0)
        
        coco_dict["images"].append({
            "id": image_id,
            "width": img_dict["width"],
            "height": img_dict["height"],
            "file_name": img_dict.get("file_name", ""),
            "license": 1,
            "date_captured": ""
        })
        
        for ann in img_dict.get("annotations", []):
            x, y, w, h = ann["bbox"]
            
            coco_dict["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": float(w * h),
                "iscrowd": ann.get("iscrowd", 0),
                "segmentation": []  # Empty for bbox-only evaluation
            })
            ann_id += 1
    
    # Save to file
    import json
    with open(output_path, 'w') as f:
        json.dump(coco_dict, f)
    
    print(f" Created ground truth: {output_path}")
    return output_path


def evaluate_memory_efficient(cfg, args):
    """Memory-efficient evaluation"""
    print("\n" + "="*80)
    print("MEMORY-EFFICIENT EVALUATION")
    print("="*80)
    
    # Setup predictor
    print("\n1. Loading model...")
    from mask2former_mars.modeling.meta_arch.mars_mask_former_head_gem import MaskFormerWithMARS
    
    model = MaskFormerWithMARS(cfg)
    model.eval()
    
    # Load weights
    checkpoint = torch.load(args.weights, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"   Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   Model loaded on CPU")
    
    # Get dataset
    dataset_name = cfg.DATASETS.TEST[0]
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    print(f"\n2. Dataset: {dataset_name}")
    print(f"   Images: {len(dataset_dicts)}")
    print(f"   Classes: {len(metadata.thing_classes)}")
    
    # Create ground truth COCO JSON
    print("\n3. Creating ground truth annotations...")
    output_dir = args.output_dir or "evaluation_output"
    os.makedirs(output_dir, exist_ok=True)
    gt_json_path = os.path.join(output_dir, f"{dataset_name}_gt.json")
    create_coco_format_gt(dataset_name, gt_json_path)
    
    # Load COCO ground truth
    coco_gt = COCO(gt_json_path)
    
    # Run inference
    print("\n4. Running inference (processing one image at a time)...")
    coco_results = []
    
    from detectron2.data import detection_utils as utils
    
    for img_dict in tqdm(dataset_dicts, desc="Evaluating"):
        # Load image
        image = utils.read_image(img_dict["file_name"], format="RGB")
        
        # Prepare input
        height, width = image.shape[:2]
        image_tensor = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        # Run inference
        with torch.no_grad():
            inputs = [{
                "image": image_tensor,
                "height": height,
                "width": width
            }]
            
            outputs = model(inputs)
        
        # Extract predictions
        if outputs and "instances" in outputs[0]:
            instances = outputs[0]["instances"].to("cpu")
            
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            
            # Convert boxes from XYXY to XYWH
            boxes_xywh = np.zeros_like(boxes)
            boxes_xywh[:, 0] = boxes[:, 0]
            boxes_xywh[:, 1] = boxes[:, 1]
            boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
            
            # Add to results
            for k in range(len(instances)):
                coco_results.append({
                    "image_id": img_dict["image_id"],
                    "category_id": int(classes[k]),
                    "bbox": boxes_xywh[k].tolist(),
                    "score": float(scores[k])
                })
        
        # Clear GPU cache every 10 images
        if torch.cuda.is_available() and len(dataset_dicts) % 10 == 0:
            torch.cuda.empty_cache()
    
    print(f"\n5. Collected {len(coco_results)} predictions")
    
    # Evaluate
    if len(coco_results) == 0:
        print("\nNo predictions to evaluate!")
        return
    
    print("\n6. Computing COCO metrics...")
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80 + "\n")
    
    coco_eval.summarize()
    
    # Save results
    results = {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "APs": float(coco_eval.stats[3]),
        "APm": float(coco_eval.stats[4]),
        "APl": float(coco_eval.stats[5])
    }
    
    results_path = os.path.join(output_dir, "evaluation_results.json")
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print(f"Results saved to: {results_path}")
    print(f"Ground truth saved to: {gt_json_path}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Memory-Efficient Model Evaluation")
    
    parser.add_argument("--config", required=True,
                       help="Path to config file")
    parser.add_argument("--weights", required=True,
                       help="Path to model weights")
    parser.add_argument("--output-dir", default="evaluation_output",
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Register the 10-class dataset (same as training script)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    #from train_mars_10class_gem import register_coco_10class_subset
    #register_coco_10class_subset(samples_per_class=100)
    
    # Setup config
    cfg = setup(args)
    
    # Run evaluation
    evaluate_memory_efficient(cfg, args)
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
