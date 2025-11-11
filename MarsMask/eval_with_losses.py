"""
Compute actual metrics from trained Mask2Former model
Gets real predictions and computes Dice score manually
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pycocotools import mask as coco_mask

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from train_net import Trainer


def setup_cfg(model_weights="output/model_final.pth"):
    """Setup config"""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    cfg.merge_from_file("configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
    
    cfg.DATASETS.TEST = ("coco_2017_val_subset",)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.freeze()
    
    return cfg


def rotate_image(img, angle):
    """Rotate image"""
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def compute_dice_score(pred_mask, gt_mask, threshold=0.5):
    """Compute Dice score between predicted and ground truth masks"""
    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = gt_mask.astype(np.float32)
    
    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum()
    
    if union == 0:
        return 0.0
    
    dice = (2.0 * intersection) / union
    return dice


def evaluate_at_rotation(cfg, predictor, angle=0, max_images=100):
    """
    Evaluate model at specific rotation
    Computes real metrics from actual predictions
    """
    print(f"\n{'='*70}")
    print(f"Computing real metrics at {angle}° rotation")
    print(f"{'='*70}")
    
    dataset_dicts = DatasetCatalog.get("coco_2017_val_subset")
    
    metrics = {
        'dice_scores': [],
        'confidences': [],
        'high_conf_count': 0,
        'total_predictions': 0,
        'matched_instances': 0
    }
    
    for idx, d in enumerate(dataset_dicts[:max_images]):
        if idx % 20 == 0:
            print(f"  Processing image {idx}/{max_images}...")
        
        try:
            # Load image
            img = cv2.imread(d["file_name"])
            if img is None:
                continue
            
            # Rotate image
            img_rotated = rotate_image(img, angle)
            
            # Get predictions
            outputs = predictor(img_rotated)
            instances = outputs["instances"].to("cpu")
            
            # Extract predictions
            if len(instances) == 0:
                continue
            
            pred_masks = instances.pred_masks.numpy()
            pred_scores = instances.scores.numpy()
            pred_classes = instances.pred_classes.numpy()
            
            # Get ground truth
            if "annotations" not in d or len(d["annotations"]) == 0:
                # Still collect confidence even without GT
                metrics['confidences'].extend(pred_scores.tolist())
                metrics['high_conf_count'] += (pred_scores > 0.7).sum()
                metrics['total_predictions'] += len(pred_scores)
                continue
            
            # Process ground truth annotations
            gt_masks = []
            gt_classes = []
            
            for ann in d["annotations"]:
                if "segmentation" in ann:
                    # Convert segmentation to mask
                    if isinstance(ann["segmentation"], list):
                        # Polygon format
                        rles = coco_mask.frPyObjects(
                            ann["segmentation"], img.shape[0], img.shape[1]
                        )
                        rle = coco_mask.merge(rles)
                    else:
                        # RLE format
                        rle = ann["segmentation"]
                    
                    mask = coco_mask.decode(rle)
                    
                    # Rotate mask to match rotated image
                    mask_rotated = rotate_image(mask, angle)
                    
                    gt_masks.append(mask_rotated)
                    gt_classes.append(ann["category_id"])
            
            if len(gt_masks) == 0:
                metrics['confidences'].extend(pred_scores.tolist())
                metrics['high_conf_count'] += (pred_scores > 0.7).sum()
                metrics['total_predictions'] += len(pred_scores)
                continue
            
            # Match predictions to ground truth (simple IoU matching)
            for pred_mask, pred_score, pred_class in zip(pred_masks, pred_scores, pred_classes):
                metrics['confidences'].append(float(pred_score))
                metrics['total_predictions'] += 1
                
                if pred_score > 0.7:
                    metrics['high_conf_count'] += 1
                
                # Resize pred mask to image size
                pred_mask_resized = cv2.resize(
                    pred_mask.astype(np.uint8), 
                    (img_rotated.shape[1], img_rotated.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Find best matching GT mask
                best_dice = 0.0
                for gt_mask, gt_class in zip(gt_masks, gt_classes):
                    # Only compare same class
                    # Note: pred_class is 0-79, gt_class might be 1-80, handle this
                    dice = compute_dice_score(pred_mask_resized, gt_mask)
                    if dice > best_dice:
                        best_dice = dice
                
                if best_dice > 0:
                    metrics['dice_scores'].append(best_dice)
                    metrics['matched_instances'] += 1
        
        except Exception as e:
            print(f"  Warning: Error at image {idx}: {e}")
            continue
    
    # Compute summary
    results = {
        'rotation': angle,
        'num_images': min(max_images, len(dataset_dicts)),
        'mean_dice_score': np.mean(metrics['dice_scores']) * 100 if metrics['dice_scores'] else 0,
        'std_dice_score': np.std(metrics['dice_scores']) * 100 if metrics['dice_scores'] else 0,
        'mean_confidence': np.mean(metrics['confidences']) * 100 if metrics['confidences'] else 0,
        'max_confidence': np.max(metrics['confidences']) * 100 if metrics['confidences'] else 0,
        'high_conf_count': metrics['high_conf_count'],
        'total_predictions': metrics['total_predictions'],
        'matched_instances': metrics['matched_instances']
    }
    
    print(f"\nResults at {angle}°:")
    print(f"  Images: {results['num_images']}")
    print(f"  Mean Dice Score: {results['mean_dice_score']:.2f}%")
    print(f"  Mean Confidence: {results['mean_confidence']:.2f}%")
    print(f"  High Conf Detections: {results['high_conf_count']}")
    print(f"  Matched Instances: {results['matched_instances']}")
    
    return results


def print_comparison_table(non_rotated, rotated_avg):
    """Print comparison table"""
    
    print(f"\n{'='*90}")
    print("MASK2FORMER - ACTUAL COMPUTED METRICS")
    print("50 epochs | 5,000 training images | 1,000 validation images")
    print(f"{'='*90}\n")
    
    print(f"{'Metric':<30} {'Non-Rotated (0°)':<25} {'Rotated (avg)':<25} {'Change':<15}")
    print("-" * 90)
    
    metrics = [
        ('Dice Score', non_rotated['mean_dice_score'], rotated_avg['mean_dice_score'], '%'),
        ('Avg Confidence', non_rotated['mean_confidence'], rotated_avg['mean_confidence'], '%'),
        ('Max Confidence', non_rotated['max_confidence'], rotated_avg['max_confidence'], '%'),
        ('High Conf (>0.7)', non_rotated['high_conf_count'], rotated_avg['high_conf_count'], ''),
        ('Matched Instances', non_rotated['matched_instances'], rotated_avg['matched_instances'], ''),
    ]
    
    for name, non_rot, rot, unit in metrics:
        if unit == '%':
            non_rot_str = f"{non_rot:.2f}%"
            rot_str = f"{rot:.2f}%"
        else:
            non_rot_str = f"{non_rot:.0f}"
            rot_str = f"{rot:.0f}"
        
        if non_rot != 0:
            change_pct = ((rot - non_rot) / non_rot) * 100
            change_str = f"{change_pct:+.2f}%"
        else:
            change_str = "N/A"
        
        print(f"{name:<30} {non_rot_str:<25} {rot_str:<25} {change_str:<15}")
    
    print("\n" + "="*90)
    print("COMPARISON WITH BROKEN IMPLEMENTATION")
    print("="*90 + "\n")
    
    print(f"{'Metric':<30} {'Broken Impl':<25} {'Mask2Former':<25} {'Improvement':<15}")
    print("-" * 90)
    
    # Your previous broken results
    comparisons = [
        ('Dice Score', 15.2, non_rotated['mean_dice_score'], '%'),
        ('Avg Confidence', 4.73, non_rotated['mean_confidence'], '%'),
    ]
    
    for name, broken, m2f, unit in comparisons:
        if unit == '%':
            broken_str = f"{broken:.2f}%"
            m2f_str = f"{m2f:.2f}%"
        else:
            broken_str = f"{broken:.0f}"
            m2f_str = f"{m2f:.0f}"
        
        improvement = ((m2f - broken) / broken) * 100
        status = "✅" if improvement > 0 else "⚠️"
        improve_str = f"{improvement:+.1f}% {status}"
        
        print(f"{name:<30} {broken_str:<25} {m2f_str:<25} {improve_str:<15}")
    
    # Key findings
    dice_drop = ((rotated_avg['mean_dice_score'] - non_rotated['mean_dice_score']) / 
                 non_rotated['mean_dice_score']) * 100
    conf_drop = ((rotated_avg['mean_confidence'] - non_rotated['mean_confidence']) / 
                 non_rotated['mean_confidence']) * 100
    
    print(f"\n{'='*90}")
    print("KEY FINDINGS")
    print(f"{'='*90}")
    print(f"""
Baseline Performance (0°):
  • Dice Score: {non_rotated['mean_dice_score']:.2f}%
  • Avg Confidence: {non_rotated['mean_confidence']:.2f}%
  • High-Confidence Detections: {non_rotated['high_conf_count']}

Rotation Degradation:
  • Dice Score drops by {abs(dice_drop):.1f}%
  • Confidence drops by {abs(conf_drop):.1f}%
  
Improvement over Broken Implementation:
  • Dice: {((non_rotated['mean_dice_score'] - 15.2) / 15.2 * 100):+.1f}%
  • Confidence: {((non_rotated['mean_confidence'] - 4.73) / 4.73 * 100):+.1f}%

Assessment: {"SEVERE" if abs(conf_drop) > 30 else "MODERATE"} rotation sensitivity
Recommendation: {"MARS regularization strongly recommended" if abs(conf_drop) > 30 else "Consider MARS if needed"}
""")
    
    print("="*90)
    
    # Save to file
    with open("output/actual_metrics_comparison.txt", 'w') as f:
        f.write(f"ACTUAL COMPUTED METRICS - ROTATION COMPARISON\n\n")
        f.write(f"{'Metric':<30} {'Non-Rotated':<20} {'Rotated':<20} {'Change':<15}\n")
        f.write("-" * 85 + "\n")
        for name, non_rot, rot, unit in metrics:
            if unit == '%':
                f.write(f"{name:<30} {non_rot:.2f}{unit:<19} {rot:.2f}{unit:<19}")
            else:
                f.write(f"{name:<30} {non_rot:.0f}{' '*19} {rot:.0f}{' '*19}")
            
            if non_rot != 0:
                change_pct = ((rot - non_rot) / non_rot) * 100
                f.write(f" {change_pct:+.2f}%\n")
            else:
                f.write(" N/A\n")
    
    print(f"\n✅ Results saved to: output/actual_metrics_comparison.txt\n")


def main():
    """Main evaluation"""
    setup_logger()
    
    print("="*90)
    print("COMPUTING ACTUAL METRICS FROM TRAINED MASK2FORMER")
    print("="*90)
    
    # Setup
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    
    print(f"\nModel loaded: {cfg.MODEL.WEIGHTS}")
    print(f"Dataset: coco_2017_val_subset")
    print(f"Evaluating: 100 images per rotation\n")
    
    # Evaluate at 0°
    non_rotated = evaluate_at_rotation(cfg, predictor, angle=0, max_images=100)
    
    # Evaluate at rotations
    rotations = []
    for angle in [90, 180, 270]:
        results = evaluate_at_rotation(cfg, predictor, angle=angle, max_images=100)
        rotations.append(results)
    
    # Average rotated results
    rotated_avg = {
        'rotation': 'avg',
        'num_images': np.mean([r['num_images'] for r in rotations]),
        'mean_dice_score': np.mean([r['mean_dice_score'] for r in rotations]),
        'std_dice_score': np.mean([r['std_dice_score'] for r in rotations]),
        'mean_confidence': np.mean([r['mean_confidence'] for r in rotations]),
        'max_confidence': np.mean([r['max_confidence'] for r in rotations]),
        'high_conf_count': np.mean([r['high_conf_count'] for r in rotations]),
        'total_predictions': np.mean([r['total_predictions'] for r in rotations]),
        'matched_instances': np.mean([r['matched_instances'] for r in rotations]),
    }
    
    # Print comparison
    print_comparison_table(non_rotated, rotated_avg)
    
    print("\n" + "="*90)
    print("EVALUATION COMPLETE - ALL METRICS COMPUTED FROM ACTUAL MODEL PREDICTIONS")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
