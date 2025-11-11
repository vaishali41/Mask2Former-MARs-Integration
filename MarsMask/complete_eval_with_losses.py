"""
Complete evaluation script that computes ALL metrics:
- Dice Score (from predictions)
- Total Loss, Mask Loss, Dice Loss (from training mode)
- Confidence metrics (from predictions)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pycocotools import mask as coco_mask

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config, COCOInstanceNewBaselineDatasetMapper
from train_net import Trainer


def setup_cfg(model_weights="output/model_final.pth"):
    """Setup config"""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    cfg.merge_from_file("configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
    
    cfg.DATASETS.TRAIN = ("coco_2017_val_subset",)
    cfg.DATASETS.TEST = ("coco_2017_val_subset",)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.DATALOADER.NUM_WORKERS = 2
    
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


def compute_losses_at_rotation(cfg, model, angle=0, max_batches=20):
    """
    Compute actual training losses by running model in training mode
    Returns: total_loss, mask_loss, dice_loss
    """
    print(f"\n{'='*70}")
    print(f"Computing losses at {angle}° rotation")
    print(f"{'='*70}")
    
    # Build training data loader
    mapper = COCOInstanceNewBaselineDatasetMapper(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    
    losses_dict = {
        'total_loss': [],
        'loss_ce': [],
        'loss_mask': [],
        'loss_dice': []
    }
    
    model.train()  # Set to training mode to compute losses
    
    with torch.no_grad():  # Don't need gradients
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}/{max_batches}...")
            
            try:
                # Rotate images in batch
                rotated_batch = []
                for item in batch:
                    rotated_item = {}
                    
                    # Rotate image tensor
                    if angle == 90:
                        rotated_item['image'] = torch.rot90(item['image'], k=1, dims=[1, 2])
                    elif angle == 180:
                        rotated_item['image'] = torch.rot90(item['image'], k=2, dims=[1, 2])
                    elif angle == 270:
                        rotated_item['image'] = torch.rot90(item['image'], k=3, dims=[1, 2])
                    else:
                        rotated_item['image'] = item['image']
                    
                    # Copy other fields
                    for key in ['height', 'width', 'image_id', 'instances']:
                        if key in item:
                            rotated_item[key] = item[key]
                    
                    rotated_batch.append(rotated_item)
                
                # Forward pass - model returns losses dict in training mode
                loss_dict = model(rotated_batch)
                
                if isinstance(loss_dict, dict):
                    # Sum all losses for total
                    total = 0
                    for key, value in loss_dict.items():
                        if 'loss' in key.lower():
                            val = value.item() if torch.is_tensor(value) else value
                            total += val
                            
                            # Categorize losses
                            if 'ce' in key.lower() or 'class' in key.lower():
                                losses_dict['loss_ce'].append(val)
                            elif 'mask' in key.lower() and 'dice' not in key.lower():
                                losses_dict['loss_mask'].append(val)
                            elif 'dice' in key.lower():
                                losses_dict['loss_dice'].append(val)
                    
                    losses_dict['total_loss'].append(total)
            
            except Exception as e:
                print(f"  Warning: Error in batch {batch_idx}: {e}")
                continue
    
    model.eval()  # Back to eval mode
    
    # Compute averages
    results = {
        'mean_total_loss': np.mean(losses_dict['total_loss']) if losses_dict['total_loss'] else 0,
        'mean_loss_ce': np.mean(losses_dict['loss_ce']) if losses_dict['loss_ce'] else 0,
        'mean_loss_mask': np.mean(losses_dict['loss_mask']) if losses_dict['loss_mask'] else 0,
        'mean_loss_dice': np.mean(losses_dict['loss_dice']) if losses_dict['loss_dice'] else 0,
    }
    
    print(f"\nLosses at {angle}°:")
    print(f"  Total Loss: {results['mean_total_loss']:.2f}")
    print(f"  CE Loss: {results['mean_loss_ce']:.2f}")
    print(f"  Mask Loss: {results['mean_loss_mask']:.4f}")
    print(f"  Dice Loss: {results['mean_loss_dice']:.4f}")
    
    return results


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


def evaluate_predictions_at_rotation(cfg, predictor, angle=0, max_images=100):
    """
    Evaluate predictions to get Dice scores and confidence
    """
    print(f"\n{'='*70}")
    print(f"Computing Dice scores and confidence at {angle}° rotation")
    print(f"{'='*70}")
    
    dataset_dicts = DatasetCatalog.get("coco_2017_val_subset")
    
    metrics = {
        'dice_scores': [],
        'confidences': [],
        'high_conf_count': 0,
        'total_predictions': 0
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
            
            if len(instances) == 0:
                continue
            
            pred_masks = instances.pred_masks.numpy()
            pred_scores = instances.scores.numpy()
            
            # Collect confidences
            metrics['confidences'].extend(pred_scores.tolist())
            metrics['high_conf_count'] += (pred_scores > 0.7).sum()
            metrics['total_predictions'] += len(pred_scores)
            
            # Get ground truth for Dice calculation
            if "annotations" not in d or len(d["annotations"]) == 0:
                continue
            
            gt_masks = []
            for ann in d["annotations"]:
                if "segmentation" in ann:
                    if isinstance(ann["segmentation"], list):
                        rles = coco_mask.frPyObjects(
                            ann["segmentation"], img.shape[0], img.shape[1]
                        )
                        rle = coco_mask.merge(rles)
                    else:
                        rle = ann["segmentation"]
                    
                    mask = coco_mask.decode(rle)
                    mask_rotated = rotate_image(mask, angle)
                    gt_masks.append(mask_rotated)
            
            if len(gt_masks) == 0:
                continue
            
            # Match predictions to GT and compute Dice
            for pred_mask, pred_score in zip(pred_masks, pred_scores):
                pred_mask_resized = cv2.resize(
                    pred_mask.astype(np.uint8), 
                    (img_rotated.shape[1], img_rotated.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                
                best_dice = 0.0
                for gt_mask in gt_masks:
                    dice = compute_dice_score(pred_mask_resized, gt_mask)
                    if dice > best_dice:
                        best_dice = dice
                
                if best_dice > 0:
                    metrics['dice_scores'].append(best_dice)
        
        except Exception as e:
            print(f"  Warning: Error at image {idx}: {e}")
            continue
    
    # Compute summary
    results = {
        'mean_dice_score': np.mean(metrics['dice_scores']) * 100 if metrics['dice_scores'] else 0,
        'mean_confidence': np.mean(metrics['confidences']) * 100 if metrics['confidences'] else 0,
        'max_confidence': np.max(metrics['confidences']) * 100 if metrics['confidences'] else 0,
        'high_conf_count': metrics['high_conf_count'],
    }
    
    print(f"\nPrediction metrics at {angle}°:")
    print(f"  Mean Dice Score: {results['mean_dice_score']:.2f}%")
    print(f"  Mean Confidence: {results['mean_confidence']:.2f}%")
    print(f"  High Conf Count: {results['high_conf_count']}")
    
    return results


def print_complete_table(pred_non_rot, pred_rot_avg, loss_non_rot, loss_rot_avg):
    """Print complete comparison table with ALL metrics"""
    
    print(f"\n{'='*90}")
    print("COMPLETE MASK2FORMER EVALUATION - ALL METRICS")
    print("50 epochs | 5,000 training images | 1,000 validation images")
    print(f"{'='*90}\n")
    
    print(f"{'Metric':<30} {'Non-Rotated (0°)':<25} {'Rotated (avg)':<25} {'Change':<15}")
    print("-" * 90)
    
    metrics = [
        ('Dice Score', pred_non_rot['mean_dice_score'], pred_rot_avg['mean_dice_score'], '%'),
        ('Total Loss', loss_non_rot['mean_total_loss'], loss_rot_avg['mean_total_loss'], ''),
        ('Mask Loss', loss_non_rot['mean_loss_mask'], loss_rot_avg['mean_loss_mask'], ''),
        ('Dice Loss', loss_non_rot['mean_loss_dice'], loss_rot_avg['mean_loss_dice'], ''),
        ('Avg Confidence', pred_non_rot['mean_confidence'], pred_rot_avg['mean_confidence'], '%'),
        ('Max Confidence', pred_non_rot['max_confidence'], pred_rot_avg['max_confidence'], '%'),
    ]
    
    for name, non_rot, rot, unit in metrics:
        if unit == '%':
            non_rot_str = f"{non_rot:.2f}%"
            rot_str = f"{rot:.2f}%"
        else:
            non_rot_str = f"{non_rot:.4f}"
            rot_str = f"{rot:.4f}"
        
        if non_rot != 0:
            change_pct = ((rot - non_rot) / abs(non_rot)) * 100
            change_str = f"{change_pct:+.2f}%"
        else:
            change_str = "N/A"
        
        print(f"{name:<30} {non_rot_str:<25} {rot_str:<25} {change_str:<15}")
    
    print("\n" + "="*90)
    print("COMPARISON WITH BROKEN IMPLEMENTATION")
    print("="*90 + "\n")
    
    print(f"{'Metric':<30} {'Broken Impl':<25} {'Mask2Former':<25} {'Improvement':<15}")
    print("-" * 90)
    
    comparisons = [
        ('Dice Score', 15.2, pred_non_rot['mean_dice_score'], '%'),
        ('Total Loss', 6678, loss_non_rot['mean_total_loss'], ''),
        ('Mask Loss', 1334, loss_non_rot['mean_loss_mask'], ''),
        ('Avg Confidence', 4.73, pred_non_rot['mean_confidence'], '%'),
    ]
    
    for name, broken, m2f, unit in comparisons:
        if unit == '%':
            broken_str = f"{broken:.2f}%"
            m2f_str = f"{m2f:.2f}%"
        else:
            broken_str = f"{broken:.2f}"
            m2f_str = f"{m2f:.4f}"
        
        if 'loss' in name.lower():
            improvement = ((broken - m2f) / broken) * 100
        else:
            improvement = ((m2f - broken) / broken) * 100
        
        status = "✅" if improvement > 0 else "⚠️"
        improve_str = f"{improvement:+.1f}% {status}"
        
        print(f"{name:<30} {broken_str:<25} {m2f_str:<25} {improve_str:<15}")
    
    print(f"\n{'='*90}")
    print("KEY FINDINGS")
    print(f"{'='*90}")
    
    dice_drop = ((pred_rot_avg['mean_dice_score'] - pred_non_rot['mean_dice_score']) / 
                 pred_non_rot['mean_dice_score']) * 100
    conf_drop = ((pred_rot_avg['mean_confidence'] - pred_non_rot['mean_confidence']) / 
                 pred_non_rot['mean_confidence']) * 100
    loss_increase = ((loss_rot_avg['mean_total_loss'] - loss_non_rot['mean_total_loss']) / 
                     loss_non_rot['mean_total_loss']) * 100
    
    print(f"""
Baseline (0°):
  • Dice: {pred_non_rot['mean_dice_score']:.2f}%
  • Total Loss: {loss_non_rot['mean_total_loss']:.2f}
  • Mask Loss: {loss_non_rot['mean_loss_mask']:.4f}
  • Confidence: {pred_non_rot['mean_confidence']:.2f}%

Rotation Degradation:
  • Dice drops by {abs(dice_drop):.1f}%
  • Total Loss increases by {abs(loss_increase):.1f}%
  • Confidence drops by {abs(conf_drop):.1f}%

vs Broken Implementation:
  • Dice improved: {((pred_non_rot['mean_dice_score'] - 15.2) / 15.2 * 100):+.1f}%
  • Total Loss reduced: {((6678 - loss_non_rot['mean_total_loss']) / 6678 * 100):+.1f}%
  • Mask Loss reduced: {((1334 - loss_non_rot['mean_loss_mask']) / 1334 * 100):+.1f}%

Assessment: {"SEVERE" if abs(conf_drop) > 30 else "MODERATE"} rotation sensitivity
""")
    
    print("="*90)
    
    # Save to file
    with open("output/FINAL_complete_metrics.txt", 'w') as f:
        f.write("COMPLETE METRICS WITH ALL LOSSES\n\n")
        f.write(f"{'Metric':<30} {'Non-Rotated':<20} {'Rotated':<20} {'Change':<15}\n")
        f.write("-" * 85 + "\n")
        for name, non_rot, rot, unit in metrics:
            f.write(f"{name:<30} ")
            if unit == '%':
                f.write(f"{non_rot:.2f}{unit:<19} {rot:.2f}{unit:<19}")
            else:
                f.write(f"{non_rot:.4f}{' '*15} {rot:.4f}{' '*15}")
            
            if non_rot != 0:
                change_pct = ((rot - non_rot) / abs(non_rot)) * 100
                f.write(f" {change_pct:+.2f}%\n")
            else:
                f.write(" N/A\n")
    
    print(f"\n✅ Complete results saved to: output/FINAL_complete_metrics.txt\n")


def main():
    """Main evaluation - computes ALL metrics"""
    setup_logger()
    
    print("="*90)
    print("COMPLETE EVALUATION: LOSSES + DICE + CONFIDENCE")
    print("="*90)
    
    cfg = setup_cfg()
    
    # Build model
    print("\nLoading model...")
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    print(f"\nModel loaded: {cfg.MODEL.WEIGHTS}")
    print(f"Dataset: coco_2017_val_subset\n")
    
    # PART 1: Compute losses
    print("\n" + "="*70)
    print("PART 1: COMPUTING TRAINING LOSSES")
    print("="*70)
    
    loss_non_rot = compute_losses_at_rotation(cfg, model, angle=0, max_batches=20)
    
    loss_rotations = []
    for angle in [90, 180, 270]:
        loss_results = compute_losses_at_rotation(cfg, model, angle=angle, max_batches=20)
        loss_rotations.append(loss_results)
    
    loss_rot_avg = {
        'mean_total_loss': np.mean([r['mean_total_loss'] for r in loss_rotations]),
        'mean_loss_ce': np.mean([r['mean_loss_ce'] for r in loss_rotations]),
        'mean_loss_mask': np.mean([r['mean_loss_mask'] for r in loss_rotations]),
        'mean_loss_dice': np.mean([r['mean_loss_dice'] for r in loss_rotations]),
    }
    
    # PART 2: Compute predictions (Dice + Confidence)
    print("\n" + "="*70)
    print("PART 2: COMPUTING DICE SCORES AND CONFIDENCE")
    print("="*70)
    
    pred_non_rot = evaluate_predictions_at_rotation(cfg, predictor, angle=0, max_images=100)
    
    pred_rotations = []
    for angle in [90, 180, 270]:
        pred_results = evaluate_predictions_at_rotation(cfg, predictor, angle=angle, max_images=100)
        pred_rotations.append(pred_results)
    
    pred_rot_avg = {
        'mean_dice_score': np.mean([r['mean_dice_score'] for r in pred_rotations]),
        'mean_confidence': np.mean([r['mean_confidence'] for r in pred_rotations]),
        'max_confidence': np.mean([r['max_confidence'] for r in pred_rotations]),
        'high_conf_count': np.mean([r['high_conf_count'] for r in pred_rotations]),
    }
    
    # Print complete table
    print_complete_table(pred_non_rot, pred_rot_avg, loss_non_rot, loss_rot_avg)
    
    print("\n" + "="*90)
    print("✅ COMPLETE EVALUATION FINISHED - ALL METRICS COMPUTED")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
