"""
Evaluate Mask2Former rotation robustness
Tests model performance on rotated images (0°, 90°, 180°, 270°)
"""

import torch
import numpy as np
import cv2
import os
import logging
from collections import defaultdict

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

# Import Mask2Former config
from mask2former import add_maskformer2_config

# Import dataset registration from train_net
from train_net import register_coco_subset

# Setup logging
logger = logging.getLogger("detectron2")


def setup_cfg(model_weights="output/model_final.pth"):
    """Setup config using same method as train_net.py"""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Load base config
    cfg.merge_from_file("configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
    
    # Override for evaluation
    cfg.DATASETS.TEST = ("coco_2017_val_subset",)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    cfg.freeze()
    return cfg


def rotate_image(image, angle):
    """Rotate image by specified angle"""
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image


def evaluate_at_rotation(cfg, angle, max_images=100):
    """
    Evaluate model at a specific rotation angle
    
    Args:
        cfg: Configuration
        angle: Rotation angle (0, 90, 180, 270)
        max_images: Number of images to evaluate (for speed)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f"Evaluating at {angle}° rotation")
    print(f"{'='*70}")
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    # Get validation dataset
    dataset_dicts = DatasetCatalog.get("coco_2017_val_subset")
    
    # Metrics to track
    metrics = {
        'num_images': 0,
        'total_detections': 0,
        'confidences': [],
        'num_instances_per_image': [],
        'class_predictions': defaultdict(int)
    }
    
    # Evaluate
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
            
            # Predict
            outputs = predictor(img_rotated)
            instances = outputs["instances"].to("cpu")
            
            # Collect metrics
            num_detections = len(instances)
            metrics['num_images'] += 1
            metrics['total_detections'] += num_detections
            metrics['num_instances_per_image'].append(num_detections)
            
            if num_detections > 0:
                scores = instances.scores.numpy()
                classes = instances.pred_classes.numpy()
                
                metrics['confidences'].extend(scores.tolist())
                for cls in classes:
                    metrics['class_predictions'][int(cls)] += 1
        
        except Exception as e:
            logger.warning(f"Error processing image {idx}: {e}")
            continue
    
    # Compute summary statistics
    results = {
        'rotation': angle,
        'num_images': metrics['num_images'],
        'total_detections': metrics['total_detections'],
        'avg_detections_per_image': metrics['total_detections'] / max(metrics['num_images'], 1),
        'avg_confidence': np.mean(metrics['confidences']) if metrics['confidences'] else 0.0,
        'std_confidence': np.std(metrics['confidences']) if metrics['confidences'] else 0.0,
        'max_confidence': np.max(metrics['confidences']) if metrics['confidences'] else 0.0,
        'min_confidence': np.min(metrics['confidences']) if metrics['confidences'] else 0.0,
        'high_conf_detections': sum(1 for c in metrics['confidences'] if c > 0.7),
        'medium_conf_detections': sum(1 for c in metrics['confidences'] if 0.5 <= c <= 0.7),
        'low_conf_detections': sum(1 for c in metrics['confidences'] if c < 0.5),
    }
    
    return results


def print_results_table(all_results):
    """Print formatted results table"""
    print(f"\n{'='*80}")
    print("ROTATION ROBUSTNESS EVALUATION RESULTS")
    print(f"{'='*80}\n")
    
    # Get baseline (0°)
    baseline = all_results[0]
    
    # Print header
    print(f"{'Metric':<30} {'0°':<12} {'90°':<12} {'180°':<12} {'270°':<12}")
    print(f"{'-'*80}")
    
    # Metrics to display
    metrics_to_show = [
        ('avg_detections_per_image', 'Avg Detections/Image', '{:.2f}'),
        ('avg_confidence', 'Avg Confidence', '{:.3f}'),
        ('std_confidence', 'Std Confidence', '{:.3f}'),
        ('max_confidence', 'Max Confidence', '{:.3f}'),
        ('high_conf_detections', 'High Conf (>0.7)', '{:.0f}'),
        ('medium_conf_detections', 'Medium Conf (0.5-0.7)', '{:.0f}'),
        ('low_conf_detections', 'Low Conf (<0.5)', '{:.0f}'),
    ]
    
    for metric_key, metric_name, fmt in metrics_to_show:
        row = f"{metric_name:<30}"
        for result in all_results:
            value = result[metric_key]
            row += f" {fmt.format(value):<12}"
        print(row)
    
    # Calculate and display degradation from baseline
    print(f"\n{'-'*80}")
    print("PERFORMANCE DEGRADATION vs 0° (baseline)")
    print(f"{'-'*80}\n")
    
    print(f"{'Metric':<30} {'90°':<15} {'180°':<15} {'270°':<15}")
    print(f"{'-'*80}")
    
    degradation_metrics = [
        ('avg_detections_per_image', 'Detections/Image'),
        ('avg_confidence', 'Avg Confidence'),
    ]
    
    for metric_key, metric_name in degradation_metrics:
        row = f"{metric_name:<30}"
        baseline_val = baseline[metric_key]
        
        for result in all_results[1:]:  # Skip 0°
            current_val = result[metric_key]
            if baseline_val != 0:
                pct_change = ((current_val - baseline_val) / baseline_val) * 100
                row += f" {pct_change:+.2f}%{' '*8}"
            else:
                row += f" N/A{' '*11}"
        print(row)
    
    # Calculate rotation variance (lower is better)
    detection_values = [r['avg_detections_per_image'] for r in all_results]
    confidence_values = [r['avg_confidence'] for r in all_results]
    
    detection_variance = np.var(detection_values)
    confidence_variance = np.var(confidence_values)
    
    print(f"\n{'-'*80}")
    print("ROTATION INVARIANCE METRICS (lower variance = better)")
    print(f"{'-'*80}")
    print(f"Detection Variance:   {detection_variance:.4f}")
    print(f"Confidence Variance:  {confidence_variance:.4f}")
    
    # Interpretation
    print(f"\n{'-'*80}")
    print("INTERPRETATION")
    print(f"{'-'*80}")
    
    avg_degradation = np.mean([
        abs((all_results[i]['avg_confidence'] - baseline['avg_confidence']) / baseline['avg_confidence'] * 100)
        for i in range(1, 4)
    ])
    
    if avg_degradation < 5:
        assessment = "EXCELLENT - Model is highly rotation-invariant"
    elif avg_degradation < 10:
        assessment = "GOOD - Model handles rotations reasonably well"
    elif avg_degradation < 20:
        assessment = "MODERATE - Noticeable performance drop with rotation"
    else:
        assessment = "POOR - Significant rotation sensitivity, MARS might help"
    
    print(f"Average degradation: {avg_degradation:.2f}%")
    print(f"Assessment: {assessment}")
    
    if avg_degradation > 15:
        print("\n⚠️  Recommendation: Consider adding MARS regularization to improve rotation robustness")
    else:
        print("\n✅ Recommendation: Current model is sufficiently rotation-robust for most applications")


def main():
    """Main evaluation function"""
    setup_logger()
    
    print("="*80)
    print("MASK2FORMER ROTATION ROBUSTNESS EVALUATION")
    print("="*80)
    print(f"Model: output/model_final.pth")
    print(f"Dataset: coco_2017_val_subset (1000 images)")
    print(f"Test: Evaluating at 0°, 90°, 180°, 270° rotations")
    print("="*80)
    
    # Setup config
    cfg = setup_cfg()
    
    # Evaluate at each rotation
    all_results = []
    for angle in [0, 90, 180, 270]:
        results = evaluate_at_rotation(cfg, angle, max_images=100)
        all_results.append(results)
        
        print(f"\nResults at {angle}°:")
        print(f"  Images processed: {results['num_images']}")
        print(f"  Total detections: {results['total_detections']}")
        print(f"  Avg detections/image: {results['avg_detections_per_image']:.2f}")
        print(f"  Avg confidence: {results['avg_confidence']:.3f}")
    
    # Print comprehensive results table
    print_results_table(all_results)
    
    # Save results to file
    output_file = "output/rotation_robustness_results.txt"
    with open(output_file, 'w') as f:
        f.write("ROTATION ROBUSTNESS EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        for result in all_results:
            f.write(f"Rotation {result['rotation']}°:\n")
            for key, value in result.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"\n✅ Results saved to: {output_file}")
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
