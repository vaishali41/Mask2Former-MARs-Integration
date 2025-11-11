"""
Enhanced Evaluation Script for MARS + Mask2Former
Evaluates model at 0°, 90°, 180°, 270° rotations
Includes: AP metrics + Loss metrics + Dice Score + Confidence stats

File: tools/eval_mars_rotations.py
"""

import os
import torch
import numpy as np
import json
from tabulate import tabulate
from collections import defaultdict
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
import torch.nn.functional as F

from mask2former import add_maskformer2_config
from mask2former_mars.modeling.meta_arch.mars_mask_former_head import add_mars_config

class RotatedDatasetMapper:
    """Dataset mapper that applies rotation to images"""
    
    def __init__(self, cfg, is_train=False, rotation_angle=0):
        from mask2former.data.dataset_mappers.mask_former_instance_dataset_mapper import (
            MaskFormerInstanceDatasetMapper,
        )
        self.base_mapper = MaskFormerInstanceDatasetMapper(cfg, is_train=is_train)
        self.rotation_angle = rotation_angle
    
    def __call__(self, dataset_dict):
        """Apply rotation to image before base mapper processing"""
        from detectron2.data import detection_utils as utils
        
        dataset_dict = dataset_dict.copy()
        
        # Read image
        image = utils.read_image(dataset_dict["file_name"], format="RGB")
        
        # Apply rotation to raw image
        if self.rotation_angle == 90:
            image = np.rot90(image, k=1, axes=(0, 1)).copy()
        elif self.rotation_angle == 180:
            image = np.rot90(image, k=2, axes=(0, 1)).copy()
        elif self.rotation_angle == 270:
            image = np.rot90(image, k=3, axes=(0, 1)).copy()
        
        # Update height and width for 90/270 degree rotations
        if self.rotation_angle in [90, 270]:
            h, w = dataset_dict.get("height", image.shape[0]), dataset_dict.get("width", image.shape[1])
            dataset_dict["height"], dataset_dict["width"] = w, h
        
        # Save rotated image temporarily
        import tempfile
        import cv2
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            original_file = dataset_dict["file_name"]
            dataset_dict["file_name"] = tmp.name
        
        # Let base mapper do the rest
        result = self.base_mapper(dataset_dict)
        
        # Restore original filename and clean up
        dataset_dict["file_name"] = original_file
        try:
            os.unlink(tmp.name)
        except:
            pass
        
        return result
"""
class RotatedDatasetMapper(DatasetMapper):
    #Dataset mapper that applies rotation to images
    
    def __init__(self, cfg, is_train=False, rotation_angle=0):
        super().__init__(cfg, is_train=is_train)
        self.rotation_angle = rotation_angle
    
    def __call__(self, dataset_dict):
        #Apply rotation to image
        dataset_dict = dataset_dict.copy()
        
        # Read image
        image = self._read_image(dataset_dict, "RGB")
        
        # Apply rotation
        if self.rotation_angle == 90:
            image = np.rot90(image, k=1, axes=(0, 1)).copy()
        elif self.rotation_angle == 180:
            image = np.rot90(image, k=2, axes=(0, 1)).copy()
        elif self.rotation_angle == 270:
            image = np.rot90(image, k=3, axes=(0, 1)).copy()
        
        # Update dimensions after rotation
        if self.rotation_angle in [90, 270]:
            dataset_dict["height"], dataset_dict["width"] = dataset_dict["width"], dataset_dict["height"]
        
        dataset_dict["image"] = image
        
        return super().__call__(dataset_dict)
"""

class DetailedMetricsEvaluator:
    """
    Computes detailed metrics including losses, dice score, and confidence
    """
    
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.reset()
    
    def reset(self):
        """Reset metric accumulators"""
        self.total_losses = []
        self.mask_losses = []
        self.dice_losses = []
        self.cls_losses = []
        self.dice_scores = []
        self.avg_confidences = []
        self.max_confidences = []
        self.num_samples = 0
    
    def compute_dice_score(self, pred_masks, gt_masks):
        """
        Compute Dice score between predicted and ground truth masks
        
        Args:
            pred_masks: [N, H, W] predicted masks
            gt_masks: [N, H, W] ground truth masks
        
        Returns:
            Average Dice score
        """
        pred_masks = (pred_masks > 0.5).float()
        gt_masks = gt_masks.float()
        
        intersection = (pred_masks * gt_masks).sum(dim=(1, 2))
        union = pred_masks.sum(dim=(1, 2)) + gt_masks.sum(dim=(1, 2))
        
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        return dice.mean().item()
    
    def process_batch(self, batched_inputs):
        """
        Process a batch and compute all metrics
        
        Args:
            batched_inputs: Batch of images and annotations
        """
        with torch.no_grad():
            # Set model to training mode to compute losses
            self.model.train()
            
            # Forward pass with loss computation
            try:
                loss_dict = self.model(batched_inputs)
                
                # Extract losses
                if isinstance(loss_dict, dict):
                    total_loss = sum([v for k, v in loss_dict.items() if 'loss' in k]).item()
                    mask_loss = loss_dict.get('loss_mask', torch.tensor(0.0)).item()
                    dice_loss = loss_dict.get('loss_dice', torch.tensor(0.0)).item()
                    cls_loss = loss_dict.get('loss_cls', torch.tensor(0.0)).item()
                    
                    self.total_losses.append(total_loss)
                    self.mask_losses.append(mask_loss)
                    self.dice_losses.append(dice_loss)
                    self.cls_losses.append(cls_loss)
            except:
                # If loss computation fails, skip
                pass
            
            # Set model to eval mode for predictions
            self.model.eval()
            
            # Get predictions
            images = self.model.preprocess_image(batched_inputs)
            features = self.model.backbone(images.tensor)
            outputs = self.model.sem_seg_head(features)
            
            # Process each item in batch
            for idx, (output, input_data) in enumerate(zip(outputs, batched_inputs)):
                # Compute confidence statistics
                if 'pred_logits' in output:
                    logits = output['pred_logits']  # [Q, C]
                    probs = F.softmax(logits, dim=-1)
                    confidences, _ = probs.max(dim=-1)  # [Q]
                    
                    self.avg_confidences.append(confidences.mean().item() * 100)
                    self.max_confidences.append(confidences.max().item() * 100)
                
                # Compute Dice score if GT available
                if 'pred_masks' in output and 'instances' in input_data:
                    try:
                        pred_masks = output['pred_masks']  # [Q, H, W]
                        
                        if hasattr(input_data['instances'], 'gt_masks'):
                            gt_masks = input_data['instances'].gt_masks
                            
                            # Match dimensions if needed
                            if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                                pred_masks = F.interpolate(
                                    pred_masks.unsqueeze(0),
                                    size=gt_masks.shape[-2:],
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze(0)
                            
                            dice = self.compute_dice_score(pred_masks[:len(gt_masks)], gt_masks)
                            self.dice_scores.append(dice)
                    except:
                        pass
            
            self.num_samples += len(batched_inputs)
    
    def get_metrics(self):
        """
        Compute final aggregated metrics
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'total_loss': np.mean(self.total_losses) if self.total_losses else 0.0,
            'mask_loss': np.mean(self.mask_losses) if self.mask_losses else 0.0,
            'dice_loss': np.mean(self.dice_losses) if self.dice_losses else 0.0,
            'cls_loss': np.mean(self.cls_losses) if self.cls_losses else 0.0,
            'dice_score': np.mean(self.dice_scores) * 100 if self.dice_scores else 0.0,
            'avg_confidence': np.mean(self.avg_confidences) if self.avg_confidences else 0.0,
            'max_confidence': np.mean(self.max_confidences) if self.max_confidences else 0.0,
            'num_samples': self.num_samples,
        }
        return metrics


def evaluate_at_rotation_detailed(cfg, model, dataset_name: str, rotation_angle: int):
    """
    Evaluate model at specific rotation angle with detailed metrics
    
    Args:
        cfg: Configuration
        model: Model to evaluate
        dataset_name: Dataset name
        rotation_angle: Rotation angle (0, 90, 180, 270)
    
    Returns:
        Dictionary with AP metrics and detailed metrics
    """
    print(f"\n{'='*70}")
    print(f"Evaluating at {rotation_angle}° rotation")
    print(f"{'='*70}")
    
    # Create data loader with rotation
    mapper = RotatedDatasetMapper(cfg, is_train=False, rotation_angle=rotation_angle)
    data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    
    # Detailed metrics evaluator
    detailed_evaluator = DetailedMetricsEvaluator(model, cfg)
    
    # Process all batches for detailed metrics
    print("Computing detailed metrics (loss, dice, confidence)...")
    for batch in tqdm(data_loader, desc=f"Processing {rotation_angle}°"):
        detailed_evaluator.process_batch(batch)
    
    detailed_metrics = detailed_evaluator.get_metrics()
    
    # Standard AP evaluation
    print("Computing AP metrics...")
    model.eval()
    coco_evaluator = COCOEvaluator(
        dataset_name,
        output_dir=os.path.join(cfg.OUTPUT_DIR, f"inference_rotation_{rotation_angle}"),
        tasks=("segm",),
    )
    
    ap_results = inference_on_dataset(model, data_loader, coco_evaluator)
    
    # Combine results
    results = {
        'ap_metrics': ap_results,
        'detailed_metrics': detailed_metrics,
    }
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Results at {rotation_angle}°:")
    print(f"{'='*70}")
    
    if "segm" in ap_results:
        print(f"AP Metrics:")
        print(f"  AP:     {ap_results['segm']['AP']:.2f}")
        print(f"  AP50:   {ap_results['segm']['AP50']:.2f}")
        print(f"  AP75:   {ap_results['segm']['AP75']:.2f}")
    
    print(f"\nDetailed Metrics:")
    print(f"  Total Loss:      {detailed_metrics['total_loss']:.4f}")
    print(f"  Mask Loss:       {detailed_metrics['mask_loss']:.4f}")
    print(f"  Dice Loss:       {detailed_metrics['dice_loss']:.4f}")
    print(f"  Dice Score:      {detailed_metrics['dice_score']:.2f}%")
    print(f"  Avg Confidence:  {detailed_metrics['avg_confidence']:.2f}%")
    print(f"  Max Confidence:  {detailed_metrics['max_confidence']:.2f}%")
    
    return results


def evaluate_all_rotations_detailed(cfg, model_path: str, dataset_name: str = "coco_2017_val_subset"):
    """
    Evaluate model at all rotation angles with detailed metrics
    
    Args:
        cfg: Configuration
        model_path: Path to model checkpoint
        dataset_name: Dataset name
    
    Returns:
        Dictionary with results at all rotations
    """
    print("\n" + "="*70)
    print("MARS + MASK2FORMER - DETAILED EVALUATION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name}")
    print("="*70)
    
    # Build and load model
    from mask2former_mars.modeling.meta_arch.mars_mask_former_head import MaskFormerWithMARS
    model = MaskFormerWithMARS(cfg)
    
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(model_path)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Evaluate at each rotation
    all_results = {}
    rotation_angles = [0, 90, 180, 270]
    
    for angle in rotation_angles:
        results = evaluate_at_rotation_detailed(cfg, model, dataset_name, angle)
        all_results[f"{angle}deg"] = results
    
    # Create summary tables
    print("\n" + "="*70)
    print("SUMMARY TABLES")
    print("="*70)
    
    # Table 1: AP Metrics
    print("\n" + "="*70)
    print("Table 1: Average Precision (AP)")
    print("="*70)
    
    ap_table_data = []
    ap_headers = ["Rotation", "AP", "AP50", "AP75", "APs", "APm", "APl"]
    
    for angle in rotation_angles:
        if "segm" in all_results[f"{angle}deg"]["ap_metrics"]:
            segm = all_results[f"{angle}deg"]["ap_metrics"]["segm"]
            ap_table_data.append([
                f"{angle}°",
                f"{segm['AP']:.2f}",
                f"{segm['AP50']:.2f}",
                f"{segm['AP75']:.2f}",
                f"{segm['APs']:.2f}",
                f"{segm['APm']:.2f}",
                f"{segm['APl']:.2f}",
            ])
    
    # Add averages
    if ap_table_data:
        ap_values = [all_results[f"{angle}deg"]["ap_metrics"]["segm"]["AP"] for angle in rotation_angles]
        avg_ap = np.mean(ap_values)
        drop = ap_values[0] - min(ap_values)
        
        ap_table_data.append([
            "Average",
            f"{avg_ap:.2f}",
            "-",
            "-",
            "-",
            "-",
            "-",
        ])
        
        ap_table_data.append([
            "Max Drop",
            f"{drop:.2f}",
            "-",
            "-",
            "-",
            "-",
            "-",
        ])
    
    print("\n" + tabulate(ap_table_data, headers=ap_headers, tablefmt="grid"))
    
    # Table 2: Detailed Metrics (Your Requested Format!)
    print("\n" + "="*70)
    print("Table 2: Detailed Metrics (Non-Rotated vs Rotated)")
    print("="*70)
    
    # Get metrics
    metrics_0 = all_results["0deg"]["detailed_metrics"]
    
    # Average rotated metrics (90, 180, 270)
    metrics_rotated = {}
    for key in metrics_0.keys():
        if key != 'num_samples':
            values = [
                all_results["90deg"]["detailed_metrics"][key],
                all_results["180deg"]["detailed_metrics"][key],
                all_results["270deg"]["detailed_metrics"][key],
            ]
            metrics_rotated[key] = np.mean(values)
    
    # Create table
    detailed_table_data = []
    
    # Dice Score
    dice_change = ((metrics_rotated['dice_score'] - metrics_0['dice_score']) / metrics_0['dice_score'] * 100) if metrics_0['dice_score'] != 0 else 0
    detailed_table_data.append([
        "Dice Score",
        f"{metrics_0['dice_score']:.2f}%",
        f"{metrics_rotated['dice_score']:.2f}%",
        f"{dice_change:+.2f}%"
    ])
    
    # Total Loss
    loss_change = ((metrics_rotated['total_loss'] - metrics_0['total_loss']) / metrics_0['total_loss'] * 100) if metrics_0['total_loss'] != 0 else 0
    detailed_table_data.append([
        "Total Loss",
        f"{metrics_0['total_loss']:.4f}",
        f"{metrics_rotated['total_loss']:.4f}",
        f"{loss_change:+.2f}%"
    ])
    
    # Mask Loss
    mask_change = ((metrics_rotated['mask_loss'] - metrics_0['mask_loss']) / metrics_0['mask_loss'] * 100) if metrics_0['mask_loss'] != 0 else 0
    detailed_table_data.append([
        "Mask Loss",
        f"{metrics_0['mask_loss']:.4f}",
        f"{metrics_rotated['mask_loss']:.4f}",
        f"{mask_change:+.2f}%"
    ])
    
    # Dice Loss
    if metrics_0['dice_loss'] != 0:
        dice_loss_change = ((metrics_rotated['dice_loss'] - metrics_0['dice_loss']) / metrics_0['dice_loss'] * 100)
        dice_loss_change_str = f"{dice_loss_change:+.2f}%"
    else:
        dice_loss_change_str = "N/A"
    detailed_table_data.append([
        "Dice Loss",
        f"{metrics_0['dice_loss']:.4f}",
        f"{metrics_rotated['dice_loss']:.4f}",
        dice_loss_change_str
    ])
    
    # Avg Confidence
    conf_change = ((metrics_rotated['avg_confidence'] - metrics_0['avg_confidence']) / metrics_0['avg_confidence'] * 100) if metrics_0['avg_confidence'] != 0 else 0
    detailed_table_data.append([
        "Avg Confidence",
        f"{metrics_0['avg_confidence']:.2f}%",
        f"{metrics_rotated['avg_confidence']:.2f}%",
        f"{conf_change:+.2f}%"
    ])
    
    # Max Confidence
    max_conf_change = ((metrics_rotated['max_confidence'] - metrics_0['max_confidence']) / metrics_0['max_confidence'] * 100) if metrics_0['max_confidence'] != 0 else 0
    detailed_table_data.append([
        "Max Confidence",
        f"{metrics_0['max_confidence']:.2f}%",
        f"{metrics_rotated['max_confidence']:.2f}%",
        f"{max_conf_change:+.2f}%"
    ])
    
    detailed_headers = ["Metric", "Non-Rotated", "Rotated", "Change"]
    print("\n" + "-" * 85)
    print(f"{'Metric':<30} {'Non-Rotated':<20} {'Rotated':<20} {'Change':<15}")
    print("-" * 85)
    for row in detailed_table_data:
        print(f"{row[0]:<30} {row[1]:<20} {row[2]:<20} {row[3]:<15}")
    print("-" * 85)
    
    # Save all results to JSON
    output_file = os.path.join(cfg.OUTPUT_DIR, "detailed_evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    if ap_table_data:
        print(f"Rotation Robustness:")
        print(f"  AP drop:           {drop:.2f} ({((drop/ap_values[0])*100):.1f}%)")
        print(f"  Dice drop:         {metrics_0['dice_score'] - metrics_rotated['dice_score']:.2f}%")
        print(f"  Loss increase:     {metrics_rotated['total_loss'] - metrics_0['total_loss']:.4f}")
        print(f"  Confidence drop:   {metrics_0['avg_confidence'] - metrics_rotated['avg_confidence']:.2f}%")
    print("="*70 + "\n")
    
    return all_results


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate MARS model with detailed metrics")
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", default="coco_2017_val_subset", help="Dataset name")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--opts",
        help="Modify config options",
        default=[],
        nargs=argparse.REMAINDER,
    )
    
    args = parser.parse_args()
    
    # Setup logger
    setup_logger()
    
    # Load config
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_mars_config(cfg)
    
    # Add custom config keys that might be in the training config
    from detectron2.config import CfgNode as CN
    if not hasattr(cfg.SOLVER, 'GRADIENT_ACCUMULATION_STEPS'):
        cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 16
    if not hasattr(cfg.SOLVER, 'AMP'):
        cfg.SOLVER.AMP = CN()
        cfg.SOLVER.AMP.ENABLED = True


    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    
    cfg.freeze()
    
    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Register datasets
    from train_mars import register_coco_subset
    register_coco_subset(train_images=5000, val_images=1000)
    
    # Run evaluation
    results = evaluate_all_rotations_detailed(
        cfg=cfg,
        model_path=args.model_path,
        dataset_name=args.dataset
    )
    
    print("\n✅ Evaluation complete!")
    
    return results


if __name__ == "__main__":
    main()
