"""
MARS + Mask2Former Evaluation (Rotation-Stable)
Fixes:
  ✅ Correct bounding-box rotation math
  ✅ Keeps class 0 predictions (no over-filtering)
  ✅ Ensures rotated masks and image dims align
"""

import os
import sys
# CRITICAL: Add path BEFORE importing detectron2
sys.path.insert(1, os.path.join(sys.path[0], '..'))
print("Python path:", sys.path)
import torch
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
import torch.nn.functional as F

from mask2former import add_maskformer2_config
from mask2former_mars.modeling.meta_arch.mars_mask_former_head import add_mars_config


# ======================================================
#  ✅ Rotation-Aware Dataset Mapper
# ======================================================
class RotatedDatasetMapper:
    """Rotate both image and annotations consistently"""

    def __init__(self, cfg, rotation_angle=0):
        self.base_mapper = DatasetMapper(cfg, is_train=False, augmentations=[])
        self.rotation_angle = rotation_angle

    def __call__(self, dataset_dict):
        result = self.base_mapper(dataset_dict)
        if self.rotation_angle == 0:
            return result

        k = self.rotation_angle // 90
        result["image"] = torch.rot90(result["image"], k=k, dims=[1, 2])

        # Swap height/width for 90°/270°
        if self.rotation_angle in [90, 270]:
            result["height"], result["width"] = result["width"], result["height"]

        # Rotate annotations if available
        if "instances" in result:
            result["instances"] = self._rotate_instances(
                result["instances"], k, result["height"], result["width"]
            )
        return result

    def _rotate_instances(self, instances, k, new_h, new_w):
        from detectron2.structures import Instances, Boxes, BitMasks
        import copy

        if k == 0:
            return copy.deepcopy(instances)

        rotated = Instances((new_h, new_w))

        # --- Rotate boxes ---
        if instances.has("gt_boxes"):
            boxes = instances.gt_boxes.tensor.clone()
            rotated_boxes = self._rotate_boxes(boxes, k, new_h, new_w)
            rotated.gt_boxes = Boxes(rotated_boxes)

        # --- Rotate masks ---
        if instances.has("gt_masks"):
            masks = instances.gt_masks
            mask_tensor = (
                masks.tensor if hasattr(masks, "tensor") else masks.clone()
            )
            rotated_masks = torch.rot90(mask_tensor, k=k, dims=[1, 2])
            rotated.gt_masks = BitMasks(rotated_masks)

        # --- Copy other fields ---
        for f, v in instances.get_fields().items():
            if f not in ["gt_boxes", "gt_masks"]:
                rotated.set(f, copy.deepcopy(v))

        return rotated

    def _rotate_boxes(self, boxes, k, h, w):
        """
        Correct COCO-format rotation (counter-clockwise)
        """
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        if k == 1:  # 90° CCW
            new_x1, new_y1 = y1, w - x2
            new_x2, new_y2 = y2, w - x1
        elif k == 2:  # 180°
            new_x1, new_y1 = w - x2, h - y2
            new_x2, new_y2 = w - x1, h - y1
        elif k == 3:  # 270° CCW
            new_x1, new_y1 = h - y2, x1
            new_x2, new_y2 = h - y1, x2
        else:
            new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
        return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1)


# ======================================================
#  ✅ Helper to fix boxes/masks before evaluation
# ======================================================
def compute_boxes_from_masks(instances):
    """Compute boxes from masks if missing or invalid."""
    from detectron2.structures import Boxes

    if not instances.has("pred_masks") or len(instances) == 0:
        return instances

    masks = instances.pred_masks > 0.5
    boxes = []
    for mask in masks:
        ys, xs = torch.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            boxes.append([0, 0, 0, 0])
        else:
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=masks.device)
    instances.pred_boxes = Boxes(boxes_tensor)
    return instances


class ModelWithBoxFix:
    """Ensure predictions have valid boxes & categories"""

    def __init__(self, model):
        self.model = model.eval()

    def __call__(self, batched_inputs):
        outputs = self.model(batched_inputs)
        for out in outputs:
            if "instances" not in out:
                continue
            inst = out["instances"]
            inst = compute_boxes_from_masks(inst)

            # Keep all non-empty predictions
            if inst.has("pred_boxes"):
                b = inst.pred_boxes.tensor
                area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                valid = area > 0
                if inst.has("scores"):
                    valid &= inst.scores > 0.05
                inst = inst[valid]

            # ✅ Keep class 0 (no forced filtering)
            out["instances"] = inst
        return outputs

    def eval(self):
        self.model.eval()
        return self


# ======================================================
#  ✅ Evaluation Function
# ======================================================
def evaluate_at_rotation(cfg, model, dataset_name, rotation_angle):
    print(f"\n{'='*70}")
    print(f"Evaluating at {rotation_angle}° rotation")
    print(f"{'='*70}")

    # Create rotation-aware dataloader
    mapper = RotatedDatasetMapper(cfg, rotation_angle=rotation_angle)
    loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    wrapped_model = ModelWithBoxFix(model)
    evaluator = COCOEvaluator(
        dataset_name,
        output_dir=os.path.join(cfg.OUTPUT_DIR, f"inference_{rotation_angle}deg"),
        tasks=("segm",),
    )

    # Debug one batch
    test_batch = next(iter(loader))
    test_out = wrapped_model(test_batch)
    inst = test_out[0]["instances"]
    if inst.has("pred_boxes"):
        b = inst.pred_boxes.tensor
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        print(f"✅ Non-zero boxes: {(area > 0).sum()} / {len(b)}")
    if inst.has("pred_masks"):
        m = inst.pred_masks
        print(f"✅ Non-empty masks: {(m.sum(dim=(1,2))>0).sum()} / {len(m)}")

    # Run COCO evaluation
    print("Running COCO evaluation...")
    ap_results = inference_on_dataset(wrapped_model, loader, evaluator)
    print(f"✅ Rotation {rotation_angle}° results:")
    print(ap_results)
    print("=" * 70)
    return ap_results

# ======================================================
# ✅ Register COCO subset datasets (same as training)
# ======================================================
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

def register_coco_subset(train_images=5000, val_images=1000):
    """Register COCO subset datasets for evaluation"""
    coco_root = "datasets/coco"
    train_json = os.path.join(coco_root, "annotations/instances_train2017.json")
    train_image_root = os.path.join(coco_root, "train2017")
    val_json = os.path.join(coco_root, "annotations/instances_val2017.json")
    val_image_root = os.path.join(coco_root, "val2017")

    # Load datasets
    train_dicts = load_coco_json(train_json, train_image_root, "coco_2017_train")
    val_dicts = load_coco_json(val_json, val_image_root, "coco_2017_val")

    # Subsets
    train_subset = train_dicts[:train_images]
    val_subset = val_dicts[:val_images]

    # Metadata
    train_metadata = MetadataCatalog.get("coco_2017_train")
    val_metadata = MetadataCatalog.get("coco_2017_val")

    DatasetCatalog.register("coco_2017_train_subset", lambda: train_subset)
    MetadataCatalog.get("coco_2017_train_subset").set(
        thing_classes=train_metadata.thing_classes
    )

    DatasetCatalog.register("coco_2017_val_subset", lambda: val_subset)
    MetadataCatalog.get("coco_2017_val_subset").set(
        thing_classes=val_metadata.thing_classes
    )

    print("✅ Registered:")
    print(f"  coco_2017_train_subset ({len(train_subset)} images)")
    print(f"  coco_2017_val_subset ({len(val_subset)} images)")


# ======================================================
#  ✅ Run All Rotations
# ======================================================
def evaluate_all_rotations(cfg, model_path, dataset_name="coco_2017_val_subset"):
    from mask2former_mars.modeling.meta_arch.mars_mask_former_head import MaskFormerWithMARS

    print("\n" + "=" * 70)
    print("MARS + Mask2Former: Rotation Evaluation")
    print("=" * 70)

    model = MaskFormerWithMARS(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    results = {}
    for angle in [0, 90, 180, 270]:
        results[angle] = evaluate_at_rotation(cfg, model, dataset_name, angle)

    # Summary table
    ap_table = []
    for angle, res in results.items():
        if "segm" in res:
            ap_table.append([f"{angle}°", res["segm"]["AP"], res["segm"]["AP50"], res["segm"]["AP75"]])

    print("\n" + tabulate(ap_table, headers=["Rotation", "AP", "AP50", "AP75"], tablefmt="grid"))
    return results


# ======================================================
#  ✅ Entry Point
# ======================================================
if __name__ == "__main__":
    from detectron2.engine import default_argument_parser

    args = default_argument_parser().parse_args()
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_mars_config(cfg)

    # --- prevent missing-key errors ---
    if not hasattr(cfg.SOLVER, "GRADIENT_ACCUMULATION_STEPS"):
        cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1
    if not hasattr(cfg.SOLVER, "CLIP_GRADIENTS"):
        cfg.SOLVER.CLIP_GRADIENTS = CN()
    if not hasattr(cfg.SOLVER.CLIP_GRADIENTS, "ENABLED"):
        cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
        cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
        cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
        cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

    cfg.merge_from_file("configs/coco/mars_5k_50ep_corrected.yaml")
    cfg.freeze()
    register_coco_subset(train_images=5000, val_images=1000)
    evaluate_all_rotations(cfg, model_path="./output_mars_fixed_5k_50ep/model_final.pth")
