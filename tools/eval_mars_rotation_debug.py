"""
MARS + Mask2Former Rotation Evaluation (Enhanced Debug Version)
---------------------------------------------------------------
Adds:
  ✅ auto dataset registration
  ✅ preprocessing alignment fix
  ✅ rotation direction sanity
  ✅ visual overlay for predictions
  ✅ CSV summary export
"""
import warnings
warnings.filterwarnings("ignore", message=".*floor_divide is deprecated.*")
from detectron2.modeling.postprocessing import detector_postprocess
import os
import sys
# CRITICAL: Add path BEFORE importing detectron2
sys.path.insert(1, os.path.join(sys.path[0], '..'))
print("Python path:", sys.path)
import csv, cv2, torch, numpy as np
from tqdm import tqdm
from tabulate import tabulate

from detectron2.config import get_cfg, CfgNode as CN
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import DatasetMapper, build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
from detectron2.utils.visualizer import Visualizer
from mask2former import add_maskformer2_config
from mask2former_mars.modeling.meta_arch.mars_mask_former_head import add_mars_config
import torch.nn.functional as F

# ======================================================
# ✅ Dataset Registration
# ======================================================
def register_coco_subset(train_images=5000, val_images=1000):
    coco_root = "datasets/coco"
    ann_train = os.path.join(coco_root, "annotations/instances_train2017.json")
    ann_val = os.path.join(coco_root, "annotations/instances_val2017.json")
    img_train = os.path.join(coco_root, "train2017")
    img_val = os.path.join(coco_root, "val2017")

    train = load_coco_json(ann_train, img_train, "coco_2017_train")
    val = load_coco_json(ann_val, img_val, "coco_2017_val")

    DatasetCatalog.register("coco_2017_train_subset", lambda: train[:train_images])
    DatasetCatalog.register("coco_2017_val_subset", lambda: val[:val_images])
    MetadataCatalog.get("coco_2017_val_subset").set(
        thing_classes=MetadataCatalog.get("coco_2017_train").thing_classes
    )
    print(f"✅ Registered subset datasets: train={train_images}, val={val_images}")


# ======================================================
# ✅ Rotation Mapper (corrected direction)
# ======================================================
class RotatedDatasetMapper:
    """Rotate images + annotations consistently (CCW rotation)"""
    def __init__(self, cfg, rotation_angle=0):
        # Use normal preprocessing — no augmentations
        self.base_mapper = DatasetMapper(cfg, is_train=False)
        self.rotation_angle = rotation_angle

    def __call__(self, dataset_dict):
        result = self.base_mapper(dataset_dict)
        if self.rotation_angle == 0:
            return result
        k = self.rotation_angle // 90
        # rotate counter-clockwise (to match GT orientation)
        result["image"] = torch.rot90(result["image"], k=-k, dims=[1, 2])

        if self.rotation_angle in [90, 270]:
            result["height"], result["width"] = result["width"], result["height"]
        return result


# ======================================================
# ✅ Box + Mask Fix Wrapper
# ======================================================
def compute_boxes_from_masks(instances):
    from detectron2.structures import Boxes
    if not instances.has("pred_masks") or len(instances) == 0:
        return instances
    masks = (instances.pred_masks > 0.5)
    boxes = []
    for m in masks:
        ys, xs = torch.where(m)
        if len(xs) == 0: boxes.append([0,0,0,0]); continue
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    instances.pred_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32, device=masks.device))
    return instances

class ModelWithBoxFix:
    def __init__(self, model): self.model = model.eval()
    def __call__(self, batched_inputs):
        outs = self.model(batched_inputs)
        for o in outs:
            if "instances" not in o: continue
            inst = compute_boxes_from_masks(o["instances"])
            if inst.has("scores"):
                mask = (inst.scores > 0.05)
                o["instances"] = inst[mask]
        return outs


# ======================================================
# ✅ Evaluation per Rotation
# ======================================================
def evaluate_at_rotation(cfg, model, dataset_name, rotation_angle, vis_dir):
    print(f"\n{'='*70}\nEvaluating at {rotation_angle}° rotation\n{'='*70}")
    mapper = RotatedDatasetMapper(cfg, rotation_angle)
    loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    wrapped = ModelWithBoxFix(model)
    evaluator = COCOEvaluator(
        dataset_name,
        output_dir=os.path.join(cfg.OUTPUT_DIR, f"inference_{rotation_angle}deg"),
        tasks=("segm",),
    )

    # Quick sample visualization
    os.makedirs(vis_dir, exist_ok=True)
    import torch.nn.functional as F

    batch = next(iter(loader))
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            raw_outputs = wrapped(batch)


        # True image dimensions after mapper + rotation
        img_tensor = batch[0]["image"]
        _, img_h, img_w = img_tensor.shape

        outputs = []
        for out in raw_outputs:
            if "instances" in out:
                inst = out["instances"]
                if inst.has("pred_masks"):
                    masks = inst.pred_masks.float()
                    # Resize masks to match the *actual image tensor* dimensions
                    resized_masks = F.interpolate(
                        masks.unsqueeze(1),
                        size=(img_h, img_w),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(1)
                    inst.pred_masks = (resized_masks > 0.5).cpu()
                out["instances"] = inst
            outputs.append(out)

    # Use the rotated + preprocessed image for visualization
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    meta = MetadataCatalog.get(dataset_name)
    v = Visualizer(img, meta, scale=0.8)
    out_vis = v.draw_instance_predictions(outputs[0]["instances"].to("cpu"))
    save_path = os.path.join(vis_dir, f"rotation_{rotation_angle}.jpg")
    cv2.imwrite(save_path, out_vis.get_image()[:, :, ::-1])
    print(f"🖼 Saved visualization: {save_path}")
    # Run COCO evaluation
    results = inference_on_dataset(wrapped, loader, evaluator)
    print(f"✅ Rotation {rotation_angle}° results: {results['segm']['AP']:.6f}")
    return results


# ======================================================
# ✅ Run all rotations & export CSV
# ======================================================
def evaluate_all_rotations(cfg, model_path, dataset_name="coco_2017_val_subset"):
    from mask2former_mars.modeling.meta_arch.mars_mask_former_head import MaskFormerWithMARS
    model = MaskFormerWithMARS(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(model_path)
    if torch.cuda.is_available(): model = model.cuda()
    model.eval()
    print("✅ Model ready for evaluation\n")

    vis_dir = os.path.join(cfg.OUTPUT_DIR, "rotation_visuals")
    results = {}
    for angle in [0, 90, 180, 270]:
        results[angle] = evaluate_at_rotation(cfg, model, dataset_name, angle, vis_dir)
        torch.cuda.empty_cache()

    # CSV summary
    csv_path = os.path.join(cfg.OUTPUT_DIR, "rotation_AP_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Rotation", "AP", "AP50", "AP75"])
        for angle, res in results.items():
            seg = res["segm"]
            w.writerow([angle, seg["AP"], seg["AP50"], seg["AP75"]])
    print(f"\n📄 Saved CSV summary: {csv_path}\n")

    # Pretty table
    table = [[f"{a}°", r["segm"]["AP"], r["segm"]["AP50"], r["segm"]["AP75"]] for a,r in results.items()]
    print(tabulate(table, headers=["Rotation","AP","AP50","AP75"], tablefmt="grid"))


# ======================================================
# ✅ Main
# ======================================================
if __name__ == "__main__":
    from detectron2.engine import default_argument_parser
    args = default_argument_parser().parse_args()

    cfg = get_cfg()
    add_deeplab_config(cfg); add_maskformer2_config(cfg); add_mars_config(cfg)
    if not hasattr(cfg.SOLVER, "GRADIENT_ACCUMULATION_STEPS"):
        cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1
    if not hasattr(cfg.SOLVER, "CLIP_GRADIENTS"):
        cfg.SOLVER.CLIP_GRADIENTS = CN()
    if not hasattr(cfg.SOLVER.CLIP_GRADIENTS, "ENABLED"):
        cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
        cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
        cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
        cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0


    cfg.TEST.IMS_PER_BATCH = 1
    cfg.INPUT.MIN_SIZE_TEST = 400
    cfg.INPUT.MAX_SIZE_TEST = 800
    cfg.merge_from_file("configs/coco/mars_5k_50ep_corrected.yaml")
    cfg.freeze()

    register_coco_subset(train_images=5000, val_images=1000)

    evaluate_all_rotations(cfg, model_path="./output_mars_fixed_5k_50ep/model_final.pth")
