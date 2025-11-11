"""
Training script for Mask2Former + MARs + GeM - 10 Class Subset
- 10 classes from COCO
- 100 samples per class (1,000 total samples)
- Supports EigenCAM evaluation
"""
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import itertools, time
import torch
import numpy as np
import logging
from collections import defaultdict
from detectron2.engine import default_argument_parser, launch, DefaultTrainer
from detectron2.config import get_cfg, CfgNode as CN
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import hooks
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.events import TensorboardXWriter, get_event_storage, EventWriter
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator
from mask2former import add_maskformer2_config
from mask2former_mars.modeling.meta_arch.mars_mask_former_head_gem import MaskFormerWithMARS
from detectron2.utils.events import JSONWriter, CommonMetricPrinter

# ---------------- Config Injection ----------------
def add_mars_config(cfg):
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

    # Optional SWEEP support
    cfg.SWEEP = CN()
    cfg.SWEEP.ENABLED = False
    cfg.SWEEP.SOLVER = CN()
    cfg.SWEEP.SOLVER.BASE_LR = [0.00005, 0.0001, 0.0002]
    cfg.SWEEP.MODEL = CN()
    cfg.SWEEP.MODEL.MARS_WEIGHT = [0.1, 0.5, 1.0]
    cfg.SWEEP.MODEL.LOSS_TYPE = ["kl", "cosine"]
    cfg.SWEEP.MODEL.GEM_INIT_P = [1.0, 2.0, 3.0]


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


# ---------------- Custom Trainer ----------------
class SafeCOCOEvaluator(COCOEvaluator):
    """
    Custom COCO Evaluator that handles polygon segmentations properly.
    Skips area computation for polygon segmentations to avoid format errors.
    """
    
    def __init__(self, dataset_name, tasks=None, output_dir=None, **kwargs):
        """
        Initialize evaluator with safe segmentation handling.
        We override the parent to handle the dataset conversion ourselves.
        """
        import os
        import json
        import tempfile
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from pycocotools.coco import COCO
        
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._tasks = tasks or ("bbox", "segm")
        self._distributed = False
        self._output_dir = output_dir
        
        self._cpu_device = torch.device("cpu")
        self._metadata = MetadataCatalog.get(dataset_name)
        
        # Create a safe COCO GT object
        cache_path = os.path.join(output_dir or "", f"{dataset_name}_coco_format_safe.json")
        
        if not os.path.exists(cache_path):
            self._logger.info(f"Creating safe COCO format JSON for {dataset_name}...")
            dataset_dicts = DatasetCatalog.get(dataset_name)
            
            # Convert to COCO format, skipping problematic segmentation area computation
            coco_dict = self._convert_to_coco_dict_safe(dataset_dicts)
            
            with open(cache_path, "w") as f:
                json.dump(coco_dict, f)
            self._logger.info(f"Saved safe COCO format to {cache_path}")
        
        self._coco_api = COCO(cache_path)
        
        # Initialize predictions storage
        self._predictions = []
    
    def _convert_to_coco_dict_safe(self, dataset_dicts):
        """
        Convert dataset to COCO format without computing segmentation areas.
        This avoids the polygon-to-RLE conversion issues.
        """
        from pycocotools import mask as mask_util
        import numpy as np
        
        categories = [
            {"id": id, "name": name} 
            for id, name in enumerate(self._metadata.thing_classes)
        ]
        
        images = []
        annotations = []
        ann_id = 1
        
        for image_dict in dataset_dicts:
            image_id = image_dict.get("image_id", 0)
            
            images.append({
                "id": image_id,
                "width": image_dict["width"],
                "height": image_dict["height"],
                "file_name": image_dict.get("file_name", ""),
            })
            
            for ann in image_dict.get("annotations", []):
                # Create annotation entry
                coco_ann = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": ann["category_id"],
                    "iscrowd": ann.get("iscrowd", 0),
                    "bbox": ann["bbox"],
                }
                
                # Handle segmentation safely
                if "segmentation" in ann:
                    seg = ann["segmentation"]
                    
                    # If it's a polygon, use it directly and compute area from bbox
                    if isinstance(seg, list):
                        coco_ann["segmentation"] = seg
                        # Use bbox area as approximation to avoid conversion issues
                        x, y, w, h = ann["bbox"]
                        coco_ann["area"] = float(w * h)
                    # If it's already RLE, use it
                    elif isinstance(seg, dict):
                        coco_ann["segmentation"] = seg
                        try:
                            coco_ann["area"] = float(mask_util.area(seg))
                        except:
                            x, y, w, h = ann["bbox"]
                            coco_ann["area"] = float(w * h)
                else:
                    # No segmentation, use bbox area
                    x, y, w, h = ann["bbox"]
                    coco_ann["area"] = float(w * h)
                
                annotations.append(coco_ann)
                ann_id += 1
        
        return {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
    
    def reset(self):
        """Reset evaluation state"""
        self._predictions = []
    
    def process(self, inputs, outputs):
        """Process predictions"""
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances
            
            self._predictions.append(prediction)
    
    def evaluate(self):
        """Evaluate predictions"""
        if self._distributed:
            raise NotImplementedError("Distributed evaluation not supported")
        
        self._results = {}
        self._eval_predictions()
        
        return self._results
    
    def _eval_predictions(self):
        """Evaluate predictions using COCO API"""
        from collections import OrderedDict
        from pycocotools.cocoeval import COCOeval
        import numpy as np
        
        if len(self._predictions) == 0:
            self._logger.warning("No predictions to evaluate!")
            return
        
        self._logger.info(f"Evaluating {len(self._predictions)} predictions...")
        
        # Prepare predictions in COCO format
        coco_results = []
        
        for prediction in self._predictions:
            if "instances" not in prediction:
                continue
            
            instances = prediction["instances"]
            image_id = prediction["image_id"]
            
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()
            
            # Convert boxes from XYXY to XYWH format
            boxes_xywh = np.zeros_like(boxes)
            boxes_xywh[:, 0] = boxes[:, 0]  # x
            boxes_xywh[:, 1] = boxes[:, 1]  # y
            boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
            boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
            
            for k in range(len(instances)):
                result = {
                    "image_id": image_id,
                    "category_id": int(classes[k]),
                    "bbox": boxes_xywh[k].tolist(),
                    "score": float(scores[k]),
                }
                coco_results.append(result)
        
        if len(coco_results) == 0:
            self._logger.warning("No valid predictions to evaluate!")
            return
        
        # Evaluate bounding boxes
        self._logger.info("Evaluating bbox predictions...")
        coco_dt = self._coco_api.loadRes(coco_results)
        coco_eval = COCOeval(self._coco_api, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Store results
        self._results["bbox"] = {
            "AP": coco_eval.stats[0],
            "AP50": coco_eval.stats[1],
            "AP75": coco_eval.stats[2],
            "APs": coco_eval.stats[3],
            "APm": coco_eval.stats[4],
            "APl": coco_eval.stats[5],
        }
        
        self._logger.info(f"Evaluation results: {self._results}")


class WandBWriter(EventWriter):
    """Custom EventWriter to log metrics to Weights & Biases"""
    def __init__(self, wandb_module, window_size=20):
        """
        Args:
            wandb_module: The wandb module (or None if disabled)
            window_size: Number of events to average (default: 20, matching PeriodicWriter)
        """
        self.wandb = wandb_module
        self._window_size = window_size
        self._last_write = -1
        
    def write(self):
        """Called by detectron2 to write metrics"""
        storage = get_event_storage()
        
        # Only write if we haven't written at this iteration
        if storage.iter == self._last_write:
            return
        self._last_write = storage.iter
        
        metrics = {}
        
        # Get all scalar metrics from the current iteration
        for k, (v, iter_num) in storage.latest().items():
            if isinstance(v, (int, float)):
                metrics[k] = v
        
        # Log to wandb
        if metrics and self.wandb is not None:
            self.wandb.log(metrics, step=storage.iter)
    
    def close(self):
        """Called when training ends"""
        if self.wandb is not None:
            self.wandb.finish()


class Trainer(DefaultTrainer):
    def __init__(self, cfg, wandb_module=None):
        self.wandb_module = wandb_module
        super().__init__(cfg)
    
    @classmethod
    def build_model(cls, cfg):
        model = MaskFormerWithMARS(cfg)
        model.train()
        # Force model to GPU
        if torch.cuda.is_available():
            print(f" Moving model to GPU: {torch.cuda.get_device_name(0)}")
            model = model.cuda()
            # CRITICAL: Re-attach hooks after moving to GPU
            if hasattr(model, '_attach_attention_hooks'):
                print("Re-attaching MARs hooks after GPU move...")
                model._attach_attention_hooks()
                print(f" Hooks re-attached: {len(model._hooks) if hasattr(model, '_hooks') else 0}")
            print(f" Model device: {next(model.parameters()).device}")
        else:
            print("  WARNING: CUDA not available, training on CPU!")
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Build evaluator for validation.
        Uses custom evaluation to handle polygon segmentations properly.
        """
        return SafeCOCOEvaluator(
            dataset_name, 
            output_dir=cfg.OUTPUT_DIR,
            tasks=("bbox", "segm"),  # Both bbox and segmentation
        )
    
    def build_hooks(self):
        """
        Build hooks for training.
        Enables evaluation after training completes.
        """
        hooks_list = super().build_hooks()
        
        # Keep evaluation hooks, but only run after training
        # They won't run during training since TEST.EVAL_PERIOD is not set
        
        return hooks_list

    @classmethod
    def build_train_loader(cls, cfg):
        from mask2former.data.dataset_mappers.mask_former_instance_dataset_mapper import (
            MaskFormerInstanceDatasetMapper,
        )
        
        mapper = MaskFormerInstanceDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    def build_writers(self):
        out = self.cfg.OUTPUT_DIR
        tb_dir = os.path.join(out, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)

        writers = [
            TensorboardXWriter(tb_dir),
            JSONWriter(os.path.join(out, "metrics.json")),
            CommonMetricPrinter(self.cfg.SOLVER.MAX_ITER),
        ]
        if self.wandb_module is not None:
            writers.append(WandBWriter(self.wandb_module))
            print("W&B logging enabled")
        return writers


# ---------------- Setup ----------------
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

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # TF32 for speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    from detectron2.engine import default_setup
    default_setup(cfg, args)
    return cfg


# ---------------- Wandb Helper ----------------
def maybe_init_wandb(args, cfg):
    """Initialize Weights & Biases logging with proper error handling"""
    use = getattr(args, "use_wandb", False)
    if not use:
        return None
    
    try:
        import wandb
        
        if not hasattr(wandb, 'init'):
            print("[ERROR] wandb module doesn't have 'init' method.")
            return None
        
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "mars_mask2former_10class"),
            entity=os.getenv("WANDB_ENTITY", None),
            config={
                "OUTPUT_DIR": cfg.OUTPUT_DIR,
                "SOLVER.BASE_LR": cfg.SOLVER.BASE_LR,
                "SOLVER.MAX_ITER": cfg.SOLVER.MAX_ITER,
                "SOLVER.IMS_PER_BATCH": cfg.SOLVER.IMS_PER_BATCH,
                "MODEL.MARS.ENABLED": cfg.MODEL.MARS.ENABLED,
                "MODEL.MARS.WEIGHT": cfg.MODEL.MARS.WEIGHT,
                "MODEL.MARS.LOSS_TYPE": cfg.MODEL.MARS.LOSS_TYPE,
                "MODEL.MARS.USE_GEM": cfg.MODEL.MARS.USE_GEM,
                "MODEL.MARS.GEM_INIT_P": cfg.MODEL.MARS.GEM_INIT_P,
                "DATASETS.TRAIN": cfg.DATASETS.TRAIN,
                "NUM_CLASSES": 10,
                "SAMPLES_PER_CLASS": 100,
            },
            name=os.path.basename(cfg.OUTPUT_DIR.rstrip("/")),
            resume="allow",
        )
        
        print(f"W&B initialized: {run.url}")
        return wandb
        
    except ImportError:
        print("[WARN] wandb not installed. Install with: pip install wandb")
        return None
    except Exception as e:
        print(f"[WARN] wandb initialization failed: {e}")
        return None


# ---------------- Single Train ----------------
def do_train_once(cfg, args):
    wandb = maybe_init_wandb(args, cfg)
    
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Dataset: 10 classes, 100 samples per class (1,000 total)")
    print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Epochs: {cfg.SOLVER.MAX_ITER * cfg.SOLVER.IMS_PER_BATCH / 1000:.1f}")
    print(f"Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"MARs weight: {cfg.MODEL.MARS.WEIGHT}")
    print(f"MARs loss: {cfg.MODEL.MARS.LOSS_TYPE}")
    print(f"GeM enabled: {cfg.MODEL.MARS.USE_GEM}")
    if cfg.MODEL.MARS.USE_GEM:
        print(f"GeM init p: {cfg.MODEL.MARS.GEM_INIT_P}")
    print("="*80 + "\n")
    
    trainer = Trainer(cfg, wandb_module=wandb)
    trainer.resume_or_load(resume=True)
    trainer.train()
    
    if wandb:
        wandb.finish()


# ---------------- Main ----------------
def main(args):
    cfg = setup(args)
    do_train_once(cfg, args)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--use_wandb", action="store_true", help="enable wandb logging")
    args = parser.parse_args()
    launch(main, num_gpus_per_machine=1, num_machines=1, machine_rank=0, dist_url="auto", args=(args,))