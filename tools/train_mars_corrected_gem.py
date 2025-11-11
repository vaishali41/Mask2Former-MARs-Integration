# -*- coding: utf-8 -*-
"""
Training script for Mask2Former + MARs + GeM
- Proper MODEL.MARS config usage
- TensorBoard logging out of the box
- Optional Weights & Biases (wandb) logging via --use_wandb
- Simple hyperparameter sweep via --sweep (LR, λ, GeM init p)
"""
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import itertools, time
import torch
from detectron2.engine import default_argument_parser, launch, DefaultTrainer
from detectron2.config import get_cfg, CfgNode as CN
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import hooks
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.events import TensorboardXWriter, get_event_storage
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
    cfg.MODEL.MARS.GEM_INIT_P = 1.0   # start from 1.0 (mean) by default; sweep will override
    cfg.MODEL.MARS.GEM_MIN_P = 1.0
    cfg.MODEL.MARS.GEM_MAX_P = 6.0
    cfg.MODEL.MARS.WARMUP_ITERS = 3000

    # Optional SWEEP support
        # Optional SWEEP support
    cfg.SWEEP = CN()
    cfg.SWEEP.ENABLED = False
    cfg.SWEEP.SOLVER = CN()
    cfg.SWEEP.SOLVER.BASE_LR = [0.00005, 0.0001, 0.0002]
    cfg.SWEEP.MODEL = CN()
    cfg.SWEEP.MODEL.MARS_WEIGHT = [0.1, 0.5, 1.0]
    cfg.SWEEP.MODEL.LOSS_TYPE = ["kl", "cosine"]     # <-- ADD THIS
    cfg.SWEEP.MODEL.GEM_INIT_P = [1.0, 2.0, 3.0]     # <-- ADD THIS


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

# ---------------- Custom Trainer ----------------
class WandBWriter:
    """Custom hook to log metrics to Weights & Biases"""
    def __init__(self, wandb_module):
        self.wandb = wandb_module
        
    def write(self):
        """Called by detectron2 to write metrics"""
        storage = get_event_storage()
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
            print(f"✅ Moving model to GPU: {torch.cuda.get_device_name(0)}")
            model = model.cuda()
            # CRITICAL: Re-attach hooks after moving to GPU
            if hasattr(model, '_attach_attention_hooks'):
                print("Re-attaching MARs hooks after GPU move...")
                model._attach_attention_hooks()
                print(f"✅ Hooks re-attached: {len(model._hooks) if hasattr(model, '_hooks') else 0}")
            print(f"✅ Model device: {next(model.parameters()).device}")
        else:
            print("⚠️  WARNING: CUDA not available, training on CPU!")
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, tasks=("segm",), output_dir=cfg.OUTPUT_DIR)

    @classmethod
    def build_train_loader(cls, cfg):
        # Import Mask2Former's data mapper that loads masks
        from mask2former.data.dataset_mappers.mask_former_instance_dataset_mapper import (
            MaskFormerInstanceDatasetMapper,
        )
        
        # Use MaskFormerInstanceDatasetMapper which loads segmentation masks properly
        mapper = MaskFormerInstanceDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    from detectron2.utils.events import JSONWriter, CommonMetricPrinter

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

    """
    def build_writers(self):
        #Build writers for logging
        out = self.cfg.OUTPUT_DIR
        tb_dir = os.path.join(out, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        
        writers = [TensorboardXWriter(tb_dir)]
        
        # Add wandb writer if available
        if self.wandb_module is not None:
            writers.append(WandBWriter(self.wandb_module))
            print("✅ W&B logging enabled")
        
        return writers
    """
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
    
    # SEM_SEG_HEAD config - critical for pixel decoder
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
        ssh.NUM_CLASSES = 80  # COCO instance segmentation
    
    # Additional defaults
    if not hasattr(cfg.INPUT, "MIN_SIZE_TRAIN_SAMPLING"):
        cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    
    # DATALOADER settings for Mask2Former
    if not hasattr(cfg.DATALOADER, "FILTER_EMPTY_ANNOTATIONS"):
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True


    add_mars_config(cfg)
    
    # Register datasets
    register_coco_subset(train_images=5000, val_images=1000)
    
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

    # TF32 for speed + AMP (Detectron2 uses AMP automatically with APEX/AMP if enabled)
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
        
        # Check if wandb has the init method
        if not hasattr(wandb, 'init'):
            print("[ERROR] wandb module doesn't have 'init' method. This might be a version issue.")
            print("        Try: pip install --upgrade wandb")
            return None
        
        # Initialize wandb
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "mars_mask2former"),
            entity=os.getenv("WANDB_ENTITY", None),  # Optional: your wandb username/team
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
                "MODEL.MARS.WARMUP_ITERS": cfg.MODEL.MARS.WARMUP_ITERS,
                "DATASETS.TRAIN": cfg.DATASETS.TRAIN,
                "MODEL.BACKBONE.NAME": cfg.MODEL.BACKBONE.NAME,
            },
            name=os.path.basename(cfg.OUTPUT_DIR.rstrip("/")),
            resume="allow",  # Allow resuming if training crashes
        )
        
        print(f"✅ W&B initialized: {run.url}")
        return wandb
        
    except ImportError:
        print("[WARN] wandb not installed. Install with: pip install wandb")
        print("       Then login with: wandb login")
        return None
    except Exception as e:
        print(f"[WARN] wandb initialization failed: {e}")
        print("       Training will continue without W&B logging")
        return None

# ---------------- Single Train ----------------
def do_train_once(cfg, args):
    wandb = maybe_init_wandb(args, cfg)
    
    # Debug: print pixel decoder configuration
    print("\n" + "="*80)
    print("PIXEL DECODER CONFIGURATION DEBUG")
    print("="*80)
    print(f"SEM_SEG_HEAD.NAME: {cfg.MODEL.SEM_SEG_HEAD.NAME}")
    print(f"SEM_SEG_HEAD.PIXEL_DECODER_NAME: {cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME}")
    print(f"SEM_SEG_HEAD.IN_FEATURES: {cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES}")
    if hasattr(cfg.MODEL.SEM_SEG_HEAD, "DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES"):
        print(f"SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES: {cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES}")
    print(f"MASK_FORMER.TRANSFORMER_IN_FEATURE: {cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE}")
    print(f"BACKBONE OUT_FEATURES: {cfg.MODEL.RESNETS.OUT_FEATURES}")
    print("="*80 + "\n")
    
    trainer = Trainer(cfg, wandb_module=wandb)
    trainer.resume_or_load(resume=True)
    trainer.train()
    
    if wandb:
        wandb.finish()

# ---------------- Sweep ----------------
def sweep(cfg, args, sweep_space):
    keys = list(sweep_space.keys())
    values = [sweep_space[k] for k in keys]

    for combo in itertools.product(*values):
        cfg2 = cfg.clone()
        cfg2.defrost()

        # apply overrides (nested: e.g., "MODEL.MARS.WEIGHT")
        for k, v in zip(keys, combo):
            node = cfg2
            parts = k.split(".")
            for p in parts[:-1]:
                node = getattr(node, p)
            setattr(node, parts[-1], v)

        tag = "_".join([f"{k.replace('.', '_')}-{v}" for k, v in zip(keys, combo)])
        cfg2.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f"sweep__{tag}")
        os.makedirs(cfg2.OUTPUT_DIR, exist_ok=True)

        cfg2.freeze()
        print(f"\n=== SWEEP: {tag} ===")
        do_train_once(cfg2, args)

# ---------------- Main ----------------
def main(args):
    cfg = setup(args)

    # default sweep space (can also read from cfg if you want)
    sweep_space = {
        "SOLVER.BASE_LR": [5e-5, 1e-4, 2e-4],
        "MODEL.MARS.WEIGHT": [0.05, 0.1, 0.2],
        "MODEL.MARS.LOSS_TYPE": ["kl", "cosine"],
        "MODEL.MARS.GEM_INIT_P": [1.0, 2.0, 3.0, 4.0],
    }

    if args.sweep:
        sweep(cfg, args, sweep_space)
    else:
        do_train_once(cfg, args)

if __name__ == "__main__":
    parser = default_argument_parser()
    #parser.add_argument("--config-file", required=True)
    parser.add_argument("--sweep", action="store_true", help="run hyperparameter sweep")
    parser.add_argument("--use_wandb", action="store_true", help="enable wandb logging")
    # passthrough extra overrides: e.g. SOLVER.BASE_LR 0.0002
    args = parser.parse_args()
    launch(main, num_gpus_per_machine=1, num_machines=1, machine_rank=0, dist_url="auto", args=(args,))
