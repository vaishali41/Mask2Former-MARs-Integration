"""
Training Script for MARS + Mask2Former
File: tools/train_mars.py
"""

import os
import torch
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from mask2former import add_maskformer2_config
from mask2former_mars.modeling.meta_arch.mars_mask_former_head import add_mars_config


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


class MARSTrainer(DefaultTrainer):
    """
    Trainer for MARS + Mask2Former
    
    Uses Detectron2's default training with AMP support.
    Gradient accumulation is handled by Detectron2's AMP implementation.
    """
    
    @classmethod
    def build_model(cls, cfg):
        """Build model with MARS"""
        from mask2former_mars.modeling.meta_arch.mars_mask_former_head import MaskFormerWithMARS
        model = MaskFormerWithMARS(cfg)
        return model
    
    @classmethod
    def build_train_loader(cls, cfg):
        """Build train loader"""
        from mask2former.data.dataset_mappers.mask_former_instance_dataset_mapper import (
            MaskFormerInstanceDatasetMapper,
        )
        from detectron2.data import build_detection_train_loader
        
        mapper = MaskFormerInstanceDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Build test loader"""
        from mask2former.data.dataset_mappers.mask_former_instance_dataset_mapper import (
            MaskFormerInstanceDatasetMapper,
        )
        from detectron2.data import build_detection_test_loader
        
        mapper = MaskFormerInstanceDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


def setup(args):
    """Create config and perform setup"""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_mars_config(cfg)
    
    # Add solver enhancements
    from detectron2.config import CfgNode as CN
    
    # Mixed precision
    if not hasattr(cfg.SOLVER, 'AMP'):
        cfg.SOLVER.AMP = CN()
    cfg.SOLVER.AMP.ENABLED = True
    
    # Gradient accumulation (handled by AMP)
    if not hasattr(cfg.SOLVER, 'GRADIENT_ACCUMULATION_STEPS'):
        cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1
    
    # Gradient clipping
    if not hasattr(cfg.SOLVER, 'CLIP_GRADIENTS'):
        cfg.SOLVER.CLIP_GRADIENTS = CN()
    if not hasattr(cfg.SOLVER.CLIP_GRADIENTS, 'ENABLED'):
        cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
        cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
        cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
        cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    
    # Merge from file and command line
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    default_setup(cfg, args)
    
    # Register datasets
    register_coco_subset(train_images=5000, val_images=1000)
    
    # Print configuration
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"MARS Enabled:       {cfg.MODEL.MARS.ENABLED}")
    print(f"MARS Weight:        {cfg.MODEL.MARS.WEIGHT}")
    print(f"Loss Type:          {cfg.MODEL.MARS.LOSS_TYPE}")
    print(f"Mixed Precision:    {cfg.SOLVER.AMP.ENABLED}")
    print(f"Batch Size:         {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Grad Accum:         {cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective Batch:    {cfg.SOLVER.IMS_PER_BATCH * cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Base LR:            {cfg.SOLVER.BASE_LR}")
    print(f"Max Iterations:     {cfg.SOLVER.MAX_ITER}")
    print(f"Output Dir:         {cfg.OUTPUT_DIR}")
    print("="*70 + "\n")
    
    return cfg

def main(args):
    import sys
    import traceback
    
    try:
        print("="*70)
        print("🔍 DEBUG: Starting main()", flush=True)
        sys.stdout.flush()
        
        print("🔍 DEBUG: Loading config...", flush=True)
        cfg = setup(args)
        print(f"✅ Config loaded. Output: {cfg.OUTPUT_DIR}", flush=True)
        
        if args.eval_only:
            model = MARSTrainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = MARSTrainer.test(cfg, model)
            return res
        
        print("🔍 DEBUG: Checking GPU...", flush=True)
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        print("🔍 DEBUG: Creating trainer...", flush=True)
        sys.stdout.flush()
        trainer = MARSTrainer(cfg)
        print("✅ Trainer created", flush=True)
        
        print("🔍 DEBUG: Loading/resuming checkpoint...", flush=True)
        sys.stdout.flush()
        trainer.resume_or_load(resume=args.resume)
        print("✅ Checkpoint ready", flush=True)
        
        print("🔍 DEBUG: Starting training loop...", flush=True)
        sys.stdout.flush()
        result = trainer.train()
        
        print("✅ Training completed!", flush=True)
        return result
        
    except Exception as e:
        print(f"❌ ERROR: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
"""
def main(args):
    cfg = setup(args)
    
    if args.eval_only:
        model = MARSTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MARSTrainer.test(cfg, model)
        return res
    
    # Enable PyTorch optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    trainer = MARSTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()
"""

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Starting MARS + Mask2Former training...")
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
