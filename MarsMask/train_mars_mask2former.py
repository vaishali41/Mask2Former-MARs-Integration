"""
Training script for Mask2Former with MARS regularization
Modified version of train_net.py with MARS integration
"""

import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
import detectron2.utils.comm as comm

from mask2former import add_maskformer2_config
from mars_integration import add_mars_config, MaskFormerWithMARS
from train_net import Trainer, register_coco_subset  # Reuse existing trainer


class MARSTrainer(Trainer):
    """
    Extended trainer for MARS training
    Inherits from existing Trainer, just changes model building
    """
    
    @classmethod
    def build_model(cls, cfg):
        """
        Build MaskFormer with MARS instead of standard MaskFormer
        """
        model = MaskFormerWithMARS(cfg)
        return model


def setup(args):
    """
    Create configs and perform basic setups for MARS training
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_mars_config(cfg)  # Add MARS config
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # MARS-specific settings (can be overridden by config file)
    if not hasattr(cfg.MODEL, 'MARS'):
        cfg.MODEL.MARS.ENABLED = True
        cfg.MODEL.MARS.WEIGHT = 0.1
        cfg.MODEL.MARS.NUM_VIEWS = 2
    
    cfg.freeze()
    default_setup(cfg, args)
    
    # Optional: Initialize Weights & Biases
    try:
        import wandb
        wandb.init(
            project="mask2former-mars",
            name=f"mars_training_{cfg.OUTPUT_DIR.split('/')[-1]}",
            config={
                "batch_size": cfg.SOLVER.IMS_PER_BATCH,
                "learning_rate": cfg.SOLVER.BASE_LR,
                "max_iter": cfg.SOLVER.MAX_ITER,
                "mars_enabled": cfg.MODEL.MARS.ENABLED,
                "mars_weight": cfg.MODEL.MARS.WEIGHT,
            }
        )
        print("✅ Weights & Biases initialized")
    except ImportError:
        print("⚠️  wandb not installed. Only TensorBoard will be used.")
        print("   Install with: pip install wandb")
    
    return cfg


def main(args):
    cfg = setup(args)
    
    if args.eval_only:
        # For evaluation, use standard model (no MARS needed)
        model = Trainer.build_model(cfg)
        from detectron2.checkpoint import DetectionCheckpointer
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res
    
    # Training with MARS
    trainer = MARSTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    print("Training Mask2Former WITH MARS regularization")
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
