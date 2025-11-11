"""
Training Script for MARS + Mask2Former

File: tools/train_mars.py
"""

import os
import time
import torch
from torch.cuda.amp import autocast, GradScaler

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
import detectron2.utils.comm as comm

from mask2former import add_maskformer2_config
from mask2former_mars.modeling.meta_arch.mars_mask_former_head import add_mars_config


def register_coco_subset(train_images=5000, val_images=1000):
    """
    Register COCO subset datasets
    
    Args:
        train_images: Number of training images
        val_images: Number of validation images
    """
    from detectron2.data.datasets import load_coco_json
    
    # Paths (adjust to your setup)
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
    
    # Get metadata from original dataset
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

    """
    # Register
    DatasetCatalog.register("coco_2017_train_subset", lambda: train_subset)
    MetadataCatalog.get("coco_2017_train_subset").set(
        thing_classes=MetadataCatalog.get("coco_2017_train").thing_classes,
        stuff_classes=MetadataCatalog.get("coco_2017_train").stuff_classes,
    )
    
    DatasetCatalog.register("coco_2017_val_subset", lambda: val_subset)
    MetadataCatalog.get("coco_2017_val_subset").set(
        thing_classes=MetadataCatalog.get("coco_2017_val").thing_classes,
        stuff_classes=MetadataCatalog.get("coco_2017_val").stuff_classes,
    )
    
    print(f"✅ Registered datasets:")
    print(f"   coco_2017_train_subset: {len(train_subset)} images")
    print(f"   coco_2017_val_subset: {len(val_subset)} images")
    """

class MARSTrainer(DefaultTrainer):
    """
    Trainer with mixed precision and gradient accumulation support
    """
    
    @classmethod
    def build_model(cls, cfg):
        """Build model with MARS"""
        from mask2former_mars.modeling.meta_arch.mars_mask_former_head import MaskFormerWithMARS
        model = MaskFormerWithMARS(cfg)
        return model
    
    @classmethod
    def build_train_loader(cls, cfg):
        """Build train loader with Mask2Former's data mapper"""
        from mask2former.data.dataset_mappers.mask_former_instance_dataset_mapper import (
            MaskFormerInstanceDatasetMapper,
        )
        from detectron2.data import build_detection_train_loader
        
        mapper = MaskFormerInstanceDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Build test loader with Mask2Former's data mapper"""
        from mask2former.data.dataset_mappers.mask_former_instance_dataset_mapper import (
            MaskFormerInstanceDatasetMapper,
        )
        from detectron2.data import build_detection_test_loader
        
        mapper = MaskFormerInstanceDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        """Build optimizer with gradient accumulation support"""
        from detectron2.solver import build_optimizer
        return build_optimizer(cfg, model)
"""
class MARSTrainer(DefaultTrainer):
    
    #Trainer with mixed precision and gradient accumulation support
    
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=cfg.SOLVER.AMP.ENABLED)
        
        # Gradient accumulation
        self.grad_accumulation_steps = cfg.SOLVER.get("GRADIENT_ACCUMULATION_STEPS", 1)
        self.accumulation_counter = 0
    
    @classmethod
    def build_model(cls, cfg):
        #Build model with MARS
        from mask2former_mars.modeling.meta_arch.mars_mask_former_head import MaskFormerWithMARS
        model = MaskFormerWithMARS(cfg)
        return model
    
    @classmethod
    def build_train_loader(cls, cfg):
        #Build train loader with Mask2Former's data mapper
        from mask2former.data.dataset_mappers.mask_former_instance_dataset_mapper import (
            MaskFormerInstanceDatasetMapper,
        )
        from detectron2.data import build_detection_train_loader
        
        mapper = MaskFormerInstanceDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        #Build test loader with Mask2Former's data mapper
        from mask2former.data.dataset_mappers.mask_former_instance_dataset_mapper import (
            MaskFormerInstanceDatasetMapper,
        )
        from detectron2.data import build_detection_test_loader
        
        mapper = MaskFormerInstanceDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    
    def run_step(self):
        #Training step with mixed precision and gradient accumulation
        assert self.model.training, "Model was changed to eval mode!"
        start = time.perf_counter()
        
        # Get data - initialize iterator if needed
        if not hasattr(self, '_data_loader_iter'):
            self._data_loader_iter = iter(self.data_loader)
        
        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)
        
        data_time = time.perf_counter() - start
        
        # Forward with mixed precision
        with autocast(enabled=self.cfg.SOLVER.AMP.ENABLED):
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())
            losses = losses / self.grad_accumulation_steps
        
        # Backward
        self.scaler.scale(losses).backward()
        
        self.accumulation_counter += 1
        
        # Update weights every N steps
        if self.accumulation_counter % self.grad_accumulation_steps == 0:
            # Unscale and clip gradients
            self.scaler.unscale_(self.optimizer)
            
            if self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        
        # Logging
        self._write_metrics(
            {k: v * self.grad_accumulation_steps for k, v in loss_dict.items()},
            data_time=data_time
        )
        
        # Periodic memory logging
        if self.iter % 100 == 0:
            self._log_memory()
        
    
    def run_step(self):
        #Training step with mixed precision and gradient accumulation
        assert self.model.training, "Model was changed to eval mode!"
        start = time.perf_counter()
        
        # Get data - use the parent class's data loader iterator
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        
        # Forward with mixed precision
        with autocast(enabled=self.cfg.SOLVER.AMP.ENABLED):
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())
            losses = losses / self.grad_accumulation_steps
        
        # Backward
        self.scaler.scale(losses).backward()
        
        self.accumulation_counter += 1
        
        # Update weights every N steps
        if self.accumulation_counter % self.grad_accumulation_steps == 0:
            # Unscale and clip gradients
            self.scaler.unscale_(self.optimizer)
            
            if self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        
        # Logging
        self._write_metrics(
            {k: v * self.grad_accumulation_steps for k, v in loss_dict.items()},
            data_time=data_time
        )
        
        # Periodic memory logging
        if self.iter % 100 == 0:
            self._log_memory()
    
    def _log_memory(self):
        #Log GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[Iter {self.iter}] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
"""
"""
class MARSTrainer(DefaultTrainer):
    
    #Trainer with mixed precision and gradient accumulation support
    
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=cfg.SOLVER.AMP.ENABLED)
        
        # Gradient accumulation
        self.grad_accumulation_steps = cfg.SOLVER.get("GRADIENT_ACCUMULATION_STEPS", 1)
        self.accumulation_counter = 0
    
    @classmethod
    def build_model(cls, cfg):
        #Build model with MARS
        from mask2former_mars.modeling.meta_arch.mars_mask_former_head import MaskFormerWithMARS
        model = MaskFormerWithMARS(cfg)
        return model
    
    def run_step(self):
        #Training step with mixed precision and gradient accumulation
        assert self.model.training, "Model was changed to eval mode!"
        start = time.perf_counter()
        
        # Get data
        data = next(self._trainer_data_loader_iter)
        data_time = time.perf_counter() - start
        
        # Forward with mixed precision
        with autocast(enabled=self.cfg.SOLVER.AMP.ENABLED):
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())
            losses = losses / self.grad_accumulation_steps
        
        # Backward
        self.scaler.scale(losses).backward()
        
        self.accumulation_counter += 1
        
        # Update weights every N steps
        if self.accumulation_counter % self.grad_accumulation_steps == 0:
            # Unscale and clip gradients
            self.scaler.unscale_(self.optimizer)
            
            if self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        
        # Logging
        self._write_metrics(
            {k: v * self.grad_accumulation_steps for k, v in loss_dict.items()},
            data_time=data_time
        )
        
        # Periodic memory logging
        if self.iter % 100 == 0:
            self._log_memory()
    
    def _log_memory(self):
        #Log GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[Iter {self.iter}] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    @property
    def _trainer_data_loader_iter(self):
        #Property to get data loader iterator, creating it if needed
        if not hasattr(self, '_data_loader_iter_obj'):
            self._data_loader_iter_obj = iter(self.data_loader)
        return self._data_loader_iter_obj
"""
"""
class MARSTrainer(DefaultTrainer):
    
    #Trainer with mixed precision and gradient accumulation support
    
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=cfg.SOLVER.AMP.ENABLED)
        
        # Gradient accumulation
        self.grad_accumulation_steps = cfg.SOLVER.get("GRADIENT_ACCUMULATION_STEPS", 1)
        self.accumulation_counter = 0
    
    @classmethod
    def build_model(cls, cfg):
        #Build model with MARS
        from mask2former_mars.modeling.meta_arch.mars_mask_former_head import MaskFormerWithMARS
        model = MaskFormerWithMARS(cfg)
        return model
    
    def run_step(self):
        #Training step with mixed precision and gradient accumulation
        assert self.model.training
        
        # Get data
        data = next(self._data_loader_iter)
        
        # Forward with mixed precision
        with autocast(enabled=self.cfg.SOLVER.AMP.ENABLED):
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())
            losses = losses / self.grad_accumulation_steps
        
        # Backward
        self.scaler.scale(losses).backward()
        
        self.accumulation_counter += 1
        
        # Update weights every N steps
        if self.accumulation_counter % self.grad_accumulation_steps == 0:
            # Unscale and clip gradients
            self.scaler.unscale_(self.optimizer)
            
            if self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        
        # Logging
        self._write_metrics(
            {k: v * self.grad_accumulation_steps for k, v in loss_dict.items()},
            data_time=0
        )
        
        # Periodic memory logging
        if self.iter % 100 == 0:
            self._log_memory()
    
    def _log_memory(self):
        #Log GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[Iter {self.iter}] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
"""

def setup(args):
    """
    Create config and perform setup
    """
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
    
    # Gradient accumulation
    cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 4
    
    # Gradient clipping
    if not hasattr(cfg.SOLVER, 'CLIP_GRADIENTS'):
        cfg.SOLVER.CLIP_GRADIENTS = CN()
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
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"MARS Enabled:     {cfg.MODEL.MARS.ENABLED}")
    print(f"MARS Weight:      {cfg.MODEL.MARS.WEIGHT}")
    print(f"Loss Type:        {cfg.MODEL.MARS.LOSS_TYPE}")
    print(f"Mixed Precision:  {cfg.SOLVER.AMP.ENABLED}")
    print(f"Batch Size:       {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Grad Accum:       {cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective Batch:  {cfg.SOLVER.IMS_PER_BATCH * cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Base LR:          {cfg.SOLVER.BASE_LR}")
    print(f"Max Iterations:   {cfg.SOLVER.MAX_ITER}")
    print(f"Output Dir:       {cfg.OUTPUT_DIR}")
    print("="*60 + "\n")
    
    return cfg


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
