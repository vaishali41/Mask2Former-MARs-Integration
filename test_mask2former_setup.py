#!/usr/bin/env python3
"""
Quick test to verify Mask2Former installation and configuration
Run this BEFORE trying to train with MARs to isolate issues
"""

import sys
import torch

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    try:
        import detectron2
        print(f"✅ detectron2 version: {detectron2.__version__}")
    except ImportError as e:
        print(f"❌ detectron2 import failed: {e}")
        return False
    
    try:
        from detectron2.projects.deeplab import add_deeplab_config
        print("✅ DeepLab config available")
    except ImportError as e:
        print(f"❌ DeepLab config import failed: {e}")
        return False
    
    try:
        from mask2former import add_maskformer2_config
        print("✅ Mask2Former config available")
    except ImportError as e:
        print(f"❌ Mask2Former import failed: {e}")
        print("   Try: cd Mask2Former && python setup.py build develop")
        return False
    
    try:
        from mask2former.maskformer_model import MaskFormer
        print("✅ Mask2Former model available")
    except ImportError as e:
        print(f"❌ Mask2Former model import failed: {e}")
        return False
    
    return True


def test_config():
    """Test if basic configuration works"""
    print("\nTesting configuration...")
    from detectron2.config import get_cfg, CfgNode as CN
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config
    from detectron2.data import DatasetCatalog, MetadataCatalog
    
    # Register a dummy dataset for testing (only if not already registered)
    if "test_dataset" not in DatasetCatalog:
        DatasetCatalog.register("test_dataset", lambda: [])
        MetadataCatalog.get("test_dataset").set(
            thing_classes=["object"] * 80,
            thing_dataset_id_to_contiguous_id={i: i for i in range(80)},
            stuff_classes=[],
        )
    
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Set minimal required configs
    cfg.MODEL.META_ARCHITECTURE = "MaskFormer"
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    
    # SEM_SEG_HEAD config
    if not hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
        cfg.MODEL.SEM_SEG_HEAD = CN()
    cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskFormerHead"
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MSDeformAttnPixelDecoder"
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 80
    
    # MASK_FORMER config
    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 9
    
    # Dataset config (required for model instantiation)
    cfg.DATASETS.TRAIN = ("test_dataset",)
    cfg.DATASETS.TEST = ("test_dataset",)
    
    cfg.MODEL.WEIGHTS = ""  # No pretrained weights for test
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.28, 103.53]
    cfg.MODEL.PIXEL_STD = [58.395, 57.12, 57.375]
    
    print("✅ Configuration created successfully")
    
    # Print key config values
    print("\nKey configuration values:")
    print(f"  SEM_SEG_HEAD.PIXEL_DECODER_NAME: {cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME}")
    print(f"  SEM_SEG_HEAD.IN_FEATURES: {cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES}")
    print(f"  SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES: {cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES}")
    print(f"  MASK_FORMER.TRANSFORMER_IN_FEATURE: {cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE}")
    
    return cfg


def test_model(cfg):
    """Test if model can be instantiated"""
    print("\nTesting model instantiation...")
    from mask2former.maskformer_model import MaskFormer
    
    try:
        model = MaskFormer(cfg)
        print("✅ Mask2Former model created successfully")
        
        # Check model components
        if hasattr(model, 'backbone'):
            print("  ✅ Backbone initialized")
        if hasattr(model, 'sem_seg_head'):
            print("  ✅ Segmentation head initialized")
            if hasattr(model.sem_seg_head, 'pixel_decoder'):
                print("  ✅ Pixel decoder initialized")
                print(f"     Type: {type(model.sem_seg_head.pixel_decoder).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward():
    """Test a dummy forward pass"""
    print("\nTesting dummy forward pass...")
    from detectron2.config import get_cfg, CfgNode as CN
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config
    from mask2former.maskformer_model import MaskFormer
    from detectron2.data import DatasetCatalog, MetadataCatalog
    
    # Register dummy dataset
    if "test_dataset_forward" not in DatasetCatalog:
        DatasetCatalog.register("test_dataset_forward", lambda: [])
        MetadataCatalog.get("test_dataset_forward").set(
            thing_classes=["object"] * 80,
            thing_dataset_id_to_contiguous_id={i: i for i in range(80)},
            stuff_classes=[],
        )
    
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Minimal config
    cfg.MODEL.META_ARCHITECTURE = "MaskFormer"
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    
    if not hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
        cfg.MODEL.SEM_SEG_HEAD = CN()
    cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskFormerHead"
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MSDeformAttnPixelDecoder"
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 80
    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 9
    
    # Dataset config
    cfg.DATASETS.TRAIN = ("test_dataset_forward",)
    cfg.DATASETS.TEST = ("test_dataset_forward",)
    
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.28, 103.53]
    cfg.MODEL.PIXEL_STD = [58.395, 57.12, 57.375]
    
    try:
        model = MaskFormer(cfg)
        model.eval()
        
        # Create dummy input
        dummy_input = [{
            "image": torch.rand(3, 224, 224),
            "height": 224,
            "width": 224
        }]
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print("✅ Dummy forward pass successful")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("Mask2Former Installation Test")
    print("="*80)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Fix imports before continuing.")
        return False
    
    # Test 2: Configuration
    try:
        cfg = test_config()
    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Model instantiation
    if not test_model(cfg):
        print("\n❌ Model instantiation failed.")
        return False
    
    # Test 4: Forward pass
    if not test_forward():
        print("\n❌ Forward pass test failed.")
        return False
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nYour Mask2Former installation is working correctly.")
    print("You can now proceed with training using MARs.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
