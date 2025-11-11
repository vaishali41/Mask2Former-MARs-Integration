import json
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
import numpy as np

def register_coco_subset(train_images=5000, val_images=1000):
    """Register COCO subset datasets"""
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

# CALL THE REGISTRATION FUNCTION
print("Registering COCO subset...")
register_coco_subset(train_images=5000, val_images=1000)

# NOW CHECK ANNOTATIONS
print("\nChecking annotations...")
dataset_dicts = DatasetCatalog.get("coco_2017_train_subset")

# Check first image
sample = dataset_dicts[0]

print("=" * 70)
print(f"Image: {sample['file_name']}")
print(f"Image ID: {sample.get('image_id', 'N/A')}")
print(f"Height x Width: {sample['height']} x {sample['width']}")
print(f"Number of annotations: {len(sample['annotations'])}")
print("=" * 70)

# Check first 5 annotations
for i, anno in enumerate(sample['annotations'][:5]):
    print(f"\nAnnotation {i}:")
    print(f"  Category ID: {anno['category_id']}")
    
    # Check bounding box
    if 'bbox' in anno:
        bbox = anno['bbox']
        print(f"  ✅ Bbox [x,y,w,h]: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        print(f"     Bbox mode: {anno.get('bbox_mode', 'XYWH_ABS (default)')}")
    else:
        print(f"  ❌ NO BBOX!")
    
    # Check segmentation
    if 'segmentation' in anno:
        seg = anno['segmentation']
        if isinstance(seg, list) and len(seg) > 0:
            print(f"  ✅ Segmentation: Polygon with {len(seg)} parts")
            if isinstance(seg[0], list):
                print(f"     First polygon has {len(seg[0])} coordinates")
        elif isinstance(seg, dict):
            print(f"  ✅ Segmentation: RLE mask")
        else:
            print(f"  ⚠️  Segmentation: Unknown format {type(seg)}")
    else:
        print(f"  ❌ NO SEGMENTATION!")
    
    # Check area
    if 'area' in anno:
        print(f"  Area: {anno['area']:.1f} pixels")
    
    # Check iscrowd
    if 'iscrowd' in anno:
        print(f"  Is crowd: {anno['iscrowd']}")

print("\n" + "=" * 70)
print("SUMMARY:")

total_with_bbox = sum(1 for anno in sample['annotations'] if 'bbox' in anno)
total_with_seg = sum(1 for anno in sample['annotations'] if 'segmentation' in anno)

print(f"Total annotations: {len(sample['annotations'])}")
print(f"With bboxes: {total_with_bbox}")
print(f"With segmentation: {total_with_seg}")
print("=" * 70)

# Check a few more images to be sure
print("\n\nChecking 5 more random images...")
import random
random_samples = random.sample(dataset_dicts[1:100], min(5, 99))

for idx, sample in enumerate(random_samples):
    n_annos = len(sample['annotations'])
    n_bbox = sum(1 for a in sample['annotations'] if 'bbox' in a)
    n_seg = sum(1 for a in sample['annotations'] if 'segmentation' in a)
    print(f"Image {idx+2}: {n_annos} annotations ({n_bbox} bbox, {n_seg} seg)")