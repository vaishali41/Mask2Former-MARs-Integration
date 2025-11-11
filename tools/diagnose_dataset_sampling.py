#!/usr/bin/env python3
"""
Diagnostic Script: Check Dataset Sampling

This script shows you exactly what your current (broken) sampling is doing
vs what it should be doing with random sampling.
"""

import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def analyze_coco_subset(coco_json, num_images, method='sequential'):
    """
    Analyze class distribution in COCO subset
    
    Args:
        coco_json: Path to COCO annotations JSON
        num_images: Number of images to sample
        method: 'sequential' (first N) or 'random'
    """
    print(f"\n{'='*70}")
    print(f"Analyzing {method.upper()} sampling of {num_images} images")
    print(f"{'='*70}\n")
    
    # Load annotations
    with open(coco_json, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    print(f"Total images in dataset: {len(images)}")
    print(f"Total annotations: {len(annotations)}")
    print(f"Total categories: {len(categories)}")
    
    # Get image IDs for subset
    if method == 'sequential':
        # OLD WAY: Just take first N images
        subset_image_ids = [img['id'] for img in images[:num_images]]
    else:
        # NEW WAY: Random sampling
        import random
        random.seed(42)
        sampled_images = random.sample(images, min(num_images, len(images)))
        subset_image_ids = [img['id'] for img in sampled_images]
    
    subset_image_ids = set(subset_image_ids)
    
    # Count categories in subset
    category_counts = defaultdict(int)
    subset_annotations = 0
    
    for ann in annotations:
        if ann['image_id'] in subset_image_ids:
            category_counts[ann['category_id']] += 1
            subset_annotations += 1
    
    # Print statistics
    print(f"\nSubset Statistics:")
    print(f"  Images selected: {len(subset_image_ids)}")
    print(f"  Annotations: {subset_annotations}")
    print(f"  Unique classes present: {len(category_counts)} / {len(categories)}")
    print(f"  Missing classes: {len(categories) - len(category_counts)}")
    
    # Show class distribution
    print(f"\nClass Distribution (Top 10 most frequent):")
    sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    for cat_id, count in sorted_cats[:10]:
        cat_name = categories[cat_id]
        print(f"  {cat_name:20s}: {count:5d} annotations")
    
    # Show missing classes
    missing_cats = set(categories.keys()) - set(category_counts.keys())
    if missing_cats:
        print(f"\n⚠️  Missing Classes ({len(missing_cats)}):")
        for cat_id in list(missing_cats)[:10]:
            print(f"  - {categories[cat_id]}")
        if len(missing_cats) > 10:
            print(f"  ... and {len(missing_cats) - 10} more")
    
    return category_counts, categories


def compare_sampling_methods():
    """
    Compare sequential vs random sampling
    """
    coco_train = "/home/vaishali/projects/Mask2Former/datasets/coco/annotations/instances_train2017.json"
    
    if not os.path.exists(coco_train):
        print(f"❌ COCO dataset not found at: {coco_train}")
        print("Please adjust the path or ensure COCO is downloaded.")
        return
    
    num_images = 5000
    
    print("\n" + "="*70)
    print("COMPARING SAMPLING METHODS")
    print("="*70)
    
    # Analyze sequential (your current broken method)
    seq_counts, categories = analyze_coco_subset(coco_train, num_images, 'sequential')
    
    # Analyze random (the fix)
    rand_counts, _ = analyze_coco_subset(coco_train, num_images, 'random')
    
    # Create comparison plot
    print("\n📊 Creating comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sequential sampling
    if seq_counts:
        sorted_seq = sorted(seq_counts.items(), key=lambda x: x[1], reverse=True)
        cat_names_seq = [categories[cat_id][:15] for cat_id, _ in sorted_seq[:20]]
        counts_seq = [count for _, count in sorted_seq[:20]]
        
        ax1.barh(range(len(counts_seq)), counts_seq, color='red', alpha=0.7)
        ax1.set_yticks(range(len(counts_seq)))
        ax1.set_yticklabels(cat_names_seq)
        ax1.set_xlabel('Number of Annotations', fontsize=12)
        ax1.set_title(f'SEQUENTIAL Sampling (First {num_images})\n❌ {len(seq_counts)}/{len(categories)} classes',
                     fontsize=14, fontweight='bold', color='red')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
    
    # Random sampling
    if rand_counts:
        sorted_rand = sorted(rand_counts.items(), key=lambda x: x[1], reverse=True)
        cat_names_rand = [categories[cat_id][:15] for cat_id, _ in sorted_rand[:20]]
        counts_rand = [count for _, count in sorted_rand[:20]]
        
        ax2.barh(range(len(counts_rand)), counts_rand, color='green', alpha=0.7)
        ax2.set_yticks(range(len(counts_rand)))
        ax2.set_yticklabels(cat_names_rand)
        ax2.set_xlabel('Number of Annotations', fontsize=12)
        ax2.set_title(f'RANDOM Sampling ({num_images} images)\n✅ {len(rand_counts)}/{len(categories)} classes',
                     fontsize=14, fontweight='bold', color='green')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset_sampling_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved comparison plot: dataset_sampling_comparison.png")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Sequential (❌ BROKEN):")
    print(f"  Classes present: {len(seq_counts)} / {len(categories)}")
    print(f"  Missing classes: {len(categories) - len(seq_counts)}")
    print(f"\nRandom (✅ FIXED):")
    print(f"  Classes present: {len(rand_counts)} / {len(categories)}")
    print(f"  Missing classes: {len(categories) - len(rand_counts)}")
    print(f"\nImprovement: +{len(rand_counts) - len(seq_counts)} classes!")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  COCO Dataset Sampling Analysis                               ║
    ║                                                               ║
    ║  This script shows you why your model has AP = 0             ║
    ║  Your current sampling is BROKEN (sequential)                ║
    ║  The fix uses RANDOM sampling                                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    compare_sampling_methods()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  NEXT STEPS:                                                  ║
    ║                                                               ║
    ║  1. Use train_mars_fixed.py instead of train_mars.py         ║
    ║  2. Re-train your model from scratch                         ║
    ║  3. Check that AP > 0 during training                        ║
    ║  4. Generate new EigenCAMs to see object-level attention     ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
