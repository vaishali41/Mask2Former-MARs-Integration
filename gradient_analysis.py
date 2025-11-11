#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient Analysis Script for MARS Loss
--------------------------------------
This script runs 1 training step and analyzes/visualizes gradients to verify
if gradients are flowing from the MARS loss.

Visualization Tools Used:
1. TensorBoard - Gradient histograms
2. Matplotlib - Gradient distribution plots  
3. Graphviz - Computation graph visualization
4. Custom gradient flow analysis
5. Grad-CAM style gradient heatmaps
"""

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import psutil

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple
import json

# Detectron2 imports
from detectron2.engine import default_argument_parser, launch, DefaultTrainer
from detectron2.config import get_cfg, CfgNode as CN
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.utils.events import get_event_storage

# Mask2Former imports
from mask2former import add_maskformer2_config
from mask2former_mars.modeling.meta_arch.mars_mask_former_head_gem import MaskFormerWithMARS


class GradientAnalyzer:
    """Comprehensive gradient analysis and visualization"""
    
    def __init__(self, output_dir="gradient_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Storage for gradient information
        self.gradients = {}
        self.gradient_norms = {}
        self.gradient_stats = defaultdict(dict)
        self.hooks = []
        
        print(f"\n{'='*80}")
        print(f"Gradient Analyzer initialized - Output: {output_dir}")
        print(f"{'='*80}\n")
    
    def register_hooks(self, model: nn.Module):
        """Register backward hooks to capture gradients"""
        print("Registering gradient hooks...")
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Hook to capture gradient
                hook = param.register_hook(
                    lambda grad, name=name: self._gradient_hook(name, grad)
                )
                self.hooks.append(hook)
        
        print(f" Registered {len(self.hooks)} gradient hooks\n")
    
    def _gradient_hook(self, name: str, grad: torch.Tensor):
        """Hook function called during backward pass"""
        if grad is not None:
            # Store gradient
            self.gradients[name] = grad.detach().cpu()
            
            # Compute statistics
            grad_np = grad.detach().cpu().numpy().flatten()
            self.gradient_stats[name] = {
                'mean': float(np.mean(grad_np)),
                'std': float(np.std(grad_np)),
                'min': float(np.min(grad_np)),
                'max': float(np.max(grad_np)),
                'norm': float(torch.norm(grad).item()),
                'num_zeros': int(np.sum(grad_np == 0)),
                'num_params': int(grad_np.size),
                'sparsity': float(np.sum(grad_np == 0) / grad_np.size),
            }
            
            # Check if gradient is from MARS-related parameters
            is_mars = any(keyword in name.lower() for keyword in ['mars', 'gem', 'attention'])
            if is_mars:
                print(f"   MARS-related gradient captured: {name}")
                print(f"     Norm: {self.gradient_stats[name]['norm']:.6e}")
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def analyze_gradient_flow(self) -> Dict:
        """Analyze gradient flow through the network"""
        print(f"\n{'='*80}")
        print("GRADIENT FLOW ANALYSIS")
        print(f"{'='*80}\n")
        
        # Categorize parameters
        mars_params = []
        backbone_params = []
        decoder_params = []
        other_params = []
        
        for name, stats in self.gradient_stats.items():
            if any(k in name.lower() for k in ['mars', 'gem']):
                mars_params.append((name, stats))
            elif 'backbone' in name.lower():
                backbone_params.append((name, stats))
            elif 'decoder' in name.lower() or 'transformer' in name.lower():
                decoder_params.append((name, stats))
            else:
                other_params.append((name, stats))
        
        # Print summary
        categories = {
            'MARS/GeM Parameters': mars_params,
            'Backbone Parameters': backbone_params,
            'Decoder Parameters': decoder_params,
            'Other Parameters': other_params,
        }
        
        summary = {}
        
        for category_name, params in categories.items():
            if params:
                print(f"\n{category_name}: {len(params)} parameters")
                print("-" * 80)
                
                norms = [p[1]['norm'] for p in params]
                means = [p[1]['mean'] for p in params]
                
                # Top 5 by gradient norm
                top_params = sorted(params, key=lambda x: x[1]['norm'], reverse=True)[:5]
                
                print(f"  Average gradient norm: {np.mean(norms):.6e}")
                print(f"  Max gradient norm: {np.max(norms):.6e}")
                print(f"  Min gradient norm: {np.min(norms):.6e}")
                print(f"\n  Top 5 by gradient norm:")
                for param_name, stats in top_params:
                    print(f"    • {param_name}: {stats['norm']:.6e}")
                
                summary[category_name] = {
                    'count': len(params),
                    'avg_norm': float(np.mean(norms)),
                    'max_norm': float(np.max(norms)),
                    'min_norm': float(np.min(norms)),
                    'avg_mean': float(np.mean(means)),
                    'top_5': [(p[0], p[1]['norm']) for p in top_params]
                }
        
        # Check if MARS gradients exist
        mars_gradient_exists = len(mars_params) > 0
        
        print(f"\n{'='*80}")
        print(f" MARS Gradient Detection: {' YES' if mars_gradient_exists else ' NO'}")
        if mars_gradient_exists:
            print(f"   Found gradients in {len(mars_params)} MARS-related parameters")
        else:
            print(f"     WARNING: No gradients found in MARS-related parameters!")
            print(f"   This suggests the MARS loss might not be contributing to backprop")
        print(f"{'='*80}\n")
        
        summary['mars_gradient_detected'] = mars_gradient_exists
        
        # Save to JSON
        with open(os.path.join(self.output_dir, 'gradient_flow_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def visualize_with_matplotlib(self):
        """Create matplotlib visualizations"""
        print("\n Creating Matplotlib visualizations...")
        # --- memory-aware safeguard ---
        mem = psutil.virtual_memory()
        if mem.total < 16 * 1024**3 or mem.available < 6 * 1024**3:
            print("  Low-memory system detected — using light visualization mode.")
            MAX_GRADS = 200_000
        else:
            MAX_GRADS = 2_000_000
        # 1. Gradient distribution histogram
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Gradient Analysis - Multiple Views', fontsize=16, fontweight='bold')
        
        # Collect all gradients
        all_grads = []
        mars_grads = []
        other_grads = []
        
        for name, grad in self.gradients.items():
            #grad_flat = grad.numpy().flatten()
            grad_flat = grad.numpy().flatten()
            if grad_flat.size > MAX_GRADS:
                grad_flat = np.random.choice(grad_flat, MAX_GRADS, replace=False)
            all_grads.extend(grad_flat)
            
            if any(k in name.lower() for k in ['mars', 'gem']):
                mars_grads.extend(grad_flat)
            else:
                other_grads.extend(grad_flat)
        
        # Plot 1: Overall gradient distribution
        axes[0, 0].hist(all_grads, bins=100, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('All Gradients Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Gradient Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: MARS vs Other gradients
        if mars_grads:
            axes[0, 1].hist(mars_grads, bins=100, alpha=0.6, label='MARS', color='red', edgecolor='black')
        if other_grads:
            axes[0, 1].hist(other_grads, bins=100, alpha=0.6, label='Other', color='blue', edgecolor='black')
        axes[0, 1].set_title('MARS vs Other Gradients', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Gradient Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Gradient norms by layer
        layer_names = []
        layer_norms = []
        layer_colors = []
        
        for name, stats in sorted(self.gradient_stats.items(), 
                                  key=lambda x: x[1]['norm'], 
                                  reverse=True)[:20]:  # Top 20
            layer_names.append(name.split('.')[-1][:15])  # Shortened name
            layer_norms.append(stats['norm'])
            
            if any(k in name.lower() for k in ['mars', 'gem']):
                layer_colors.append('red')
            else:
                layer_colors.append('blue')
        
        axes[1, 0].barh(range(len(layer_names)), layer_norms, color=layer_colors)
        axes[1, 0].set_yticks(range(len(layer_names)))
        axes[1, 0].set_yticklabels(layer_names, fontsize=8)
        axes[1, 0].set_xlabel('Gradient Norm')
        axes[1, 0].set_title('Top 20 Layers by Gradient Norm (Red=MARS)', 
                           fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Gradient sparsity
        sparsity_data = []
        sparsity_labels = []
        sparsity_colors = []
        
        for name, stats in sorted(self.gradient_stats.items(), 
                                  key=lambda x: x[1]['sparsity'], 
                                  reverse=True)[:20]:
            sparsity_data.append(stats['sparsity'] * 100)
            sparsity_labels.append(name.split('.')[-1][:15])
            
            if any(k in name.lower() for k in ['mars', 'gem']):
                sparsity_colors.append('red')
            else:
                sparsity_colors.append('blue')
        
        axes[1, 1].barh(range(len(sparsity_labels)), sparsity_data, color=sparsity_colors)
        axes[1, 1].set_yticks(range(len(sparsity_labels)))
        axes[1, 1].set_yticklabels(sparsity_labels, fontsize=8)
        axes[1, 1].set_xlabel('Sparsity (%)')
        axes[1, 1].set_title('Top 20 Layers by Gradient Sparsity (Red=MARS)', 
                           fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'gradient_distributions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: gradient_distributions.png")
        
        # 2. Gradient flow heatmap
        self._create_gradient_heatmap()
    
    def _create_gradient_heatmap(self):
        """Create a heatmap showing gradient flow"""
        # Group parameters by module
        module_gradients = defaultdict(list)
        
        for name, stats in self.gradient_stats.items():
            parts = name.split('.')
            if len(parts) > 1:
                module = '.'.join(parts[:2])  # First two levels
            else:
                module = parts[0]
            
            module_gradients[module].append(stats['norm'])
        
        # Create heatmap data
        modules = sorted(module_gradients.keys())
        data = [module_gradients[m] for m in modules]
        
        # Pad to same length
        max_len = max(len(d) for d in data)
        data_padded = [d + [0] * (max_len - len(d)) for d in data]
        
        fig, ax = plt.subplots(figsize=(16, max(8, len(modules) * 0.3)))
        
        im = ax.imshow(np.log10(np.array(data_padded) + 1e-10), 
                      aspect='auto', cmap='YlOrRd')
        
        ax.set_yticks(range(len(modules)))
        ax.set_yticklabels(modules, fontsize=8)
        ax.set_xlabel('Parameter Index within Module')
        ax.set_title('Gradient Norm Heatmap (log10 scale)', 
                    fontsize=14, fontweight='bold')
        
        # Highlight MARS modules
        for i, module in enumerate(modules):
            if any(k in module.lower() for k in ['mars', 'gem']):
                ax.get_yticklabels()[i].set_color('red')
                ax.get_yticklabels()[i].set_weight('bold')
        
        plt.colorbar(im, ax=ax, label='log10(Gradient Norm)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'gradient_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: gradient_heatmap.png")
    
    def create_gradient_flow_diagram(self):
        """Create a simplified gradient flow diagram"""
        print("\n Creating gradient flow diagram...")
        
        try:
            import graphviz
            
            dot = graphviz.Digraph(comment='Gradient Flow', format='png')
            dot.attr(rankdir='TB', size='12,16')
            dot.attr('node', shape='box', style='rounded,filled', fontsize='10')
            
            # Define node categories
            categories = {
                'Input': [],
                'Backbone': [],
                'Decoder': [],
                'MARS': [],
                'Loss': []
            }
            
            # Categorize parameters with gradients
            for name, stats in self.gradient_stats.items():
                norm = stats['norm']
                
                if 'backbone' in name.lower():
                    categories['Backbone'].append((name, norm))
                elif any(k in name.lower() for k in ['decoder', 'transformer']):
                    categories['Decoder'].append((name, norm))
                elif any(k in name.lower() for k in ['mars', 'gem']):
                    categories['MARS'].append((name, norm))
                elif 'loss' in name.lower():
                    categories['Loss'].append((name, norm))
            
            # Create nodes for each category
            for category, params in categories.items():
                if params:
                    avg_norm = np.mean([p[1] for p in params])
                    max_norm = np.max([p[1] for p in params])
                    
                    # Color based on gradient strength
                    if max_norm > 1e-2:
                        color = '#ff4444'  # Strong gradient - red
                    elif max_norm > 1e-4:
                        color = '#ffaa44'  # Medium gradient - orange
                    else:
                        color = '#aaaaaa'  # Weak gradient - gray
                    
                    if category == 'MARS':
                        color = '#44ff44' if max_norm > 1e-6 else '#ff4444'
                    
                    label = f"{category}\\n{len(params)} params\\nAvg: {avg_norm:.2e}\\nMax: {max_norm:.2e}"
                    dot.node(category, label, fillcolor=color, fontcolor='black')
            
            # Add edges
            if 'Backbone' in categories and categories['Backbone']:
                dot.edge('Input', 'Backbone')
            if 'Decoder' in categories and categories['Decoder']:
                dot.edge('Backbone', 'Decoder')
            if 'MARS' in categories and categories['MARS']:
                dot.edge('Decoder', 'MARS', color='red', penwidth='2')
                dot.edge('MARS', 'Loss', color='red', penwidth='2', 
                        label='MARS Loss')
            if 'Loss' in categories:
                dot.edge('Decoder', 'Loss', label='Main Loss')
            
            # Render
            output_path = os.path.join(self.output_dir, 'gradient_flow_diagram')
            dot.render(output_path, cleanup=True)
            
            print(f"   Saved: gradient_flow_diagram.png")
            
        except ImportError:
            print("    Graphviz not installed, skipping flow diagram")
            print("     Install with: pip install graphviz")
        except Exception as e:
            print(f"    Error creating flow diagram: {e}")
    
    def generate_report(self):
        """Generate a comprehensive text report"""
        print("\n Generating comprehensive report...")
        
        report_path = os.path.join(self.output_dir, 'gradient_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GRADIENT ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total parameters with gradients: {len(self.gradient_stats)}\n")
            
            all_norms = [s['norm'] for s in self.gradient_stats.values()]
            f.write(f"Average gradient norm: {np.mean(all_norms):.6e}\n")
            f.write(f"Max gradient norm: {np.max(all_norms):.6e}\n")
            f.write(f"Min gradient norm: {np.min(all_norms):.6e}\n\n")
            
            # MARS-specific analysis
            f.write("MARS GRADIENT ANALYSIS\n")
            f.write("-"*80 + "\n")
            
            mars_params = [(n, s) for n, s in self.gradient_stats.items() 
                          if any(k in n.lower() for k in ['mars', 'gem'])]
            
            if mars_params:
                f.write(f" MARS gradients detected: {len(mars_params)} parameters\n\n")
                f.write("MARS parameters with gradients:\n")
                for name, stats in sorted(mars_params, key=lambda x: x[1]['norm'], reverse=True):
                    f.write(f"  • {name}\n")
                    f.write(f"    Norm: {stats['norm']:.6e}, ")
                    f.write(f"Mean: {stats['mean']:.6e}, ")
                    f.write(f"Std: {stats['std']:.6e}\n")
                f.write("\n")
            else:
                f.write(" NO MARS gradients detected!\n")
                f.write("This suggests the MARS loss is not contributing to backpropagation.\n\n")
                f.write("Possible causes:\n")
                f.write("  1. MARS loss weight is zero\n")
                f.write("  2. MARS loss returns a constant (detached tensor)\n")
                f.write("  3. Attention hooks are not capturing data correctly\n")
                f.write("  4. GeM pooling issues\n\n")
            
            # Top 20 parameters by gradient norm
            f.write("TOP 20 PARAMETERS BY GRADIENT NORM\n")
            f.write("-"*80 + "\n")
            
            top_20 = sorted(self.gradient_stats.items(), 
                          key=lambda x: x[1]['norm'], 
                          reverse=True)[:20]
            
            for i, (name, stats) in enumerate(top_20, 1):
                is_mars = any(k in name.lower() for k in ['mars', 'gem'])
                marker = "🔴" if is_mars else "  "
                f.write(f"{i:2d}. {marker} {name}\n")
                f.write(f"     Norm: {stats['norm']:.6e}, ")
                f.write(f"Sparsity: {stats['sparsity']:.2%}\n")
            
            f.write("\n")
            f.write("🔴 = MARS-related parameter\n")
            f.write("\n" + "="*80 + "\n")
        
        print(f"   Saved: gradient_analysis_report.txt")
        
        # Also print key findings to console
        print(f"\n{'='*80}")
        print("KEY FINDINGS")
        print(f"{'='*80}")
        
        mars_params = [(n, s) for n, s in self.gradient_stats.items() 
                      if any(k in n.lower() for k in ['mars', 'gem'])]
        
        if mars_params:
            print(f" MARS gradients detected in {len(mars_params)} parameters")
            print(f"   Average MARS gradient norm: {np.mean([s['norm'] for _, s in mars_params]):.6e}")
        else:
            print(f" NO MARS gradients detected!")
            print(f"   The MARS loss is not contributing to backpropagation")
        
        print(f"{'='*80}\n")


def add_mars_config(cfg):
    """Add MARS configuration"""
    cfg.MODEL.MARS = CN()
    cfg.MODEL.MARS.ENABLED = True
    cfg.MODEL.MARS.WEIGHT = 1.0
    cfg.MODEL.MARS.LOSS_TYPE = "cosine"
    cfg.MODEL.MARS.USE_GEM = True
    cfg.MODEL.MARS.GEM_INIT_P = 1.0
    cfg.MODEL.MARS.GEM_MIN_P = 1.0
    cfg.MODEL.MARS.GEM_MAX_P = 6.0
    cfg.MODEL.MARS.WARMUP_ITERS = 3000


def register_coco_subset(train_images=10):
    """Register small COCO subset for quick testing"""
    from detectron2.data.datasets import load_coco_json
    
    coco_root = "datasets/coco"
    train_json = os.path.join(coco_root, "annotations/instances_train2017.json")
    train_image_root = os.path.join(coco_root, "train2017")
    
    train_dicts = load_coco_json(train_json, train_image_root, "coco_2017_train")
    train_subset = train_dicts[:train_images]
    
    train_metadata = MetadataCatalog.get("coco_2017_train")
    
    DatasetCatalog.register("coco_2017_train_minimal", lambda: train_subset)
    train_subset_metadata = MetadataCatalog.get("coco_2017_train_minimal")
    train_subset_metadata.set(thing_classes=train_metadata.thing_classes)
    if hasattr(train_metadata, "stuff_classes"):
        train_subset_metadata.set(stuff_classes=train_metadata.stuff_classes)
    
    print(f" Registered minimal dataset: {len(train_subset)} images\n")


def setup_minimal_config(args):
    """Setup minimal config for 1-step training"""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_mars_config(cfg)
    
    # Register minimal dataset
    register_coco_subset(train_images=1)  # Just 2 images for 1 batch
    
    # Minimal Mask2Former config
    if not hasattr(cfg.MODEL, "MASK_FORMER"):
        cfg.MODEL.MASK_FORMER = CN()
    
    mf = cfg.MODEL.MASK_FORMER
    mf.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"
    mf.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    mf.NUM_OBJECT_QUERIES = 100
    mf.DEC_LAYERS = 2  # Reduced from default
    mf.SIZE_DIVISIBILITY = 32
    
    # Pixel decoder config
    if not hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
        cfg.MODEL.SEM_SEG_HEAD = CN()
    
    ssh = cfg.MODEL.SEM_SEG_HEAD
    ssh.NAME = "MaskFormerHead"
    ssh.PIXEL_DECODER_NAME = "MSDeformAttnPixelDecoder"
    ssh.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    ssh.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    ssh.MASK_DIM = 256
    ssh.NUM_CLASSES = 80
    
    # Training config
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 1  # Just 1 iteration!
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.CHECKPOINT_PERIOD = 1
    
    # Gradient clipping
    if not hasattr(cfg.SOLVER, "CLIP_GRADIENTS"):
        cfg.SOLVER.CLIP_GRADIENTS = CN()
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    
    # Dataset
    cfg.DATASETS.TRAIN = ("coco_2017_train_minimal",)
    cfg.DATASETS.TEST = ()
    
    # Output
    cfg.OUTPUT_DIR = "./output_gradient_analysis"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Backbone
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.RESNETS.DEPTH = 50
    
    # Input
    cfg.INPUT.MIN_SIZE_TRAIN = (480,)
    cfg.INPUT.MAX_SIZE_TRAIN = 640
    
    cfg.freeze()
    
    return cfg


def run_one_step_with_gradient_analysis(cfg):
    """Run exactly 1 training step and analyze gradients"""
    
    print(f"\n{'='*80}")
    print("STARTING 1-STEP GRADIENT ANALYSIS")
    print(f"{'='*80}\n")
    
    # Create model
    model = MaskFormerWithMARS(cfg)
    model.train()
    
    if torch.cuda.is_available():
        print(f" Using GPU: {torch.cuda.get_device_name(0)}")
        model = model.cuda()
        if hasattr(model, '_attach_attention_hooks'):
            model._attach_attention_hooks()
    else:
        print("  Using CPU")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    
    # Create data loader
    from mask2former.data.dataset_mappers.mask_former_instance_dataset_mapper import (
        MaskFormerInstanceDatasetMapper,
    )
    mapper = MaskFormerInstanceDatasetMapper(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    
    # Initialize gradient analyzer
    analyzer = GradientAnalyzer(output_dir="gradient_analysis_output")
    
    # Register hooks before forward pass
    analyzer.register_hooks(model)
    
    # Get one batch
    print("\n Loading data batch...")
    data_iter = iter(data_loader)
    batched_inputs = next(data_iter)
    
    print(f" Loaded batch with {len(batched_inputs)} images\n")
    
    # Forward pass
    #print("  Running forward pass...")
    #loss_dict = model(batched_inputs)
    
    print("  Running forward pass...")

    from detectron2.utils.events import EventStorage
    with EventStorage(0):  # Create a dummy storage context
        loss_dict = model(batched_inputs)
        total_loss = sum(loss_dict.values())
        if "loss_mars" in loss_dict:
            mars_loss = loss_dict["loss_mars"].item()
            total_value = total_loss.item()
            ratio = mars_loss / total_value if total_value > 0 else 0
            print(f"\n📊 MARS Loss: {mars_loss:.6f} | Total Loss: {total_value:.6f} | Contribution: {ratio*100:.2f}%")
        else:
            print("\n⚠️ No MARS loss found in loss_dict!")

    print("\n Losses:")
    for k, v in loss_dict.items():
        print(f"   {k}: {v.item():.6f}")
    
    # Check if MARS loss exists
    if 'loss_mars' in loss_dict:
        mars_loss_value = loss_dict['loss_mars'].item()
        print(f"\n MARS loss found: {mars_loss_value:.6e}")
        
        if mars_loss_value == 0.0:
            print("  WARNING: MARS loss is zero!")
        
        if not loss_dict['loss_mars'].requires_grad:
            print("  WARNING: MARS loss doesn't require grad!")
    else:
        print("\n MARS loss not found in loss dict!")
    
    # Compute total loss
    total_loss = sum(loss_dict.values())
    print(f"\n   Total loss: {total_loss.item():.6f}")
    
    # Backward pass
    print("\n  Running backward pass...")
    optimizer.zero_grad()
    total_loss.backward()
    
    print(" Backward pass complete\n")
    
    # Remove hooks
    analyzer.remove_hooks()
    
    # Analyze gradients
    print(f"\n{'='*80}")
    print("ANALYZING CAPTURED GRADIENTS")
    print(f"{'='*80}\n")
    
    summary = analyzer.analyze_gradient_flow()
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    analyzer.visualize_with_matplotlib()
    analyzer.create_gradient_flow_diagram()
    analyzer.generate_report()
    
    print(f"\n{'='*80}")
    print("GRADIENT ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\n All outputs saved to: gradient_analysis_output/")
    print("\nGenerated files:")
    print("  • gradient_analysis_report.txt - Detailed text report")
    print("  • gradient_distributions.png - Multiple distribution plots")
    print("  • gradient_heatmap.png - Heatmap of gradient norms")
    print("  • gradient_flow_diagram.png - Visual flow diagram")
    print("  • gradient_flow_summary.json - JSON summary data")
    print(f"\n{'='*80}\n")
    
    return summary


def main(args):
    """Main entry point"""
    cfg = setup_minimal_config(args)
    
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Model: Mask2Former + MARS")
    print(f"Dataset: {cfg.DATASETS.TRAIN}")
    print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"MARS enabled: {cfg.MODEL.MARS.ENABLED}")
    print(f"MARS weight: {cfg.MODEL.MARS.WEIGHT}")
    print(f"MARS loss type: {cfg.MODEL.MARS.LOSS_TYPE}")
    print(f"Use GeM: {cfg.MODEL.MARS.USE_GEM}")
    print("="*80 + "\n")
    
    # Run analysis
    summary = run_one_step_with_gradient_analysis(cfg)
    
    return summary


if __name__ == "__main__":
    parser = default_argument_parser()
    #parser.add_argument("--config-file", default="", help="Optional config file")
    args = parser.parse_args()
    
    # Run on single GPU
    launch(main, num_gpus_per_machine=1, num_machines=1, machine_rank=0, 
           dist_url="auto", args=(args,))
