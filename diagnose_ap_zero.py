#!/usr/bin/env python3
"""
Ultimate Diagnostic: Why is AP = 0?

This script checks EVERYTHING that could cause AP=0:
1. Training completion status
2. Checkpoint validity
3. Loss values during training
4. Model architecture match
5. Learning rate issues
6. MARS loss magnitude
"""

import os
import sys
import torch
import json
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"{text:^70}")
    print("="*70)

def check_training_logs(output_dir):
    """Check if training completed successfully"""
    print_header("1. TRAINING COMPLETION CHECK")
    
    log_file = os.path.join(output_dir, "log.txt")
    
    if not os.path.exists(log_file):
        print(f"❌ No log file found at: {log_file}")
        return False
    
    print(f"✅ Log file found: {log_file}")
    
    # Read last 100 lines
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Check for completion indicators
    completed = False
    final_iter = None
    
    for line in lines[-100:]:
        if "Total training time" in line:
            completed = True
            print(f"✅ Training completed! {line.strip()}")
        if "model_final.pth" in line and "Saving" in line:
            print(f"✅ Final model saved: {line.strip()}")
        if "iter:" in line or "iteration:" in line.lower():
            # Extract iteration number
            try:
                if "iter:" in line:
                    final_iter = int(line.split("iter:")[1].split()[0].strip().split('/')[0])
            except:
                pass
    
    if final_iter:
        print(f"📊 Final iteration reached: {final_iter}")
    
    if not completed:
        print("⚠️  WARNING: No 'Total training time' found - training may not have completed!")
        print("\nLast 10 lines of log:")
        for line in lines[-10:]:
            print(f"  {line.rstrip()}")
    
    return completed


def check_loss_values(output_dir):
    """Check loss values during training"""
    print_header("2. LOSS VALUES CHECK")
    
    metrics_file = os.path.join(output_dir, "metrics.json")
    
    if not os.path.exists(metrics_file):
        print(f"⚠️  No metrics.json found at: {metrics_file}")
        print("Checking log.txt for loss values...")
        
        log_file = os.path.join(output_dir, "log.txt")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Find loss values
            losses = []
            mars_losses = []
            
            for line in lines:
                if "total_loss:" in line:
                    try:
                        loss_str = line.split("total_loss:")[1].split()[0]
                        losses.append(float(loss_str))
                    except:
                        pass
                if "loss_mars:" in line:
                    try:
                        mars_str = line.split("loss_mars:")[1].split()[0]
                        mars_losses.append(float(mars_str))
                    except:
                        pass
            
            if losses:
                print(f"\n📉 Total Loss Trajectory:")
                print(f"   First loss: {losses[0]:.4f}")
                print(f"   Last loss:  {losses[-1]:.4f}")
                print(f"   Change:     {losses[-1] - losses[0]:.4f}")
                
                if losses[-1] > losses[0]:
                    print("   ❌ PROBLEM: Loss INCREASED during training!")
                elif losses[-1] > 2.0:
                    print("   ⚠️  WARNING: Loss is still very high (>2.0)")
                elif abs(losses[-1] - losses[0]) < 0.1:
                    print("   ⚠️  WARNING: Loss barely changed - model may not be learning!")
                else:
                    print("   ✅ Loss decreased (good!)")
            
            if mars_losses:
                print(f"\n🔴 MARS Loss:")
                print(f"   Average MARS loss: {sum(mars_losses)/len(mars_losses):.6f}")
                print(f"   Last MARS loss:    {mars_losses[-1]:.6f}")
                
                avg_main_loss = losses[-1] if losses else 0
                avg_mars = mars_losses[-1] if mars_losses else 0
                
                if avg_mars > avg_main_loss * 0.5:
                    print(f"   ⚠️  WARNING: MARS loss is {(avg_mars/avg_main_loss)*100:.1f}% of main loss!")
                    print(f"   This might be too high and interfering with training")
        
        return False
    
    # Read metrics
    with open(metrics_file, 'r') as f:
        metrics = [json.loads(line) for line in f]
    
    print(f"✅ Found {len(metrics)} training iterations")
    
    # Extract losses
    total_losses = [m.get('total_loss', 0) for m in metrics if 'total_loss' in m]
    mars_losses = [m.get('loss_mars', 0) for m in metrics if 'loss_mars' in m]
    
    if total_losses:
        print(f"\n📉 Total Loss:")
        print(f"   Start: {total_losses[0]:.4f}")
        print(f"   End:   {total_losses[-1]:.4f}")
        print(f"   Change: {total_losses[-1] - total_losses[0]:.4f}")
        
        if total_losses[-1] > 5.0:
            print("   ❌ CRITICAL: Loss > 5.0 - Model didn't learn!")
        elif total_losses[-1] > 2.0:
            print("   ⚠️  WARNING: Loss > 2.0 - Training incomplete?")
    
    if mars_losses:
        print(f"\n🔴 MARS Loss:")
        print(f"   Average: {sum(mars_losses)/len(mars_losses):.6f}")
        print(f"   Last:    {mars_losses[-1]:.6f}")
    
    return True


def check_checkpoint(checkpoint_path):
    """Check checkpoint validity"""
    print_header("3. CHECKPOINT VALIDITY CHECK")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint NOT found at: {checkpoint_path}")
        return None
    
    print(f"✅ Checkpoint exists: {checkpoint_path}")
    print(f"   Size: {os.path.getsize(checkpoint_path) / (1024*1024):.1f} MB")
    
    # Load checkpoint
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Checkpoint loaded successfully")
        
        print(f"\n📦 Checkpoint contents:")
        for key in ckpt.keys():
            print(f"   - {key}")
        
        if 'iteration' in ckpt:
            print(f"\n🔢 Training iteration: {ckpt['iteration']}")
        
        if 'model' in ckpt:
            model_keys = list(ckpt['model'].keys())
            print(f"\n🏗️  Model has {len(model_keys)} parameters")
            print(f"   First 5 keys: {model_keys[:5]}")
            
            # Check if MARS is in the model
            has_mars = any('mars' in k.lower() for k in model_keys)
            has_attention = any('attention' in k.lower() for k in model_keys)
            
            print(f"\n🔍 Model components:")
            print(f"   Has 'mars' in keys: {has_mars}")
            print(f"   Has 'attention' in keys: {has_attention}")
            
            # Check parameter statistics
            param_values = []
            for k, v in ckpt['model'].items():
                if isinstance(v, torch.Tensor):
                    param_values.extend(v.flatten().tolist()[:1000])  # Sample
            
            if param_values:
                import numpy as np
                param_array = np.array(param_values)
                print(f"\n📊 Parameter statistics:")
                print(f"   Mean: {param_array.mean():.6f}")
                print(f"   Std:  {param_array.std():.6f}")
                print(f"   Min:  {param_array.min():.6f}")
                print(f"   Max:  {param_array.max():.6f}")
                
                # Check if weights are random/untrained
                if abs(param_array.mean()) < 0.01 and param_array.std() < 0.1:
                    print("   ⚠️  WARNING: Weights look like random initialization!")
        
        return ckpt
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None


def check_config(config_path):
    """Check training configuration"""
    print_header("4. CONFIGURATION CHECK")
    
    if not os.path.exists(config_path):
        print(f"⚠️  Config file not found: {config_path}")
        return
    
    print(f"✅ Config file found: {config_path}")
    
    # Try to parse as YAML
    try:
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        # Look for key settings
        for line in lines:
            line = line.strip()
            if any(keyword in line.upper() for keyword in 
                   ['BASE_LR', 'MAX_ITER', 'IMS_PER_BATCH', 'MARS']):
                print(f"   {line}")
    except Exception as e:
        print(f"⚠️  Could not parse config: {e}")


def test_model_forward():
    """Test if model can run forward pass"""
    print_header("5. MODEL FORWARD PASS TEST")
    
    try:
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from mask2former import add_maskformer2_config
        
        # Try to import MARS
        try:
            from mask2former_mars.modeling.meta_arch.mars_mask_former_head import add_mars_config, MaskFormerWithMARS
            print("✅ MARS module imported successfully")
        except ImportError as e:
            print(f"❌ Cannot import MARS module: {e}")
            return
        
        # Create minimal config
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        add_mars_config(cfg)
        
        print("✅ Config created successfully")
        
    except Exception as e:
        print(f"❌ Error setting up model: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         ULTIMATE DIAGNOSTIC: WHY IS AP = 0?                   ║
    ║                                                               ║
    ║  This script will check EVERYTHING that could cause AP=0     ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Get paths from user or use defaults
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = input("Enter output directory path (or press Enter for './output'): ").strip()
        if not output_dir:
            output_dir = "./output"
    
    print(f"\n🔍 Analyzing: {output_dir}")
    
    # Run all checks
    training_completed = check_training_logs(output_dir)
    loss_ok = check_loss_values(output_dir)
    
    # Check for checkpoint
    checkpoint_paths = [
        os.path.join(output_dir, "model_final.pth"),
        os.path.join(output_dir, "model_0004999.pth"),
        os.path.join(output_dir, "model_best.pth"),
    ]
    
    checkpoint = None
    for ckpt_path in checkpoint_paths:
        if os.path.exists(ckpt_path):
            checkpoint = check_checkpoint(ckpt_path)
            break
    
    if checkpoint is None:
        print("\n❌ NO CHECKPOINT FOUND!")
        print(f"   Searched in: {output_dir}")
        print(f"   Looking for: model_final.pth, model_0004999.pth, model_best.pth")
    
    # Check config
    config_paths = [
        "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
        os.path.join(output_dir, "config.yaml"),
    ]
    
    for cfg_path in config_paths:
        if os.path.exists(cfg_path):
            check_config(cfg_path)
            break
    
    # Test model
    test_model_forward()
    
    # DIAGNOSIS
    print_header("🔬 DIAGNOSIS")
    
    issues_found = []
    
    if not training_completed:
        issues_found.append("Training did not complete")
    
    if checkpoint is None:
        issues_found.append("No checkpoint found")
    
    if not loss_ok:
        issues_found.append("Cannot verify loss values")
    
    if issues_found:
        print("\n❌ PROBLEMS FOUND:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        
        print("\n💡 RECOMMENDED ACTIONS:")
        
        if "Training did not complete" in issues_found:
            print("   1. Re-run training and wait for completion")
            print("   2. Check for errors in training logs")
            print("   3. Ensure sufficient GPU memory")
        
        if "No checkpoint found" in issues_found:
            print("   1. Training may have crashed before saving")
            print("   2. Check OUTPUT_DIR in your training command")
            print("   3. Look for checkpoints in other directories")
        
        if "Cannot verify loss values" in issues_found:
            print("   1. Training may not have started properly")
            print("   2. Check if GPU is available")
            print("   3. Verify dataset paths")
    else:
        print("\n✅ All basic checks passed")
        print("\n❓ BUT AP is still 0?")
        print("\n   Possible causes:")
        print("   1. Using WRONG checkpoint for evaluation")
        print("      → Check: Are you using pretrained weights instead of trained model?")
        print("   2. Model architecture mismatch")
        print("      → Check: Does eval use same MaskFormerWithMARS class?")
        print("   3. MARS weight too high, overwhelming main loss")
        print("      → Check: Is MARS loss > 50% of total loss?")
        print("   4. Training iterations too few (didn't converge)")
        print("      → Check: Did you train for full 5000 iterations?")
    
    print("\n" + "="*70)
    print("💬 NEXT STEPS:")
    print("="*70)
    print("""
Please share:
1. Last 50 lines of training log:
   tail -50 output/log.txt

2. What checkpoint you used for evaluation:
   ls -lh output/model*.pth

3. Your exact evaluation command:
   python eval_... --model-path ???

This will help pinpoint the exact issue!
""")


if __name__ == "__main__":
    main()
