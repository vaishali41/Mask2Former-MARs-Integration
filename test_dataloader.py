from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from train_mars_corrected_gem import register_coco_subset, Trainer

# Setup
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("mars_5k_50ep_fixed_v2.yaml")
cfg.DATALOADER.NUM_WORKERS = 0
cfg.SOLVER.IMS_PER_BATCH = 1

# Register datasets
print("🔵 Registering datasets...")
register_coco_subset(train_images=100, val_images=10)  # Small subset!

# Build data loader
print("🔵 Building data loader...")
data_loader = Trainer.build_train_loader(cfg)

# Try to load one batch
print("🔵 Loading first batch...")
import sys
sys.stdout.flush()

for i, batch in enumerate(data_loader):
    print(f"✅ Loaded batch {i}: {len(batch)} images")
    if i >= 2:
        break

print("✅ Data loader works!")
