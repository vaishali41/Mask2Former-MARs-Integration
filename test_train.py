import sys
import os
sys.path.insert(0, "tools")
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, launch
from train_mars_corrected_gem import Trainer, setup

def main(args):
    print("🔵 Step 1: Setting up config...")
    cfg = setup(args)
    
    print("🔵 Step 2: Creating trainer...")
    trainer = Trainer(cfg)
    
    print("🔵 Step 3: Starting training...")
    print("🔵 About to call trainer.train()...")
    sys.stdout.flush()
    
    trainer.train()

if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = "configs/coco/mars_5k_50ep_corrected_gem.yaml"
    args.num_gpus = 1
    args.opts = ["SOLVER.IMS_PER_BATCH", "1", 
                 "DATALOADER.NUM_WORKERS", "0",
                 "SOLVER.MAX_ITER", "10",
                 "MODEL.MARS.ENABLED", "False"]
    
    print("🟢 Starting test...")
    launch(main, num_gpus_per_machine=1, num_machines=1, machine_rank=0, dist_url="auto", args=(args,))
