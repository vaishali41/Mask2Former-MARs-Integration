# Quick start script for Mask2Former + MARs 10-class training
# Usage: ./quick_start.sh [train|resume|eval|eigencam]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="mars_10class_200ep_gem.yaml"
TRAIN_SCRIPT="train_mars_10class_gem.py"
EIGENCAM_SCRIPT="eigencam_evaluation.py"
OUTPUT_DIR="output_mars_10class_200ep_gem_detach"
COCO_ROOT="datasets/coco"

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if dataset exists
check_dataset() {
    print_info "Checking COCO dataset..."
    
    if [ ! -d "$COCO_ROOT" ]; then
        print_error "COCO dataset not found at $COCO_ROOT"
        echo "Please download COCO dataset first:"
        echo "  1. Create directory: mkdir -p $COCO_ROOT"
        echo "  2. Download and extract COCO 2017 dataset"
        echo "  3. See README_10class_training.md for detailed instructions"
        exit 1
    fi
    
    if [ ! -f "$COCO_ROOT/annotations/instances_train2017.json" ]; then
        print_error "COCO annotations not found!"
        exit 1
    fi
    
    if [ ! -d "$COCO_ROOT/train2017" ]; then
        print_error "COCO train2017 images not found!"
        exit 1
    fi
    
    print_info "Dataset check passed"
}

# Function to check dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    python3 -c "import torch" 2>/dev/null || {
        print_error "PyTorch not installed! Install with: pip install torch torchvision"
        exit 1
    }
    
    python3 -c "import detectron2" 2>/dev/null || {
        print_error "Detectron2 not installed! See README for installation instructions"
        exit 1
    }
    
    python3 -c "import mask2former" 2>/dev/null || {
        print_error "Mask2Former not installed! Install with: pip install git+https://github.com/facebookresearch/Mask2Former.git"
        exit 1
    }
    
    print_info "✓ All dependencies installed"
}

# Function to train model
train_model() {
    print_info "Starting training..."
    print_info "Config: $CONFIG_FILE"
    print_info "Output: $OUTPUT_DIR"
    
    # Check if user wants to use wandb
    read -p "Enable Weights & Biases logging? (y/n): " -n 1 -r
    echo
    
    WANDB_FLAG=""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        WANDB_FLAG="--use_wandb"
        print_info "W&B logging enabled"
    fi
    
    # Start training
    python3 $TRAIN_SCRIPT \
        --config-file $CONFIG_FILE \
        --num-gpus 1 \
        $WANDB_FLAG
    
    print_info " Training complete!"
}

# Function to resume training
resume_training() {
    print_info "Resuming training from checkpoint..."
    
    if [ ! -d "$OUTPUT_DIR" ]; then
        print_error "No checkpoint found at $OUTPUT_DIR"
        print_info "Starting fresh training instead..."
        train_model
        return
    fi
    
    python3 $TRAIN_SCRIPT \
        --config-file $CONFIG_FILE \
        --num-gpus 1 \
        --resume
    
    print_info "Training resumed and complete!"
}

# Function to evaluate model
evaluate_model() {
    print_info "Evaluating model..."
    
    if [ ! -f "$OUTPUT_DIR/model_final.pth" ]; then
        print_error "No trained model found at $OUTPUT_DIR/model_final.pth"
        exit 1
    fi
    
    python3 $TRAIN_SCRIPT \
        --config-file $CONFIG_FILE \
        --eval-only \
        MODEL.WEIGHTS $OUTPUT_DIR/model_final.pth
    
    print_info " Evaluation complete!"
}

# Function to generate EigenCAM visualizations
generate_eigencam() {
    print_info "Generating EigenCAM visualizations..."
    
    if [ ! -f "$OUTPUT_DIR/model_final.pth" ]; then
        print_error "No trained model found at $OUTPUT_DIR/model_final.pth"
        exit 1
    fi
    
    # Ask for number of images
    read -p "Number of images to process (default: 50): " NUM_IMAGES
    NUM_IMAGES=${NUM_IMAGES:-50}
    
    # Create output directory
    EIGENCAM_OUTPUT="eigencam_results"
    mkdir -p $EIGENCAM_OUTPUT
    
    python3 $EIGENCAM_SCRIPT \
        --config $CONFIG_FILE \
        --weights $OUTPUT_DIR/model_final.pth \
        --input $COCO_ROOT/val2017 \
        --output $EIGENCAM_OUTPUT \
        --num-images $NUM_IMAGES
    
    print_info "✓ EigenCAM generation complete!"
    print_info "Results saved to: $EIGENCAM_OUTPUT"
}

# Function to show training status
show_status() {
    print_info "Training Status"
    echo "================================"
    
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Output directory: $OUTPUT_DIR"
        
        if [ -f "$OUTPUT_DIR/model_final.pth" ]; then
            echo "Status: ✓ Training complete"
            
            # Show model size
            SIZE=$(du -h "$OUTPUT_DIR/model_final.pth" | cut -f1)
            echo "Final model size: $SIZE"
        else
            # Count checkpoints
            NUM_CHECKPOINTS=$(ls -1 $OUTPUT_DIR/model_*.pth 2>/dev/null | wc -l)
            echo "Status: Training in progress"
            echo "Checkpoints saved: $NUM_CHECKPOINTS"
            
            if [ -f "$OUTPUT_DIR/log.txt" ]; then
                echo "Last log entry:"
                tail -n 1 $OUTPUT_DIR/log.txt
            fi
        fi
        
        # Show disk usage
        DISK_USAGE=$(du -sh $OUTPUT_DIR | cut -f1)
        echo "Disk usage: $DISK_USAGE"
    else
        echo "Status: Not started"
        echo "Run './quick_start.sh train' to begin training"
    fi
    echo "================================"
}

# Main script
main() {
    echo "================================================"
    echo "  Mask2Former + MARs: 10-Class Training"
    echo "================================================"
    echo ""
    
    # Parse command
    COMMAND=${1:-help}
    
    case $COMMAND in
        train)
            check_dependencies
            check_dataset
            train_model
            ;;
        resume)
            check_dependencies
            check_dataset
            resume_training
            ;;
        eval)
            check_dependencies
            evaluate_model
            ;;
        eigencam)
            check_dependencies
            generate_eigencam
            ;;
        status)
            show_status
            ;;
        help|*)
            echo "Usage: ./quick_start.sh [command]"
            echo ""
            echo "Commands:"
            echo "  train      - Start fresh training (200 epochs)"
            echo "  resume     - Resume training from checkpoint"
            echo "  eval       - Evaluate trained model"
            echo "  eigencam   - Generate EigenCAM visualizations"
            echo "  status     - Show training status"
            echo "  help       - Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./quick_start.sh train        # Start training"
            echo "  ./quick_start.sh resume       # Continue training"
            echo "  ./quick_start.sh eigencam     # Generate attention maps"
            echo ""
            echo "For more details, see README_10class_training.md"
            ;;
    esac
}

# Run main
main "$@"
