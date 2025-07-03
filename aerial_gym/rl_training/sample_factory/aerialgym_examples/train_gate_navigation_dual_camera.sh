#!/bin/bash

# Train Gate Navigation with Dual Camera (Drone + Static) and GPU Monitoring
# This script runs gate navigation training with memory-optimized dual camera system
#
# Usage:
#   ./train_gate_navigation_dual_camera.sh [EXPERIMENT_NAME] [--view] [--gifs]
#
# EXPERIMENT_NAME: Optional custom experiment name
#   If not provided, auto-generates based on timestamp
#
# --view: Optional flag to enable visualization (slower training)
#   If not provided, runs in headless mode for maximum performance
#
# --gifs: Optional flag to save episode GIFs (drone + static camera)
#   If not provided, no GIF saving (faster training)
#
# Examples:
#   ./train_gate_navigation_dual_camera.sh                           # Headless training
#   ./train_gate_navigation_dual_camera.sh --view                    # Training with visualization  
#   ./train_gate_navigation_dual_camera.sh --gifs                    # Headless training with GIF saving
#   ./train_gate_navigation_dual_camera.sh --view --gifs             # Training with visualization and GIF saving
#   ./train_gate_navigation_dual_camera.sh my_experiment             # Headless with custom name
#   ./train_gate_navigation_dual_camera.sh my_experiment --view      # Viewing with custom name
#   ./train_gate_navigation_dual_camera.sh my_experiment --gifs      # Headless with custom name and GIF saving

set -e  # Exit on any error

# Parse arguments
EXPERIMENT_NAME=""
ENABLE_VIEWER=false
ENABLE_GIFS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Usage: $0 [EXPERIMENT_NAME] [--view] [--gifs]"
            echo ""
            echo "This script trains gate navigation with dual cameras:"
            echo "  - Environment: gate_env"
            echo "  - Robot: X500 with D455 camera"
            echo "  - Static camera: D455 behind gate (270x480 resolution)"
            echo "  - Observation space: 145D (17D basic + 64D drone VAE + 64D static camera VAE)"
            echo "  - Action space: 3D (forward, lateral, yaw_rate)"
            echo "  - Memory optimization: Shared VAE model reduces GPU usage by ~50%"
            echo ""
            echo "Arguments:"
            echo "  EXPERIMENT_NAME: Optional custom experiment name"
            echo "  --view:          Enable visualization (slower but visual feedback)"
            echo "  --gifs:          Save episode GIFs for both drone and static cameras"
            echo ""
            echo "Examples:"
            echo "  $0                           # Headless training"
            echo "  $0 --view                    # Training with visualization"
            echo "  $0 --gifs                    # Headless training with GIF saving"
            echo "  $0 --view --gifs             # Training with visualization and GIF saving"
            echo "  $0 my_experiment             # Headless with custom name"
            echo "  $0 my_experiment --view      # Viewing with custom name"
            echo "  $0 my_experiment --gifs      # Headless with custom name and GIF saving"
            exit 0
            ;;
        --view)
            ENABLE_VIEWER=true
            shift
            ;;
        --gifs)
            ENABLE_GIFS=true
            shift
            ;;
        *)
            if [ -z "$EXPERIMENT_NAME" ]; then
                EXPERIMENT_NAME="$1"
            else
                echo "Error: Unknown argument $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Gate Navigation Training Configuration (16 environments to match standard config)
CONFIG_NAME="Gate Navigation Dual Camera Configuration (16 environments)"
ENV_AGENTS=16
BATCH_SIZE=2048
NUM_BATCHES_TO_ACCUMULATE=2
CONFIG_PREFIX="gate_nav_dual_cam"

# Set experiment name - use provided name or auto-generate
if [ -n "$EXPERIMENT_NAME" ]; then
    echo "Using custom experiment name: $EXPERIMENT_NAME"
else
    EXPERIMENT_NAME="${CONFIG_PREFIX}_$(date +%Y%m%d_%H%M%S)"
    echo "Auto-generated experiment name: $EXPERIMENT_NAME"
fi

TRAIN_STEPS=100000000
GPU_MONITOR_INTERVAL=10
GPU_LOG_FILE="../../../logs/gpu_usage_gate_nav_${EXPERIMENT_NAME}.csv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Gate Navigation Dual Camera Training with GPU Monitoring ===${NC}"
echo -e "${YELLOW}Environment: gate_env${NC}"
echo -e "${YELLOW}Robot: X500 with D455 camera${NC}"
echo -e "${YELLOW}Static camera: D455 behind gate (270x480 resolution)${NC}"
echo -e "${YELLOW}Observation space: 145D (17D basic + 64D drone VAE + 64D static camera VAE)${NC}"
echo -e "${GREEN}⚡ MEMORY OPTIMIZATION: Shared VAE model reduces GPU usage by ~50%${NC}"
echo -e "${YELLOW}Action space: 3D (forward, lateral, yaw_rate)${NC}"
echo -e "${GREEN}Environments: ${ENV_AGENTS} (standard configuration)${NC}"
if [ "$ENABLE_VIEWER" = true ]; then
    echo -e "${GREEN}🖥️  Viewer: ENABLED (slower training, visual feedback)${NC}"
else
    echo -e "${YELLOW}⚡ Viewer: DISABLED (maximum performance, prevents Isaac Gym conflicts)${NC}"
fi

if [ "$ENABLE_GIFS" = true ]; then
    echo -e "${GREEN}📹 GIF Saving: ENABLED (drone + static camera episodes saved as GIFs)${NC}"
else
    echo -e "${YELLOW}⚡ GIF Saving: DISABLED (faster training, no GIF generation)${NC}"
fi
echo ""
echo -e "${YELLOW}Using fresh experiment name: ${EXPERIMENT_NAME}${NC}"
echo -e "${YELLOW}This ensures no configuration conflicts with previous runs${NC}"
echo ""
echo -e "${YELLOW}GPU Monitoring Interval: ${GPU_MONITOR_INTERVAL}s${NC}"
echo -e "${YELLOW}GPU Log File: ${GPU_LOG_FILE}${NC}"

# Create logs directory
mkdir -p ../../../logs

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}Cleaning up background processes...${NC}"
    if [[ ! -z $GPU_MONITOR_PID ]]; then
        kill $GPU_MONITOR_PID 2>/dev/null || true
        wait $GPU_MONITOR_PID 2>/dev/null || true
        echo -e "${GREEN}GPU monitoring stopped${NC}"
    fi
}

# Set trap to cleanup on script exit
trap cleanup EXIT

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. GPU monitoring will not work.${NC}"
    exit 1
fi

# Start GPU monitoring in background
echo -e "${GREEN}Starting GPU monitoring...${NC}"
python ../../../logs/monitor_gpu.py --interval $GPU_MONITOR_INTERVAL --output "$GPU_LOG_FILE" &
GPU_MONITOR_PID=$!

# Give GPU monitor time to start
sleep 2

# Check if GPU monitor is running
if ! kill -0 $GPU_MONITOR_PID 2>/dev/null; then
    echo -e "${RED}Error: Failed to start GPU monitoring${NC}"
    exit 1
fi

echo -e "${GREEN}GPU monitoring started (PID: $GPU_MONITOR_PID)${NC}"
echo -e "${BLUE}Monitor GPU usage in real-time: ${NC}watch -n 1 nvidia-smi"
echo ""
echo "================================================================================"
echo ""

# Display current GPU status
python -c "
import subprocess
import time

def show_gpu_status():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        print(f'[{time.strftime(\"%H:%M:%S\")}] GPU Status:')
        print('-' * 80)
        for i, line in enumerate(lines):
            name, util, mem_used, mem_total, temp, power = line.split(', ')
            mem_used_mb = int(mem_used)
            mem_total_mb = int(mem_total)
            mem_used_gb = mem_used_mb / 1024
            mem_total_gb = mem_total_mb / 1024
            mem_percent = (mem_used_mb / mem_total_mb) * 100
            
            print(f'GPU {i} ({name}):')
            print(f'  VRAM: {mem_used_mb}MB/{mem_total_gb:.1f}GB ({mem_percent:.1f}%)')
            print(f'  Utilization: {util}%')
            print(f'  Temperature: {temp}°C')
            print(f'  Power: {power}W')
            
            # Visual VRAM bar
            bar_length = 40
            filled = int((mem_percent / 100) * bar_length)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f'  VRAM: [{bar}] {mem_percent:.1f}%')
    except:
        print('Could not get GPU status')

show_gpu_status()
"

# Clear existing training directory for fresh start
echo -e "${GREEN}✓ Cleared any existing training directory for fresh start${NC}"
rm -rf ./train_dir

# Clear GPU cache before training
echo -e "${YELLOW}Clearing GPU cache...${NC}"
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Export environment variables for Sample Factory training
export SF_ENV_AGENTS=${ENV_AGENTS}
echo -e "${GREEN}Set SF_ENV_AGENTS=${ENV_AGENTS} environment variable for all processes (STANDARD CONFIG)${NC}"

# Export headless setting for both main process and worker processes
if [ "$ENABLE_VIEWER" = true ]; then
    export SF_HEADLESS=false
    echo -e "${GREEN}Set SF_HEADLESS=false environment variable for viewer mode${NC}"
else
    export SF_HEADLESS=true
    echo -e "${YELLOW}Set SF_HEADLESS=true environment variable for headless mode${NC}"
fi

echo -e "${GREEN}Starting Gate Navigation training...${NC}"
echo -e "${GREEN}CRITICAL: Using shared VAE model to prevent GPU memory overflow${NC}"
if [ "$ENABLE_VIEWER" = true ]; then
    echo -e "${GREEN}Training with visualization enabled${NC}"
else
    echo -e "${YELLOW}Training in headless mode for maximum performance and Isaac Gym compatibility${NC}"
fi

# Build training command with proper headless parameter
TRAIN_CMD="python train_aerialgym_custom_net_gate.py \
    --env=quad_with_obstacles_gate \
    --experiment=\"${EXPERIMENT_NAME}\" \
    --train_dir=./train_dir \
    --num_workers=1 \
    --num_envs_per_worker=1 \
    --env_agents=${ENV_AGENTS} \
    --obs_key=\"observations\" \
    --batch_size=${BATCH_SIZE} \
    --num_batches_to_accumulate=${NUM_BATCHES_TO_ACCUMULATE} \
    --num_batches_per_epoch=8 \
    --num_epochs=4 \
    --rollout=32 \
    --learning_rate=0.0003 \
    --use_rnn=true \
    --rnn_size=64 \
    --rnn_num_layers=1 \
    --encoder_mlp_layers 512 256 64 \
    --gamma=0.98 \
    --reward_scale=0.1 \
    --max_grad_norm=1.0 \
    --async_rl=true \
    --normalize_input=true \
    --use_env_info_cache=false \
    --with_wandb=true \
    --wandb_project=\"gate_navigation_dual_camera\" \
    --wandb_user=\"ziya-ruso-ucl\" \
    --wandb_group=\"gate_navigation_training\" \
    --wandb_tags \"aerial_gym\" \"gate_navigation\" \"dual_camera\" \"x500\" \"sample_factory\" \"memory_optimized\" \
    --save_every_sec=120 \
    --save_best_every_sec=5 \
    --train_for_env_steps=${TRAIN_STEPS}"

# Add headless parameter based on viewer preference
if [ "$ENABLE_VIEWER" = false ]; then
    TRAIN_CMD="$TRAIN_CMD --headless=true"
else
    TRAIN_CMD="$TRAIN_CMD --headless=false"
fi

# Add GIF saving parameter if requested
if [ "$ENABLE_GIFS" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --save_gifs=true"
fi

echo -e "${YELLOW}Training command:${NC}"
echo "$TRAIN_CMD"
echo ""

# Run training with error handling
if eval $TRAIN_CMD; then
    echo -e "\n${GREEN}✓ Training completed successfully!${NC}"
    echo -e "${BLUE}GPU usage log saved to: $GPU_LOG_FILE${NC}"
    echo -e "${GREEN}Check wandb for training metrics and curves${NC}"
else
    echo -e "\n${RED}❌ Training failed or was interrupted${NC}"
    echo -e "${BLUE}GPU usage log saved to: $GPU_LOG_FILE${NC}"
    echo -e "${RED}Check the log for CUDA memory issues - optimization may need further tuning${NC}"
    exit 1
fi

echo -e "\n${GREEN}All processes completed successfully${NC}"