#!/bin/bash

# Train DCE Navigation with GPU Monitoring (Optimized Configuration)
# This script runs training with 16 environments and GPU monitoring in parallel
#
# Usage:
#   ./train_with_monitoring.sh [EXPERIMENT_NAME] [--view]
#
# EXPERIMENT_NAME: Optional custom experiment name
#   If not provided, auto-generates based on timestamp
#
# --view: Optional flag to enable visualization (slower training)
#   If not provided, runs in headless mode for maximum performance
#
# Examples:
#   ./train_with_monitoring.sh                           # Headless training
#   ./train_with_monitoring.sh --view                    # Training with visualization  
#   ./train_with_monitoring.sh my_experiment             # Headless with custom name
#   ./train_with_monitoring.sh my_experiment --view      # Viewing with custom name

set -e  # Exit on any error

# Parse arguments
EXPERIMENT_NAME=""
ENABLE_VIEWER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Usage: $0 [EXPERIMENT_NAME] [--view]"
            echo ""
            echo "This script uses the ORIGINAL DCE configuration:"
            echo "  - 16 parallel environments"
            echo "  - 2048 batch size"  
            echo "  - Maximum performance (requires high VRAM)"
            echo ""
            echo "Arguments:"
            echo "  EXPERIMENT_NAME: Optional custom experiment name"
            echo "  --view:          Enable visualization (slower but visual feedback)"
            echo ""
            echo "Examples:"
            echo "  $0                           # Headless training"
            echo "  $0 --view                    # Training with visualization"
            echo "  $0 my_experiment             # Headless with custom name"
            echo "  $0 my_experiment --view      # Viewing with custom name"
            exit 0
            ;;
        --view)
            ENABLE_VIEWER=true
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

# Original DCE Configuration (16 environments, maximum performance)
CONFIG_NAME="Original DCE Configuration (16 environments, 2048 batch size)"
ENV_AGENTS=16
BATCH_SIZE=2048
CONFIG_PREFIX="original_dce_config"

# Set experiment name - use provided name or auto-generate
if [ -n "$EXPERIMENT_NAME" ]; then
    echo "Using custom experiment name: $EXPERIMENT_NAME"
else
    EXPERIMENT_NAME="${CONFIG_PREFIX}_$(date +%Y%m%d_%H%M%S)"
    echo "Auto-generated experiment name: $EXPERIMENT_NAME"
fi

echo "Configuration: $CONFIG_NAME"
if [ "$ENABLE_VIEWER" = true ]; then
    echo "🖥️  Viewer: ENABLED (slower training, visual feedback)"
else
    echo "⚡ Viewer: DISABLED (maximum performance)"
fi

TRAIN_STEPS=100000000
GPU_MONITOR_INTERVAL=10
GPU_LOG_FILE="logs/gpu_usage_${EXPERIMENT_NAME}.csv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== DCE Navigation Training with GPU Monitoring ===${NC}"
echo -e "${YELLOW}Experiment: ${EXPERIMENT_NAME}${NC}"
echo -e "${YELLOW}Configuration: ${CONFIG_NAME}${NC}"
if [ "$ENABLE_VIEWER" = true ]; then
    echo -e "${GREEN}Visualization: ENABLED${NC}"
else
    echo -e "${YELLOW}Visualization: DISABLED (use --view to enable)${NC}"
fi
echo -e "${YELLOW}GPU Monitoring Interval: ${GPU_MONITOR_INTERVAL}s${NC}"
echo -e "${YELLOW}GPU Log File: ${GPU_LOG_FILE}${NC}"
echo ""

# Create logs directory
mkdir -p logs

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
python monitor_gpu.py --interval $GPU_MONITOR_INTERVAL --output "$GPU_LOG_FILE" &
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

# Clear GPU cache before training
echo -e "${YELLOW}Clearing GPU cache...${NC}"
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Start training
echo -e "${GREEN}Starting DCE Navigation training...${NC}"

# Build training command with conditional headless parameter
TRAIN_CMD="python train_aerialgym_custom_net.py --env=quad_with_obstacles --train_for_env_steps=$TRAIN_STEPS --experiment=$EXPERIMENT_NAME --async_rl=True --use_env_info_cache=False --normalize_input=True"

# Add headless parameter based on viewer preference
if [ "$ENABLE_VIEWER" = false ]; then
    TRAIN_CMD="$TRAIN_CMD --headless=True"
    echo -e "${YELLOW}Training in headless mode for maximum performance${NC}"
else
    TRAIN_CMD="$TRAIN_CMD --headless=False"
    echo -e "${GREEN}Training with visualization enabled${NC}"
fi

echo -e "${YELLOW}Training command:${NC}"
echo "$TRAIN_CMD"
echo ""

# Export environment variables for the training process
export SF_ENV_AGENTS=${ENV_AGENTS}
echo "Set SF_ENV_AGENTS=${ENV_AGENTS} environment variable for all processes (ORIGINAL DCE CONFIG)"

# Run training with error handling
if $TRAIN_CMD; then
    
    echo -e "\n${GREEN}Training completed successfully!${NC}"
    echo -e "${BLUE}GPU usage log saved to: $GPU_LOG_FILE${NC}"
else
    echo -e "\n${RED}Training failed or was interrupted${NC}"
    echo -e "${BLUE}GPU usage log saved to: $GPU_LOG_FILE${NC}"
    exit 1
fi

echo -e "\n${GREEN}All processes completed${NC}" 
 