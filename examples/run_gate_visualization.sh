#!/bin/bash

# Set up environment for Isaac Gym
export LD_LIBRARY_PATH=/home/ziyar/miniforge3/envs/aerialgym/lib:$LD_LIBRARY_PATH

# Add Isaac Gym to Python path
export PYTHONPATH=/home/ziyar/aerialgym/IsaacGym_Preview_4_Package/isaacgym/python:$PYTHONPATH

# Change to the examples directory
cd "$(dirname "$0")"

echo "Running Gate Environment Visualization..."
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "Current directory: $(pwd)"

# Activate conda environment if not already active
if [[ "$CONDA_DEFAULT_ENV" != "aerialgym" ]]; then
    echo "Activating conda environment..."
    source /home/ziyar/miniforge3/etc/profile.d/conda.sh
    conda activate aerialgym
fi

# Run the visualization script
python3 simple_gate_visualization.py 