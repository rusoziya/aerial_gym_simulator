#!/bin/bash

# Quick test script for COMPLETE OBSERVATION INFLUENCE ANALYSIS
# Tests all 150D observation components for neural network influence

echo "🔬 Starting COMPLETE OBSERVATION INFLUENCE ANALYSIS test..."
echo "📊 Will analyze ALL observation components: position, orientation, velocities, actions, cameras"

# Clear any existing cache thoroughly
echo "🧹 Clearing Sample Factory cache..."
rm -rf /tmp/sf2_*
rm -rf ~/.cache/sample_factory_cache*
rm -rf ./train_dir/*/cache*

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/home/ziyar/aerialgym/aerialgym_ws/src/aerial_gym_simulator"

# Override environment variable to ensure 16 agents (not default 128)
export SF_ENV_AGENTS=16
echo "🔧 Set SF_ENV_AGENTS=16 to override default"

# Run with complete observation analysis enabled
# Use 16 environments for testing (not the default 128)
# Run for more steps to see step counter progression
/home/ziyar/miniforge3/envs/aerialgym/bin/python /home/ziyar/aerialgym/aerialgym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/sample_factory/aerialgym_examples/train_aerialgym_custom_net_gate.py \
    --env=quad_with_obstacles_gate \
    --train_for_env_steps=5000 \
    --enable_gradient_monitoring=True \
    --gradient_log_interval=25 \
    --gradient_print_interval=25 \
    --env_agents=16 \
    --recurrence=32 \
    --headless=True \
    --use_env_info_cache=False

echo "🎯 Complete observation analysis test completed!" 