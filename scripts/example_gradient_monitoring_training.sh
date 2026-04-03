#!/bin/bash

# Example training script with gradient monitoring enabled
# This demonstrates how to use the gradient flow analysis feature

echo "🔬 Starting Gate Navigation Training with Gradient Monitoring"
echo "================================================================"

# Configuration
EXPERIMENT_NAME="gate_nav_gradient_test_$(date +%Y%m%d_%H%M%S)"
ENV_AGENTS=16
TRAIN_STEPS=10000000

echo "📋 Configuration:"
echo "   Experiment: $EXPERIMENT_NAME"
echo "   Environments: $ENV_AGENTS"
echo "   Training steps: $TRAIN_STEPS"
echo "   Gradient monitoring: ENABLED"
echo ""

# Change to the training directory
cd aerial_gym/rl_training/sample_factory/aerialgym_examples

# Run training with gradient monitoring enabled
python train_aerialgym_custom_net_gate.py \
    --env=quad_with_obstacles_gate \
    --experiment="$EXPERIMENT_NAME" \
    --train_dir=./train_dir \
    --num_workers=1 \
    --num_envs_per_worker=1 \
    --env_agents=$ENV_AGENTS \
    --obs_key="observations" \
    --batch_size=2048 \
    --num_batches_to_accumulate=2 \
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
    --wandb_project="gradient_monitoring_test" \
    --wandb_user="your_username" \
    --wandb_group="gradient_analysis" \
    --train_for_env_steps=$TRAIN_STEPS \
    --headless=true \
    \
    --enable_gradient_monitoring=true \
    --gradient_log_interval=100 \
    --gradient_print_interval=1000

echo "✅ Training completed!"
echo ""
echo "📊 What to look for in the logs:"
echo "   🔍 'Gradient monitor successfully attached to model' - confirms monitoring is active"
echo "   📈 Periodic gradient analysis summaries every 1000 steps"
echo "   📋 Final gradient flow analysis at end of training"
echo "   🌐 Gradient metrics in wandb dashboard (if enabled)"
echo ""
echo "📖 Key metrics to monitor:"
echo "   • gradients/static_camera/mean_abs - absolute gradient magnitude for static camera"
echo "   • gradients/drone_camera/mean_abs - absolute gradient magnitude for drone camera"
echo "   • importance/static_to_drone_camera - ratio indicating relative usage"
echo "   • gradients/static_camera_active - binary indicator if static camera is being used"
echo ""
echo "🎯 Interpretation:"
echo "   ✅ Ratio > 0.1: Static camera is being actively used by the network"
echo "   ⚠️  Ratio 0.01-0.1: Static camera has limited usage"
echo "   ❌ Ratio < 0.01: Static camera is likely ignored by the network" 