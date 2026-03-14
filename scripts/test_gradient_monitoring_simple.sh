#!/bin/bash

# Simple gradient monitoring test with fixed configurations
cd aerial_gym/rl_training/sample_factory/aerialgym_examples

echo "🧪 Simple Gradient Monitoring Test"
echo "Testing with corrected configurations..."

# Clear all caches first
rm -rf /tmp/sf2_*

/home/ziyar/miniforge3/envs/aerialgym/bin/python train_aerialgym_custom_net_gate.py \
    --env=quad_with_obstacles_gate \
    --experiment=simple_gradient_test \
    --train_dir=./train_dir \
    --num_workers=1 \
    --num_envs_per_worker=1 \
    --env_agents=16 \
    --obs_key=observations \
    --batch_size=256 \
    --num_batches_to_accumulate=1 \
    --num_batches_per_epoch=2 \
    --num_epochs=2 \
    --rollout=32 \
    --recurrence=32 \
    --learning_rate=0.0003 \
    --use_rnn=true \
    --rnn_size=64 \
    --rnn_num_layers=1 \
    --encoder_mlp_layers 128 64 \
    --gamma=0.98 \
    --reward_scale=0.1 \
    --max_grad_norm=1.0 \
    --normalize_input=true \
    --use_env_info_cache=false \
    --save_every_sec=3600 \
    --save_best_every_sec=3600 \
    --train_for_env_steps=2000 \
    --headless=true \
    --enable_gradient_monitoring=true \
    --gradient_log_interval=50 \
    --gradient_print_interval=50

echo "🏁 Simple test completed" 