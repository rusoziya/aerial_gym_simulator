#!/usr/bin/env bash

# Run inference for DRONE-ONLY inputs (ablated proprio 0:22 and static latents 86:150)
# Levels: 3, 13, 23, 33
# Seeds: five fixed seeds reused across all levels

set -euo pipefail

# Directory constants
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR="$SCRIPT_DIR/../../.."

# Checkpoint path (do not change unless explicitly edited)
export DCE_MODEL="/home/ziyar/aerialgym/aeriabgym_ws/src/aerial_gym_simulator/aerial_gym/examples/dce_rl_navigation/selected_network/checkpoint_p0/checkpoint_000003968_2031616.pth"

# Levels and seeds
LEVELS=(3 13 23 33)
SEEDS=(123 231 321 456 789)

# Common environment variables for all runs
export VISIBILITY_DEBUG=true
export PRINT_ENV0_LATENTS_ONCE_NORM=false
export PRINT_ENV0_LATENTS_ONCE=false
export DISABLE_NORM=false
export NORM_TAP_DEBUG=false
export ENC_TAP_DEBUG=false
export EVAL_STRETCH_ENABLED=1
export EVAL_STRETCH_END_LEVEL=33
export WANDB_PROJECT=final_inference_indiviual_policies_2
export WANDB_DISABLED=false
export WANDB_MODE=online
export ABLATE_DEBUG=true
export ABLATE_ZERO_RNN=false
# DRONE-ONLY ablation: zero proprio (0:22) and static camera latents (86:150)
export ABLATE_OBS_RANGES='0:22=zero,86:150=zero'
export CUBLAS_WORKSPACE_CONFIG=:16:8

# Paths
ENV_NAME=dce_navigation_task_gate
TRAIN_DIR="/home/ziyar/aerialgym/aerialgym_ws/src/aerial_gym_simulator/aerial_gym/examples/dce_rl_navigation"
EXPERIMENT=selected_network

# Inference options
ENV_AGENTS=128
MAX_EPISODES=512
SAVE_GIFS=false
HEADLESS=false

echo "Starting DRONE-ONLY inference runs (proprio + static latents ablated)..."

for SEED in "${SEEDS[@]}"; do
  for LVL in "${LEVELS[@]}"; do
    RUN_NAME="drone_only_L${LVL}_SEED${SEED}"
    export WANDB_RUN_NAME="${RUN_NAME}"
    echo "\n=== Running ${RUN_NAME} ==="

    python -B -m aerial_gym.examples.dce_rl_navigation.dce_nn_navigation_gate \
      --rnn_warmup_steps=50 \
      --env ${ENV_NAME} \
      --train_dir=${TRAIN_DIR} \
      --experiment=${EXPERIMENT} \
      --seed=${SEED} \
      --env_agents=${ENV_AGENTS} \
      --max_num_episodes=${MAX_EPISODES} \
      --eval_deterministic=true \
      --save_gifs=${SAVE_GIFS} \
      --headless=${HEADLESS} \
      --disable_curriculum_multiplier=true \
      --disable_gate_size_randomization=false \
      --fixed_gate_scale_percent=100 \
      --disable_obstacle_randomization=false \
      --fixed_obstacles_behind_gate=1 \
      --disable_static_camera_orientation_randomization=true \
      --disable_drone_camera_noise_randomization=false \
      --disable_static_camera_noise_randomization=false \
      --disable_drone_camera_frame_dropout=false \
      --disable_static_camera_frame_dropout=false \
      --disable_state_noise_randomization=false \
      --force_curriculum_level=${LVL} \
      --disable_spawn_position_randomization=false \
      --disable_spawn_orientation_randomization=false \
      --enable_dynamic_camera_following=false \
      --enable_static_camera_yaw_sweep=false \
      --static_camera_yaw_sweep_speed_deg=180 \
      --static_camera_base_y=-2.0 \
      --static_camera_base_z=adaptive \
      --enable_static_camera_locked=false \
      --enable_static_camera_arc_follow=false
  done
done

echo "All DRONE-ONLY inference runs completed."
