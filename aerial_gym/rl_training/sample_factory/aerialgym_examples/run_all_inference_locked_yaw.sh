#!/usr/bin/env bash

# Run inference for static locked-yaw camera modality across multiple levels and seeds
# Levels: 3, 13, 23, 33
# Seeds: five fixed seeds reused across all levels

set -euo pipefail

# Directory constants
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR="$SCRIPT_DIR/../../.."

# Resolve policy checkpoint from FINAL POLICIES
POLICY_BASE_DIR="/home/ziyar/aerialgym/aerialgym_ws/src/aerial_gym_simulator/aerial_gym/examples/dce_rl_navigation/FINAL POLICIES"
POLICY_SUBDIR="drone_and_static_locked_following_curriculum_L13_23"
CKPT=$(ls -1t "$POLICY_BASE_DIR/$POLICY_SUBDIR/checkpoint_p0"/best_*.pth 2>/dev/null | head -n 1 || true)
if [[ -z "$CKPT" ]]; then
  CKPT=$(ls -1t "$POLICY_BASE_DIR/$POLICY_SUBDIR/checkpoint_p0"/checkpoint_*.pth 2>/dev/null | head -n 1 || true)
fi
export DCE_MODEL="$CKPT"

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
export ABLATE_OBS_RANGES='0:22=zero'
export CUBLAS_WORKSPACE_CONFIG=:16:8

# Paths
ENV_NAME=dce_navigation_task_gate
TRAIN_DIR="$POLICY_BASE_DIR"
EXPERIMENT="$POLICY_SUBDIR"

# Inference options
ENV_AGENTS=128
MAX_EPISODES=512
SAVE_GIFS=false
HEADLESS=false

echo "Starting static locked-yaw camera inference runs..."

for SEED in "${SEEDS[@]}"; do
  for LVL in "${LEVELS[@]}"; do
    RUN_NAME="locked_yaw_L${LVL}_SEED${SEED}"
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
      --enable_static_camera_locked=true \
      --enable_static_camera_arc_follow=false
  done
done

echo "All static locked-yaw camera inference runs completed."
