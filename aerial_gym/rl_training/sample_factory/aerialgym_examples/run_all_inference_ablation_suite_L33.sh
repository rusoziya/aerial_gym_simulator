#!/usr/bin/env bash

# Inference ablation suite @ L33 only
# Scenarios (5 seeds each; same policy):
# 1) Drone-only (exocentric ablated)
# 2) Static-only (egocentric ablated)
# 3) Dual (no ablation)
# 4) Dual (drone cam noise+dropout disabled)
# 5) Dual (static cam noise+dropout disabled)

set -euo pipefail

# Directory constants (workspace: aerialgym_ws)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR="$SCRIPT_DIR/../../.."

# Resolve policy checkpoint from FINAL POLICIES (dynamic-follow policy)
POLICY_BASE_DIR="/home/ziyar/aerialgym/aerialgym_ws/src/aerial_gym_simulator/aerial_gym/examples/dce_rl_navigation/FINAL POLICIES"
POLICY_SUBDIR="drone_and_static_dynamic_following_curriculum_L13_23"
CKPT=$(ls -1t "$POLICY_BASE_DIR/$POLICY_SUBDIR/checkpoint_p0"/best_*.pth 2>/dev/null | head -n 1 || true)
if [[ -z "$CKPT" ]]; then
  CKPT=$(ls -1t "$POLICY_BASE_DIR/$POLICY_SUBDIR/checkpoint_p0"/checkpoint_*.pth 2>/dev/null | head -n 1 || true)
fi
export DCE_MODEL="$CKPT"

if [[ -z "${DCE_MODEL:-}" ]]; then
  echo "ERROR: Could not resolve checkpoint under $POLICY_BASE_DIR/$POLICY_SUBDIR/checkpoint_p0" >&2
  exit 1
fi

# Single evaluation level: 33 (held-out harder level)
LEVELS=(33)

# Five seeds
SEEDS=(123 231 321 456 789)

# Common environment variables
export VISIBILITY_DEBUG=false
export PRINT_ENV0_LATENTS_ONCE_NORM=false
export PRINT_ENV0_LATENTS_ONCE=false
export DISABLE_NORM=false
export NORM_TAP_DEBUG=false
export ENC_TAP_DEBUG=false
export EVAL_STRETCH_ENABLED=1
export EVAL_STRETCH_END_LEVEL=33
export WANDB_PROJECT=final_inference_ablation_suite_L33
export WANDB_DISABLED=false
export WANDB_MODE=online
export ABLATE_DEBUG=true
export ABLATE_ZERO_RNN=false
export CUBLAS_WORKSPACE_CONFIG=:16:8
# Enable RNN warm-up via env (recognized by dce_nn_navigation_gate.py)
export RNN_WARMUP_STEPS=50
export DISABLE_WALLS=false

# Paths
ENV_NAME=dce_navigation_task_gate
TRAIN_DIR="$POLICY_BASE_DIR"
EXPERIMENT="$POLICY_SUBDIR"

# Inference options
ENV_AGENTS=16
MAX_EPISODES=512
SAVE_GIFS=false
HEADLESS=false

echo "Starting ablation suite inference runs @ L33 (2 scenarios × 5 seeds)..."

# Scenario list
SCENARIOS=(
  "dual_no_drone_noise"   # dual; disable noise+dropout on drone cam (noise asymmetry)
  "dual_no_static_noise"  # dual; disable noise+dropout on static cam (noise asymmetry)
)

for SCEN in "${SCENARIOS[@]}"; do
  # Defaults (noise/dropout enabled as in baseline eval)
  DISABLE_DRONE_NOISE=false
  DISABLE_STATIC_NOISE=false
  DISABLE_DRONE_DROPOUT=false
  DISABLE_STATIC_DROPOUT=false

  case "$SCEN" in
    drone_only)
      export ABLATE_OBS_RANGES='0:22=zero,86:150=zero'
      ;;
    static_only)
      export ABLATE_OBS_RANGES='0:22=zero,22:86=zero'
      ;;
    dual)
      export ABLATE_OBS_RANGES='0:22=zero'
      ;;
    dual_no_drone_noise)
      export ABLATE_OBS_RANGES='0:22=zero'
      DISABLE_STATIC_NOISE=false
      DISABLE_STATIC_DROPOUT=false
      DISABLE_DRONE_NOISE=true
      DISABLE_DRONE_DROPOUT=true
      ;;
    dual_no_static_noise)
      export ABLATE_OBS_RANGES='0:22=zero'
      DISABLE_STATIC_NOISE=true
      DISABLE_STATIC_DROPOUT=true
      DISABLE_DRONE_NOISE=false
      DISABLE_DRONE_DROPOUT=false
      ;;
    *)
      echo "Unknown scenario: $SCEN" >&2; exit 1;
      ;;
  esac

  for SEED in "${SEEDS[@]}"; do
    for LVL in "${LEVELS[@]}"; do
      RUN_NAME="${SCEN}_L${LVL}_SEED${SEED}"
      export WANDB_RUN_NAME="$RUN_NAME"
      echo -e "\n=== Running ${RUN_NAME} ==="

      python -B -m aerial_gym.examples.dce_rl_navigation.dce_nn_navigation_gate \
        --env ${ENV_NAME} \
        --train_dir="${TRAIN_DIR}" \
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
        --disable_static_camera_orientation_randomization=false \
        --disable_drone_camera_noise_randomization=${DISABLE_DRONE_NOISE} \
        --disable_static_camera_noise_randomization=${DISABLE_STATIC_NOISE} \
        --disable_drone_camera_frame_dropout=${DISABLE_DRONE_DROPOUT} \
        --disable_static_camera_frame_dropout=${DISABLE_STATIC_DROPOUT} \
        --disable_state_noise_randomization=false \
        --force_curriculum_level=${LVL} \
        --disable_spawn_position_randomization=false \
        --disable_spawn_orientation_randomization=false \
        --enable_dynamic_camera_following=true \
        --dynamic_camera_follow_y_offset_m=-1.0 \
        --enable_static_camera_yaw_sweep=false \
        --static_camera_yaw_sweep_speed_deg=180 \
        --static_camera_base_y=-2.0 \
        --static_camera_base_z=adaptive \
        --enable_static_camera_locked=false \
        --enable_static_camera_arc_follow=false
    done
  done
done

echo "All ablation suite inference runs completed."
