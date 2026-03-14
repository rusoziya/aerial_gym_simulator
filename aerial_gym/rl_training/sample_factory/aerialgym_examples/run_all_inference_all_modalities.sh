#!/usr/bin/env bash

# Unified runner: evaluate all camera modalities across seeds and levels,
# automatically selecting the correct policy checkpoint from FINAL POLICIES.
#
# Modalities handled:
#  - sweeping (static yaw sweep)
#  - dynamic_follow (dynamic camera following)
#  - locked_yaw (static locked orientation)
#  - static_random (static randomized orientation)
#  - drone_only (proprio + static latent ablation)
#  - arc_follow (static arc-follow around gate)

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR="$SCRIPT_DIR/../../.."

# Base directory with trained policies (note the space in the folder name)
# Policies live under examples/dce_rl_navigation/FINAL POLICIES
POLICY_BASE_DIR=""

# Mapping of modality -> policy folder (relative to POLICY_BASE_DIR)
declare -A POLICY_DIRS=(
  [sweeping]="drone_and_static_sweeping_curriculum_L13_23_TEST6"
  [dynamic_follow]="drone_and_static_dynamic_following_curriculum_L13_23"
  [locked_yaw]="drone_and_static_locked_following_curriculum_L13_23"
  [static_random]="drone_and_static_fixed_orientation_2"  # randomized orientation policy suite
  [drone_only]="drone_only_curriculum_L13_23"
  [arc_follow]="drone_and_static_arc_following_curriculum_L13_23_4"
)

LEVELS=(3 13 23 33)
SEEDS=(123 231 321 456 789)

# Common environment settings
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
export ABLATE_DEBUG=false
export ABLATE_ZERO_RNN=false
export CUBLAS_WORKSPACE_CONFIG=:16:8
# Enable RNN warm-up via env (avoid CLI parsing conflicts)
export RNN_WARMUP_STEPS=50
# Default ablation: zero proprio slice [0:22] for all modalities
export ABLATE_OBS_RANGES='0:22=zero'

ENV_NAME=dce_navigation_task_gate
TRAIN_DIR="$ROOT_DIR/examples/dce_rl_navigation"
# Now that base is resolved, set the policy directory containing all policies
POLICY_BASE_DIR="$TRAIN_DIR/FINAL POLICIES"
EXPERIMENT=selected_network
ENV_AGENTS=16
MAX_EPISODES=512
SAVE_GIFS=false
HEADLESS=false

find_checkpoint() {
  local policy_dir="$1"
  local ckpt_dir="$policy_dir/checkpoint_p0"
  # Prefer best_*.pth, otherwise latest checkpoint_*.pth by mtime
  local best
  best=$(ls -1t "$ckpt_dir"/best_*.pth 2>/dev/null | head -n 1 || true)
  if [[ -n "$best" ]]; then
    echo "$best"
    return 0
  fi
  local latest
  latest=$(ls -1t "$ckpt_dir"/checkpoint_*.pth 2>/dev/null | head -n 1 || true)
  if [[ -n "$latest" ]]; then
    echo "$latest"
    return 0
  fi
  return 1
}

run_one() {
  local modality="$1"; shift
  local lvl="$1"; shift
  local seed="$1"; shift

  local policy_rel="${POLICY_DIRS[$modality]}"
  local policy_dir="$POLICY_BASE_DIR/$policy_rel"
  if [[ ! -d "$policy_dir" ]]; then
    echo "[WARN] Missing policy folder for modality=$modality → $policy_dir" >&2
    return 0
  fi
  local ckpt
  if ! ckpt=$(find_checkpoint "$policy_dir"); then
    echo "[WARN] No checkpoint found in $policy_dir/checkpoint_p0" >&2
    return 0
  fi
  export DCE_MODEL="$ckpt"

  # Reset ablations to default per run; refine per modality below
  export ABLATE_OBS_RANGES='0:22=zero'

  # Per-modality CLI flags
  local cli=(
    --env "${ENV_NAME}"
    # Point train_dir/experiment to this policy folder so checkpoint config is loaded
    --train_dir="${POLICY_BASE_DIR}"
    --experiment="${policy_rel}"
    --seed="${seed}"
    --env_agents="${ENV_AGENTS}"
    --max_num_episodes="${MAX_EPISODES}"
    --eval_deterministic=true
    --save_gifs="${SAVE_GIFS}"
    --headless="${HEADLESS}"
    --disable_curriculum_multiplier=true
    --disable_gate_size_randomization=false
    --fixed_gate_scale_percent=100
    --disable_obstacle_randomization=false
    --fixed_obstacles_behind_gate=1
    --disable_static_camera_orientation_randomization=false
    --disable_drone_camera_noise_randomization=false
    --disable_static_camera_noise_randomization=false
    --disable_drone_camera_frame_dropout=false
    --disable_static_camera_frame_dropout=false
    --disable_state_noise_randomization=false
    --force_curriculum_level="${lvl}"
    --disable_spawn_position_randomization=false
    --disable_spawn_orientation_randomization=false
    --static_camera_yaw_sweep_speed_deg=180
    --static_camera_base_y=-2.0
    --static_camera_base_z=adaptive
  )

  case "$modality" in
    sweeping)
      cli+=( --enable_dynamic_camera_following=false \
             --enable_static_camera_yaw_sweep=true \
             --enable_static_camera_locked=false \
             --enable_static_camera_arc_follow=false )
      ;;
    dynamic_follow)
      cli+=( --enable_dynamic_camera_following=true \
             --dynamic_camera_follow_y_offset_m=-1.0 \
             --enable_static_camera_yaw_sweep=false \
             --enable_static_camera_locked=false \
             --enable_static_camera_arc_follow=false )
      ;;
    locked_yaw)
      cli+=( --disable_static_camera_orientation_randomization=true \
             --enable_dynamic_camera_following=false \
             --enable_static_camera_yaw_sweep=false \
             --enable_static_camera_locked=true \
             --enable_static_camera_arc_follow=false )
      ;;
    static_random)
      cli+=( --disable_static_camera_orientation_randomization=false \
             --enable_dynamic_camera_following=false \
             --enable_static_camera_yaw_sweep=false \
             --enable_static_camera_locked=false \
             --enable_static_camera_arc_follow=false )
      ;;
    drone_only)
      # DRONE-ONLY: ablate proprio (0:22) and static latents (86:150)
      export ABLATE_OBS_RANGES='0:22=zero,86:150=zero'
      cli+=( --disable_static_camera_orientation_randomization=true \
             --enable_dynamic_camera_following=false \
             --enable_static_camera_yaw_sweep=false \
             --enable_static_camera_locked=false \
             --enable_static_camera_arc_follow=false )
      ;;
    arc_follow)
      cli+=( --enable_dynamic_camera_following=false \
             --enable_static_camera_yaw_sweep=false \
             --enable_static_camera_locked=false \
             --enable_static_camera_arc_follow=true \
             --static_camera_arc_radius_m=2.0 )
      ;;
    *)
      echo "[WARN] Unknown modality $modality" >&2
      return 0
      ;;
  esac

  local run_name="${modality}_L${lvl}_SEED${seed}"
  export WANDB_RUN_NAME="$run_name"
  echo -e "\n=== Running $run_name (ckpt: $DCE_MODEL) ==="
  python -B -m aerial_gym.examples.dce_rl_navigation.dce_nn_navigation_gate "${cli[@]}"
}

main() {
  local modalities=(sweeping dynamic_follow locked_yaw static_random drone_only arc_follow)
  for m in "${modalities[@]}"; do
    echo "\n>>> Modality: $m"
    for s in "${SEEDS[@]}"; do
      for l in "${LEVELS[@]}"; do
        run_one "$m" "$l" "$s"
      done
    done
  done
  echo "\nAll modalities completed."
}

main "$@"


