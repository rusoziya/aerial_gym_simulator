#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

echo "[run_all] Starting 4 runs..."

echo "[1/4] Camera + static sweeping curriculum"
WANDB_DISABLED=false WANDB_MODE=online ABLATE_DEBUG=true ABLATE_OBS_RANGES='0:22=zero' \
./train_gate_navigation_dual_camera.sh drone_and_static_sweeping_curriculum_1.5lateral_2 \
  --seed=123 --train_steps=2012416 --headless=false --view --envs=256 \
  --disable_gate_size_randomization=false --fixed_gate_scale_percent=100 \
  --disable_obstacle_randomization=false --fixed_obstacles_behind_gate=1 \
  --disable_static_camera_orientation_randomization=false \
  --disable_drone_camera_noise_randomization=false --disable_static_camera_noise_randomization=false \
  --disable_drone_camera_frame_dropout=false --disable_static_camera_frame_dropout=false \
  --disable_state_noise_randomization=false --disable_spawn_position_randomization=false \
  --disable_spawn_orientation_randomization=false --disable_curriculum_multiplier=true \
  --force_curriculum_level=none --max_curriculum_level=23 \
  --enable_dynamic_camera_following=false --fusion=gated --gate_per_feature=1 \
  --enable_static_camera_yaw_sweep=true --static_camera_yaw_sweep_speed_deg=180 \
  --static_camera_base_y=-2.0 --static_camera_base_z=adaptive --enable_static_camera_locked=false |& tee -a logs/run1_$(date +%F_%H-%M-%S).log

echo "[2/4] Camera + static locked"
WANDB_DISABLED=false WANDB_MODE=online ABLATE_DEBUG=true ABLATE_OBS_RANGES='0:22=zero' \
./train_gate_navigation_dual_camera.sh drone_and_static_locked_curriculum_1.5lateral \
  --seed=123 --train_steps=2012416 --headless=false --view --envs=256 \
  --disable_gate_size_randomization=false --fixed_gate_scale_percent=100 \
  --disable_obstacle_randomization=false --fixed_obstacles_behind_gate=1 \
  --disable_static_camera_orientation_randomization=false \
  --disable_drone_camera_noise_randomization=false --disable_static_camera_noise_randomization=false \
  --disable_drone_camera_frame_dropout=false --disable_static_camera_frame_dropout=false \
  --disable_state_noise_randomization=false --disable_spawn_position_randomization=false \
  --disable_spawn_orientation_randomization=false --disable_curriculum_multiplier=true \
  --force_curriculum_level=none --max_curriculum_level=23 \
  --enable_dynamic_camera_following=false --fusion=gated --gate_per_feature=1 \
  --enable_static_camera_yaw_sweep=false --static_camera_yaw_sweep_speed_deg=180 \
  --static_camera_base_y=-2.0 --static_camera_base_z=adaptive --enable_static_camera_locked=true |& tee -a logs/run2_$(date +%F_%H-%M-%S).log

echo "[3/4] Camera + static dynamic following curriculum"
WANDB_DISABLED=false WANDB_MODE=online ABLATE_DEBUG=true ABLATE_OBS_RANGES='0:22=zero' \
./train_gate_navigation_dual_camera.sh drone_and_static_dynamic_curriculum_1.5lateral \
  --seed=123 --train_steps=2012416 --headless=false --view --envs=256 \
  --disable_gate_size_randomization=false --fixed_gate_scale_percent=100 \
  --disable_obstacle_randomization=false --fixed_obstacles_behind_gate=1 \
  --disable_static_camera_orientation_randomization=false \
  --disable_drone_camera_noise_randomization=false --disable_static_camera_noise_randomization=false \
  --disable_drone_camera_frame_dropout=false --disable_static_camera_frame_dropout=false \
  --disable_state_noise_randomization=false --disable_spawn_position_randomization=false \
  --disable_spawn_orientation_randomization=false --disable_curriculum_multiplier=true \
  --force_curriculum_level=none --max_curriculum_level=23 \
  --enable_dynamic_camera_following=true --fusion=gated --gate_per_feature=1 \
  --enable_static_camera_yaw_sweep=false --static_camera_yaw_sweep_speed_deg=180 \
  --static_camera_base_y=-2.0 --static_camera_base_z=adaptive --enable_static_camera_locked=false |& tee -a logs/run3_$(date +%F_%H-%M-%S).log

echo "[4/4] Camera only curriculum"
WANDB_DISABLED=false WANDB_MODE=online ABLATE_DEBUG=true ABLATE_OBS_RANGES='0:22=zero,86:150=zero' \
./train_gate_navigation_dual_camera.sh drone_only_curriculum_1.5lateral \
  --seed=123 --train_steps=2012416 --headless=false --view --envs=256 \
  --disable_gate_size_randomization=false --fixed_gate_scale_percent=100 \
  --disable_obstacle_randomization=false --fixed_obstacles_behind_gate=1 \
  --disable_static_camera_orientation_randomization=false \
  --disable_drone_camera_noise_randomization=false --disable_static_camera_noise_randomization=false \
  --disable_drone_camera_frame_dropout=false --disable_static_camera_frame_dropout=false \
  --disable_state_noise_randomization=false --disable_spawn_position_randomization=false \
  --disable_spawn_orientation_randomization=false --disable_curriculum_multiplier=true \
  --force_curriculum_level=none --max_curriculum_level=23 \
  --enable_dynamic_camera_following=false --fusion=gated --gate_per_feature=1 \
  --enable_static_camera_yaw_sweep=false --static_camera_yaw_sweep_speed_deg=180 \
  --static_camera_base_y=-2.0 --static_camera_base_z=adaptive --enable_static_camera_locked=false |& tee -a logs/run4_$(date +%F_%H-%M-%S).log

echo "[run_all] All runs completed."
