#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

echo "[run_all] Starting 3 runs..."

echo "[1/3] Fixed orientation (no sweep, no dynamic follow)"
TRAIN_ENV0_LATENTS_NORM=false WANDB_DISABLED=false WANDB_MODE=online ABLATE_DEBUG=true \
./train_gate_navigation_dual_camera.sh drone_and_static_fixed_orientation_1 \
  --seed=123 --train_steps=2012416 --headless=false --view --envs=128 \
  --disable_gate_size_randomization=false --fixed_gate_scale_percent=100 \
  --disable_obstacle_randomization=false --fixed_obstacles_behind_gate=1 \
  --disable_static_camera_orientation_randomization=false \
  --disable_drone_camera_noise_randomization=false --disable_static_camera_noise_randomization=false \
  --disable_drone_camera_frame_dropout=false --disable_static_camera_frame_dropout=false \
  --disable_state_noise_randomization=false --disable_spawn_position_randomization=false \
  --disable_spawn_orientation_randomization=false --disable_curriculum_multiplier=true \
  --force_curriculum_level=none --min_curriculum_level=13 --max_curriculum_level=23 \
  --enable_dynamic_camera_following=false --fusion=gated --gate_per_feature=1 \
  --enable_static_camera_yaw_sweep=false --static_camera_yaw_sweep_speed_deg=180 \
  --static_camera_base_y=-2.0 --static_camera_base_z=adaptive --enable_static_camera_locked=false |& tee -a logs/run1_$(date +%F_%H-%M-%S).log

echo "[2/3] Arc-follow static camera (no dynamic follow)"
TRAIN_ENV0_LATENTS_NORM=false WANDB_DISABLED=false WANDB_MODE=online ABLATE_DEBUG=true ABLATE_OBS_RANGES='0:22=zero'\
./train_gate_navigation_dual_camera.sh drone_and_static_arc_following_curriculum_L13_23 \
  --seed=123 --train_steps=2012416 --headless=false --view --envs=128 \
  --disable_gate_size_randomization=false --fixed_gate_scale_percent=100 \
  --disable_obstacle_randomization=false --fixed_obstacles_behind_gate=1 \
  --disable_static_camera_orientation_randomization=false \
  --disable_drone_camera_noise_randomization=false --disable_static_camera_noise_randomization=false \
  --disable_drone_camera_frame_dropout=false --disable_static_camera_frame_dropout=false \
  --disable_state_noise_randomization=false --disable_spawn_position_randomization=false \
  --disable_spawn_orientation_randomization=false --disable_curriculum_multiplier=true \
  --force_curriculum_level=none --min_curriculum_level=13 --max_curriculum_level=23 \
  --enable_dynamic_camera_following=false --fusion=gated --gate_per_feature=1 \
  --enable_static_camera_yaw_sweep=false --static_camera_yaw_sweep_speed_deg=180 \
  --static_camera_base_y=-2.0 --static_camera_base_z=adaptive --enable_static_camera_locked=false \
  --enable_static_camera_arc_follow=true |& tee -a logs/run2_$(date +%F_%H-%M-%S).log

echo "[3/3] Dynamic following (2 m), no arc-follow"
TRAIN_ENV0_LATENTS_NORM=false WANDB_DISABLED=false WANDB_MODE=online ABLATE_DEBUG=true \
./train_gate_navigation_dual_camera.sh drone_and_static_dynamic_following_curriculum_L13_23_2metres \
  --seed=123 --train_steps=2012416 --headless=false --view --envs=128 \
  --disable_gate_size_randomization=false --fixed_gate_scale_percent=100 \
  --disable_obstacle_randomization=false --fixed_obstacles_behind_gate=1 \
  --disable_static_camera_orientation_randomization=false \
  --disable_drone_camera_noise_randomization=false --disable_static_camera_noise_randomization=false \
  --disable_drone_camera_frame_dropout=false --disable_static_camera_frame_dropout=false \
  --disable_state_noise_randomization=false --disable_spawn_position_randomization=false \
  --disable_spawn_orientation_randomization=false --disable_curriculum_multiplier=true \
  --force_curriculum_level=none --min_curriculum_level=13 --max_curriculum_level=23 \
  --enable_dynamic_camera_following=true --fusion=gated --gate_per_feature=1 \
  --enable_static_camera_yaw_sweep=false --static_camera_yaw_sweep_speed_deg=180 \
  --static_camera_base_y=-2.0 --static_camera_base_z=adaptive --enable_static_camera_locked=false \
  --enable_static_camera_arc_follow=false --dynamic_camera_follow_y_offset_m=-2.0 |& tee -a logs/run3_$(date +%F_%H-%M-%S).log

echo "[run_all] All runs completed."
