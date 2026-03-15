from __future__ import annotations

import os
import torch

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("navigation_task_gate_curriculum_logging")


class CurriculumLogging:
    def __init__(self, task: object) -> None:
        self.task = task

    def _log_curriculum_details(
        self,
        success_rate: float,
        crash_rate: float,
        timeout_rate: float,
        obstacles_behind_gate: int,
        total_obstacles_in_env: int,
    ) -> None:
        """Log comprehensive curriculum state after level update."""
        self.task.log_curriculum_update(f"Gate Navigation Curriculum Level: {self.task.curriculum_level}, Progress: {self.task.curriculum_progress_fraction:.3f}")
        self.task.log_curriculum_update(f"\nSuccess Rate: {success_rate:.3f}\nCrash Rate: {crash_rate:.3f}\nTimeout Rate: {timeout_rate:.3f}")

        self.task.log_curriculum_update(f"\nCURRICULUM APPLIED:")
        # Report yaw sweep status in curriculum update debug (takes precedence over orientation randomization)
        try:
            yaw_enabled = str(os.environ.get('SF_ENABLE_STATIC_CAMERA_YAW_SWEEP', 'false')).lower() == 'true'
            yaw_speed = float(os.environ.get('SF_STATIC_CAMERA_YAW_SWEEP_SPEED_DEG', '10.0'))
        except (ValueError, TypeError):
            yaw_enabled = False
            yaw_speed = 10.0
        # Determine dynamic camera effective state (needed below)
        try:
            dyn_cfg = self.task.task_config.curriculum.enable_dynamic_camera_following
            dyn_dis = bool(self.task.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False))
            dynamic_effective = bool(dyn_cfg and not dyn_dis)
        except (KeyError, TypeError):
            dynamic_effective = False
        # Report sweep with effective status and orientation/dynamic interactions
        if yaw_enabled and not dynamic_effective:
            self.task.log_curriculum_update(f"   3. STATIC CAMERA YAW SWEEP: ENABLED (speed={yaw_speed:.1f} deg/s)")
        elif yaw_enabled and dynamic_effective:
            self.task.log_curriculum_update(f"   3. STATIC CAMERA YAW SWEEP: ENABLED but IGNORED (dynamic camera active)")
        else:
            self.task.log_curriculum_update(f"   3. STATIC CAMERA YAW SWEEP: DISABLED")
        # Report arc-follow status
        try:
            arc_follow_enabled = bool(self.task.sim_env.global_tensor_dict.get('static_camera/arc_follow_enabled', False))
            arc_radius = float(self.task.sim_env.global_tensor_dict.get('static_camera/arc_follow_radius_m', 2.0))
        except (ValueError, TypeError):
            arc_follow_enabled = False
            arc_radius = 2.0
        if arc_follow_enabled:
            self.task.log_curriculum_update(f"   3b. STATIC CAMERA ARC-FOLLOW: ENABLED (radius={arc_radius:.1f} m)")
        if obs_dis:
            self.task.log_curriculum_update(f"   1. OBSTACLES: fixed to {obstacles_behind_gate} behind gate (total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)")
        else:
            self.task.log_curriculum_update(f"   1. OBSTACLES: {obstacles_behind_gate} behind gate (total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)")
        try:
            baseline_level = int(self.task.task_config.curriculum.min_level)
            pos_dis = bool(self.task.sim_env.global_tensor_dict.get('spawn_randomization/position_disabled', False))
            yaw_dis = bool(self.task.sim_env.global_tensor_dict.get('spawn_randomization/orientation_disabled', False))
            sr_active = self.task.task_config.curriculum.get_spawn_ranges(self.task.curriculum_level)
            sr_base = self.task.task_config.curriculum.get_spawn_ranges(baseline_level)
            sr_use = {
                'x_half_span_m': sr_base['x_half_span_m'] if pos_dis else sr_active['x_half_span_m'],
                'y_center_m':    sr_base['y_center_m']    if pos_dis else sr_active['y_center_m'],
                'y_half_span_m': sr_base['y_half_span_m'] if pos_dis else sr_active['y_half_span_m'],
                'z_center_m':    sr_base['z_center_m']    if pos_dis else sr_active['z_center_m'],
                'z_half_span_m': sr_base['z_half_span_m'] if pos_dis else sr_active['z_half_span_m'],
                'yaw_abs_rad':   sr_base['yaw_abs_rad']   if yaw_dis else sr_active['yaw_abs_rad'],
            }
            if pos_dis or yaw_dis:
                status_pos = "DISABLED" if pos_dis else "ENABLED"
                status_yaw = "DISABLED" if yaw_dis else "ENABLED"
                self.task.log_curriculum_update(f"   2. SPAWN RANDOMIZATION: position={status_pos}, orientation={status_yaw}")
            self.task.log_curriculum_update(
                f"   2. SPAWN: X∈[{(-sr_use['x_half_span_m']):.1f}, {(+sr_use['x_half_span_m']):.1f}] m, "
                f"Y∈[{(sr_use['y_center_m']-sr_use['y_half_span_m']):.1f}, {(sr_use['y_center_m']+sr_use['y_half_span_m']):.1f}] m, "
                f"Z∈[{(sr_use['z_center_m']-sr_use['z_half_span_m']):.1f}, {(sr_use['z_center_m']+sr_use['z_half_span_m']):.1f}] m; yaw ±{(sr_use['yaw_abs_rad']*57.2958):.1f}°"
            )
        except (ValueError, TypeError) as e:
            self.task.log_curriculum_update(f"   2. SPAWN: (fallback) Using fixed LMF2 config due to: {e}")
        # Get current randomized angle for first environment (representative)
        current_angle = 0.0
        if hasattr(self.task.static_camera_manager, 'current_camera_angles'):
            current_angle = self.task.static_camera_manager.current_camera_angles[0] if self.task.static_camera_manager.current_camera_angles else 0.0
        # Report static camera orientation randomization status (only relevant when yaw sweep is DISABLED)
        try:
            cam_orient_disabled = bool(self.task.sim_env.global_tensor_dict.get('static_camera_randomization/orientation_disabled', False))
        except (KeyError, TypeError):
            cam_orient_disabled = False
        if yaw_enabled and not dynamic_effective:
            self.task.log_curriculum_update(f"   4. CAMERA ANGLE: overridden by yaw sweep (env0 current: {current_angle:.1f}°)")
        elif dynamic_effective:
            self.task.log_curriculum_update(f"   4. CAMERA ANGLE: suppressed (dynamic camera following active)")
        elif cam_orient_disabled:
            self.task.log_curriculum_update(f"   4. CAMERA ANGLE: randomization DISABLED, fixed at 0.0° (env0: {current_angle:.1f}°)")
        else:
            self.task.log_curriculum_update(f"   4. CAMERA ANGLE: ±{self.task.max_camera_angle:.1f}deg max range, env0: {current_angle:.1f}deg (fixed per episode)")

        # 4. GATE SIZE UNLOCKS (Curriculum-gated randomization) or Fixed (ablation)
        gate_names = []
        if len(self.task.sim_env.global_tensor_dict.get("gate_variant_names_per_env", [])) > 0:
            gate_names = self.task.sim_env.global_tensor_dict["gate_variant_names_per_env"][0]
        # Report fixed mode if enabled
        disable_flag = self.task.sim_env.global_tensor_dict.get('gate_randomization/disabled', False)
        try:
            if hasattr(disable_flag, 'item'):
                disable_flag = bool(disable_flag.item())
            else:
                disable_flag = bool(disable_flag)
        except (ValueError, TypeError, RuntimeError):
            disable_flag = False
        if disable_flag:
            try:
                fixed_scale = self.task.sim_env.global_tensor_dict.get('gate_randomization/fixed_scale_percent', 100)
                if hasattr(fixed_scale, 'item'):
                    fixed_scale = int(fixed_scale.item())
                else:
                    fixed_scale = int(fixed_scale)
            except (ValueError, TypeError):
                fixed_scale = 100
            self.task.log_curriculum_update(f"   4. GATE SIZE: randomization disabled, fixed scale = {fixed_scale}%")
        else:
            # Compute linear threshold from 80 -> 60 over levels 3..23
            # If EVAL_STRETCH_ENABLED, extend further to 50% by eval_end_level
            stretch_enabled = _os.environ.get("EVAL_STRETCH_ENABLED", "0").strip() in ("1", "true", "True")
            eval_end = int(_os.environ.get("EVAL_STRETCH_END_LEVEL", str(self.task.task_config.curriculum.eval_stretch_end_level)))
            level = int(self.task.curriculum_level)
            if level <= 3:
                min_scale = 80
            elif level <= 23:
                frac = (level - 3) / (23 - 3)
                raw = 80 - frac * (80 - 60)
                min_scale = int((int(raw) // 2) * 2)
            elif stretch_enabled:
                # Extend 23->eval_end: 60% -> 50% linearly
                if level >= eval_end:
                    min_scale = 50
                else:
                    extra_frac = (level - 23) / max(1, (eval_end - 23))
                    raw = 60 - extra_frac * (60 - 50)
                    min_scale = int((int(raw) // 2) * 2)
            else:
                min_scale = 60
            if min_scale < 50:
                min_scale = 50
            if min_scale > 100:
                min_scale = 100
            # Collect scales meeting threshold
            scales = []
            for n in gate_names:
                if isinstance(n, str) and "gate_scale_" in n:
                    try:
                        s = int(n.replace("gate_scale_", ""))
                        if s >= min_scale:
                            scales.append(s)
                    except (ValueError, TypeError):
                        pass
            # Report unique scales only (avoid duplicates from config classes)
            scales = sorted(list(set(scales)), reverse=True)
            self.task.log_curriculum_update(f"   4. GATE SIZE: unlocked scales >= {min_scale}% -> {scales if scales else [100]} (uniform across unique scales)")

        # 5. CAMERA NOISE PROGRESSION (D455 Simulation)
        camera_gaussian_std, camera_dropout_rate = self.task.task_config.curriculum.get_camera_noise(self.task.curriculum_level)
        try:
            cam_noise_disabled = bool(self.task.sim_env.global_tensor_dict.get('camera_randomization/noise_disabled', False))
        except (KeyError, TypeError):
            cam_noise_disabled = False
        # Per-camera overrides for noise (presence-based overrides)
        try:
            gtd = self.task.sim_env.global_tensor_dict
            drone_noise_key_present = 'camera_randomization/drone_noise_disabled' in gtd
            static_noise_key_present = 'camera_randomization/static_noise_disabled' in gtd
            drone_noise_flag = bool(gtd.get('camera_randomization/drone_noise_disabled', False)) if drone_noise_key_present else cam_noise_disabled
            static_noise_flag = bool(gtd.get('camera_randomization/static_noise_disabled', False)) if static_noise_key_present else cam_noise_disabled
        except (KeyError, TypeError):
            drone_noise_flag = cam_noise_disabled
            static_noise_flag = cam_noise_disabled
        # Level-3 fallbacks when disabled
        d_std_min, d_drop_min = self.task.task_config.curriculum.get_camera_noise(3)
        eff_drone_std = camera_gaussian_std if not drone_noise_flag else d_std_min
        eff_static_std = camera_gaussian_std if not static_noise_flag else d_std_min
        eff_drone_drop = camera_dropout_rate if not drone_noise_flag else d_drop_min
        eff_static_drop = camera_dropout_rate if not static_noise_flag else d_drop_min
        self.task.log_curriculum_update(
            f"   5. CAMERA NOISE: drone(std={eff_drone_std:.4f}, pixel_drop={eff_drone_drop*100:.1f}%), static(std={eff_static_std:.4f}, pixel_drop={eff_static_drop*100:.1f}%)"
        )

        # 6. CAMERA FRAME DROPOUT (entire-frame)
        fd = self.task.task_config.curriculum.get_camera_frame_dropout(self.task.curriculum_level)
        try:
            cam_fd_disabled = bool(self.task.sim_env.global_tensor_dict.get('camera_randomization/frame_dropout_disabled', False))
        except (KeyError, TypeError):
            cam_fd_disabled = False
        # Per-camera overrides for frame dropout (presence-based overrides)
        try:
            gtd = self.task.sim_env.global_tensor_dict
            drone_fd_key_present = 'camera_randomization/drone_frame_dropout_disabled' in gtd
            static_fd_key_present = 'camera_randomization/static_frame_dropout_disabled' in gtd
            drone_fd_flag = bool(gtd.get('camera_randomization/drone_frame_dropout_disabled', False)) if drone_fd_key_present else cam_fd_disabled
            static_fd_flag = bool(gtd.get('camera_randomization/static_frame_dropout_disabled', False)) if static_fd_key_present else cam_fd_disabled
        except (KeyError, TypeError):
            drone_fd_flag = cam_fd_disabled
            static_fd_flag = cam_fd_disabled
        # After change: when disabled, show level-3 minimum totals instead of 0
        fd_min = self.task.task_config.curriculum.get_camera_frame_dropout(3)
        eff_drone_tot = fd['drone_total'] if not drone_fd_flag else fd_min['drone_total']
        eff_static_tot = fd['static_total'] if not static_fd_flag else fd_min['static_total']
        eff_drone_freeze = fd['drone_freeze'] if not drone_fd_flag else fd_min['drone_freeze']
        eff_drone_blank = fd['drone_blank'] if not drone_fd_flag else fd_min['drone_blank']
        eff_static_freeze = fd['static_freeze'] if not static_fd_flag else fd_min['static_freeze']
        eff_static_blank = fd['static_blank'] if not static_fd_flag else fd_min['static_blank']
        self.task.log_curriculum_update(
            f"   6. CAMERA FRAME DROPOUT: drone_total={eff_drone_tot*100:.1f}% (freeze {eff_drone_freeze*100:.1f}%, blank {eff_drone_blank*100:.1f}%), static_total={eff_static_tot*100:.1f}% (freeze {eff_static_freeze*100:.1f}%, blank {eff_static_blank*100:.1f}%)"
        )

        # 7. STATE NOISE (pose)
        if self.task.task_config.curriculum.enable_state_noise:
            try:
                state_noise_disabled = bool(self.task.sim_env.global_tensor_dict.get('state_randomization/noise_disabled', False))
            except (KeyError, TypeError):
                state_noise_disabled = False
            if state_noise_disabled:
                self.task.log_curriculum_update("   7. STATE NOISE: DISABLED (all std=0)")
            else:
                sn = self.task.task_config.curriculum.get_state_noise(self.task.curriculum_level)
                self.task.log_curriculum_update(
                    f"   7. STATE NOISE: drone_pos_std={sn['drone_pos_std_m']:.4f} m, drone_orient_std={sn['drone_orient_std_rad']*57.2958:.3f} deg, "
                    f"static_pos_std={sn['static_pos_std_m']:.4f} m, static_orient_std={sn['static_orient_std_rad']*57.2958:.3f} deg"
                )
        else:
            self.task.log_curriculum_update("   7. STATE NOISE: disabled")

        # 8. DYNAMIC CAMERA FOLLOWING
        dynamic_enabled = self.task.task_config.curriculum.enable_dynamic_camera_following
        try:
            dynamic_disabled = bool(self.task.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False))
            config_overridden = bool(self.task.sim_env.global_tensor_dict.get('dynamic_camera_following/config_overridden', False))
        except (KeyError, TypeError):
            dynamic_disabled = False
            config_overridden = False

        if dynamic_enabled and not dynamic_disabled:
            if config_overridden:
                self.task.log_curriculum_update("   8. DYNAMIC CAMERA: ENABLED by flag (--enable_dynamic_camera_following=true) - camera follows drone with adaptive gate targeting")
            else:
                self.task.log_curriculum_update("   8. DYNAMIC CAMERA: ENABLED (camera follows drone with adaptive gate targeting)")
        elif dynamic_enabled and dynamic_disabled:
            self.task.log_curriculum_update("   8. DYNAMIC CAMERA: DISABLED by flag (--disable_dynamic_camera_following=true)")
        else:
            if config_overridden:
                self.task.log_curriculum_update("   8. DYNAMIC CAMERA: DISABLED by flag (--enable_dynamic_camera_following=false)")
            else:
                self.task.log_curriculum_update("   8. DYNAMIC CAMERA: DISABLED (static camera mode - curriculum-based positioning)")

        # Curriculum multiplier debug (update block)
        cm_disabled = read_env_bool("SF_DISABLE_CURRICULUM_MULTIPLIER", self.task.task_config.disable_curriculum_multiplier)
        if not cm_disabled:
            cm_disabled = bool(self.task.task_config.disable_curriculum_multiplier)
        frac_eff = 0.0 if cm_disabled else float(self.task.curriculum_progress_fraction)
        factor = 1.0 + 0.5 * frac_eff
        self.task.log_curriculum_update(f"   8. CURRICULUM MULTIPLIER: {'DISABLED' if cm_disabled else 'ENABLED'} (factor={factor:.3f})")

        self.task.log_curriculum_update(f"[CURRICULUM UPDATE] FINAL STATE:")
        self.task.log_curriculum_update(f"[CURRICULUM UPDATE]   Level: {self.task.curriculum_level} (range: {self.task.task_config.curriculum.min_level}-{self.task.task_config.curriculum.max_level})")
        self.task.log_curriculum_update(f"[CURRICULUM UPDATE]   Max level reached: {self.task.max_curriculum_level_reached} (DECREASE ENABLED)")
        self.task.log_curriculum_update(f"[CURRICULUM UPDATE]   Progress: {self.task.curriculum_progress_fraction:.3f}")
        self.task.log_curriculum_update(f"[CURRICULUM UPDATE]   Obstacles behind gate: {obstacles_behind_gate} (total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)")
        self.task.log_curriculum_update(f"[CURRICULUM UPDATE]   Asset manager: Updated both obs_dict and global_tensor_dict with count {total_obstacles_in_env}")
        # Report spawn ablation status with effective ranges
        try:
            baseline_level = int(self.task.task_config.curriculum.min_level)
            pos_dis = bool(self.task.sim_env.global_tensor_dict.get('spawn_randomization/position_disabled', False))
            yaw_dis = bool(self.task.sim_env.global_tensor_dict.get('spawn_randomization/orientation_disabled', False))
            sr_active = self.task.task_config.curriculum.get_spawn_ranges(self.task.curriculum_level)
            sr_base = self.task.task_config.curriculum.get_spawn_ranges(baseline_level)
            sr_use = {
                'x_half_span_m': sr_base['x_half_span_m'] if pos_dis else sr_active['x_half_span_m'],
                'y_center_m':    sr_base['y_center_m']    if pos_dis else sr_active['y_center_m'],
                'y_half_span_m': sr_base['y_half_span_m'] if pos_dis else sr_active['y_half_span_m'],
                'z_center_m':    sr_base['z_center_m']    if pos_dis else sr_active['z_center_m'],
                'z_half_span_m': sr_base['z_half_span_m'] if pos_dis else sr_active['z_half_span_m'],
                'yaw_abs_rad':   sr_base['yaw_abs_rad']   if yaw_dis else sr_active['yaw_abs_rad'],
            }
            status_pos = "DISABLED" if pos_dis else "ENABLED"
            status_yaw = "DISABLED" if yaw_dis else "ENABLED"
            self.task.log_curriculum_update(f"[CURRICULUM UPDATE]   Spawn randomization: position={status_pos}, orientation={status_yaw}")
            self.task.log_curriculum_update(
                f"[CURRICULUM UPDATE]   Spawn ranges: X∈[{(-sr_use['x_half_span_m']):.1f}, {(+sr_use['x_half_span_m']):.1f}] m, "
                f"Y∈[{(sr_use['y_center_m']-sr_use['y_half_span_m']):.1f}, {(sr_use['y_center_m']+sr_use['y_half_span_m']):.1f}] m, "
                f"Z∈[{(sr_use['z_center_m']-sr_use['z_half_span_m']):.1f}, {(sr_use['z_center_m']+sr_use['z_half_span_m']):.1f}] m; yaw ±{(sr_use['yaw_abs_rad']*57.2958):.1f}°"
            )
        except (ValueError, TypeError):
            self.task.log_curriculum_update(f"[CURRICULUM UPDATE]   Spawn difficulty: LMF2 config (fallback)")
        # When yaw sweep is enabled and dynamic camera is not active, suppress static camera angle randomization message
        try:
            yaw_enabled = str(os.environ.get('SF_ENABLE_STATIC_CAMERA_YAW_SWEEP', 'false')).lower() == 'true'
        except (KeyError, TypeError):
            yaw_enabled = False
        dynamic_effective = False
        try:
            dyn_cfg = self.task.task_config.curriculum.enable_dynamic_camera_following
            dyn_dis = bool(self.task.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False))
            dynamic_effective = bool(dyn_cfg and not dyn_dis)
        except (KeyError, TypeError):
            dynamic_effective = False
        if yaw_enabled and not dynamic_effective:
            # Already logged as overridden by yaw sweep earlier
            pass
        else:
            self.task.log_curriculum_update(f"[CURRICULUM UPDATE]   Camera angle: ±{self.task.max_camera_angle:.1f}deg max range (randomized per episode reset, fixed during episode)")


    def _populate_curriculum_infos(
        self,
        success_rate: float,
        crash_rate: float,
        timeout_rate: float,
        obstacles_behind_gate: int,
        total_obstacles_in_env: int,
    ) -> None:
        """Populate self.task.infos with curriculum metrics for wandb logging."""
        # Add comprehensive curriculum metrics to infos for wandb logging
        self.task.infos["curriculum/level"] = torch.as_tensor(self.task.curriculum_level, dtype=torch.float32)
        self.task.infos["curriculum/progress"] = torch.as_tensor(self.task.curriculum_progress_fraction, dtype=torch.float32)
        self.task.infos["curriculum/success_rate"] = torch.as_tensor(success_rate, dtype=torch.float32)
        self.task.infos["curriculum/crash_rate"] = torch.as_tensor(crash_rate, dtype=torch.float32)
        self.task.infos["curriculum/timeout_rate"] = torch.as_tensor(timeout_rate, dtype=torch.float32)

        # Add curriculum metrics
        self.task.infos["curriculum/obstacles_behind_gate"] = torch.as_tensor(obstacles_behind_gate, dtype=torch.float32)
        self.task.infos["curriculum/total_assets"] = torch.as_tensor(total_obstacles_in_env, dtype=torch.float32)
        self.task.infos["curriculum/max_level_reached"] = torch.as_tensor(self.task.max_curriculum_level_reached, dtype=torch.float32)

        # Add camera noise metrics (D455 simulation) — report effective per-camera values in logs above
        self.task.infos["curriculum/camera_gaussian_std"] = torch.as_tensor(camera_gaussian_std, dtype=torch.float32)
        self.task.infos["curriculum/camera_dropout_rate"] = torch.as_tensor(camera_dropout_rate, dtype=torch.float32)
        # Per-camera effective values (respecting per-camera disable overrides) — level-3 fallback when disabled
        gtd = self.task.sim_env.global_tensor_dict
        drone_noise_dis = bool(gtd.get('camera_randomization/drone_noise_disabled', False))
        static_noise_dis = bool(gtd.get('camera_randomization/static_noise_disabled', False))
        # Level-3 minimums
        d_std_min, d_drop_min = self.task.task_config.curriculum.get_camera_noise(3)
        # Effective Gaussian std per camera
        eff_gauss_drone = camera_gaussian_std if not drone_noise_dis else d_std_min
        eff_gauss_static = camera_gaussian_std if not static_noise_dis else d_std_min
        # Effective pixel dropout per camera
        eff_drop_drone = camera_dropout_rate if not drone_noise_dis else d_drop_min
        eff_drop_static = camera_dropout_rate if not static_noise_dis else d_drop_min
        self.task.infos["curriculum/camera_noise_drone_gaussian_std"] = torch.tensor(eff_gauss_drone, dtype=torch.float32)
        self.task.infos["curriculum/camera_noise_static_gaussian_std"] = torch.tensor(eff_gauss_static, dtype=torch.float32)
        self.task.infos["curriculum/camera_noise_drone_dropout_rate"] = torch.tensor(eff_drop_drone, dtype=torch.float32)
        self.task.infos["curriculum/camera_noise_static_dropout_rate"] = torch.tensor(eff_drop_static, dtype=torch.float32)
        # Add camera frame dropout metrics (effective per-camera, with level-3 fallback when disabled)
        fd_sched = self.task.task_config.curriculum.get_camera_frame_dropout(self.task.curriculum_level)
        try:
            gtd = self.task.sim_env.global_tensor_dict
            drone_fd_flag = bool(gtd.get('camera_randomization/drone_frame_dropout_disabled', False))
            static_fd_flag = bool(gtd.get('camera_randomization/static_frame_dropout_disabled', False))
        except (KeyError, TypeError):
            drone_fd_flag = False
            static_fd_flag = False
        fd_min = self.task.task_config.curriculum.get_camera_frame_dropout(3)
        eff = {
            'drone_total':  fd_sched['drone_total']  if not drone_fd_flag else fd_min['drone_total'],
            'static_total': fd_sched['static_total'] if not static_fd_flag else fd_min['static_total'],
            'drone_freeze': fd_sched['drone_freeze'] if not drone_fd_flag else fd_min['drone_freeze'],
            'drone_blank':  fd_sched['drone_blank']  if not drone_fd_flag else fd_min['drone_blank'],
            'static_freeze':fd_sched['static_freeze']if not static_fd_flag else fd_min['static_freeze'],
            'static_blank': fd_sched['static_blank'] if not static_fd_flag else fd_min['static_blank'],
        }
        self.task.infos["curriculum/camera_frame_dropout_drone_total"] = torch.tensor(eff["drone_total"], dtype=torch.float32)
        self.task.infos["curriculum/camera_frame_dropout_static_total"] = torch.tensor(eff["static_total"], dtype=torch.float32)
        self.task.infos["curriculum/camera_frame_freeze_drone"] = torch.tensor(eff["drone_freeze"], dtype=torch.float32)
        self.task.infos["curriculum/camera_frame_blank_drone"] = torch.tensor(eff["drone_blank"], dtype=torch.float32)
        self.task.infos["curriculum/camera_frame_freeze_static"] = torch.tensor(eff["static_freeze"], dtype=torch.float32)
        self.task.infos["curriculum/camera_frame_blank_static"] = torch.tensor(eff["static_blank"], dtype=torch.float32)

        # Add camera angle metrics
        self.task.infos["curriculum/camera_max_angle"] = torch.tensor(self.task.max_camera_angle, dtype=torch.float32)
        # Use first environment's angle as representative for wandb tracking
        current_angle = 0.0
        if hasattr(self.task.static_camera_manager, 'current_camera_angles'):
            current_angle = self.task.static_camera_manager.current_camera_angles[0] if self.task.static_camera_manager.current_camera_angles else 0.0
        self.task.infos["curriculum/camera_current_angle"] = torch.tensor(current_angle, dtype=torch.float32)
        # Track ablation flag in infos
        try:
            cam_orient_disabled = bool(self.task.sim_env.global_tensor_dict.get('static_camera_randomization/orientation_disabled', False))
        except (KeyError, TypeError):
            cam_orient_disabled = False
        self.task.infos["curriculum/camera_orientation_randomization_disabled"] = torch.tensor(1.0 if cam_orient_disabled else 0.0, dtype=torch.float32)

        # Add state noise metrics
        if self.task.task_config.curriculum.enable_state_noise:
            sn = self.task.task_config.curriculum.get_state_noise(self.task.curriculum_level)
            self.task.infos["curriculum/state_noise_drone_pos_std_m"] = torch.tensor(sn["drone_pos_std_m"], dtype=torch.float32)
            self.task.infos["curriculum/state_noise_drone_orient_std_deg"] = torch.tensor(sn["drone_orient_std_rad"]*57.2958, dtype=torch.float32)
            self.task.infos["curriculum/state_noise_static_pos_std_m"] = torch.tensor(sn["static_pos_std_m"], dtype=torch.float32)
            self.task.infos["curriculum/state_noise_static_orient_std_deg"] = torch.tensor(sn["static_orient_std_rad"]*57.2958, dtype=torch.float32)



