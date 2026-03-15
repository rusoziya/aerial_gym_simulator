from __future__ import annotations

import os

import torch

from aerial_gym.task.navigation_task_gate.curriculum_infos import CurriculumInfos
from aerial_gym.utils.env_flag_utils import read_env_bool
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("navigation_task_gate_curriculum_logging")

RAD_TO_DEG: float = 57.2958


class CurriculumLogging:
    def __init__(self, task: object) -> None:
        self.task = task
        self._infos = CurriculumInfos(self)

    def _get_gtd(self) -> dict[str, object]:
        return self.task.sim_env.global_tensor_dict

    def _get_obs_disabled(self) -> bool:
        return bool(self._get_gtd().get("obstacles_randomization/disabled", False))

    def _get_fixed_assets_visible(self) -> int:
        visible_gates = 1
        walls = 6
        robot = 1
        return visible_gates + walls + robot

    def _get_yaw_sweep_state(self) -> tuple[bool, float]:
        try:
            enabled = (
                str(os.environ.get("SF_ENABLE_STATIC_CAMERA_YAW_SWEEP", "false")).lower() == "true"
            )
            speed = float(os.environ.get("SF_STATIC_CAMERA_YAW_SWEEP_SPEED_DEG", "10.0"))
        except (ValueError, TypeError):
            enabled = False
            speed = 10.0
        return enabled, speed

    def _is_dynamic_effective(self) -> bool:
        dyn_cfg = self.task.task_config.curriculum.enable_dynamic_camera_following
        dyn_dis = bool(self._get_gtd().get("dynamic_camera_following/disabled", False))
        return bool(dyn_cfg and not dyn_dis)

    def _get_current_camera_angle(self) -> float:
        scm = self.task.static_camera_manager
        if scm.current_camera_angles:
            return scm.current_camera_angles[0]
        return 0.0

    def _get_spawn_ranges(self) -> dict[str, float] | None:
        """Compute effective spawn ranges respecting ablation flags. Returns None on error."""
        try:
            baseline_level = int(self.task.task_config.curriculum.min_level)
            gtd = self._get_gtd()
            pos_dis = bool(gtd.get("spawn_randomization/position_disabled", False))
            yaw_dis = bool(gtd.get("spawn_randomization/orientation_disabled", False))
            sr_active = self.task.task_config.curriculum.get_spawn_ranges(
                self.task.curriculum_level
            )
            sr_base = self.task.task_config.curriculum.get_spawn_ranges(baseline_level)
            return {
                "x_half_span_m": (
                    sr_base["x_half_span_m"] if pos_dis else sr_active["x_half_span_m"]
                ),
                "y_center_m": sr_base["y_center_m"] if pos_dis else sr_active["y_center_m"],
                "y_half_span_m": (
                    sr_base["y_half_span_m"] if pos_dis else sr_active["y_half_span_m"]
                ),
                "z_center_m": sr_base["z_center_m"] if pos_dis else sr_active["z_center_m"],
                "z_half_span_m": (
                    sr_base["z_half_span_m"] if pos_dis else sr_active["z_half_span_m"]
                ),
                "yaw_abs_rad": sr_base["yaw_abs_rad"] if yaw_dis else sr_active["yaw_abs_rad"],
                "pos_disabled": pos_dis,
                "yaw_disabled": yaw_dis,
            }
        except (ValueError, TypeError):
            return None

    def _get_camera_noise_effective(
        self,
    ) -> tuple[float, float, float, float, float, float]:
        """Return (gaussian_std, dropout_rate, eff_drone_std, eff_static_std, eff_drone_drop, eff_static_drop)."""
        gaussian_std, dropout_rate = self.task.task_config.curriculum.get_camera_noise(
            self.task.curriculum_level
        )
        gtd = self._get_gtd()
        cam_noise_disabled = bool(gtd.get("camera_randomization/noise_disabled", False))
        drone_noise_flag = bool(
            gtd.get("camera_randomization/drone_noise_disabled", cam_noise_disabled)
        )
        static_noise_flag = bool(
            gtd.get("camera_randomization/static_noise_disabled", cam_noise_disabled)
        )
        d_std_min, d_drop_min = self.task.task_config.curriculum.get_camera_noise(3)
        return (
            gaussian_std,
            dropout_rate,
            gaussian_std if not drone_noise_flag else d_std_min,
            gaussian_std if not static_noise_flag else d_std_min,
            dropout_rate if not drone_noise_flag else d_drop_min,
            dropout_rate if not static_noise_flag else d_drop_min,
        )

    def _get_frame_dropout_effective(
        self,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return (effective_fd, scheduled_fd) dicts with per-camera frame dropout rates."""
        fd = self.task.task_config.curriculum.get_camera_frame_dropout(self.task.curriculum_level)
        gtd = self._get_gtd()
        global_fd_dis = bool(gtd.get("camera_randomization/frame_dropout_disabled", False))
        drone_fd_flag = bool(
            gtd.get("camera_randomization/drone_frame_dropout_disabled", global_fd_dis)
        )
        static_fd_flag = bool(
            gtd.get("camera_randomization/static_frame_dropout_disabled", global_fd_dis)
        )
        fd_min = self.task.task_config.curriculum.get_camera_frame_dropout(3)
        eff = {
            "drone_total": fd["drone_total"] if not drone_fd_flag else fd_min["drone_total"],
            "static_total": fd["static_total"] if not static_fd_flag else fd_min["static_total"],
            "drone_freeze": fd["drone_freeze"] if not drone_fd_flag else fd_min["drone_freeze"],
            "drone_blank": fd["drone_blank"] if not drone_fd_flag else fd_min["drone_blank"],
            "static_freeze": fd["static_freeze"] if not static_fd_flag else fd_min["static_freeze"],
            "static_blank": fd["static_blank"] if not static_fd_flag else fd_min["static_blank"],
        }
        return eff, fd

    def _to_tensor_scalar(self, val: object) -> bool:
        """Safely convert a value (possibly a tensor) to bool."""
        if isinstance(val, torch.Tensor):
            return bool(val.item())
        return bool(val)

    def _log_curriculum_details(
        self,
        success_rate: float,
        crash_rate: float,
        timeout_rate: float,
        obstacles_behind_gate: int,
        total_obstacles_in_env: int,
    ) -> None:
        """Log comprehensive curriculum state after level update."""
        log = self.task.log_curriculum_update
        log(
            f"Gate Navigation Curriculum Level: {self.task.curriculum_level}, "
            f"Progress: {self.task.curriculum_progress_fraction:.3f}"
        )
        log(
            f"\nSuccess Rate: {success_rate:.3f}\nCrash Rate: {crash_rate:.3f}\nTimeout Rate: {timeout_rate:.3f}"
        )
        log("\nCURRICULUM APPLIED:")

        self._log_yaw_sweep_status()
        self._log_arc_follow_status()
        self._log_obstacle_status(obstacles_behind_gate, total_obstacles_in_env)
        self._log_spawn_ranges()
        self._log_camera_angle_status()
        self._log_gate_size_unlocks()
        self._log_camera_noise_status()
        self._log_frame_dropout_status()
        self._log_state_noise_status()
        self._log_dynamic_camera_status()
        self._log_curriculum_multiplier_status()
        self._log_final_state(obstacles_behind_gate, total_obstacles_in_env)

    def _log_yaw_sweep_status(self) -> None:
        yaw_enabled, yaw_speed = self._get_yaw_sweep_state()
        dynamic_effective = self._is_dynamic_effective()
        log = self.task.log_curriculum_update
        if yaw_enabled and not dynamic_effective:
            log(f"   3. STATIC CAMERA YAW SWEEP: ENABLED (speed={yaw_speed:.1f} deg/s)")
        elif yaw_enabled and dynamic_effective:
            log("   3. STATIC CAMERA YAW SWEEP: ENABLED but IGNORED (dynamic camera active)")
        else:
            log("   3. STATIC CAMERA YAW SWEEP: DISABLED")

    def _log_arc_follow_status(self) -> None:
        gtd = self._get_gtd()
        arc_follow_enabled = bool(gtd.get("static_camera/arc_follow_enabled", False))
        if arc_follow_enabled:
            arc_radius = float(gtd.get("static_camera/arc_follow_radius_m", 2.0))
            self.task.log_curriculum_update(
                f"   3b. STATIC CAMERA ARC-FOLLOW: ENABLED (radius={arc_radius:.1f} m)"
            )

    def _log_obstacle_status(self, obstacles_behind_gate: int, total_obstacles_in_env: int) -> None:
        obs_dis = self._get_obs_disabled()
        fixed_assets_visible = self._get_fixed_assets_visible()
        log = self.task.log_curriculum_update
        if obs_dis:
            log(
                f"   1. OBSTACLES: fixed to {obstacles_behind_gate} behind gate "
                f"(total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)"
            )
        else:
            log(
                f"   1. OBSTACLES: {obstacles_behind_gate} behind gate "
                f"(total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)"
            )

    def _log_spawn_ranges(self) -> None:
        log = self.task.log_curriculum_update
        sr = self._get_spawn_ranges()
        if sr is None:
            log("   2. SPAWN: (fallback) Using fixed LMF2 config")
            return
        if sr["pos_disabled"] or sr["yaw_disabled"]:
            status_pos = "DISABLED" if sr["pos_disabled"] else "ENABLED"
            status_yaw = "DISABLED" if sr["yaw_disabled"] else "ENABLED"
            log(f"   2. SPAWN RANDOMIZATION: position={status_pos}, orientation={status_yaw}")
        log(
            f"   2. SPAWN: X∈[{(-sr['x_half_span_m']):.1f}, {(+sr['x_half_span_m']):.1f}] m, "
            f"Y∈[{(sr['y_center_m'] - sr['y_half_span_m']):.1f}, {(sr['y_center_m'] + sr['y_half_span_m']):.1f}] m, "
            f"Z∈[{(sr['z_center_m'] - sr['z_half_span_m']):.1f}, {(sr['z_center_m'] + sr['z_half_span_m']):.1f}] m; "
            f"yaw ±{(sr['yaw_abs_rad'] * RAD_TO_DEG):.1f}°"
        )

    def _log_camera_angle_status(self) -> None:
        log = self.task.log_curriculum_update
        yaw_enabled, _ = self._get_yaw_sweep_state()
        dynamic_effective = self._is_dynamic_effective()
        current_angle = self._get_current_camera_angle()
        cam_orient_disabled = bool(
            self._get_gtd().get("static_camera_randomization/orientation_disabled", False)
        )
        if yaw_enabled and not dynamic_effective:
            log(f"   4. CAMERA ANGLE: overridden by yaw sweep (env0 current: {current_angle:.1f}°)")
        elif dynamic_effective:
            log("   4. CAMERA ANGLE: suppressed (dynamic camera following active)")
        elif cam_orient_disabled:
            log(
                f"   4. CAMERA ANGLE: randomization DISABLED, fixed at 0.0° (env0: {current_angle:.1f}°)"
            )
        else:
            log(
                f"   4. CAMERA ANGLE: ±{self.task.max_camera_angle:.1f}deg max range, "
                f"env0: {current_angle:.1f}deg (fixed per episode)"
            )

    def _log_gate_size_unlocks(self) -> None:
        log = self.task.log_curriculum_update
        gtd = self._get_gtd()
        gate_names: list[str] = []
        names_per_env = gtd.get("gate_variant_names_per_env", [])
        if len(names_per_env) > 0:
            gate_names = names_per_env[0]

        disable_flag = self._to_tensor_scalar(gtd.get("gate_randomization/disabled", False))
        if disable_flag:
            fixed_scale = self._to_tensor_scalar(
                gtd.get("gate_randomization/fixed_scale_percent", 100)
            )
            log(f"   4. GATE SIZE: randomization disabled, fixed scale = {int(fixed_scale)}%")
        else:
            min_scale = self._compute_min_gate_scale()
            scales = self._collect_unlocked_scales(gate_names, min_scale)
            log(
                f"   4. GATE SIZE: unlocked scales >= {min_scale}% -> "
                f"{scales if scales else [100]} (uniform across unique scales)"
            )

    def _compute_min_gate_scale(self) -> int:
        stretch_enabled = os.environ.get("EVAL_STRETCH_ENABLED", "0").strip() in (
            "1",
            "true",
            "True",
        )
        eval_end = int(
            os.environ.get(
                "EVAL_STRETCH_END_LEVEL",
                str(self.task.task_config.curriculum.eval_stretch_end_level),
            )
        )
        level = int(self.task.curriculum_level)
        if level <= 3:
            min_scale = 80
        elif level <= 23:
            frac = (level - 3) / (23 - 3)
            raw = 80 - frac * (80 - 60)
            min_scale = int((int(raw) // 2) * 2)
        elif stretch_enabled:
            if level >= eval_end:
                min_scale = 50
            else:
                extra_frac = (level - 23) / max(1, (eval_end - 23))
                raw = 60 - extra_frac * (60 - 50)
                min_scale = int((int(raw) // 2) * 2)
        else:
            min_scale = 60
        return max(50, min(100, min_scale))

    def _collect_unlocked_scales(self, gate_names: list[str], min_scale: int) -> list[int]:
        scales: list[int] = []
        for n in gate_names:
            if isinstance(n, str) and "gate_scale_" in n:
                try:
                    s = int(n.replace("gate_scale_", ""))
                    if s >= min_scale:
                        scales.append(s)
                except (ValueError, TypeError):
                    pass
        return sorted(set(scales), reverse=True)

    def _log_camera_noise_status(self) -> None:
        _, _, eff_drone_std, eff_static_std, eff_drone_drop, eff_static_drop = (
            self._get_camera_noise_effective()
        )
        self.task.log_curriculum_update(
            f"   5. CAMERA NOISE: drone(std={eff_drone_std:.4f}, pixel_drop={eff_drone_drop * 100:.1f}%), "
            f"static(std={eff_static_std:.4f}, pixel_drop={eff_static_drop * 100:.1f}%)"
        )

    def _log_frame_dropout_status(self) -> None:
        eff, _ = self._get_frame_dropout_effective()
        self.task.log_curriculum_update(
            f"   6. CAMERA FRAME DROPOUT: drone_total={eff['drone_total'] * 100:.1f}% "
            f"(freeze {eff['drone_freeze'] * 100:.1f}%, blank {eff['drone_blank'] * 100:.1f}%), "
            f"static_total={eff['static_total'] * 100:.1f}% "
            f"(freeze {eff['static_freeze'] * 100:.1f}%, blank {eff['static_blank'] * 100:.1f}%)"
        )

    def _log_state_noise_status(self) -> None:
        log = self.task.log_curriculum_update
        if not self.task.task_config.curriculum.enable_state_noise:
            log("   7. STATE NOISE: disabled")
            return
        state_noise_disabled = bool(
            self._get_gtd().get("state_randomization/noise_disabled", False)
        )
        if state_noise_disabled:
            log("   7. STATE NOISE: DISABLED (all std=0)")
        else:
            sn = self.task.task_config.curriculum.get_state_noise(self.task.curriculum_level)
            log(
                f"   7. STATE NOISE: drone_pos_std={sn['drone_pos_std_m']:.4f} m, "
                f"drone_orient_std={sn['drone_orient_std_rad'] * RAD_TO_DEG:.3f} deg, "
                f"static_pos_std={sn['static_pos_std_m']:.4f} m, "
                f"static_orient_std={sn['static_orient_std_rad'] * RAD_TO_DEG:.3f} deg"
            )

    def _log_dynamic_camera_status(self) -> None:
        log = self.task.log_curriculum_update
        gtd = self._get_gtd()
        dynamic_enabled = self.task.task_config.curriculum.enable_dynamic_camera_following
        dynamic_disabled = bool(gtd.get("dynamic_camera_following/disabled", False))
        config_overridden = bool(gtd.get("dynamic_camera_following/config_overridden", False))

        if dynamic_enabled and not dynamic_disabled:
            suffix = (
                " by flag (--enable_dynamic_camera_following=true)" if config_overridden else ""
            )
            log(
                f"   8. DYNAMIC CAMERA: ENABLED{suffix} - camera follows drone with adaptive gate targeting"
            )
        elif dynamic_enabled and dynamic_disabled:
            log("   8. DYNAMIC CAMERA: DISABLED by flag (--disable_dynamic_camera_following=true)")
        elif config_overridden:
            log("   8. DYNAMIC CAMERA: DISABLED by flag (--enable_dynamic_camera_following=false)")
        else:
            log(
                "   8. DYNAMIC CAMERA: DISABLED (static camera mode - curriculum-based positioning)"
            )

    def _log_curriculum_multiplier_status(self) -> None:
        cm_disabled = read_env_bool(
            "SF_DISABLE_CURRICULUM_MULTIPLIER", self.task.task_config.disable_curriculum_multiplier
        )
        if not cm_disabled:
            cm_disabled = bool(self.task.task_config.disable_curriculum_multiplier)
        frac_eff = 0.0 if cm_disabled else float(self.task.curriculum_progress_fraction)
        factor = 1.0 + 0.5 * frac_eff
        self.task.log_curriculum_update(
            f"   8. CURRICULUM MULTIPLIER: {'DISABLED' if cm_disabled else 'ENABLED'} (factor={factor:.3f})"
        )

    def _log_final_state(self, obstacles_behind_gate: int, total_obstacles_in_env: int) -> None:
        log = self.task.log_curriculum_update
        fixed_assets_visible = self._get_fixed_assets_visible()
        log("[CURRICULUM UPDATE] FINAL STATE:")
        log(
            f"[CURRICULUM UPDATE]   Level: {self.task.curriculum_level} "
            f"(range: {self.task.task_config.curriculum.min_level}-{self.task.task_config.curriculum.max_level})"
        )
        log(
            f"[CURRICULUM UPDATE]   Max level reached: {self.task.max_curriculum_level_reached} (DECREASE ENABLED)"
        )
        log(f"[CURRICULUM UPDATE]   Progress: {self.task.curriculum_progress_fraction:.3f}")
        log(
            f"[CURRICULUM UPDATE]   Obstacles behind gate: {obstacles_behind_gate} "
            f"(total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)"
        )
        log(
            f"[CURRICULUM UPDATE]   Asset manager: Updated both obs_dict and global_tensor_dict with count {total_obstacles_in_env}"
        )

        self._log_final_spawn_status()
        self._log_final_camera_angle_status()

    def _log_final_spawn_status(self) -> None:
        log = self.task.log_curriculum_update
        sr = self._get_spawn_ranges()
        if sr is None:
            log("[CURRICULUM UPDATE]   Spawn difficulty: LMF2 config (fallback)")
            return
        status_pos = "DISABLED" if sr["pos_disabled"] else "ENABLED"
        status_yaw = "DISABLED" if sr["yaw_disabled"] else "ENABLED"
        log(
            f"[CURRICULUM UPDATE]   Spawn randomization: position={status_pos}, orientation={status_yaw}"
        )
        log(
            f"[CURRICULUM UPDATE]   Spawn ranges: X∈[{(-sr['x_half_span_m']):.1f}, {(+sr['x_half_span_m']):.1f}] m, "
            f"Y∈[{(sr['y_center_m'] - sr['y_half_span_m']):.1f}, {(sr['y_center_m'] + sr['y_half_span_m']):.1f}] m, "
            f"Z∈[{(sr['z_center_m'] - sr['z_half_span_m']):.1f}, {(sr['z_center_m'] + sr['z_half_span_m']):.1f}] m; "
            f"yaw ±{(sr['yaw_abs_rad'] * RAD_TO_DEG):.1f}°"
        )

    def _log_final_camera_angle_status(self) -> None:
        yaw_enabled, _ = self._get_yaw_sweep_state()
        dynamic_effective = self._is_dynamic_effective()
        if not (yaw_enabled and not dynamic_effective):
            self.task.log_curriculum_update(
                f"[CURRICULUM UPDATE]   Camera angle: ±{self.task.max_camera_angle:.1f}deg max range "
                "(randomized per episode reset, fixed during episode)"
            )

    def _populate_curriculum_infos(
        self,
        success_rate: float,
        crash_rate: float,
        timeout_rate: float,
        obstacles_behind_gate: int,
        total_obstacles_in_env: int,
    ) -> None:
        """Populate self.task.infos with curriculum metrics for wandb logging."""
        self._infos.populate(
            success_rate, crash_rate, timeout_rate, obstacles_behind_gate, total_obstacles_in_env
        )
