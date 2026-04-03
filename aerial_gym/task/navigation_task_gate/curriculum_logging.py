from __future__ import annotations

from aerial_gym.task.navigation_task_gate.curriculum_data import (
    RAD_TO_DEG,
    collect_unlocked_scales,
    compute_min_gate_scale,
    get_camera_noise_effective,
    get_current_camera_angle,
    get_fixed_assets_visible,
    get_frame_dropout_effective,
    get_global_tensor_dict,
    get_obs_disabled,
    get_spawn_ranges,
    get_yaw_sweep_state,
    is_dynamic_effective,
    to_tensor_scalar,
)
from aerial_gym.task.navigation_task_gate.curriculum_infos import CurriculumInfos
from aerial_gym.utils.env_flag_utils import read_env_bool
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("navigation_task_gate_curriculum_logging")


class CurriculumLogging:
    def __init__(self, task: object) -> None:
        self.task = task
        self._infos = CurriculumInfos(self)

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
        yaw_enabled, yaw_speed = get_yaw_sweep_state()
        dynamic_effective = is_dynamic_effective(self.task)
        log = self.task.log_curriculum_update
        if yaw_enabled and not dynamic_effective:
            log(f"   3. STATIC CAMERA YAW SWEEP: ENABLED (speed={yaw_speed:.1f} deg/s)")
        elif yaw_enabled and dynamic_effective:
            log("   3. STATIC CAMERA YAW SWEEP: ENABLED but IGNORED (dynamic camera active)")
        else:
            log("   3. STATIC CAMERA YAW SWEEP: DISABLED")

    def _log_arc_follow_status(self) -> None:
        gtd = get_global_tensor_dict(self.task)
        arc_follow_enabled = bool(gtd.get("static_camera/arc_follow_enabled", False))
        if arc_follow_enabled:
            arc_radius = float(gtd.get("static_camera/arc_follow_radius_m", 2.0))
            self.task.log_curriculum_update(
                f"   3b. STATIC CAMERA ARC-FOLLOW: ENABLED (radius={arc_radius:.1f} m)"
            )

    def _log_obstacle_status(self, obstacles_behind_gate: int, total_obstacles_in_env: int) -> None:
        obs_dis = get_obs_disabled(self.task)
        fixed_assets_visible = get_fixed_assets_visible()
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
        sr = get_spawn_ranges(self.task)
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
        yaw_enabled, _ = get_yaw_sweep_state()
        dynamic_effective = is_dynamic_effective(self.task)
        current_angle = get_current_camera_angle(self.task)
        cam_orient_disabled = bool(
            get_global_tensor_dict(self.task).get(
                "static_camera_randomization/orientation_disabled", False
            )
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
        gtd = get_global_tensor_dict(self.task)
        gate_names: list[str] = []
        names_per_env = gtd.get("gate_variant_names_per_env", [])
        if len(names_per_env) > 0:
            gate_names = names_per_env[0]

        disable_flag = to_tensor_scalar(gtd.get("gate_randomization/disabled", False))
        if disable_flag:
            fixed_scale = to_tensor_scalar(gtd.get("gate_randomization/fixed_scale_percent", 100))
            log(f"   4. GATE SIZE: randomization disabled, fixed scale = {int(fixed_scale)}%")
        else:
            min_scale = compute_min_gate_scale(self.task)
            scales = collect_unlocked_scales(gate_names, min_scale)
            log(
                f"   4. GATE SIZE: unlocked scales >= {min_scale}% -> "
                f"{scales if scales else [100]} (uniform across unique scales)"
            )

    def _log_camera_noise_status(self) -> None:
        _, _, eff_drone_std, eff_static_std, eff_drone_drop, eff_static_drop = (
            get_camera_noise_effective(self.task)
        )
        self.task.log_curriculum_update(
            f"   5. CAMERA NOISE: drone(std={eff_drone_std:.4f}, pixel_drop={eff_drone_drop * 100:.1f}%), "
            f"static(std={eff_static_std:.4f}, pixel_drop={eff_static_drop * 100:.1f}%)"
        )

    def _log_frame_dropout_status(self) -> None:
        eff, _ = get_frame_dropout_effective(self.task)
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
            get_global_tensor_dict(self.task).get("state_randomization/noise_disabled", False)
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
        gtd = get_global_tensor_dict(self.task)
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
        fixed_assets_visible = get_fixed_assets_visible()
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
        sr = get_spawn_ranges(self.task)
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
        yaw_enabled, _ = get_yaw_sweep_state()
        dynamic_effective = is_dynamic_effective(self.task)
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
