from __future__ import annotations

import math
import os

import numpy as np
import torch
from isaacgym import gymapi

from aerial_gym.env_manager.env_manager import EnvManager
from aerial_gym.sensors.static_camera_angle_computation import (
    apply_camera_transform,
    compute_spawn_aware_angle,
    compute_yaw_sweep_angle,
    read_ablation_flags_and_robot_positions,
    resample_jitter_for_env,
    resolve_base_camera_position,
)
from aerial_gym.sensors.static_camera_capture import (
    capture_images_batched,
    capture_images_single,
    generate_synthetic_camera_data,
)
from aerial_gym.sensors.static_camera_follow_modes import (
    update_arc_follow as _update_arc_follow,
)
from aerial_gym.sensors.static_camera_follow_modes import (
    update_dynamic_camera_following as _update_dynamic_following,
)
from aerial_gym.sensors.static_camera_follow_modes import (
    update_locked_follow as _update_locked_follow,
)
from aerial_gym.task.task_config_protocol import TaskConfig
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("static_camera_manager")


class StaticCameraManager:
    """Manages static camera for gate navigation using Isaac Gym native API."""

    def __init__(self, env_manager: EnvManager, task_config: TaskConfig) -> None:
        self.env_manager = env_manager
        self.task_config = task_config
        self.gym = env_manager.IGE_env.gym
        self.sim = env_manager.IGE_env.sim
        self.env_handles: list = env_manager.IGE_env.env_handles
        self.camera_handles: list = []
        self.camera_setup_success: bool = False
        self.use_synthetic_camera: bool = False
        self.device: str | torch.device = env_manager.device

        self.gate_position: list[float] = [0.0, 0.0, 0.0]
        self.env_bounds: list[list[float]] = [[-4.0, -4.0, 0.0], [4.0, 4.0, 4.0]]

        self.num_envs: int = len(self.env_handles)
        self.current_camera_angles: list[float] = [0.0] * self.num_envs
        self.last_camera_pos: list[tuple[float, float, float]] = [
            (0.0, -3.0, 1.5) for _ in range(self.num_envs)
        ]
        self.last_camera_target: list[tuple[float, float, float]] = [
            (0.0, 0.0, 1.5) for _ in range(self.num_envs)
        ]
        self.last_angle_deg: list[float] = [0.0] * self.num_envs

        self.static_cam_randomize: bool = bool(task_config.static_camera_randomize_placement)
        self.static_cam_min_t: list[float] = list(task_config.static_camera_min_translation)
        self.static_cam_max_t: list[float] = list(task_config.static_camera_max_translation)
        self.static_cam_min_euler: list[float] = list(task_config.static_camera_min_euler_deg)
        self.static_cam_max_euler: list[float] = list(task_config.static_camera_max_euler_deg)
        self._trans_jitter: list[tuple[float, float, float]] = [
            (0.0, 0.0, 0.0) for _ in range(self.num_envs)
        ]
        self._euler_jitter_deg: list[tuple[float, float, float]] = [
            (0.0, 0.0, 0.0) for _ in range(self.num_envs)
        ]

        self.sweep_phase_offsets: list[float] = [0.0] * self.num_envs
        self.sweep_directions: list[float] = [1.0] * self.num_envs
        self._lazy_setup_attempted: bool = False

        self._setup_static_camera()

    def get_average_camera_angle(self) -> float:
        """Get average camera angle across all environments for logging."""
        if not self.current_camera_angles:
            return 0.0
        return sum(self.current_camera_angles) / len(self.current_camera_angles)

    def _setup_static_camera(self) -> None:
        """Setup static camera using Isaac Gym native camera API with D455 specifications."""
        logger.info("Setting up static camera for gate navigation...")

        if self.task_config.headless:
            logger.info("Headless mode detected - proceeding with static camera GPU setup")

        try:
            camera_props = self._create_camera_properties()
            self._create_camera_sensors(camera_props)
            base_y, adaptive_z, base_z_value = self._resolve_initial_base_position()
            gate_center_per_env = self._read_gate_center_heights()
            self._position_cameras_initial(base_y, adaptive_z, base_z_value, gate_center_per_env)
            self._cache_base_values(base_y, adaptive_z, base_z_value)
            self._finalize_setup()
        except RuntimeError as e:
            logger.warning(f"Static camera setup failed: {e}")
            self.camera_setup_success = False
            self.use_synthetic_camera = False

    def _create_camera_properties(self) -> gymapi.CameraProperties:
        """Create D455 depth camera properties."""
        camera_props = gymapi.CameraProperties()
        camera_props.width = 240
        camera_props.height = 135
        camera_props.horizontal_fov = 87.0
        camera_props.near_plane = 0.4
        camera_props.far_plane = 20.0
        camera_props.enable_tensors = True
        logger.info(
            f"Static camera properties (D455 specs): {camera_props.width}x{camera_props.height}, "
            f"FOV: {camera_props.horizontal_fov}"
        )
        return camera_props

    def _create_camera_sensors(self, camera_props: gymapi.CameraProperties) -> None:
        """Create camera sensor in each environment."""
        self.camera_handles = []
        for i, env_handle in enumerate(self.env_handles):
            cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            if cam_handle >= 0:
                self.camera_handles.append(cam_handle)
                logger.info(f"Created static camera sensor {i} in environment {i}")
            else:
                logger.warning(f"Failed to create camera for environment {i}, handle: {cam_handle}")

    def _resolve_initial_base_position(self) -> tuple[float, bool, float]:
        """Resolve base_y, adaptive_z flag, and base_z_value from config/env vars."""
        base_y = float(self.task_config.static_camera_base_y)

        adaptive_z = False
        base_z_value = 1.5
        cfg_base_z = self.task_config.static_camera_base_z
        if cfg_base_z is not None:
            if isinstance(cfg_base_z, str) and cfg_base_z.strip().lower() == "adaptive":
                adaptive_z = True
            else:
                base_z_value = float(cfg_base_z)
        else:
            env_base_z = os.environ.get("SF_STATIC_CAMERA_BASE_Z", None)
            if env_base_z is not None and str(env_base_z).strip().lower() == "adaptive":
                adaptive_z = True
            elif env_base_z is not None:
                base_z_value = float(env_base_z)

        return base_y, adaptive_z, base_z_value

    def _read_gate_center_heights(self) -> torch.Tensor | None:
        """Read per-env adaptive gate center Z from global tensors."""
        try:
            gtd = self.env_manager.IGE_env.global_tensor_dict
            return gtd.get("gate/center_height_per_env", None)
        except (KeyError, TypeError, AttributeError):
            return None

    def _position_cameras_initial(
        self,
        base_y: float,
        adaptive_z: bool,
        base_z_value: float,
        gate_center_per_env: torch.Tensor | None,
    ) -> None:
        """Set initial camera transform for each environment."""
        for i, (env_handle, cam_handle) in enumerate(zip(self.env_handles, self.camera_handles)):
            if self.static_cam_randomize:
                t, e = resample_jitter_for_env(
                    i,
                    self.device,
                    self.static_cam_min_t,
                    self.static_cam_max_t,
                    self.static_cam_min_euler,
                    self.static_cam_max_euler,
                )
                self._trans_jitter[i] = t
                self._euler_jitter_deg[i] = e

            if adaptive_z and gate_center_per_env is not None and i < len(gate_center_per_env):
                env_base_z = float(gate_center_per_env[i].item())
            else:
                env_base_z = float(base_z_value)

            jx, jy, jz = self._trans_jitter[i]
            camera_pos = gymapi.Vec3(0.0 + jx, float(base_y) + jy, env_base_z + jz)
            camera_target = gymapi.Vec3(0.0, 0.0, env_base_z)
            self.gym.set_camera_location(cam_handle, env_handle, camera_pos, camera_target)

            self.last_camera_pos[i] = (
                float(camera_pos.x),
                float(camera_pos.y),
                float(camera_pos.z),
            )
            self.last_camera_target[i] = (
                float(camera_target.x),
                float(camera_target.y),
                float(camera_target.z),
            )
            self.last_angle_deg[i] = 0.0

    def _cache_base_values(self, base_y: float, adaptive_z: bool, base_z_value: float) -> None:
        """Cache base values into global tensor dict for downstream updates."""
        try:
            gtd = self.env_manager.IGE_env.global_tensor_dict
            gtd["static_camera/base_y"] = float(base_y)
            gtd["static_camera/base_z"] = "adaptive" if adaptive_z else float(base_z_value)
        except AttributeError:
            pass

    def _finalize_setup(self) -> None:
        """Check setup completeness and log result."""
        if len(self.camera_handles) == len(self.env_handles):
            logger.info("Static camera setup complete with configured positioning")
            self.camera_setup_success = True
            self.use_synthetic_camera = False
        else:
            logger.error(
                f"Static camera setup incomplete: "
                f"created {len(self.camera_handles)}/{len(self.env_handles)} cameras"
            )
            self.camera_setup_success = False
            self.use_synthetic_camera = False

    def _update_synthetic_camera_angles(self, curriculum_level: int, env_ids: torch.Tensor) -> None:
        """Update camera angles for synthetic camera mode."""
        max_angle_range, _, _ = self.task_config.curriculum.get_static_camera_difficulty(
            curriculum_level
        )
        disable_flag, rp = read_ablation_flags_and_robot_positions(self.env_manager)
        import random

        horizontal_fov = 87.0
        half_fov = horizontal_fov * 0.5
        margin = 5.0
        for env_idx in env_ids:
            if env_idx >= len(self.current_camera_angles):
                continue
            if disable_flag or max_angle_range <= 0:
                ang = 0.0
            elif rp is not None and env_idx < rp.shape[0]:
                ang = self._compute_synthetic_fov_angle(
                    env_idx, rp, half_fov, margin, max_angle_range
                )
            else:
                ang = random.uniform(-max_angle_range, max_angle_range)
            self.current_camera_angles[env_idx] = ang
        logger.debug(f"Synthetic camera mode - updated angles for envs {env_ids.tolist()}")

    def _compute_synthetic_fov_angle(
        self,
        env_idx: int,
        rp: torch.Tensor,
        half_fov: float,
        margin: float,
        max_angle_range: float,
    ) -> float:
        """Compute FOV-constrained angle for synthetic camera mode."""
        import random

        cam_x, cam_y = 0.0, -3.0
        dx = float(rp[env_idx, 0].item()) - cam_x
        dy = float(rp[env_idx, 1].item()) - cam_y
        theta_r = math.degrees(math.atan2(dx, dy))
        gate_low, gate_high = -half_fov + margin, half_fov - margin
        rob_low = theta_r - (half_fov - margin)
        rob_high = theta_r + (half_fov - margin)
        low = max(gate_low, rob_low, -max_angle_range)
        high = min(gate_high, rob_high, max_angle_range)
        if high > low:
            return random.uniform(low, high)
        target = max(min(theta_r, gate_high), gate_low)
        return max(-max_angle_range, min(max_angle_range, target))

    def update_camera_positions(self, curriculum_level: int, env_ids: torch.Tensor) -> None:
        """Update static camera orientation ONLY for resetting environments."""
        if self.use_synthetic_camera:
            self._update_synthetic_camera_angles(curriculum_level, env_ids)
            return

        if not self.camera_setup_success or len(self.camera_handles) == 0:
            return

        _max_angle_range, _, _ = self.task_config.curriculum.get_static_camera_difficulty(
            curriculum_level
        )
        disable_flag, rp = read_ablation_flags_and_robot_positions(self.env_manager)

        try:
            base_camera_pos, base_z, gtd = resolve_base_camera_position(self.env_manager)

            for env_idx in env_ids:
                if env_idx >= len(self.env_handles) or env_idx >= len(self.camera_handles):
                    continue

                if self.static_cam_randomize:
                    t, e = resample_jitter_for_env(
                        env_idx,
                        self.device,
                        self.static_cam_min_t,
                        self.static_cam_max_t,
                        self.static_cam_min_euler,
                        self.static_cam_max_euler,
                    )
                    self._trans_jitter[env_idx] = t
                    self._euler_jitter_deg[env_idx] = e

                sweep_enabled, sweep_speed_deg = self._read_sweep_config()

                if sweep_enabled:
                    angle_offset, debug_max = compute_yaw_sweep_angle(
                        env_idx,
                        curriculum_level,
                        sweep_speed_deg,
                        gtd,
                        self.device,
                        self.env_manager,
                        self.sweep_phase_offsets,
                        self.sweep_directions,
                    )
                else:
                    angle_offset, debug_max = compute_spawn_aware_angle(
                        env_idx,
                        curriculum_level,
                        _max_angle_range,
                        base_camera_pos,
                        disable_flag,
                        rp,
                        base_camera_pos.y,
                        self.device,
                        self.env_manager,
                    )

                apply_camera_transform(
                    env_idx,
                    angle_offset,
                    sweep_enabled,
                    disable_flag,
                    debug_max,
                    base_camera_pos,
                    base_z,
                    gtd,
                    self.gym,
                    self.env_handles,
                    self.camera_handles,
                    self.current_camera_angles,
                    self._euler_jitter_deg,
                    self._trans_jitter,
                    self.last_camera_pos,
                    self.last_camera_target,
                    self.last_angle_deg,
                )

        except RuntimeError as e:
            logger.warning(f"Failed to update static camera orientation: {e}")

    def _read_sweep_config(self) -> tuple[bool, float]:
        """Read yaw sweep configuration from global tensor dict."""
        try:
            gtd = self.env_manager.global_tensor_dict
            sweep_enabled = (
                str(gtd.get("static_camera/yaw_sweep_enabled", "false")).lower() == "true"
            )
            sweep_speed_deg = float(gtd.get("static_camera/yaw_sweep_speed_deg", 10.0))
        except (ValueError, TypeError, AttributeError):
            sweep_enabled = False
            sweep_speed_deg = 10.0
        return sweep_enabled, sweep_speed_deg

    def update_dynamic_camera_following(
        self,
        robot_positions: torch.Tensor,
        gate_positions: torch.Tensor,
        gate_center_heights: torch.Tensor,
    ) -> None:
        """Delegate to follow-mode helper for dynamic camera following."""
        if self.use_synthetic_camera or not self.camera_setup_success or not self.camera_handles:
            return
        _update_dynamic_following(
            robot_positions,
            gate_positions,
            gate_center_heights,
            self.gym,
            self.sim,
            self.env_handles,
            self.camera_handles,
            self.env_manager,
            self.last_camera_pos,
            self.last_camera_target,
            self.last_angle_deg,
        )

    def update_arc_follow(
        self,
        robot_positions: torch.Tensor,
        gate_positions: torch.Tensor,
        gate_center_heights: torch.Tensor,
        radius_m: float = 2.0,
    ) -> None:
        """Delegate to follow-mode helper for arc follow."""
        if self.use_synthetic_camera or not self.camera_setup_success or not self.camera_handles:
            return
        _update_arc_follow(
            robot_positions,
            gate_positions,
            gate_center_heights,
            radius_m,
            self.gym,
            self.sim,
            self.env_handles,
            self.camera_handles,
            self.env_manager,
        )

    def update_locked_follow(self, robot_positions: torch.Tensor) -> None:
        """Delegate to follow-mode helper for locked follow."""
        if self.use_synthetic_camera or not self.camera_setup_success or not self.camera_handles:
            return
        _update_locked_follow(
            robot_positions,
            self.gym,
            self.sim,
            self.env_handles,
            self.camera_handles,
            self.env_manager,
        )

    def capture_images(self, batched: bool = False) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Capture depth and segmentation images from static camera."""
        if self.use_synthetic_camera:
            logger.error("Static camera synthetic mode disabled; no images will be generated.")
            return None, None

        if not self.camera_setup_success or len(self.camera_handles) == 0:
            return self._attempt_lazy_capture()

        try:
            if batched:
                return capture_images_batched(
                    self.gym, self.sim, self.env_handles, self.camera_handles
                )
            return capture_images_single(self.gym, self.sim, self.env_handles, self.camera_handles)
        except RuntimeError as e:
            logger.error(f"Static camera capture error: {e}")
            return None, None

    def _attempt_lazy_capture(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Attempt one-time lazy re-initialization of camera setup."""
        if not self._lazy_setup_attempted:
            self._lazy_setup_attempted = True
            logger.warning("Static camera not set up; attempting one-time lazy initialization")
            try:
                self._setup_static_camera()
            except RuntimeError as e:
                logger.warning(f"Lazy static camera setup attempt failed: {e}")
            if self.camera_setup_success and len(self.camera_handles) > 0:
                return self.capture_images()
        logger.error("Static camera not set up; returning no images.")
        return None, None

    def _generate_synthetic_camera_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic camera data for headless training."""
        return generate_synthetic_camera_data()
