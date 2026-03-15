from __future__ import annotations

import math
import os

import numpy as np
import torch

from aerial_gym.utils.env_flag_utils import read_env_bool
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import *  # noqa: F401,F403

logger = CustomLogger("navigation_task_gate_camera")


class CameraObservations:
    def __init__(self, task: object) -> None:
        self.task = task

    def process_image_observation(self) -> None:
        """Process drone camera depth image: apply curriculum noise, frame dropout, and VAE encode."""
        image_obs = self.task.obs_dict["depth_range_pixels"].squeeze(1)

        noised_image_obs = self._apply_drone_camera_noise(image_obs)
        noised_image_obs = self._apply_drone_frame_dropout(noised_image_obs)

        self.task.obs_dict["depth_range_pixels_noised"] = noised_image_obs.unsqueeze(1)

        if self.task.task_config.vae_config.use_vae:
            self._encode_drone_vae(noised_image_obs)

    def _apply_drone_camera_noise(self, image_obs: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise and pixel dropout to drone depth image."""
        noised = image_obs.clone()
        camera_noise_disabled = bool(
            self.task.sim_env.global_tensor_dict.get(
                "camera_randomization/noise_disabled", False
            )
        )
        drone_noise_override = bool(
            self.task.sim_env.global_tensor_dict.get(
                "camera_randomization/drone_noise_disabled", False
            )
        )
        if not self.task.task_config.curriculum.enable_camera_noise:
            return noised

        if not camera_noise_disabled and not drone_noise_override:
            gaussian_std, dropout_rate = self.task.task_config.curriculum.get_camera_noise(
                self.task.curriculum_level
            )
        else:
            gaussian_std, dropout_rate = self.task.task_config.curriculum.get_camera_noise(3)

        if gaussian_std > 0:
            noised = noised + torch.randn_like(noised) * gaussian_std
        if dropout_rate > 0:
            dropout_mask = torch.rand_like(noised) < dropout_rate
            noised = noised.masked_fill(dropout_mask, 1.0)
        return torch.clamp(noised, 0.0, 1.0)

    def _apply_drone_frame_dropout(self, noised: torch.Tensor) -> torch.Tensor:
        """Apply entire-frame blank/freeze dropout to drone depth image."""
        frame_dropout_disabled = bool(
            self.task.sim_env.global_tensor_dict.get(
                "camera_randomization/frame_dropout_disabled", False
            )
        )
        drone_fd_override = bool(
            self.task.sim_env.global_tensor_dict.get(
                "camera_randomization/drone_frame_dropout_disabled", False
            )
        )

        if self.task.task_config.curriculum.enable_camera_frame_dropout:
            if not frame_dropout_disabled and not drone_fd_override:
                fd = self.task.task_config.curriculum.get_camera_frame_dropout(
                    self.task.curriculum_level
                )
            else:
                fd = self.task.task_config.curriculum.get_camera_frame_dropout(3)
            p_blank = fd.get("drone_blank", 0.0)
            p_freeze = fd.get("drone_freeze", 0.0)

            if self.task._prev_drone_depth is None:
                self.task._prev_drone_depth = noised.clone()

            if p_blank > 0.0:
                blank_mask = (
                    torch.rand(noised.shape[0], device=noised.device) < p_blank
                ).view(-1, 1, 1)
                noised = torch.where(blank_mask, torch.ones_like(noised), noised)
            if p_freeze > 0.0:
                freeze_mask = (
                    torch.rand(noised.shape[0], device=noised.device) < p_freeze
                ).view(-1, 1, 1)
                apply_freeze = (
                    freeze_mask if p_blank == 0.0 else (freeze_mask & (~blank_mask))
                )
                noised = torch.where(apply_freeze, self.task._prev_drone_depth, noised)
            self.task._prev_drone_depth = noised.clone()
        else:
            self.task._prev_drone_depth = noised.clone()

        return noised

    def _encode_drone_vae(self, noised_image_obs: torch.Tensor) -> None:
        """Encode drone depth image through VAE to get latent representation."""
        try:
            img = noised_image_obs.to(self.task.device, dtype=torch.float32).contiguous()
            if img.dim() == 2:
                img = img.unsqueeze(0).unsqueeze(0).expand(
                    self.task.sim_env.num_envs, -1, -1
                )
            elif img.dim() == 3:
                img = img.unsqueeze(1)
            self.task.image_latents[:] = self.task.shared_vae_model.encode(img)
        except RuntimeError as e:
            logger.warning(f"VAE encoding of drone camera failed: {e}")
            self.task.image_latents.zero_()

    def process_static_camera_observation(self) -> None:
        """Capture static camera depth, apply curriculum noise/dropout, and VAE encode."""
        try:
            static_depth, static_seg = self.task.static_camera_manager.capture_images(
                batched=True
            )

            if static_depth is not None and self.task.task_config.vae_config.use_vae:
                self._process_static_depth(static_depth, static_seg)
            else:
                if not self.task._no_static_logged:
                    self.task._no_static_logged = True
                    if static_depth is None:
                        logger.warning("Static camera data is None - capture failed")
                    elif not self.task.task_config.vae_config.use_vae:
                        logger.warning("VAE disabled - static camera latents will be zeros")
                self.task.static_image_latents.fill_(0.0)

        except RuntimeError as e:
            logger.error(f"Static camera processing error: {e}")
            self.task.static_image_latents.fill_(0.0)

    def _process_static_depth(
        self, static_depth: torch.Tensor | np.ndarray, static_seg: object
    ) -> None:
        """Apply noise, frame dropout, and VAE encoding to static camera depth."""
        if isinstance(static_depth, np.ndarray):
            clean = static_depth.copy()
        else:
            clean = static_depth.clone()

        noised = self._apply_static_camera_noise(clean)
        noised = self._apply_static_frame_dropout(noised)

        # Store env0 images for GIF/debug
        self.task.obs_dict["static_depth_clean"] = clean[0]
        self.task.obs_dict["static_depth_noised"] = (
            noised[0] if noised.ndim == 3 else noised
        )
        self.task.obs_dict["static_seg"] = static_seg

        self._encode_static_vae(noised)

    def _apply_static_camera_noise(
        self, depth: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Apply Gaussian noise and pixel dropout to static camera depth."""
        if isinstance(depth, np.ndarray):
            noised = depth.copy()
        else:
            noised = depth.clone()

        if not self.task.task_config.curriculum.enable_camera_noise:
            return noised

        static_noise_override = bool(
            self.task.sim_env.global_tensor_dict.get(
                "camera_randomization/static_noise_disabled", False
            )
        )
        global_noise_disabled = bool(
            self.task.sim_env.global_tensor_dict.get(
                "camera_randomization/noise_disabled", False
            )
        )

        if not global_noise_disabled and not static_noise_override:
            gaussian_std, dropout_rate = self.task.task_config.curriculum.get_camera_noise(
                self.task.curriculum_level
            )
        else:
            gaussian_std, dropout_rate = self.task.task_config.curriculum.get_camera_noise(3)

        if isinstance(noised, np.ndarray):
            if gaussian_std > 0:
                noised = noised + np.random.normal(0.0, gaussian_std, size=noised.shape)
            if dropout_rate > 0:
                mask = np.random.rand(*noised.shape) < dropout_rate
                noised[mask] = 1.0
            return np.clip(noised, 0.0, 1.0)
        else:
            if gaussian_std > 0:
                noised = noised + torch.randn_like(noised) * gaussian_std
            if dropout_rate > 0:
                mask = torch.rand_like(noised) < dropout_rate
                noised = noised.masked_fill(mask, 1.0)
            return torch.clamp(noised, 0.0, 1.0)

    def _apply_static_frame_dropout(
        self, noised: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Apply entire-frame blank/freeze dropout to static camera depth."""
        if not self.task.task_config.curriculum.enable_camera_frame_dropout:
            return noised

        static_fd_override = bool(
            self.task.sim_env.global_tensor_dict.get(
                "camera_randomization/static_frame_dropout_disabled", False
            )
        )
        global_fd_disabled = bool(
            self.task.sim_env.global_tensor_dict.get(
                "camera_randomization/frame_dropout_disabled", False
            )
        )

        if not global_fd_disabled and not static_fd_override:
            fd = self.task.task_config.curriculum.get_camera_frame_dropout(
                self.task.curriculum_level
            )
        else:
            fd = self.task.task_config.curriculum.get_camera_frame_dropout(3)
        p_blank = fd.get("static_blank", 0.0)
        p_freeze = fd.get("static_freeze", 0.0)

        if self.task._prev_static_depth is None:
            if isinstance(noised, np.ndarray):
                self.task._prev_static_depth = noised.copy()
            else:
                self.task._prev_static_depth = noised.clone()

        if isinstance(noised, np.ndarray):
            if p_blank > 0.0 and np.random.rand() < p_blank:
                noised[...] = 1.0
            elif p_freeze > 0.0 and np.random.rand() < p_freeze:
                noised = self.task._prev_static_depth.copy()
            self.task._prev_static_depth = noised.copy()
        else:
            do_blank = torch.rand(1, device=noised.device).item() < p_blank
            if do_blank:
                noised = torch.ones_like(noised)
            else:
                do_freeze = torch.rand(1, device=noised.device).item() < p_freeze
                if do_freeze:
                    noised = self.task._prev_static_depth.clone()
            self.task._prev_static_depth = noised.clone()

        return noised

    def _encode_static_vae(self, noised: torch.Tensor | np.ndarray) -> None:
        """Convert static camera depth to tensor and encode through shared VAE."""
        try:
            if isinstance(noised, np.ndarray):
                depth_tensor = torch.from_numpy(noised).float().to(self.task.device)
            else:
                depth_tensor = noised

            if depth_tensor.dim() == 2:
                depth_tensor = depth_tensor.unsqueeze(0).expand(
                    self.task.sim_env.num_envs, -1, -1
                )
            elif (
                depth_tensor.dim() == 3
                and depth_tensor.shape[0] != self.task.sim_env.num_envs
            ):
                n = depth_tensor.shape[0]
                if n < self.task.sim_env.num_envs:
                    reps = (self.task.sim_env.num_envs + n - 1) // n
                    depth_tensor = depth_tensor.repeat(reps, 1, 1)[
                        : self.task.sim_env.num_envs
                    ]
                else:
                    depth_tensor = depth_tensor[: self.task.sim_env.num_envs]

            self.task.static_image_latents[:] = self.task.shared_vae_model.encode(
                depth_tensor
            )
        except RuntimeError as e:
            logger.warning(f"VAE encoding of static camera failed: {e}")

    def _compute_visibility_metrics(
        self, infos_to_return: dict[str, torch.Tensor]
    ) -> None:
        """Compute geometric gate visibility and static FOV metrics for logging."""
        self._compute_geometric_visibility(infos_to_return)
        self._compute_fov_metrics(infos_to_return)

    def _get_camera_pose_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get camera position and target tensors from StaticCameraManager caches."""
        scm = self.task.static_camera_manager
        if (
            scm is None
            or len(scm.last_camera_pos) < self.task.num_envs
            or len(scm.last_camera_target) < self.task.num_envs
        ):
            return None
        cam_pos = torch.tensor(
            scm.last_camera_pos, dtype=torch.float32, device=self.task.device
        )
        cam_tgt = torch.tensor(
            scm.last_camera_target, dtype=torch.float32, device=self.task.device
        )
        return cam_pos, cam_tgt

    def _compute_camera_basis(
        self, cam_pos: torch.Tensor, cam_tgt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute right/up/forward basis vectors from camera position and target."""
        up_world = (
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.task.device)
            .view(1, 3)
            .expand(self.task.num_envs, 3)
        )
        fwd = cam_tgt - cam_pos
        fwd = fwd / (torch.norm(fwd, dim=1, keepdim=True) + 1e-8)
        right = torch.cross(fwd, up_world)
        right = right / (torch.norm(right, dim=1, keepdim=True) + 1e-8)
        up = torch.cross(right, fwd)
        return right, up, fwd

    def _compute_geometric_visibility(
        self, infos_to_return: dict[str, torch.Tensor]
    ) -> None:
        """Compute grid-based geometric gate visibility fraction (disabled by default)."""
        try:
            gtd = self.task.sim_env.global_tensor_dict
            enabled = (
                read_env_bool("SF_ENABLE_GEOM_VISIBILITY")
                or bool(gtd.get("static_visibility/enable", False))
                or read_env_bool("VISIBILITY_DEBUG")
            )
            if not enabled:
                return

            N = max(4, int(gtd.get("static_visibility/N", 30)))
            M = max(4, int(gtd.get("static_visibility/M", 30)))

            pose = self._get_camera_pose_tensors()
            if pose is None:
                return
            cam_pos, cam_tgt = pose
            right, up, fwd = self._compute_camera_basis(cam_pos, cam_tgt)

            vis_frac, frustum_frac = self._compute_grid_visibility(
                cam_pos, right, up, fwd, N, M, gtd
            )
            eff = torch.where(
                frustum_frac > 1e-6,
                vis_frac / torch.clamp(frustum_frac, min=1e-6),
                torch.zeros_like(frustum_frac),
            )

            for key, val in [
                ("static_visibility/abs", vis_frac),
                ("static_visibility/frustum", frustum_frac),
                ("static_visibility/eff", eff),
            ]:
                self.task.infos[key] = val.detach()
                infos_to_return[key] = val.detach()

        except (ValueError, TypeError) as e:
            logger.debug(f"[VIS] Geometric visibility computation skipped: {e}")

    def _compute_grid_visibility(
        self,
        cam_pos: torch.Tensor,
        right: torch.Tensor,
        up: torch.Tensor,
        fwd: torch.Tensor,
        N: int,
        M: int,
        gtd: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute frustum and occlusion-aware visibility fractions on an NxM gate grid."""
        ne = self.task.num_envs
        dev = self.task.device

        W = self.task.gate_width.view(ne, 1, 1)
        H = self.task.gate_height.view(ne, 1, 1)
        gx = self.task.gate_position[:, 0].view(ne, 1, 1)
        gy = self.task.gate_position[:, 1].view(ne, 1, 1)
        gz = self.task.gate_position[:, 2].view(ne, 1, 1)

        xi = (torch.arange(N, device=dev, dtype=torch.float32) + 0.5) / float(N) - 0.5
        zj = (torch.arange(M, device=dev, dtype=torch.float32) + 0.5) / float(M)

        X = (gx + W * xi.view(1, N, 1)).expand(ne, N, M)
        Z = (gz + H * zj.view(1, 1, M)).expand(ne, N, M)
        Y = gy.expand(ne, N, M)

        cam_pos_ = cam_pos.view(ne, 1, 1, 3)
        Pw = torch.stack([X, Y, Z], dim=3) - cam_pos_

        x_c = torch.sum(Pw * right.view(ne, 1, 1, 3), dim=3)
        y_c = torch.sum(Pw * up.view(ne, 1, 1, 3), dim=3)
        z_c = torch.sum(Pw * fwd.view(ne, 1, 1, 3), dim=3)

        half_angle = math.radians(87.0 * 0.5)
        tan_half = math.tan(half_angle)
        z_ok = (z_c > 0.4) & (z_c < 20.0)
        nx = torch.abs(x_c) / torch.clamp(z_c, min=1e-6)
        ny = torch.abs(y_c) / torch.clamp(z_c, min=1e-6)
        frustum_mask = z_ok & (nx <= tan_half) & (ny <= tan_half)

        # Occlusion by drone (sphere test)
        drone_pos = self.task.obs_dict["robot_position"]
        qpts = torch.stack([X, Y, Z], dim=3)
        seg = qpts - cam_pos_
        vv = torch.sum(seg * seg, dim=3)
        w = drone_pos.view(ne, 1, 1, 3) - cam_pos_
        t = torch.clamp(torch.sum(w * seg, dim=3) / torch.clamp(vv, min=1e-6), 0.0, 1.0)
        p_closest = cam_pos_ + t.unsqueeze(3) * seg
        d2 = torch.sum((p_closest - drone_pos.view(ne, 1, 1, 3)) ** 2, dim=3)
        r = float(gtd.get("static_visibility/drone_radius_m", 0.25))
        occluded = d2 <= (r * r)

        visible_mask = frustum_mask & (~occluded)
        total_cells = float(N * M)
        vis_frac = torch.sum(visible_mask, dim=(1, 2)).to(torch.float32) / total_cells
        frustum_frac = torch.sum(frustum_mask, dim=(1, 2)).to(torch.float32) / total_cells
        return vis_frac, frustum_frac

    def _compute_fov_metrics(
        self, infos_to_return: dict[str, torch.Tensor]
    ) -> None:
        """Compute static camera FOV score for the drone (non-reward metric)."""
        try:
            pose = self._get_camera_pose_tensors()
            if pose is None:
                return
            cam_pos, cam_tgt = pose
            right, up, fwd = self._compute_camera_basis(cam_pos, cam_tgt)

            robot_position = self.task.obs_dict["robot_position"]
            pw = robot_position - cam_pos
            x_c = torch.sum(pw * right, dim=1)
            y_c = torch.sum(pw * up, dim=1)
            z_c = torch.sum(pw * fwd, dim=1)

            half_fov_rad = math.radians(87.0 * 0.5)
            horiz_angle = torch.atan2(torch.abs(x_c), torch.clamp(z_c, min=1e-6))
            vert_angle = torch.atan2(torch.abs(y_c), torch.clamp(z_c, min=1e-6))
            visible = (
                (z_c > 0.1) & (horiz_angle <= half_fov_rad) & (vert_angle <= half_fov_rad)
            )

            h_norm = torch.clamp(horiz_angle / half_fov_rad, 0.0, 1.0)
            v_norm = torch.clamp(vert_angle / half_fov_rad, 0.0, 1.0)
            m_norm = torch.maximum(h_norm, v_norm)
            fov_alpha = float(
                self.task.task_config.reward_parameters.get(
                    "static_fov_visibility_exponent", 2.0
                )
            )
            fov_score = torch.pow(torch.clamp(1.0 - m_norm, min=0.0), fov_alpha)

            for key, val in [
                ("static_fov/visible", visible.float()),
                ("static_fov/horiz_angle_rad", horiz_angle),
                ("static_fov/vert_angle_rad", vert_angle),
                ("static_fov/score", fov_score),
            ]:
                self.task.infos[key] = val
                infos_to_return[key] = val

        except (ValueError, TypeError):
            pass

    def _get_static_camera_pose_relative_to_drone(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-env static camera pose relative to the drone in body frame.

        Returns (relative_position, relative_orientation_euler).
        """
        device = self.task.device
        num_envs = self.task.num_envs

        cam_world = self._compute_camera_world_positions(device, num_envs)
        robot_pos, q = self._get_robot_pose(device, num_envs)

        rel_world = cam_world - robot_pos
        rel_pos_body = quat_rotate_inverse(q, rel_world)

        target_world = self._compute_camera_target(cam_world, num_envs, device)

        fwd = target_world - cam_world
        fwd = fwd / (torch.norm(fwd, dim=1, keepdim=True) + 1e-8)
        fx, fy, fz = fwd[:, 0], fwd[:, 1], fwd[:, 2]
        yaw_cam = torch.atan2(fx, torch.clamp(fy, min=1e-8))
        hyp = torch.sqrt(torch.clamp(fx * fx + fy * fy, min=1e-8))
        pitch_cam = torch.atan2(fz, hyp)
        roll_cam = torch.zeros_like(yaw_cam)
        eul_cam = torch.stack([roll_cam, pitch_cam, yaw_cam], dim=1)
        q_cam = quat_from_euler_xyz_tensor(eul_cam)

        q_drone_conj = torch.stack(
            [-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3]], dim=1
        )
        q_rel = quat_mul(q_drone_conj, q_cam)
        rel_orient_euler = ssa(get_euler_xyz_tensor(q_rel))

        self.task._debug_cam_world = cam_world.detach().clone()
        self.task._debug_rel_pos = rel_pos_body.detach().clone()
        self.task._debug_rel_eul = rel_orient_euler.detach().clone()
        self.task._debug_cam_eul = eul_cam.detach().clone()
        self.task._debug_drone_eul = ssa(get_euler_xyz_tensor(q)).detach().clone()

        return rel_pos_body, rel_orient_euler

    def _compute_camera_world_positions(
        self, device: str, num_envs: int
    ) -> torch.Tensor:
        """Compute per-env camera world positions based on dynamic/static mode."""
        base_y = float(self.task.task_config.static_camera_base_y)
        gate_center_z = self._resolve_camera_z(device, num_envs)

        dynamic_effective = self._is_dynamic_camera_effective()

        cam_world = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        if dynamic_effective:
            try:
                from aerial_gym.config.task_config.navigation_task_config_gate import (
                    task_config as _tc,
                )

                x_off, y_off, z_off = _tc.curriculum.get_dynamic_camera_follow_offset()
            except ImportError:
                x_off, y_off, z_off = 0.0, -1.0, 0.0
            robot_pos = self.task.obs_dict["robot_position"].to(
                device=device, dtype=torch.float32
            )
            cam_world[:, 0] = robot_pos[:, 0] + float(x_off)
            cam_world[:, 1] = robot_pos[:, 1] + float(y_off)
            cam_world[:, 2] = robot_pos[:, 2] + float(z_off)
        else:
            cam_world[:, 0] = 0.0
            cam_world[:, 1] = float(base_y)
            cam_world[:, 2] = gate_center_z

        return cam_world

    def _resolve_camera_z(self, device: str, num_envs: int) -> torch.Tensor:
        """Resolve per-env camera Z coordinate (adaptive to gate or fixed)."""
        cfg_base_z = self.task.task_config.static_camera_base_z
        adaptive_z = False
        base_z_value = 1.5

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

        if adaptive_z:
            gcz = self.task.gate_center_height
            if gcz is None:
                return torch.full((num_envs,), 1.5, device=device, dtype=torch.float32)
            if not torch.is_tensor(gcz):
                return torch.full(
                    (num_envs,), float(gcz), device=device, dtype=torch.float32
                )
            return gcz.to(device=device, dtype=torch.float32).view(-1)
        return torch.full((num_envs,), float(base_z_value), device=device, dtype=torch.float32)

    def _is_dynamic_camera_effective(self) -> bool:
        """Check if dynamic camera following is active (enabled and not disabled)."""
        dynamic_enabled = bool(
            self.task.task_config.curriculum.enable_dynamic_camera_following
        )
        dyn_dis = bool(
            self.task.sim_env.global_tensor_dict.get(
                "dynamic_camera_following/disabled", False
            )
        )
        arc_follow = bool(
            self.task.sim_env.global_tensor_dict.get(
                "static_camera/arc_follow_enabled", False
            )
        )
        return dynamic_enabled and not dyn_dis and not arc_follow

    def _get_robot_pose(
        self, device: str, num_envs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get robot world position and vehicle orientation quaternion."""
        robot_pos = self.task.obs_dict["robot_position"].to(
            device=device, dtype=torch.float32
        )
        q = self.task.obs_dict.get("robot_vehicle_orientation", None)
        if q is None:
            q = self.task.obs_dict.get("robot_orientation", None)
        if q is None:
            q = torch.zeros((num_envs, 4), device=device, dtype=torch.float32)
            q[:, 3] = 1.0
        else:
            q = q.to(device=device, dtype=torch.float32)
        return robot_pos, q

    def _compute_camera_target(
        self, cam_world: torch.Tensor, num_envs: int, device: str
    ) -> torch.Tensor:
        """Compute per-env camera look-at target (adaptive gate center or yaw-swept)."""
        target_world = self.task.gate_position.clone()
        target_world[:, 2] = self.task.gate_position[:, 2] + self.task.gate_center_height

        if not self._is_dynamic_camera_effective():
            scm = self.task.static_camera_manager
            if (
                scm is not None
                and len(scm.current_camera_angles) >= num_envs
            ):
                target_distance = torch.abs(cam_world[:, 1])
                angles_deg = torch.tensor(
                    scm.current_camera_angles[:num_envs],
                    dtype=torch.float32,
                    device=device,
                )
                ang = angles_deg * (3.141592653589793 / 180.0)
                target_world[:, 0] = cam_world[:, 0] + target_distance * torch.sin(ang)
                target_world[:, 1] = cam_world[:, 1] + target_distance * torch.cos(ang)

        return target_world
