from __future__ import annotations

import math
import os

import torch

from aerial_gym.utils.env_flag_utils import read_env_bool
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import *  # noqa: F401,F403
from aerial_gym.utils.tensor_utils import normalize_safe

logger = CustomLogger("navigation_task_gate_camera_pose_visibility")


class CameraPoseAndVisibility:
    """Handles static camera pose computation and gate/drone visibility metrics."""

    def __init__(self, task: object) -> None:
        self.task = task

    def _compute_visibility_metrics(self, infos_to_return: dict[str, torch.Tensor]) -> None:
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
        cam_pos = torch.tensor(scm.last_camera_pos, dtype=torch.float32, device=self.task.device)
        cam_tgt = torch.tensor(scm.last_camera_target, dtype=torch.float32, device=self.task.device)
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
        fwd = normalize_safe(fwd)
        right = torch.cross(fwd, up_world)
        right = normalize_safe(right)
        up = torch.cross(right, fwd)
        return right, up, fwd

    def _compute_geometric_visibility(self, infos_to_return: dict[str, torch.Tensor]) -> None:
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

    def _compute_fov_metrics(self, infos_to_return: dict[str, torch.Tensor]) -> None:
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
            visible = (z_c > 0.1) & (horiz_angle <= half_fov_rad) & (vert_angle <= half_fov_rad)

            h_norm = torch.clamp(horiz_angle / half_fov_rad, 0.0, 1.0)
            v_norm = torch.clamp(vert_angle / half_fov_rad, 0.0, 1.0)
            m_norm = torch.maximum(h_norm, v_norm)
            fov_alpha = float(
                self.task.task_config.reward_parameters.get("static_fov_visibility_exponent", 2.0)
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
        """Compute per-env static camera pose relative to the drone in body frame."""
        device = self.task.device
        num_envs = self.task.num_envs

        cam_world = self._compute_camera_world_positions(device, num_envs)
        robot_pos, q = self._get_robot_pose(device, num_envs)

        rel_world = cam_world - robot_pos
        rel_pos_body = quat_rotate_inverse(q, rel_world)

        target_world = self._compute_camera_target(cam_world, num_envs, device)

        fwd = target_world - cam_world
        fwd = normalize_safe(fwd)
        fx, fy, fz = fwd[:, 0], fwd[:, 1], fwd[:, 2]
        yaw_cam = torch.atan2(fx, torch.clamp(fy, min=1e-8))
        hyp = torch.sqrt(torch.clamp(fx * fx + fy * fy, min=1e-8))
        pitch_cam = torch.atan2(fz, hyp)
        roll_cam = torch.zeros_like(yaw_cam)
        eul_cam = torch.stack([roll_cam, pitch_cam, yaw_cam], dim=1)
        q_cam = quat_from_euler_xyz_tensor(eul_cam)

        q_drone_conj = torch.stack([-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3]], dim=1)
        q_rel = quat_mul(q_drone_conj, q_cam)
        rel_orient_euler = ssa(get_euler_xyz_tensor(q_rel))

        self.task._debug_cam_world = cam_world.detach().clone()
        self.task._debug_rel_pos = rel_pos_body.detach().clone()
        self.task._debug_rel_eul = rel_orient_euler.detach().clone()
        self.task._debug_cam_eul = eul_cam.detach().clone()
        self.task._debug_drone_eul = ssa(get_euler_xyz_tensor(q)).detach().clone()

        return rel_pos_body, rel_orient_euler

    def _compute_camera_world_positions(self, device: str, num_envs: int) -> torch.Tensor:
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
            robot_pos = self.task.obs_dict["robot_position"].to(device=device, dtype=torch.float32)
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
                return torch.full((num_envs,), float(gcz), device=device, dtype=torch.float32)
            return gcz.to(device=device, dtype=torch.float32).view(-1)
        return torch.full((num_envs,), float(base_z_value), device=device, dtype=torch.float32)

    def _is_dynamic_camera_effective(self) -> bool:
        """Check if dynamic camera following is active (enabled and not disabled)."""
        dynamic_enabled = bool(self.task.task_config.curriculum.enable_dynamic_camera_following)
        dyn_dis = bool(
            self.task.sim_env.global_tensor_dict.get("dynamic_camera_following/disabled", False)
        )
        arc_follow = bool(
            self.task.sim_env.global_tensor_dict.get("static_camera/arc_follow_enabled", False)
        )
        return dynamic_enabled and not dyn_dis and not arc_follow

    def _get_robot_pose(self, device: str, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get robot world position and vehicle orientation quaternion."""
        robot_pos = self.task.obs_dict["robot_position"].to(device=device, dtype=torch.float32)
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
            if scm is not None and len(scm.current_camera_angles) >= num_envs:
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
