from __future__ import annotations

import math
import os

import torch
from isaacgym import gymapi

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("static_camera_follow_modes")


def update_dynamic_camera_following(
    robot_positions: torch.Tensor,
    gate_positions: torch.Tensor,
    gate_center_heights: torch.Tensor,
    gym: object,
    sim: object,
    env_handles: list,
    camera_handles: list,
    env_manager: object,
    last_camera_pos: list[tuple[float, float, float]],
    last_camera_target: list[tuple[float, float, float]],
    last_angle_deg: list[float],
) -> None:
    """Keep camera 1 m behind the drone, same height, look at the drone.

    If the gate is far outside the view, minimally steer toward the gate.
    """
    try:
        try:
            gtd = env_manager.IGE_env.global_tensor_dict
            y_off = float(gtd.get("dynamic_camera_following/offset_y_m", -1.0))
        except (ValueError, TypeError, AttributeError):
            y_off = -1.0
        x_off, z_off = 0.0, 0.0
        half_fov = 87.0 * 0.5
        margin = 5.0

        num = min(len(env_handles), len(camera_handles), robot_positions.shape[0])
        for env_idx in range(num):
            drone = robot_positions[env_idx]
            cam_x = float(drone[0].item()) + x_off
            cam_y = float(drone[1].item()) + y_off
            cam_z = float(drone[2].item()) + z_off
            camera_pos = gymapi.Vec3(cam_x, cam_y, cam_z)

            target_drone = gymapi.Vec3(
                float(drone[0].item()), float(drone[1].item()), float(drone[2].item())
            )

            gate = gate_positions[env_idx]
            gate_cz = float(gate_center_heights[env_idx].item())
            yaw_d = math.degrees(
                math.atan2(target_drone.x - cam_x, target_drone.y - cam_y)
            )
            yaw_g = math.degrees(
                math.atan2(float(gate[0].item()) - cam_x, float(gate[1].item()) - cam_y)
            )
            delta = yaw_g - yaw_d
            while delta > 180.0:
                delta -= 360.0
            while delta < -180.0:
                delta += 360.0

            disable_blend = False
            try:
                disable_blend = bool(
                    env_manager.IGE_env.global_tensor_dict.get(
                        "dynamic_camera_following/disable_gate_blending", False
                    )
                )
            except (KeyError, TypeError, AttributeError):
                pass

            if not disable_blend and abs(delta) > (half_fov - margin):
                w = 0.2
                tgx = (1.0 - w) * target_drone.x + w * float(gate[0].item())
                tgy = (1.0 - w) * target_drone.y + w * float(gate[1].item())
                tgz = (1.0 - w) * target_drone.z + w * gate_cz
                camera_target = gymapi.Vec3(tgx, tgy, tgz)
            else:
                camera_target = target_drone

            cam_handle = camera_handles[env_idx]
            env_handle = env_handles[env_idx]
            gym.set_camera_location(cam_handle, env_handle, camera_pos, camera_target)

            last_camera_pos[env_idx] = (
                float(camera_pos.x), float(camera_pos.y), float(camera_pos.z)
            )
            last_camera_target[env_idx] = (
                float(camera_target.x), float(camera_target.y), float(camera_target.z)
            )
            last_angle_deg[env_idx] = 0.0

    except RuntimeError as e:
        logger.warning(f"Failed to update dynamic camera following: {e}")


def update_arc_follow(
    robot_positions: torch.Tensor,
    gate_positions: torch.Tensor,
    gate_center_heights: torch.Tensor,
    radius_m: float,
    gym: object,
    sim: object,
    env_handles: list,
    camera_handles: list,
    env_manager: object,
) -> None:
    """Arc-follow: constrain camera to a circular arc around the gate center."""
    try:
        sim_steps_fn = _make_sim_steps_getter(env_manager)

        omega = 2.0 * math.pi / 600.0
        num = min(len(env_handles), len(camera_handles), robot_positions.shape[0])
        for env_idx in range(num):
            steps = sim_steps_fn(env_idx)
            gx = float(gate_positions[env_idx, 0].item())
            gy = float(gate_positions[env_idx, 1].item())
            gz_center = float(gate_center_heights[env_idx].item())

            theta = omega * steps + (0.17 * env_idx)
            arc_x = gx + radius_m * math.sin(theta)
            arc_y = gy - radius_m * math.cos(theta)
            cam_z = gz_center
            camera_pos = gymapi.Vec3(arc_x, arc_y, cam_z)

            drone = robot_positions[env_idx]
            w = 0.3
            tgt_x = (1.0 - w) * float(drone[0].item()) + w * gx
            tgt_y = (1.0 - w) * float(drone[1].item()) + w * gy
            tgt_z = (1.0 - w) * float(drone[2].item()) + w * gz_center
            camera_target = gymapi.Vec3(tgt_x, tgt_y, tgt_z)

            cam_handle = camera_handles[env_idx]
            env_handle = env_handles[env_idx]
            gym.set_camera_location(cam_handle, env_handle, camera_pos, camera_target)
    except RuntimeError as e:
        logger.warning(f"Failed to update arc-follow camera: {e}")


def update_locked_follow(
    robot_positions: torch.Tensor,
    gym: object,
    sim: object,
    env_handles: list,
    camera_handles: list,
    env_manager: object,
) -> None:
    """Keep camera position fixed; rotate to always center the drone."""
    try:
        base_y = float(os.environ.get("SF_STATIC_CAMERA_BASE_Y", -3.0))
        base_z_env = os.environ.get("SF_STATIC_CAMERA_BASE_Z", "1.5")

        gate_center_per_env = None
        if isinstance(base_z_env, str) and base_z_env.strip().lower() == "adaptive":
            try:
                gtd = env_manager.IGE_env.global_tensor_dict
                gate_center_per_env = gtd.get("gate/center_height_per_env", None)
            except (KeyError, TypeError, AttributeError):
                pass

        num = min(len(env_handles), len(camera_handles), robot_positions.shape[0])
        for env_idx in range(num):
            cam_x = 0.0
            cam_y = base_y
            if gate_center_per_env is not None and env_idx < len(gate_center_per_env):
                cam_z = float(gate_center_per_env[env_idx].item())
            else:
                try:
                    cam_z = float(base_z_env)
                except (ValueError, TypeError):
                    cam_z = 1.5

            camera_pos = gymapi.Vec3(cam_x, cam_y, cam_z)
            drone = robot_positions[env_idx]
            target = gymapi.Vec3(
                float(drone[0].item()), float(drone[1].item()), float(drone[2].item())
            )
            cam_handle = camera_handles[env_idx]
            env_handle = env_handles[env_idx]
            gym.set_camera_location(cam_handle, env_handle, camera_pos, target)
    except RuntimeError as e:
        logger.warning(f"Failed to update locked-follow camera: {e}")


def _make_sim_steps_getter(env_manager: object) -> callable:
    """Create a function that returns sim_steps for a given env index."""
    try:
        gtd = env_manager.IGE_env.global_tensor_dict
        sim_steps_obj = gtd.get("sim_steps", 0)
        if isinstance(sim_steps_obj, torch.Tensor):
            if sim_steps_obj.ndim > 0:

                def get_step(idx: int) -> int:
                    return (
                        int(sim_steps_obj[idx].item())
                        if idx < sim_steps_obj.shape[0]
                        else int(sim_steps_obj.item())
                    )
            else:
                val = int(sim_steps_obj.item())

                def get_step(idx: int) -> int:
                    return val
        else:
            val = int(sim_steps_obj)

            def get_step(idx: int) -> int:
                return val

        return get_step
    except (ValueError, TypeError, AttributeError):

        def get_step(idx: int) -> int:
            return 0

        return get_step
