from __future__ import annotations

import numpy as np

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("static_camera_capture")


def capture_images_batched(
    gym: object,
    sim: object,
    env_handles: list,
    camera_handles: list,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Capture depth and segmentation images from all environments.

    Returns:
        A tuple of (depth_stack, seg_img_env0) where depth_stack has shape (N, H, W)
        normalized to [0, 1], and seg_img_env0 is the segmentation for env 0.
    """
    from isaacgym import gymapi, gymtorch

    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    depth_imgs: list[np.ndarray] = []
    seg_imgs: list[np.ndarray] = []
    for env_handle, cam_handle in zip(env_handles, camera_handles):
        depth_tensor = gym.get_camera_image_gpu_tensor(
            sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH
        )
        depth_img = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
        depth_imgs.append(depth_img)

        seg_tensor = gym.get_camera_image_gpu_tensor(
            sim, env_handle, cam_handle, gymapi.IMAGE_SEGMENTATION
        )
        seg_img = gymtorch.wrap_tensor(seg_tensor).cpu().numpy()
        seg_imgs.append(seg_img)

    gym.end_access_image_tensors(sim)

    if len(depth_imgs) > 0 and depth_imgs[0] is not None:
        depth_stack = np.stack(depth_imgs, axis=0)
        depth_stack = _normalize_depth(depth_stack)
    else:
        depth_stack = None

    seg_img0 = seg_imgs[0] if len(seg_imgs) > 0 else None
    return depth_stack, seg_img0


def capture_images_single(
    gym: object,
    sim: object,
    env_handles: list,
    camera_handles: list,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Capture depth and segmentation images from env 0 only.

    Returns:
        A tuple of (depth_img, seg_img) for env 0, with depth normalized to [0, 1].
    """
    from isaacgym import gymapi, gymtorch

    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    env_handle = env_handles[0]
    cam_handle = camera_handles[0]

    depth_tensor = gym.get_camera_image_gpu_tensor(
        sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH
    )
    depth_img = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()

    seg_tensor = gym.get_camera_image_gpu_tensor(
        sim, env_handle, cam_handle, gymapi.IMAGE_SEGMENTATION
    )
    seg_img = gymtorch.wrap_tensor(seg_tensor).cpu().numpy()

    gym.end_access_image_tensors(sim)

    if depth_img is not None:
        depth_img = _normalize_depth(depth_img)

    return depth_img, seg_img


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize Isaac Gym depth image to [0, 1] range using D455 specs."""
    near_plane = 0.4
    far_plane = 20.0
    result = depth.copy()
    result[result == -np.inf] = far_plane
    result = np.abs(result)
    result = np.clip(result, near_plane, far_plane)
    result = (result - near_plane) / (far_plane - near_plane)
    return result.astype(np.float32)


def generate_synthetic_camera_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic camera data for headless training."""
    height, width = 135, 240
    depth_img = np.full((height, width), 0.5, dtype=np.float32)

    gate_w = max(1, width // 4)
    gate_h = max(1, height // 3)
    gate_x_start = width // 2 - gate_w // 2
    gate_x_end = width // 2 + gate_w // 2
    gate_y_start = height // 2 - gate_h // 2
    gate_y_end = height // 2 + gate_h // 2

    depth_img[gate_y_start:gate_y_end, gate_x_start:gate_x_end] = 0.8

    frame_thickness = max(1, min(width, height) // 24)
    depth_img[
        gate_y_start - frame_thickness : gate_y_start,
        gate_x_start - frame_thickness : gate_x_end + frame_thickness,
    ] = 0.2
    depth_img[
        gate_y_end : gate_y_end + frame_thickness,
        gate_x_start - frame_thickness : gate_x_end + frame_thickness,
    ] = 0.2
    depth_img[
        gate_y_start:gate_y_end, gate_x_start - frame_thickness : gate_x_start
    ] = 0.2
    depth_img[
        gate_y_start:gate_y_end, gate_x_end : gate_x_end + frame_thickness
    ] = 0.2

    noise = np.random.normal(0, 0.02, (height, width)).astype(np.float32)
    depth_img = np.clip(depth_img + noise, 0.0, 1.0)

    seg_img = np.zeros((height, width), dtype=np.uint8)
    seg_img[gate_y_start:gate_y_end, gate_x_start:gate_x_end] = 1

    return depth_img, seg_img
