"""Helper module for static camera management and image processing.

Provides StaticCameraManager for Isaac Gym native camera setup/capture,
and utility functions for robot camera capture and combined image visualization
using the DCE RL Navigation processing pipeline.
"""

from __future__ import annotations

import cv2
import matplotlib.cm
import numpy as np
import numpy.typing as npt
from isaacgym import gymapi, gymtorch

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)


class StaticCameraManager:
    """Manages static camera setup and capture using Isaac Gym native API."""

    def __init__(self, env_manager: object) -> None:
        """Initialize static camera manager."""
        self.env_manager = env_manager
        self.gym = env_manager.IGE_env.gym
        self.sim = env_manager.IGE_env.sim
        self.env_handles: list[object] = env_manager.IGE_env.env_handles
        self.camera_handles: list[object] = []
        self.camera_setup_success: bool = False

        self._setup_static_camera()

    def _setup_static_camera(self) -> None:
        """Setup static camera using Isaac Gym native camera API with D455 specifications."""
        logger.info("Setting up static camera using Isaac Gym native API...")

        try:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1280
            camera_props.height = 720
            camera_props.horizontal_fov = 87.0
            camera_props.near_plane = 0.4
            camera_props.far_plane = 20.0
            camera_props.enable_tensors = True

            logger.info(
                f"Static camera properties (D455 specs): {camera_props.width}x{camera_props.height}, FOV: {camera_props.horizontal_fov}°"
            )
            logger.info(
                f"Static camera depth range: {camera_props.near_plane}m - {camera_props.far_plane}m"
            )

            self.camera_handles = []
            for i, env_handle in enumerate(self.env_handles):
                cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                self.camera_handles.append(cam_handle)
                logger.info(f"Created static camera sensor {i} in environment {i}")

            # Position camera to face the gate directly
            camera_pos = gymapi.Vec3(0.0, -3.0, 1.5)
            camera_target = gymapi.Vec3(0.0, 0.0, 1.5)

            for i, (env_handle, cam_handle) in enumerate(
                zip(self.env_handles, self.camera_handles)
            ):
                self.gym.set_camera_location(cam_handle, env_handle, camera_pos, camera_target)
                logger.info(
                    f"Set static camera {i} to look from ({camera_pos.x}, {camera_pos.y}, {camera_pos.z}) toward ({camera_target.x}, {camera_target.y}, {camera_target.z})"
                )

            logger.info("✓ Static cameras positioned to face gate directly")
            self.camera_setup_success = True

        except Exception as e:
            logger.error(f"❌ ERROR: Isaac Gym static camera setup failed: {e}")
            import traceback

            traceback.print_exc()
            self.camera_setup_success = False

    def capture_static_camera_images(
        self,
    ) -> tuple[npt.NDArray[np.uint8] | None, npt.NDArray[np.float32] | None]:
        """Capture depth and segmentation images from static camera."""
        if not self.camera_setup_success or len(self.camera_handles) == 0:
            return None, None

        try:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            env_handle = self.env_handles[0]
            cam_handle = self.camera_handles[0]

            depth_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH
            )
            depth_img = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()

            seg_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, env_handle, cam_handle, gymapi.IMAGE_SEGMENTATION
            )
            seg_img = gymtorch.wrap_tensor(seg_tensor).cpu().numpy()

            self.gym.end_access_image_tensors(self.sim)

            # Normalize depth to [0,1] then convert to uint8 for DCE processing pipeline
            if depth_img is not None:
                depth_normalized = depth_img.copy()
                depth_normalized[depth_normalized == -np.inf] = 20.0
                depth_normalized = np.abs(depth_normalized)
                depth_normalized = np.clip(depth_normalized, 0.4, 20.0)
                depth_normalized = (depth_normalized - 0.4) / (20.0 - 0.4)
                depth_img = (255.0 * depth_normalized).astype(np.uint8)

            return depth_img, seg_img

        except Exception as e:
            logger.debug(f"Error capturing static camera images: {e}")
            return None, None


def capture_robot_camera_images(
    env_manager: object,
) -> tuple[npt.NDArray[np.uint8] | None, npt.NDArray[np.float32] | None]:
    """Capture depth and segmentation images from robot camera using global tensor dictionary.

    Uses the exact same method as DCE RL navigation for consistency.
    """
    try:
        env_manager.render(render_components="sensors")

        try:
            global_tensor_dict = env_manager.global_tensor_dict
        except AttributeError:
            return None, None

        depth_img = None
        seg_img = None

        if (
            "depth_range_pixels" in global_tensor_dict
            and global_tensor_dict["depth_range_pixels"] is not None
        ):
            depth_tensor = global_tensor_dict["depth_range_pixels"][0, 0]
            if depth_tensor is not None:
                depth_img = (255.0 * depth_tensor.cpu().numpy()).astype(np.uint8)

        if (
            "segmentation_pixels" in global_tensor_dict
            and global_tensor_dict["segmentation_pixels"] is not None
        ):
            seg_tensor = global_tensor_dict["segmentation_pixels"][0, 0]
            if seg_tensor is not None:
                seg_img = seg_tensor.cpu().numpy()

        return depth_img, seg_img

    except Exception as e:
        logger.debug(f"Robot camera capture error: {e}")
        return None, None


def create_combined_image(
    depth_img: npt.NDArray[np.uint8],
    seg_img: npt.NDArray[np.float32],
    title: str = "Camera",
) -> npt.NDArray[np.uint8] | None:
    """Create combined visualization of depth and segmentation images.

    Uses the exact same processing pipeline as DCE RL navigation for consistency:
    - Depth: Normalized [0,1] -> uint8 -> JET colormap
    - Segmentation: Raw segment IDs -> 3-step DCE processing -> Plasma colormap
    """
    if depth_img is None or seg_img is None:
        return None

    depth_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)

    seg_image_processed = seg_img.copy()

    # Step 1: Fix the error when there are no positive values (DCE method)
    if np.any(seg_image_processed > 0):
        min_positive = seg_image_processed[seg_image_processed > 0].min()
        seg_image_processed[seg_image_processed <= 0] = min_positive
    else:
        seg_image_processed[:] = 0.1

    # Step 2: Normalize to [0,1] range (DCE method)
    seg_normalized = (seg_image_processed - seg_image_processed.min()) / (
        seg_image_processed.max() - seg_image_processed.min() + 1e-8
    )

    # Step 3: Apply plasma colormap (DCE method)
    seg_colored_float = matplotlib.cm.plasma(seg_normalized)
    seg_colored = (seg_colored_float[:, :, :3] * 255.0).astype(np.uint8)

    # Create side-by-side layout
    h, w = depth_colored.shape[:2]
    combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined[:, :w] = depth_colored
    combined[:, w:] = seg_colored

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Depth", (10, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(combined, "Segmentation", (w + 10, 20), font, 0.5, (255, 255, 255), 1)

    return combined
