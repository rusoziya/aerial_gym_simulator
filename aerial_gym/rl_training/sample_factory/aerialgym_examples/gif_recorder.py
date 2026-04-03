"""GIF recording helpers for dual-camera episode visualization."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor

VERBOSE = os.environ.get("TRAIN_VERBOSE", "false").lower() == "true"


def process_camera_image(
    image_data: Tensor,
    camera_type: str = "depth",
) -> Optional[Image.Image]:
    """Convert a single-channel tensor to a PIL Image for GIF saving."""
    if camera_type == "depth":
        image = (255.0 * image_data.cpu().numpy()).astype(np.uint8)
        return Image.fromarray(image)
    if camera_type == "segmentation":
        import matplotlib.cm

        seg_image = image_data.cpu().numpy()
        seg_min, seg_max = seg_image.min(), seg_image.max()
        if seg_max - seg_min < 1e-8:
            seg_norm = np.zeros_like(seg_image, dtype=np.float32)
        else:
            seg_norm = (seg_image - seg_min) / (seg_max - seg_min)
        seg_image_plasma = matplotlib.cm.plasma(seg_norm)
        return Image.fromarray((seg_image_plasma * 255.0).astype(np.uint8))
    return None


def _to_2d_tensor(data: object) -> Optional[Tensor]:
    """Coerce numpy/tensor data to a 2-D tensor, or return None."""
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data.astype(np.float32))
    else:
        tensor = data.float() if not data.is_floating_point() else data
    if tensor.dim() > 2:
        tensor = tensor.squeeze()
    return tensor


class GifRecorder:
    """Manages per-environment frame buffers and GIF saving for dual cameras."""

    def __init__(self, num_agents: int, output_dir: str = "./gif_episodes") -> None:
        self.num_agents = num_agents
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.episode_counter: int = 0

        self.drone_depth_frames: list[list[Image.Image]] = [[] for _ in range(num_agents)]
        self.drone_seg_frames: list[list[Image.Image]] = [[] for _ in range(num_agents)]
        self.static_depth_frames: list[list[Image.Image]] = [[] for _ in range(num_agents)]
        self.static_seg_frames: list[list[Image.Image]] = [[] for _ in range(num_agents)]
        self.merged_frames: list[list[Image.Image]] = [[] for _ in range(num_agents)]

        self.drone_depth_noised_frames: list[list[Image.Image]] = [[] for _ in range(num_agents)]
        self.static_depth_noised_frames: list[list[Image.Image]] = [[] for _ in range(num_agents)]
        self.merged_noised_frames: list[list[Image.Image]] = [[] for _ in range(num_agents)]

    def clear_frames(self, env_id: int = 0) -> None:
        """Clear all frame buffers for the given environment."""
        if env_id >= self.num_agents:
            return
        self.drone_depth_frames[env_id] = []
        self.drone_seg_frames[env_id] = []
        self.static_depth_frames[env_id] = []
        self.static_seg_frames[env_id] = []
        self.merged_frames[env_id] = []
        self.drone_depth_noised_frames[env_id] = []
        self.static_depth_noised_frames[env_id] = []
        self.merged_noised_frames[env_id] = []

    def collect_frames(self, obs_dict: dict[str, Tensor], task: object) -> None:
        """Collect frames from both drone and static cameras (clean + noised)."""
        try:
            self._collect_drone_frames(task)
            self._collect_static_frames(task)
            self._collect_noised_frames(task)
            self._build_merged_frames()
        except (ValueError, TypeError) as e:
            if VERBOSE:
                import traceback

                print(f"[GIF] Warning: Failed to collect frames: {e}")
                print(f"[GIF] Traceback: {traceback.format_exc()}")

    def _collect_drone_frames(self, task: object) -> None:
        if "depth_range_pixels" in task.obs_dict:
            drone_depth = task.obs_dict["depth_range_pixels"][0, 0]
            img = process_camera_image(drone_depth, "depth")
            if img is not None:
                self.drone_depth_frames[0].append(img)

        if "segmentation_pixels" in task.obs_dict:
            drone_seg = task.obs_dict["segmentation_pixels"][0, 0]
            img = process_camera_image(drone_seg, "segmentation")
            if img is not None:
                self.drone_seg_frames[0].append(img)

    def _collect_static_frames(self, task: object) -> None:
        if "static_depth_clean" in task.obs_dict:
            tensor = _to_2d_tensor(task.obs_dict["static_depth_clean"])
            if tensor is not None:
                img = process_camera_image(tensor, "depth")
                if img is not None:
                    self.static_depth_frames[0].append(img)

        if "static_seg" in task.obs_dict:
            tensor = _to_2d_tensor(task.obs_dict["static_seg"])
            if tensor is not None:
                img = process_camera_image(tensor, "segmentation")
                if img is not None:
                    self.static_seg_frames[0].append(img)

    def _collect_noised_frames(self, task: object) -> None:
        if "depth_range_pixels_noised" in task.obs_dict:
            drone_noised = task.obs_dict["depth_range_pixels_noised"][0, 0]
            img = process_camera_image(drone_noised, "depth")
            if img is not None:
                self.drone_depth_noised_frames[0].append(img)

        if "static_depth_noised" in task.obs_dict:
            tensor = _to_2d_tensor(task.obs_dict["static_depth_noised"])
            if tensor is not None:
                img = process_camera_image(tensor, "depth")
                if img is not None:
                    self.static_depth_noised_frames[0].append(img)

    def _build_merged_frames(self) -> None:
        self._merge_pair(
            self.drone_depth_frames[0],
            self.static_depth_frames[0],
            self.merged_frames[0],
        )
        self._merge_pair(
            self.drone_depth_noised_frames[0],
            self.static_depth_noised_frames[0],
            self.merged_noised_frames[0],
        )

    @staticmethod
    def _merge_pair(
        left_list: list[Image.Image],
        right_list: list[Image.Image],
        target_list: list[Image.Image],
    ) -> None:
        if not left_list or not right_list:
            return
        if len(left_list) != len(right_list):
            return
        left_arr = np.array(left_list[-1])
        right_arr = np.array(right_list[-1])
        if left_arr.shape != right_arr.shape:
            right_img = Image.fromarray(right_arr).resize((left_arr.shape[1], left_arr.shape[0]))
            right_arr = np.array(right_img)
        merged = np.concatenate((left_arr, right_arr), axis=1)
        target_list.append(Image.fromarray(merged))

    def save_episode_gifs(self, env_id: int = 0, level_suffix: str = "") -> None:
        """Save all collected frame buffers as GIFs for the given environment."""
        if env_id >= self.num_agents:
            return
        episode_num = self.episode_counter
        self.episode_counter += 1
        try:
            self._save_single_gif(
                self.drone_depth_frames[env_id],
                f"episode_{episode_num:04d}_drone_depth{level_suffix}.gif",
                "drone depth",
            )
            self._save_single_gif(
                self.drone_seg_frames[env_id],
                f"episode_{episode_num:04d}_drone_seg{level_suffix}.gif",
                "drone segmentation",
            )
            self._save_single_gif(
                self.static_depth_frames[env_id],
                f"episode_{episode_num:04d}_static_depth{level_suffix}.gif",
                "static depth",
            )
            self._save_single_gif(
                self.static_seg_frames[env_id],
                f"episode_{episode_num:04d}_static_seg{level_suffix}.gif",
                "static segmentation",
            )
            self._save_single_gif(
                self.merged_frames[env_id],
                f"episode_{episode_num:04d}_merged{level_suffix}.gif",
                "merged",
            )
            self._save_single_gif(
                self.drone_depth_noised_frames[env_id],
                f"episode_{episode_num:04d}_drone_depth_D455_NOISED{level_suffix}.gif",
                "drone depth (D455 NOISED)",
            )
            self._save_single_gif(
                self.static_depth_noised_frames[env_id],
                f"episode_{episode_num:04d}_static_depth_D455_NOISED{level_suffix}.gif",
                "static depth (D455 NOISED)",
            )
            self._save_single_gif(
                self.merged_noised_frames[env_id],
                f"episode_{episode_num:04d}_merged_D455_NOISED{level_suffix}.gif",
                "merged (D455 NOISED)",
            )
        except OSError as e:
            if VERBOSE:
                print(f"[GIF] Warning: Failed to save GIFs for episode {episode_num}: {e}")

    def _save_single_gif(
        self,
        frames: list[Image.Image],
        filename: str,
        label: str,
    ) -> None:
        if not frames:
            return
        gif_path = os.path.join(self.output_dir, filename)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        if VERBOSE:
            print(f"[GIF] Saved {label}: {gif_path}")
