"""GIF frame collection and saving for env-0 episode recordings."""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

from aerial_gym.examples.dce_rl_navigation.inference_utils import to_pil_gray, save_gif


class GifRecorder:
    """Collects depth/segmentation frames for env 0 and saves GIFs on episode end."""

    def __init__(self, enabled: bool, output_dir: str, save_every_n: int = 1) -> None:
        self.enabled: bool = enabled
        self.output_dir: str = output_dir
        self.save_every_n: int = save_every_n
        self.episode_counter: int = 0
        self._drone_noised: list[Image.Image] = []
        self._drone_clean: list[Image.Image] = []
        self._drone_seg: list[Image.Image] = []
        self._static_noised: list[Image.Image] = []
        self._static_clean: list[Image.Image] = []
        self._static_seg: list[Image.Image] = []
        if enabled:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError:
                self.enabled = False

    def collect_frames(self, obs_dict: dict[str, torch.Tensor]) -> None:
        """Collect one step of frames from the observation dict for env 0."""
        if not self.enabled or not isinstance(obs_dict, dict):
            return
        self._collect_drone_depth(obs_dict)
        self._collect_drone_seg(obs_dict)
        self._collect_static_depth(obs_dict)

    def _collect_drone_depth(self, d: dict[str, torch.Tensor]) -> None:
        noised = d.get("depth_range_pixels_noised")
        if isinstance(noised, torch.Tensor) and noised.ndim >= 3:
            pil = to_pil_gray(noised[0, 0])
            if pil is not None:
                self._drone_noised.append(pil)
        clean = d.get("depth_range_pixels")
        pil_clean = _extract_env0_pil(clean)
        if pil_clean is not None:
            self._drone_clean.append(pil_clean)

    def _collect_drone_seg(self, d: dict[str, torch.Tensor]) -> None:
        ds = d.get("segmentation_pixels")
        if not isinstance(ds, torch.Tensor):
            return
        if ds.ndim == 4:
            ds0 = ds[0, 0]
        elif ds.ndim == 3:
            ds0 = ds[0]
        else:
            ds0 = ds
        pil = _normalize_seg_to_pil(ds0.detach().cpu().numpy())
        if pil is not None:
            self._drone_seg.append(pil)

    def _collect_static_depth(self, d: dict[str, torch.Tensor]) -> None:
        sd = d.get("static_depth_noised")
        if sd is not None:
            pil = _extract_env0_pil(sd)
            if pil is not None:
                self._static_noised.append(pil)
        sc = d.get("static_depth")
        pil_sc = _extract_env0_pil(sc)
        if pil_sc is not None:
            self._static_clean.append(pil_sc)

    def collect_static_camera_frames(self, static_camera_manager: object) -> None:
        """Collect static camera depth/segmentation from camera manager."""
        if not self.enabled or static_camera_manager is None:
            return
        if not hasattr(static_camera_manager, "capture_images"):
            return
        try:
            depth, seg = static_camera_manager.capture_images(batched=False)
            if depth is not None:
                pil_d = _extract_env0_pil(depth)
                if pil_d is not None:
                    self._static_clean.append(pil_d)
            if seg is not None:
                seg_arr = seg
                if isinstance(seg_arr, np.ndarray) and seg_arr.ndim > 2:
                    seg_arr = np.squeeze(seg_arr)
                pil_seg = _normalize_seg_to_pil(seg_arr)
                if pil_seg is not None:
                    self._static_seg.append(pil_seg)
        except (ValueError, TypeError):
            pass

    def on_episode_end(self) -> None:
        """Save GIFs and reset frame buffers when env 0 episode ends."""
        if not self.enabled:
            return
        self.episode_counter += 1
        if (self.episode_counter % self.save_every_n) == 0:
            tag = f"episode_{self.episode_counter:04d}"
            save_gif(self._drone_noised, self.output_dir, f"{tag}_drone_depth_D455_NOISED.gif")
            save_gif(self._drone_clean, self.output_dir, f"{tag}_drone_depth_D455_CLEAN.gif")
            save_gif(self._drone_seg, self.output_dir, f"{tag}_drone_seg.gif")
            save_gif(self._static_noised, self.output_dir, f"{tag}_static_depth_D455_NOISED.gif")
            save_gif(self._static_clean, self.output_dir, f"{tag}_static_depth_D455_CLEAN.gif")
            save_gif(self._static_seg, self.output_dir, f"{tag}_static_seg.gif")
        self._clear_buffers()

    def _clear_buffers(self) -> None:
        self._drone_noised = []
        self._drone_clean = []
        self._drone_seg = []
        self._static_noised = []
        self._static_clean = []
        self._static_seg = []


def _extract_env0_pil(data: torch.Tensor | np.ndarray | None) -> Image.Image | None:
    """Extract env-0 slice from a tensor/array and convert to grayscale PIL."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        if data.ndim == 4:
            return to_pil_gray(data[0, 0])
        if data.ndim == 3:
            return to_pil_gray(data[0])
        if data.ndim == 2:
            return to_pil_gray(data)
        return None
    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            return to_pil_gray(data[0])
        return to_pil_gray(data)
    return to_pil_gray(data)


def _normalize_seg_to_pil(seg: np.ndarray) -> Image.Image | None:
    """Normalize segmentation array to [0,1] and convert to grayscale PIL."""
    seg_min = np.min(seg)
    seg_max = np.max(seg)
    if (seg_max - seg_min) > 0:
        seg_norm = (seg - seg_min) / float(seg_max - seg_min)
    else:
        seg_norm = np.zeros_like(seg, dtype=np.float32)
    return to_pil_gray(seg_norm)
