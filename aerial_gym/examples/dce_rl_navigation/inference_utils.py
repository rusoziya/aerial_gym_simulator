"""Shared utilities for DCE RL navigation inference scripts.

Extracted from dce_nn_navigation_gate.py to be reusable across
inference variants (base, gate, future).
"""

from __future__ import annotations

import os
from math import sqrt

import numpy as np
import torch
from PIL import Image


def to_pil_gray(image_data: torch.Tensor | np.ndarray) -> Image.Image | None:
    """Convert a depth/segmentation tensor or array to a grayscale PIL Image.

    Clips to [0, 1] and scales to uint8.  Returns None on failure.
    """
    try:
        if isinstance(image_data, torch.Tensor):
            img = image_data.detach().cpu().numpy()
        else:
            img = np.array(image_data)
        if img.ndim > 2:
            img = np.squeeze(img)
        img_u8 = (img.clip(0.0, 1.0) * 255.0).astype("uint8")
        return Image.fromarray(img_u8)
    except (RuntimeError, ValueError):
        return None


def save_gif(
    frames: list[Image.Image],
    output_dir: str,
    filename: str,
    duration_ms: int = 100,
) -> None:
    """Save a list of PIL Images as an animated GIF."""
    if not frames or len(frames) < 1:
        return
    try:
        frames[0].save(
            os.path.join(output_dir, filename),
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
    except OSError:
        pass


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion.

    Returns (lower_bound, upper_bound) clamped to [0, 1].
    """
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    centre = p + z * z / (2 * total)
    rad = z * sqrt((p * (1.0 - p) + z * z / (4 * total)) / total)
    lo = (centre - rad) / denom
    hi = (centre + rad) / denom
    return max(0.0, lo), min(1.0, hi)


def sanitize_obs(obs_tensor: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf values in observation tensor with zeros."""
    from aerial_gym.utils.tensor_utils import sanitize_tensor
    return sanitize_tensor(obs_tensor)


def apply_obs_ablation(obs_tensor: torch.Tensor) -> torch.Tensor:
    """Apply observation ablation based on environment variables.

    Supports:
        ABLATE_DRONE_POS=1          — zero out indices 0:3
        ABLATE_OBS_RANGES=a:b=mode  — modify index ranges
            Modes: zero, zerograd, shuffle, noise:std

    Returns the (potentially modified) tensor. Does not modify in-place.
    """
    obs = obs_tensor.clone()

    # Zero out drone position if requested
    if os.environ.get("ABLATE_DRONE_POS", "").strip().lower() in ("1", "true", "yes"):
        obs[:, 0:3] = 0.0

    spec_str = os.environ.get("ABLATE_OBS_RANGES", "").strip()
    if not spec_str:
        return obs

    for spec in spec_str.split(","):
        spec = spec.strip()
        if not spec or "=" not in spec:
            continue
        lhs, rhs = spec.split("=", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if ":" not in lhs:
            continue
        try:
            a_str, b_str = lhs.split(":", 1)
            a, b = int(a_str), int(b_str)
        except ValueError:
            continue

        if rhs == "zero":
            obs[:, a:b] = 0.0
        elif rhs == "zerograd":
            obs[:, a:b] = 0.0
        elif rhs == "shuffle":
            perm = torch.randperm(obs.shape[0], device=obs.device)
            obs[:, a:b] = obs[perm, a:b]
        elif rhs.startswith("noise:"):
            try:
                std = float(rhs.split(":", 1)[1])
            except ValueError:
                std = 0.1
            obs[:, a:b] = obs[:, a:b] + torch.randn_like(obs[:, a:b]) * std

    return obs
