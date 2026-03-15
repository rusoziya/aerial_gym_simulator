"""Shared tensor utilities used across the codebase.

Consolidates NaN/Inf sanitization, tensor validation, and common
tensor operations that were previously duplicated across modules.
"""

from __future__ import annotations

import torch


def sanitize_tensor(
    t: torch.Tensor, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0
) -> torch.Tensor:
    """Replace NaN/Inf values in a tensor. Returns a new tensor."""
    return torch.nan_to_num(t, nan=nan, posinf=posinf, neginf=neginf)


def has_invalid(t: torch.Tensor) -> bool:
    """Check if a tensor contains any NaN or Inf values."""
    return bool(torch.any(torch.isnan(t) | torch.isinf(t)).item())


def invalid_mask_per_env(t: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask of shape (num_envs,) indicating which envs have NaN/Inf.

    Works for tensors of shape (num_envs,) or (num_envs, ...).
    """
    bad = torch.isnan(t) | torch.isinf(t)
    if bad.ndim > 1:
        return torch.any(bad, dim=tuple(range(1, bad.ndim)))
    return bad


def clamp_actions(actions: torch.Tensor, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    """Sanitize and clamp actions: replace NaN/Inf then clamp to [low, high]."""
    return sanitize_tensor(actions).clamp_(low, high)
