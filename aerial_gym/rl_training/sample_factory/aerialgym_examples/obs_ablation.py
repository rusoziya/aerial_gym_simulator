"""Observation ablation utilities for return-drop experiments."""

from __future__ import annotations

import os

import torch
from torch import Tensor

OBS_SLICES: dict[str, tuple[int, int]] = {
    "drone_position": (0, 3),
    "static_camera_pos": (3, 6),
    "static_camera_orient": (6, 9),
    "drone_orientation": (9, 12),
    "drone_linear_vel": (12, 15),
    "drone_angular_vel": (15, 18),
    "drone_actions": (18, 22),
    "drone_camera_vae": (22, 86),
    "static_camera_vae": (86, 150),
}


class ObsAblation:
    """Applies return-drop ablation controlled by environment variables.

    Supported env vars:
      - ABLATE_DRONE_POS=true           -> zero [0:3]
      - ABLATE_OBS_RANGES="0:3=zero,22:86=shuffle,86:150=noise:0.1,0:22=zerograd"
    Ops:
      - zero     : set slice to 0
      - zerograd : replace with constant zeros detached from graph
      - shuffle  : permute values across envs for this slice
      - noise:std: add Gaussian noise with given std
    """

    def __init__(self) -> None:
        self._debug_count: int = 0

    def apply(self, obs_tensor: Tensor) -> Tensor:
        if obs_tensor is None:
            return obs_tensor
        debug = os.environ.get("ABLATE_DEBUG", "false").lower() == "true"
        obs_tensor = self._apply_drone_pos_ablation(obs_tensor, debug)
        obs_tensor = self._apply_range_specs(obs_tensor, debug)
        return obs_tensor

    def _apply_drone_pos_ablation(self, obs_tensor: Tensor, debug: bool) -> Tensor:
        if os.environ.get("ABLATE_DRONE_POS", "false").lower() != "true":
            return obs_tensor
        start, end = OBS_SLICES.get("drone_position", (0, 3))
        obs_tensor[:, start:end] = 0.0
        if debug and self._debug_count < 10:
            v = obs_tensor[:, start:end]
            print(
                f"[ABLATE_DEBUG] applied: {start}:{end}=zero | "
                f"min={v.min().item():.3e} max={v.max().item():.3e} "
                f"mean={v.mean().item():.3e} nonzero={int((v != 0).sum().item())}"
            )
            self._debug_count += 1
        return obs_tensor

    def _apply_range_specs(self, obs_tensor: Tensor, debug: bool) -> Tensor:
        spec_str = os.environ.get("ABLATE_OBS_RANGES", "").strip()
        if not spec_str:
            return obs_tensor
        grad_mask: Tensor | None = None
        zero_ranges: list[tuple[int, int]] = []
        zerograd_ranges: list[tuple[int, int]] = []
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
                start_s, end_s = lhs.split(":", 1)
                start, end = int(start_s), int(end_s)
            except (ValueError, TypeError):
                continue
            obs_tensor, grad_mask = self._apply_single_op(
                obs_tensor, rhs, start, end, debug, grad_mask, zero_ranges, zerograd_ranges
            )
        if grad_mask is not None:
            obs_tensor = obs_tensor * grad_mask
            if debug:
                for s, e in zero_ranges:
                    if self._debug_count >= 10:
                        break
                    v = obs_tensor[:, s:e]
                    print(
                        f"[ABLATE_DEBUG] applied: {s}:{e}=zero | "
                        f"min={v.min().item():.3e} max={v.max().item():.3e} "
                        f"mean={v.mean().item():.3e} nonzero={int((v != 0).sum().item())}"
                    )
                    self._debug_count += 1
        for s, e in zerograd_ranges:
            zero_slice = torch.zeros_like(obs_tensor[:, s:e])
            left = obs_tensor[:, :s]
            right = obs_tensor[:, e:]
            obs_tensor = torch.cat([left, zero_slice, right], dim=-1)
            if debug and self._debug_count < 10:
                v = obs_tensor[:, s:e]
                print(
                    f"[ABLATE_DEBUG] applied: {s}:{e}=zerograd | "
                    f"min={v.min().item():.3e} max={v.max().item():.3e} "
                    f"mean={v.mean().item():.3e} nonzero={int((v != 0).sum().item())}"
                )
                self._debug_count += 1
        return obs_tensor

    def _apply_single_op(
        self,
        obs_tensor: Tensor,
        op: str,
        start: int,
        end: int,
        debug: bool,
        grad_mask: Tensor | None,
        zero_ranges: list[tuple[int, int]],
        zerograd_ranges: list[tuple[int, int]],
    ) -> tuple[Tensor, Tensor | None]:
        if op == "zero":
            if grad_mask is None:
                grad_mask = torch.ones_like(obs_tensor)
            grad_mask[:, start:end] = 0.0
            zero_ranges.append((start, end))
        elif op == "zerograd":
            zerograd_ranges.append((start, end))
        elif op == "shuffle":
            if obs_tensor.shape[0] > 1:
                perm = torch.randperm(obs_tensor.shape[0], device=obs_tensor.device)
                obs_tensor[:, start:end] = obs_tensor[perm, start:end]
            if debug and self._debug_count < 10:
                v = obs_tensor[:, start:end]
                env1 = v[1].detach().cpu().numpy() if v.shape[0] > 1 else "NA"
                print(
                    f"[ABLATE_DEBUG] applied: {start}:{end}=shuffle | "
                    f"sample_env0={v[0].detach().cpu().numpy()} sample_env1={env1}"
                )
                self._debug_count += 1
        elif op.startswith("noise:"):
            try:
                std = float(op.split(":", 1)[1])
            except (ValueError, TypeError):
                std = 0.0
            if std > 0.0:
                obs_tensor[:, start:end] += torch.randn_like(obs_tensor[:, start:end]) * std
            if debug and self._debug_count < 10:
                v = obs_tensor[:, start:end]
                print(
                    f"[ABLATE_DEBUG] applied: {start}:{end}=noise:{std} | "
                    f"std_est={v.std().item():.3e} mean={v.mean().item():.3e}"
                )
                self._debug_count += 1
        return obs_tensor, grad_mask
