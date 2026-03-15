from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def to_meters(t_norm: torch.Tensor, near: float, far: float) -> torch.Tensor:
    """Convert normalised [0,1] depth to meters."""
    return near + t_norm * (far - near)


def to_norm(meters: torch.Tensor, near: float, far: float) -> torch.Tensor:
    """Convert depth in meters back to normalised [0,1]."""
    return torch.clamp((meters - near) / (far - near), 0.0, 1.0)


def compute_inflation_kernel(
    hfov_deg: float,
    vfov_deg: float,
    image_w: int,
    image_h: int,
    robot_width_m: float,
    robot_height_m: float,
    ref_dist_m: float,
) -> tuple[int, int]:
    """Compute (kh, kw) inflation kernel sizes at a reference distance."""
    hfov = math.radians(hfov_deg)
    vfov = math.radians(vfov_deg)
    px_per_rad_w = image_w / hfov
    px_per_rad_h = image_h / vfov
    ang_w = 2.0 * math.atan(robot_width_m / (2.0 * max(ref_dist_m, 1e-3)))
    ang_h = 2.0 * math.atan(robot_height_m / (2.0 * max(ref_dist_m, 1e-3)))
    kw = max(1, int(2 * round(px_per_rad_w * ang_w / 2.0) + 1))
    kh = max(1, int(2 * round(px_per_rad_h * ang_h / 2.0) + 1))
    kw = int(min(max(kw, 3), 31) | 1)
    kh = int(min(max(kh, 3), 31) | 1)
    return kh, kw


def _collision_image_3d(
    d: torch.Tensor,
    near: float,
    far: float,
    hfov_deg: float,
    vfov_deg: float,
    image_w: int,
    image_h: int,
    robot_width_m: float,
    robot_height_m: float,
    z_bins: int,
) -> torch.Tensor:
    """Full 3D camera-space dilation (uvz) for collision image target."""
    B, C, H, W = d.shape
    out_coll: list[torch.Tensor] = []
    z_edges_global = torch.linspace(near, far, steps=z_bins + 1, device=d.device, dtype=d.dtype)
    z_centers_global = 0.5 * (z_edges_global[:-1] + z_edges_global[1:])
    # precompute 2D kernel sizes per z
    kw_z: list[int] = []
    kh_z: list[int] = []
    for zc in z_centers_global:
        px_w = (2.0 * float(zc)) * math.tan(math.radians(hfov_deg) / 2.0) / float(image_w)
        px_h = (2.0 * float(zc)) * math.tan(math.radians(vfov_deg) / 2.0) / float(image_h)
        kw_b = int(min(max(int(math.ceil(robot_width_m / max(px_w, 1e-6))), 3), 51) | 1)
        kh_b = int(min(max(int(math.ceil(robot_height_m / max(px_h, 1e-6))), 3), 51) | 1)
        kw_z.append(kw_b)
        kh_z.append(kh_b)
    for bi in range(B):
        d_b = d[bi : bi + 1]  # [1,1,H,W]
        z_edges = z_edges_global
        z_centers = z_centers_global
        d2 = d_b.squeeze(0).squeeze(0)  # [H,W]
        bin_idx = torch.bucketize(d2, z_edges) - 1
        bin_idx = torch.clamp(bin_idx, 0, z_bins - 1)
        occ = torch.zeros((z_bins, H, W), device=d.device, dtype=torch.bool)
        h_idx = torch.arange(H, device=d.device).view(H, 1).expand(H, W)
        w_idx = torch.arange(W, device=d.device).view(1, W).expand(H, W)
        occ[bin_idx.flatten(), h_idx.flatten(), w_idx.flatten()] = True
        dil = torch.zeros_like(occ)
        for zi in range(z_bins):
            if not occ[zi].any():
                continue
            kh_b = kh_z[zi]
            kw_b = kw_z[zi]
            slice_in = occ[zi].float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            slice_out = F.max_pool2d(
                slice_in,
                kernel_size=(kh_b, kw_b),
                stride=1,
                padding=(kh_b // 2, kw_b // 2),
            )
            dil[zi] = slice_out.squeeze(0).squeeze(0) > 0.5
        dil_shift_up = torch.zeros_like(dil)
        dil_shift_down = torch.zeros_like(dil)
        dil_shift_up[1:] = dil[:-1]
        dil_shift_down[:-1] = dil[1:]
        dil = dil | dil_shift_up | dil_shift_down
        inf = torch.full((z_bins, H, W), float("inf"), device=d.device)
        idx_vals = torch.arange(z_bins, device=d.device, dtype=torch.float32).view(z_bins, 1, 1)
        masked = torch.where(dil, idx_vals, inf)
        min_idx_vals, _ = torch.min(masked, dim=0)  # [H,W]
        valid = torch.isfinite(min_idx_vals)
        reproj = torch.full((H, W), far, device=d.device, dtype=d.dtype)
        min_idx_long = torch.clamp(min_idx_vals.long(), 0, z_bins - 1)
        reproj[valid] = z_centers[min_idx_long[valid]]
        out_coll.append(to_norm(reproj.unsqueeze(0).unsqueeze(0), near, far))
    return torch.cat(out_coll, dim=0)


def _collision_image_multibin(
    d: torch.Tensor,
    near: float,
    far: float,
    hfov_deg: float,
    vfov_deg: float,
    image_w: int,
    image_h: int,
    robot_width_m: float,
    robot_height_m: float,
    depth_bins: int,
) -> torch.Tensor:
    """Multi-bin variable kernel min-pooling inflation."""
    z_edges = torch.linspace(near, far, steps=depth_bins + 1, device=d.device, dtype=d.dtype)
    pooled_all = torch.zeros_like(d)
    for b in range(depth_bins):
        z0, z1 = z_edges[b], z_edges[b + 1]
        mid = 0.5 * (z0 + z1)
        px_w_m = (2.0 * mid) * math.tan(math.radians(hfov_deg) / 2.0) / float(image_w)
        px_h_m = (2.0 * mid) * math.tan(math.radians(vfov_deg) / 2.0) / float(image_h)
        kw_b = int(min(max(int(math.ceil(robot_width_m / max(px_w_m, 1e-6))), 3), 51) | 1)
        kh_b = int(min(max(int(math.ceil(robot_height_m / max(px_h_m, 1e-6))), 3), 51) | 1)
        d_neg = -d
        pooled_b = -F.max_pool2d(
            d_neg,
            kernel_size=(kh_b, kw_b),
            stride=1,
            padding=(kh_b // 2, kw_b // 2),
        )
        mask_b = (d >= z0) & (d < z1)
        pooled_all = torch.where(mask_b, pooled_b, pooled_all)
    return to_norm(pooled_all, near, far)


def collision_image(
    depth_norm: torch.Tensor,
    near: float,
    far: float,
    hfov_deg: float,
    vfov_deg: float,
    image_w: int,
    image_h: int,
    robot_width_m: float,
    robot_height_m: float,
    kh_const: int,
    kw_const: int,
    depth_bins: int,
    dilate3d: bool,
    z_bins: int,
) -> torch.Tensor:
    """Convert normalised depth to collision image via min-pooling inflation.

    All parameters that were previously captured from ``args`` are now explicit.
    """
    d = to_meters(depth_norm, near, far)
    if dilate3d:
        return _collision_image_3d(
            d,
            near,
            far,
            hfov_deg,
            vfov_deg,
            image_w,
            image_h,
            robot_width_m,
            robot_height_m,
            z_bins,
        )
    if depth_bins <= 1:
        d_neg = -d
        pooled = -F.max_pool2d(
            d_neg,
            kernel_size=(kh_const, kw_const),
            stride=1,
            padding=(kh_const // 2, kw_const // 2),
        )
        return to_norm(pooled, near, far)
    return _collision_image_multibin(
        d,
        near,
        far,
        hfov_deg,
        vfov_deg,
        image_w,
        image_h,
        robot_width_m,
        robot_height_m,
        depth_bins,
    )
