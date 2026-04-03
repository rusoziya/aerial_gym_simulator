from __future__ import annotations

import torch
import torch.nn.functional as F


def kld(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence: 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar ) per sample."""
    return 0.5 * torch.sum(torch.exp(logvar) + mu * mu - 1.0 - logvar, dim=1)


def ssim_local(
    x: torch.Tensor,
    y: torch.Tensor,
    window: int = 7,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
) -> torch.Tensor:
    """Local SSIM between x and y, both [B,1,H,W] in [0,1]. Returns per-sample score."""
    pad = window // 2
    mu_x = F.avg_pool2d(x, kernel_size=window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, kernel_size=window, stride=1, padding=pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x = F.avg_pool2d(x * x, window, 1, pad) - mu_x2
    sigma_y = F.avg_pool2d(y * y, window, 1, pad) - mu_y2
    sigma_xy = F.avg_pool2d(x * y, window, 1, pad) - mu_xy
    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = num / (den + 1e-12)
    return ssim_map.mean(dim=[1, 2, 3])


def sobel_edges(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Sobel edge maps for [B,1,H,W] tensor. Returns (edges_x, edges_y)."""
    gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=t.dtype, device=t.device).view(
        1, 1, 3, 3
    )
    gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=t.dtype, device=t.device).view(
        1, 1, 3, 3
    )
    ex = F.conv2d(t, gx, padding=1)
    ey = F.conv2d(t, gy, padding=1)
    return ex, ey
