from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def load_index(index_csv, split):
    rows = []
    with open(index_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["split"] == split:
                rows.append(r)
    return rows


def read_image_gray(path, resize_wh):
    img = Image.open(path).convert("L").resize(resize_wh, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None, ...]  # [1,H,W]
    return t


def save_strip(panels, titles, out_path):
    # panels: list of [1,H,W] tensors in [0,1]
    H, W = panels[0].shape[-2:]
    n = len(panels)
    title_h = 22
    canvas = Image.new("L", (W * n, H + title_h), color=0)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    # paste images
    for i, p in enumerate(panels):
        img = Image.fromarray((p.squeeze().cpu().numpy() * 255).astype(np.uint8))
        canvas.paste(img, (i * W, title_h))
        # draw titles if provided
        if titles and i < len(titles):
            text = str(titles[i])
            if font is not None:
                # Use textbbox (Pillow >=9) to avoid deprecated textsize
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            else:
                tw, th = (len(text) * 6, 10)
            tx = i * W + max(2, (W - tw) // 2)
            ty = max(2, (title_h - th) // 2)
            draw.text((tx, ty), text, fill=200, font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def sobel_edges(t):
    gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=t.dtype, device=t.device).view(1, 1, 3, 3)
    gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=t.dtype, device=t.device).view(1, 1, 3, 3)
    ex = F.conv2d(t, gx, padding=1)
    ey = F.conv2d(t, gy, padding=1)
    mag = torch.sqrt(ex * ex + ey * ey + 1e-12)
    # Normalize to [0,1] per-image for visualization
    m = mag - mag.amin(dim=[1, 2, 3], keepdim=True)
    denom = (m.amax(dim=[1, 2, 3], keepdim=True) + 1e-6)
    return m / denom


def compute_inflation_kernel(image_w, image_h, hfov_deg, vfov_deg, robot_w_m, robot_h_m, ref_dist_m):
    hfov = math.radians(hfov_deg)
    vfov = math.radians(vfov_deg)
    px_per_rad_w = image_w / hfov
    px_per_rad_h = image_h / vfov
    ang_w = 2.0 * math.atan(robot_w_m / (2.0 * max(ref_dist_m, 1e-3)))
    ang_h = 2.0 * math.atan(robot_h_m / (2.0 * max(ref_dist_m, 1e-3)))
    kw = max(1, int(2 * round(px_per_rad_w * ang_w / 2.0) + 1))
    kh = max(1, int(2 * round(px_per_rad_h * ang_h / 2.0) + 1))
    kw = int(min(max(kw, 3), 31) | 1)
    kh = int(min(max(kh, 3), 31) | 1)
    return kh, kw


def to_meters(depth_norm, near, far):
    return near + depth_norm * (far - near)


def to_norm(meters, near, far):
    return torch.clamp((meters - near) / (far - near), 0.0, 1.0)


def collision_image_from_depth(depth_norm, near, far, kh, kw):
    d = to_meters(depth_norm, near, far)
    d_neg = -d
    pooled = -F.max_pool2d(d_neg, kernel_size=(kh, kw), stride=1, padding=(kh // 2, kw // 2))
    return to_norm(pooled, near, far)


def main():
    p = argparse.ArgumentParser(description="Visualize collision-image pipeline (depth → noise/dropout → edges → inflated depth → collision image)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--index_csv", type=str, help="Index CSV from extractor")
    src.add_argument("--images_glob", type=str, nargs="+", help="Glob(s) for input images")
    p.add_argument("--split", type=str, default="val", choices=["train","val","test"])
    p.add_argument("--out_dir", type=str, default="aerial_gym/utils/vae/val_reports/collision_viz")
    p.add_argument("--num_random", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--image_w", type=int, default=480)
    p.add_argument("--image_h", type=int, default=270)
    p.add_argument("--near", type=float, default=0.4)
    p.add_argument("--far", type=float, default=20.0)
    p.add_argument("--hfov_deg", type=float, default=87.0)
    p.add_argument("--vfov_deg", type=float, default=56.2)
    p.add_argument("--robot_width_m", type=float, default=0.5)
    p.add_argument("--robot_height_m", type=float, default=0.25)
    p.add_argument("--ref_dist_m", type=float, default=2.0)
    p.add_argument("--pixel_dropout", type=float, default=0.02)
    p.add_argument("--depth_noise_sigma", type=float, default=0.01)
    p.add_argument("--depth_bins", type=int, default=6, help="Number of depth bins for variable inflation (closer to paper)")
    p.add_argument("--dilate3d", action="store_true", help="Enable full 3D dilation in camera space (uvz voxel). Slower but closer to paper")
    p.add_argument("--z_bins", type=int, default=64, help="Number of depth voxels for 3D dilation mode")
    args = p.parse_args()

    resize_wh = (args.image_w, args.image_h)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # gather image paths
    img_paths = []
    if args.index_csv:
        rows = load_index(args.index_csv, args.split)
        img_paths = [Path(r["image_path"]) for r in rows]
    else:
        for g in args.images_glob:
            img_paths.extend(sorted(Path().glob(g)))

    if not img_paths:
        print("No input images found.")
        return

    if args.num_random and len(img_paths) > args.num_random:
        rng = random.Random(args.seed)
        img_paths = rng.sample(img_paths, args.num_random)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kh, kw = compute_inflation_kernel(args.image_w, args.image_h, args.hfov_deg, args.vfov_deg,
                                      args.robot_width_m, args.robot_height_m, args.ref_dist_m)
    print(f"Inflation kernel (kh,kw)=({kh},{kw})")

    for pth in img_paths:
        depth = read_image_gray(pth, resize_wh).to(device)  # [1, H, W]
        depth_b = depth.unsqueeze(0)  # [B=1,1,H,W]

        # step 1: edges on raw depth
        edges = sobel_edges(depth_b)

        # step 2: noise + dropout
        drop = torch.rand_like(depth_b).le(args.pixel_dropout).float()
        sigma = args.depth_noise_sigma * depth_b
        noise = torch.normal(mean=0.0, std=1.0, size=depth_b.shape, device=device) * sigma
        depth_noisy = torch.clamp(depth_b + noise, 0.0, 1.0)
        mask = (1.0 - drop)
        depth_noisy_masked = depth_noisy * mask

        # step 2: projection & mesh creation (approx): show kernel size map (depth-dependent)
        z_m = args.near + depth_b * (args.far - args.near)
        # per-pixel kernel (rounded later). Normalize map for visualization
        px_w_m = (2.0 * z_m) * math.tan(math.radians(args.hfov_deg) / 2.0) / args.image_w
        px_h_m = (2.0 * z_m) * math.tan(math.radians(args.vfov_deg) / 2.0) / args.image_h
        kw_map = torch.clamp((args.robot_width_m / (px_w_m + 1e-6)).ceil(), 3, 51)
        kh_map = torch.clamp((args.robot_height_m / (px_h_m + 1e-6)).ceil(), 3, 51)
        # force odd
        kw_map = (kw_map.to(torch.int32) | 1).float()
        kh_map = (kh_map.to(torch.int32) | 1).float()
        kvis = (kw_map - 3.0) / (51.0 - 3.0)

        # step 3: virtual mesh rendering
        if not args.dilate3d:
            # multi-bin variable inflation in image plane
            bins = max(1, args.depth_bins)
            z_lin = torch.linspace(args.near, args.far, bins + 1, device=device)
            inflated_norm = torch.zeros_like(depth_b)
            for b in range(bins):
                z0, z1 = z_lin[b], z_lin[b + 1]
                mask_b = (z_m >= z0) & (z_m < z1)
                # kernel for mid-depth of this bin
                z_mid = 0.5 * (z0 + z1)
                px_w = (2.0 * z_mid) * math.tan(math.radians(args.hfov_deg) / 2.0) / args.image_w
                px_h = (2.0 * z_mid) * math.tan(math.radians(args.vfov_deg) / 2.0) / args.image_h
                kw_b = int(min(max(int(math.ceil(args.robot_width_m / max(px_w, 1e-6))), 3), 51) | 1)
                kh_b = int(min(max(int(math.ceil(args.robot_height_m / max(px_h, 1e-6))), 3), 51) | 1)
                pooled_b = collision_image_from_depth(depth_noisy, args.near, args.far, kh_b, kw_b)
                inflated_norm = torch.where(mask_b, pooled_b, inflated_norm)
            xcoll = inflated_norm
        else:
            # full 3D dilation in camera space (uvz voxel grid)
            B, C, H, W = depth_b.shape
            z_edges = torch.linspace(args.near, args.far, args.z_bins + 1, device=device)
            z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])  # [Z]
            # occupancy grid: 1 where a voxel along ray is closer than measured depth
            # build per-pixel depth bin index
            d_m = z_m.squeeze(0).squeeze(0)  # [H,W]
            bin_idx = torch.bucketize(d_m, z_edges) - 1
            bin_idx = torch.clamp(bin_idx, 0, args.z_bins - 1)
            occ = torch.zeros((1, 1, args.z_bins, H, W), device=device)
            # mark all voxels from near up to measured bin as free, next as occupied shell
            # here we approximate obstacle at the measured bin
            occ[0, 0, bin_idx, torch.arange(H).unsqueeze(1), torch.arange(W)] = 1.0
            # compute voxel kernel sizes from robot size at each z
            kw_z = []
            kh_z = []
            for zc in z_centers:
                px_w = (2.0 * zc) * math.tan(math.radians(args.hfov_deg) / 2.0) / args.image_w
                px_h = (2.0 * zc) * math.tan(math.radians(args.vfov_deg) / 2.0) / args.image_h
                kw_b = int(min(max(int(math.ceil(args.robot_width_m / max(px_w, 1e-6))), 3), 51) | 1)
                kh_b = int(min(max(int(math.ceil(args.robot_height_m / max(px_h, 1e-6))), 3), 51) | 1)
                kw_z.append(kw_b)
                kh_z.append(kh_b)
            # 2D dilation per z-slice
            dil = torch.zeros_like(occ)
            for zi in range(args.z_bins):
                if occ[0, 0, zi].max() == 0:
                    continue
                kh_b = kh_z[zi]
                kw_b = kw_z[zi]
                dil[0, 0, zi] = F.max_pool2d(occ[0, 0, zi:zi+1], kernel_size=(kh_b, kw_b), stride=1, padding=(kh_b//2, kw_b//2)).squeeze(0)
            # 1D dilation along z (extend occupancy by one bin above/below)
            dil_z = torch.maximum(dil[:, :, 1:], dil[:, :, :-1])
            dil[:, :, 1:] = torch.maximum(dil[:, :, 1:], dil_z)
            dil[:, :, :-1] = torch.maximum(dil[:, :, :-1], dil_z)
            # reproject: for each pixel, find nearest occupied voxel depth
            occ_any = (dil > 0.5).squeeze(0).squeeze(0)  # [Z,H,W]
            # default to far plane
            reproj = torch.full((1, 1, H, W), fill_value=args.far, device=device)
            for zi in range(args.z_bins):
                mask_hit = occ_any[zi]
                reproj[0, 0][mask_hit] = torch.minimum(reproj[0, 0][mask_hit], z_centers[zi])
            xcoll = to_norm(reproj, args.near, args.far)

        panels = [depth_b[0], depth_noisy_masked[0], edges[0], kvis[0].clamp(0, 1), xcoll[0]]
        titles = ["depth_norm", "noisy+mask", "edges", "kernel_map", "collision_img"]
        out_file = out_root / f"{pth.stem}_pipeline.png"
        save_strip(panels, titles, out_file)

    print(f"Saved {len(img_paths)} pipeline visualizations to {out_root}")


if __name__ == "__main__":
    main()


