from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Import VAE directly from local file to avoid pulling aerial_gym/isaacgym
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path as _PathLocal

_vae_path = _PathLocal(__file__).with_name("VAE.py")
_spec = spec_from_file_location("VAE_local", str(_vae_path))
_mod = module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
VAE = _mod.VAE


def clean_state_dict(state_dict):
    clean = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "")
        if k.startswith("dronet."):
            k = k.replace("dronet.", "encoder.")
        clean[k] = v
    return clean


def load_index(index_csv, split):
    rows = []
    with open(index_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["split"] != split:
                continue
            rows.append(r)
    return rows


def read_image_gray(path, resize_wh=None):
    # Returns float32 tensor in [0,1] with shape [1, H, W]
    img = Image.open(path).convert("L")
    if resize_wh is not None:
        img = img.resize(resize_wh, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None, ...]  # [1,H,W]
    return t


def psnr(x, y, max_val=1.0):
    mse = F.mse_loss(x, y).item()
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def psnr_masked(x, y, mask, max_val=1.0):
    # x,y,mask: [1,H,W] in [0,1]; mask in {0,1}
    mse = ((x - y) ** 2 * mask).sum() / (mask.sum() + 1e-12)
    mse = float(mse.item())
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def ssim_torch(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    # x,y: [1,H,W] in [0,1]; simple global SSIM (not windowed)
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var(unbiased=False)
    sigma_y = y.var(unbiased=False)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return (num / (den + 1e-12)).item()


def save_grid(samples, out_path, ncols=8):
    # samples: list of (input_t, recon_t) tensors in [0,1] [1,H,W]
    if not samples:
        return
    H, W = samples[0][0].shape[-2:]
    n = len(samples)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    # Each row: input|recon pairs stacked horizontally
    cell_w = W * 2
    grid = Image.new("L", (cell_w * ncols, H * nrows))
    for idx, (inp, rec) in enumerate(samples):
        r = idx // ncols
        c = idx % ncols
        x0 = c * cell_w
        y0 = r * H
        inp_img = Image.fromarray((inp.squeeze().cpu().numpy() * 255).astype(np.uint8))
        rec_img = Image.fromarray((rec.squeeze().cpu().numpy() * 255).astype(np.uint8))
        grid.paste(inp_img, (x0, y0))
        grid.paste(rec_img, (x0 + W, y0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main():
    p = argparse.ArgumentParser(description="Validate/test VAE on static-camera depth frames.")
    p.add_argument("--index_csv", type=str, required=True, help="CSV built by extract_static_vae_frames.py")
    p.add_argument("--weights", type=str, required=True, help="Path to VAE checkpoint .pth")
    p.add_argument("--latent_dims", type=int, default=64)
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"]) 
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--image_w", type=int, default=480)
    p.add_argument("--image_h", type=int, default=270)
    p.add_argument("--max_items", type=int, default=None, help="Optional cap on items to evaluate")
    p.add_argument("--out_dir", type=str, default="aerial_gym/utils/vae/val_reports")
    p.add_argument("--save_grid", action="store_true")
    # Collision-image evaluation (match training target)
    p.add_argument("--eval_collision", action="store_true", help="Compute collision target and evaluate recon vs. it")
    p.add_argument("--near", type=float, default=0.4)
    p.add_argument("--far", type=float, default=20.0)
    p.add_argument("--hfov_deg", type=float, default=87.0)
    p.add_argument("--vfov_deg", type=float, default=56.2)
    p.add_argument("--robot_width_m", type=float, default=0.5)
    p.add_argument("--robot_height_m", type=float, default=0.25)
    p.add_argument("--depth_bins", type=int, default=16)
    # Optional: match training input (noise + dropout).
    p.add_argument("--pixel_dropout", type=float, default=0.0)
    p.add_argument("--depth_noise_sigma", type=float, default=0.0)

    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows = load_index(args.index_csv, args.split)
    if args.max_items is not None:
        rows = rows[: args.max_items]
    if not rows:
        print("No items found for split.")
        return

    # Load model
    vae = VAE(input_dim=1, latent_dim=args.latent_dims, inference_mode=True).to(device)
    state = clean_state_dict(torch.load(args.weights, map_location="cpu"))
    vae.load_state_dict(state)
    vae.eval()

    def to_meters(t_norm):
        return args.near + t_norm * (args.far - args.near)

    def to_norm(meters):
        return torch.clamp((meters - args.near) / (args.far - args.near), 0.0, 1.0)

    def collision_image(depth_norm):
        # depth-dependent multi-bin min-pooling
        B, C, H, W = depth_norm.shape
        z_m = to_meters(depth_norm)
        bins = max(1, args.depth_bins)
        z_lin = torch.linspace(args.near, args.far, bins + 1, device=depth_norm.device)
        out = torch.zeros_like(depth_norm)
        for b in range(bins):
            z0, z1 = z_lin[b], z_lin[b + 1]
            mask_b = (z_m >= z0) & (z_m < z1)
            z_mid = 0.5 * (z0 + z1)
            px_w = (2.0 * z_mid) * math.tan(math.radians(args.hfov_deg) / 2.0) / args.image_w
            px_h = (2.0 * z_mid) * math.tan(math.radians(args.vfov_deg) / 2.0) / args.image_h
            kw_b = int(min(max(int(math.ceil(args.robot_width_m / max(px_w, 1e-6))), 3), 51) | 1)
            kh_b = int(min(max(int(math.ceil(args.robot_height_m / max(px_h, 1e-6))), 3), 51) | 1)
            d = z_m
            pooled_b = -F.max_pool2d(-d, kernel_size=(kh_b, kw_b), stride=1, padding=(kh_b // 2, kw_b // 2))
            out = torch.where(mask_b, to_norm(pooled_b), out)
        return out

    # Metrics
    psnrs = []
    ssims = []
    l1s = []
    kl_mu_norms = []  # mean latent norm statistics (proxy)
    sample_pairs = []

    with torch.no_grad():
        for i, r in enumerate(rows):
            img_path = Path(r["image_path"])  # absolute or relative
            t = read_image_gray(img_path, resize_wh=(args.image_w, args.image_h)).to(device)
            t_bchw = t.unsqueeze(0)  # [1,1,H,W]

            # Optional: replicate training input noise/dropout
            if args.pixel_dropout > 0.0:
                drop = torch.rand_like(t_bchw).le(args.pixel_dropout).float()
            else:
                drop = torch.zeros_like(t_bchw)
            if args.depth_noise_sigma > 0.0:
                sigma = args.depth_noise_sigma * t_bchw
                noise = torch.normal(mean=0.0, std=1.0, size=t_bchw.shape, device=device) * sigma
            else:
                noise = torch.zeros_like(t_bchw)
            noisy = torch.clamp(t_bchw + noise, 0.0, 1.0)
            mask = (1.0 - drop).squeeze(0)  # [1,H,W]

            enc_in = noisy if args.eval_collision else t_bchw
            recon, mu, logvar, z = vae.forward(enc_in)
            target = t  # default: raw depth
            if args.eval_collision:
                target = collision_image(noisy).squeeze(0)
            recon = recon.squeeze(0)  # [1,H,W]

            # Metrics per frame
            if args.eval_collision and args.pixel_dropout > 0.0:
                psnrs.append(psnr_masked(target, recon, mask))
                l1s.append((F.l1_loss(target * mask, recon * mask, reduction="sum") / (mask.sum() + 1e-12)).item())
            else:
                psnrs.append(psnr(target, recon))
                l1s.append(F.l1_loss(target, recon).item())
            ssims.append(ssim_torch(target, recon))
            kl_mu_norms.append(mu.squeeze(0).norm(p=2).item())

            if args.save_grid and len(sample_pairs) < 32:
                sample_pairs.append((target.detach().cpu(), recon.detach().cpu()))

    report = {
        "split": args.split,
        "count": len(rows),
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std": float(np.std(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_std": float(np.std(ssims)),
        "l1_mean": float(np.mean(l1s)),
        "l1_std": float(np.std(l1s)),
        "latent_mu_norm_mean": float(np.mean(kl_mu_norms)),
        "latent_mu_norm_std": float(np.std(kl_mu_norms)),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"report_{args.split}.json", "w") as f:
        json.dump(report, f, indent=2)

    if args.save_grid and sample_pairs:
        save_grid(sample_pairs, out_dir / f"grid_{args.split}.png", ncols=8)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


