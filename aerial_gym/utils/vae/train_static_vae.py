import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Import VAE directly from the local file to avoid importing the top-level
# aerial_gym package (which pulls isaacgym and enforces torch import order).
from importlib.util import spec_from_file_location, module_from_spec

_vae_path = Path(__file__).with_name("VAE.py")
_spec = spec_from_file_location("VAE_local", str(_vae_path))
_mod = module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
VAE = _mod.VAE


def load_index(index_csv, split):
    rows = []
    with open(index_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["split"] == split:
                rows.append(r)
    return rows


class StaticDepthDataset(Dataset):
    def __init__(self, rows, image_wh=(240, 135), augment=False, seed=17):
        self.rows = rows
        self.W, self.H = image_wh
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.rows)

    def _read_gray(self, path):
        img = Image.open(path).convert("L").resize((self.W, self.H), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr)[None, ...]  # [1,H,W]
        return t

    def _augment(self, t):
        # t: [1,H,W] in [0,1]
        if self.rng.rand() < 0.15:
            # Pixel dropout (Bernoulli masking)
            drop_prob = self.rng.uniform(0.01, 0.05)
            mask = torch.from_numpy((self.rng.rand(*t.shape) > drop_prob).astype(np.float32))
            t = t * mask
        if self.rng.rand() < 0.15:
            # Additive white Gaussian noise
            sigma = self.rng.uniform(0.005, 0.02)
            noise = torch.from_numpy(self.rng.normal(0.0, sigma, size=t.shape).astype(np.float32))
            t = torch.clamp(t + noise, 0.0, 1.0)
        if self.rng.rand() < 0.10:
            # Multiplicative gain noise
            gain = self.rng.uniform(0.95, 1.05)
            t = torch.clamp(t * gain, 0.0, 1.0)
        if self.rng.rand() < 0.02:
            # Frame dropout (blank)
            t = torch.zeros_like(t)
        if self.rng.rand() < 0.02:
            # Frame freeze (slight blur approximation)
            t = torch.clamp(torch.nn.functional.avg_pool2d(t, kernel_size=3, stride=1, padding=1), 0.0, 1.0)
        return t

    def __getitem__(self, idx):
        path = self.rows[idx]["image_path"]
        t = self._read_gray(path)
        if self.augment:
            t = self._augment(t)
        return t


def kld(mu, logvar):
    # 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar ) per sample
    return 0.5 * torch.sum(torch.exp(logvar) + mu * mu - 1.0 - logvar, dim=1)


def main():
    p = argparse.ArgumentParser(description="Train VAE on static-camera depth frames.")
    p.add_argument("--index_csv", type=str, required=True)
    p.add_argument("--weights_out", type=str, default="aerial_gym/utils/vae/weights")
    p.add_argument("--latent_dims", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0, help="Final KL weight (beta) after warmup")
    p.add_argument("--beta_warmup_epochs", type=int, default=10, help="Epochs to warm up KL from 0 to beta")
    p.add_argument("--lambda_ssim", type=float, default=0.0, help="Weight for SSIM loss term (1-SSIM)")
    p.add_argument("--lambda_edge", type=float, default=0.0, help="Weight for Sobel edge L1 loss")
    # Collision-image training per paper (remap depth to collision image and use masked MSE)
    p.add_argument("--collision_image", action="store_true", help="Train to reconstruct collision image instead of raw depth")
    p.add_argument("--near", type=float, default=0.4, help="Near plane in meters")
    p.add_argument("--far", type=float, default=20.0, help="Far plane in meters")
    p.add_argument("--hfov_deg", type=float, default=87.0)
    p.add_argument("--vfov_deg", type=float, default=56.2)
    p.add_argument("--robot_width_m", type=float, default=0.5, help="Robot width (meters) for inflation")
    p.add_argument("--robot_height_m", type=float, default=0.25, help="Robot height (meters) for inflation")
    p.add_argument("--ref_dist_m", type=float, default=2.0, help="Reference distance for inflation kernel computation")
    p.add_argument("--depth_bins", type=int, default=6, help="Depth bins for variable inflation (>=1). 1 = fixed kernel at ref_dist_m")
    p.add_argument("--dilate3d", action="store_true", help="Use full 3D camera-space dilation (uvz) for collision image target")
    p.add_argument("--z_bins", type=int, default=64, help="Number of depth voxels for 3D dilation mode")
    p.add_argument("--pixel_dropout", type=float, default=0.02, help="Bernoulli pixel dropout prob")
    p.add_argument("--depth_noise_sigma", type=float, default=0.01, help="Depth-dependent noise multiplier (on [0,1] normalized depth)")
    p.add_argument("--image_w", type=int, default=480)
    p.add_argument("--image_h", type=int, default=270)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_every", type=int, default=1)
    # KL stabilization options
    p.add_argument("--kl_free_bits", type=float, default=0.0, help="Minimum KL per-dimension in nats (free-bits). Set >0 to enable")
    p.add_argument("--capacity_C_final", type=float, default=0.0, help="Target KL capacity C (nats). If >0, use capacity schedule instead of beta")
    p.add_argument("--capacity_warmup_epochs", type=int, default=0, help="Epochs to increase capacity from 0 to C_final")
    p.add_argument("--capacity_gamma", type=float, default=0.0, help="Gamma weight for capacity loss |KL-C| (recommend 50-200)")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    W, H = args.image_w, args.image_h

    # Data
    train_rows = load_index(args.index_csv, "train")
    val_rows = load_index(args.index_csv, "val")
    test_rows = load_index(args.index_csv, "test")

    train_ds = StaticDepthDataset(train_rows, image_wh=(W, H), augment=args.augment)
    val_ds = StaticDepthDataset(val_rows, image_wh=(W, H), augment=False)
    test_ds = StaticDepthDataset(test_rows, image_wh=(W, H), augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    vae = VAE(input_dim=1, latent_dim=args.latent_dims).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=args.lr)

    best_val = float("inf")
    out_dir = Path(args.weights_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Collision image helpers ---
    def to_meters(t_norm):
        return args.near + t_norm * (args.far - args.near)

    def to_norm(meters):
        return torch.clamp((meters - args.near) / (args.far - args.near), 0.0, 1.0)

    def compute_inflation_kernel():
        # approximate kernel sizes at reference distance using FOV and robot size
        hfov = math.radians(args.hfov_deg)
        vfov = math.radians(args.vfov_deg)
        px_per_rad_w = args.image_w / hfov
        px_per_rad_h = args.image_h / vfov
        ang_w = 2.0 * math.atan(args.robot_width_m / (2.0 * max(args.ref_dist_m, 1e-3)))
        ang_h = 2.0 * math.atan(args.robot_height_m / (2.0 * max(args.ref_dist_m, 1e-3)))
        kw = max(1, int(2 * round(px_per_rad_w * ang_w / 2.0) + 1))
        kh = max(1, int(2 * round(px_per_rad_h * ang_h / 2.0) + 1))
        # limit to reasonable odd sizes
        kw = int(min(max(kw, 3), 31) | 1)
        kh = int(min(max(kh, 3), 31) | 1)
        return kh, kw

    kh_const, kw_const = compute_inflation_kernel()

    def collision_image(depth_norm):
        # Convert to meters and apply min-pooling inflation. Supports depth-dependent multi-bin pooling
        # depth_norm: [B,1,H,W] in [0,1]
        d = to_meters(depth_norm)
        if args.dilate3d:
            # Memory-aware per-sample 3D dilation in camera space
            B, C, H, W = d.shape
            out_coll = []
            z_edges_global = torch.linspace(args.near, args.far, steps=args.z_bins + 1, device=d.device, dtype=d.dtype)
            z_centers_global = 0.5 * (z_edges_global[:-1] + z_edges_global[1:])
            # precompute 2D kernel sizes per z
            kw_z = []
            kh_z = []
            for zc in z_centers_global:
                px_w = (2.0 * float(zc)) * math.tan(math.radians(args.hfov_deg) / 2.0) / float(args.image_w)
                px_h = (2.0 * float(zc)) * math.tan(math.radians(args.vfov_deg) / 2.0) / float(args.image_h)
                kw_b = int(min(max(int(math.ceil(args.robot_width_m / max(px_w, 1e-6))), 3), 51) | 1)
                kh_b = int(min(max(int(math.ceil(args.robot_height_m / max(px_h, 1e-6))), 3), 51) | 1)
                kw_z.append(kw_b)
                kh_z.append(kh_b)
            for bi in range(B):
                d_b = d[bi:bi+1]  # [1,1,H,W]
                z_edges = z_edges_global
                z_centers = z_centers_global
                # bin index per pixel
                d2 = d_b.squeeze(0).squeeze(0)  # [H,W]
                bin_idx = torch.bucketize(d2, z_edges) - 1
                bin_idx = torch.clamp(bin_idx, 0, args.z_bins - 1)
                # occupancy grid [Z,H,W] sparse (bool)
                occ = torch.zeros((args.z_bins, H, W), device=d.device, dtype=torch.bool)
                h_idx = torch.arange(H, device=d.device).view(H, 1).expand(H, W)
                w_idx = torch.arange(W, device=d.device).view(1, W).expand(H, W)
                occ[bin_idx.flatten(), h_idx.flatten(), w_idx.flatten()] = True
                # per-slice 2D dilation
                dil = torch.zeros_like(occ)
                for zi in range(args.z_bins):
                    if not occ[zi].any():
                        continue
                    kh_b = kh_z[zi]
                    kw_b = kw_z[zi]
                    slice_in = occ[zi].float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                    slice_out = F.max_pool2d(slice_in, kernel_size=(kh_b, kw_b), stride=1, padding=(kh_b//2, kw_b//2))
                    dil[zi] = (slice_out.squeeze(0).squeeze(0) > 0.5)
                # 1D dilation along z (neighbor bins)
                dil_shift_up = torch.zeros_like(dil)
                dil_shift_down = torch.zeros_like(dil)
                dil_shift_up[1:] = dil[:-1]
                dil_shift_down[:-1] = dil[1:]
                dil = dil | dil_shift_up | dil_shift_down
                # nearest occupied depth per pixel
                inf = torch.full((args.z_bins, H, W), float('inf'), device=d.device)
                idx_vals = torch.arange(args.z_bins, device=d.device, dtype=torch.float32).view(args.z_bins, 1, 1)
                masked = torch.where(dil, idx_vals, inf)
                min_idx_vals, _ = torch.min(masked, dim=0)  # [H,W]
                valid = torch.isfinite(min_idx_vals)
                reproj = torch.full((H, W), args.far, device=d.device, dtype=d.dtype)
                min_idx_long = torch.clamp(min_idx_vals.long(), 0, args.z_bins - 1)
                reproj[valid] = z_centers[min_idx_long[valid]]
                out_coll.append(to_norm(reproj.unsqueeze(0).unsqueeze(0)))
            return torch.cat(out_coll, dim=0)
        if args.depth_bins <= 1:
            d_neg = -d
            pooled = -F.max_pool2d(d_neg, kernel_size=(kh_const, kw_const), stride=1, padding=(kh_const//2, kw_const//2))
            return to_norm(pooled)
        # multi-bin variable kernel
        z_edges = torch.linspace(args.near, args.far, steps=args.depth_bins + 1, device=d.device, dtype=d.dtype)
        pooled_all = torch.zeros_like(d)
        for b in range(args.depth_bins):
            z0, z1 = z_edges[b], z_edges[b + 1]
            mid = 0.5 * (z0 + z1)
            # pixel size in meters at mid-depth
            px_w_m = (2.0 * mid) * math.tan(math.radians(args.hfov_deg) / 2.0) / float(args.image_w)
            px_h_m = (2.0 * mid) * math.tan(math.radians(args.vfov_deg) / 2.0) / float(args.image_h)
            kw_b = int(min(max(int(math.ceil(args.robot_width_m / max(px_w_m, 1e-6))), 3), 51) | 1)
            kh_b = int(min(max(int(math.ceil(args.robot_height_m / max(px_h_m, 1e-6))), 3), 51) | 1)
            d_neg = -d
            pooled_b = -F.max_pool2d(d_neg, kernel_size=(kh_b, kw_b), stride=1, padding=(kh_b//2, kw_b//2))
            mask_b = (d >= z0) & (d < z1)
            pooled_all = torch.where(mask_b, pooled_b, pooled_all)
        return to_norm(pooled_all)

    # --- Loss helpers: local SSIM and Sobel edge ---
    def ssim_local(x, y, window=7, C1=0.01 ** 2, C2=0.03 ** 2):
        # x,y: [B,1,H,W] in [0,1]
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
        return ssim_map.mean(dim=[1, 2, 3])  # per-sample

    def sobel_edges(t):
        # t: [B,1,H,W]
        gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=t.dtype, device=t.device).view(1, 1, 3, 3)
        gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=t.dtype, device=t.device).view(1, 1, 3, 3)
        ex = F.conv2d(t, gx, padding=1)
        ey = F.conv2d(t, gy, padding=1)
        return ex, ey

    def run_epoch(loader, train: bool, epoch_idx: int):
        vae.train(train)
        total = 0.0
        count = 0
        total_l1 = 0.0
        total_ssim = 0.0
        total_edge = 0.0
        total_kl = 0.0
        total_kl_eff = 0.0
        # KL warmup
        beta_t = float(args.beta) * min(1.0, max(0.0, epoch_idx / max(1, args.beta_warmup_epochs)))
        # Capacity schedule
        if args.capacity_C_final > 0.0 and args.capacity_gamma > 0.0:
            if args.capacity_warmup_epochs > 0:
                C_t = float(args.capacity_C_final) * min(1.0, max(0.0, epoch_idx / float(args.capacity_warmup_epochs)))
            else:
                C_t = float(args.capacity_C_final)
        else:
            C_t = 0.0
        for batch in loader:
            batch = batch.to(device)  # normalized depth in [0,1]
            if train:
                opt.zero_grad(set_to_none=True)
            # Simulate sensor noise and invalids (Bernoulli dropout and depth-dependent noise)
            if args.collision_image:
                if args.pixel_dropout > 0.0:
                    drop = torch.rand_like(batch).le(args.pixel_dropout).float()
                else:
                    drop = torch.zeros_like(batch)
                if args.depth_noise_sigma > 0.0:
                    sigma = args.depth_noise_sigma * batch
                    noise = torch.normal(mean=0.0, std=1.0, size=batch.shape, device=batch.device) * sigma
                else:
                    noise = torch.zeros_like(batch)
                noisy = torch.clamp(batch + noise, 0.0, 1.0)
                # collision target and mask
                target = collision_image(noisy)
                mask = (1.0 - drop)
            else:
                target = batch
                mask = torch.ones_like(batch)

            # Per paper: encode noisy depth, reconstruct collision image
            enc_in = noisy if args.collision_image else batch
            recon, mu, logvar, _ = vae.forward(enc_in)
            # Reconstruction loss: masked MSE (per-sample)
            mse = (recon - target) ** 2
            # avoid division by zero if mask sums to 0 (unlikely)
            denom = mask.mean(dim=[1, 2, 3]).clamp_min(1e-6)
            l1 = (mse * mask).mean(dim=[1, 2, 3]) / denom  # use MSE as primary as in paper
            # Optional structure terms (default 0)
            ssim_val = ssim_local(recon, target)  # per-sample
            l_ssim = 1.0 - ssim_val
            rx, ry = sobel_edges(recon)
            bx, by = sobel_edges(target)
            l_edge = (F.l1_loss(rx, bx, reduction="none").mean(dim=[1, 2, 3]) +
                      F.l1_loss(ry, by, reduction="none").mean(dim=[1, 2, 3])) * 0.5
            # KL per sample (sum over dims)
            # Option A: Capacity schedule -> gamma * |KL - C_t|
            # Option B: Free-bits -> clamp per-dim KL before summation
            if args.capacity_C_final > 0.0 and args.capacity_gamma > 0.0:
                kl_sum = kld(mu, logvar)
                kl_eff = args.capacity_gamma * torch.abs(kl_sum - C_t)
                loss = (l1 + args.lambda_ssim * l_ssim + args.lambda_edge * l_edge + kl_eff).mean()
                kl_for_log = kl_sum
                kl_eff_for_log = kl_eff
            else:
                if args.kl_free_bits > 0.0:
                    # per-dimension KL with free-bits threshold
                    kl_per_dim = 0.5 * (torch.exp(logvar) + mu * mu - 1.0 - logvar)  # [B, D]
                    kl_clamped = torch.clamp(kl_per_dim, min=float(args.kl_free_bits))
                    kl_sum = torch.sum(kl_clamped, dim=1)
                else:
                    kl_sum = kld(mu, logvar)
                kl_eff = beta_t * kl_sum
                loss = (l1 + args.lambda_ssim * l_ssim + args.lambda_edge * l_edge + kl_eff).mean()
                kl_for_log = kl_sum
                kl_eff_for_log = kl_eff
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
                opt.step()
            total += loss.item() * batch.size(0)
            count += batch.size(0)
            total_l1 += l1.mean().item() * batch.size(0)
            total_ssim += l_ssim.mean().item() * batch.size(0)
            total_edge += l_edge.mean().item() * batch.size(0)
            total_kl += kl_for_log.mean().item() * batch.size(0)
            total_kl_eff += kl_eff_for_log.mean().item() * batch.size(0)
        avg = total / max(count, 1)
        comps = {
            "l1": total_l1 / max(count, 1),
            "ssim": total_ssim / max(count, 1),
            "edge": total_edge / max(count, 1),
            "kl": total_kl / max(count, 1),
            "kl_eff": total_kl_eff / max(count, 1),
            "beta_t": beta_t,
            "C_t": C_t,
        }
        return avg, comps

    history = {}
    for epoch in range(1, args.epochs + 1):
        train_loss, train_comps = run_epoch(train_loader, train=True, epoch_idx=epoch)
        if epoch % args.val_every == 0:
            val_loss, val_comps = run_epoch(val_loader, train=False, epoch_idx=epoch)
        else:
            val_loss, val_comps = float("nan"), {}
        if args.capacity_C_final > 0.0 and args.capacity_gamma > 0.0:
            print(
                f"Epoch {epoch:03d}: train={train_loss:.5f} (L1={train_comps['l1']:.4f}, SSIM={train_comps['ssim']:.4f}, "
                f"EDGE={train_comps['edge']:.4f}, KL={train_comps['kl']:.4f}, KL_eff={train_comps['kl_eff']:.4f}, C={train_comps['C_t']:.2f}, gamma={args.capacity_gamma:.1f}) "
                f"val={val_loss:.5f}"
            )
        else:
            print(
                f"Epoch {epoch:03d}: train={train_loss:.5f} (L1={train_comps['l1']:.4f}, SSIM={train_comps['ssim']:.4f}, "
                f"EDGE={train_comps['edge']:.4f}, KL={train_comps['kl']:.4f}, KL_eff={train_comps['kl_eff']:.4f}, beta={train_comps['beta_t']:.3f}) "
                f"val={val_loss:.5f}"
            )
        history[epoch] = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_components": train_comps,
            "val_components": val_comps,
        }

        if not math.isnan(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_path = out_dir / f"vae_L{args.latent_dims}_beta{args.beta}_best.pth"
            torch.save(vae.state_dict(), best_path)
            print(f"Saved best to {best_path}")

    last_path = out_dir / f"vae_L{args.latent_dims}_beta{args.beta}_last.pth"
    torch.save(vae.state_dict(), last_path)
    print(f"Saved last to {last_path}")

    # Quick test loss
    test_loss, test_comps = run_epoch(test_loader, train=False, epoch_idx=args.epochs)
    report = {"best_val": best_val, "test_loss": test_loss, "epochs": args.epochs, "test_components": test_comps}
    with open(out_dir / f"train_report_L{args.latent_dims}_beta{args.beta}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


