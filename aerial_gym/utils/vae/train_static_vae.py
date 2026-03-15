from __future__ import annotations

import argparse
import json
import math

# Import VAE directly from the local file to avoid importing the top-level
# aerial_gym package (which pulls isaacgym and enforces torch import order).
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aerial_gym.utils.vae.collision_image import (
    collision_image,
    compute_inflation_kernel,
)
from aerial_gym.utils.vae.static_depth_dataset import StaticDepthDataset, load_index
from aerial_gym.utils.vae.vae_losses import kld, sobel_edges, ssim_local

_vae_path = Path(__file__).with_name("VAE.py")
_spec = spec_from_file_location("VAE_local", str(_vae_path))
_mod = module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
VAE = _mod.VAE


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train VAE on static-camera depth frames.")
    p.add_argument("--index_csv", type=str, required=True)
    p.add_argument("--weights_out", type=str, default="aerial_gym/utils/vae/weights")
    p.add_argument("--latent_dims", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0, help="Final KL weight (beta) after warmup")
    p.add_argument(
        "--beta_warmup_epochs", type=int, default=10, help="Epochs to warm up KL from 0 to beta"
    )
    p.add_argument(
        "--lambda_ssim", type=float, default=0.0, help="Weight for SSIM loss term (1-SSIM)"
    )
    p.add_argument("--lambda_edge", type=float, default=0.0, help="Weight for Sobel edge L1 loss")
    p.add_argument(
        "--collision_image",
        action="store_true",
        help="Train to reconstruct collision image instead of raw depth",
    )
    p.add_argument("--near", type=float, default=0.4, help="Near plane in meters")
    p.add_argument("--far", type=float, default=20.0, help="Far plane in meters")
    p.add_argument("--hfov_deg", type=float, default=87.0)
    p.add_argument("--vfov_deg", type=float, default=56.2)
    p.add_argument(
        "--robot_width_m", type=float, default=0.5, help="Robot width (meters) for inflation"
    )
    p.add_argument(
        "--robot_height_m", type=float, default=0.25, help="Robot height (meters) for inflation"
    )
    p.add_argument(
        "--ref_dist_m",
        type=float,
        default=2.0,
        help="Reference distance for inflation kernel computation",
    )
    p.add_argument(
        "--depth_bins",
        type=int,
        default=6,
        help="Depth bins for variable inflation (>=1). 1 = fixed kernel at ref_dist_m",
    )
    p.add_argument(
        "--dilate3d",
        action="store_true",
        help="Use full 3D camera-space dilation (uvz) for collision image target",
    )
    p.add_argument(
        "--z_bins", type=int, default=64, help="Number of depth voxels for 3D dilation mode"
    )
    p.add_argument("--pixel_dropout", type=float, default=0.02, help="Bernoulli pixel dropout prob")
    p.add_argument(
        "--depth_noise_sigma",
        type=float,
        default=0.01,
        help="Depth-dependent noise multiplier (on [0,1] normalized depth)",
    )
    p.add_argument("--image_w", type=int, default=480)
    p.add_argument("--image_h", type=int, default=270)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument(
        "--kl_free_bits",
        type=float,
        default=0.0,
        help="Minimum KL per-dimension in nats (free-bits). Set >0 to enable",
    )
    p.add_argument(
        "--capacity_C_final",
        type=float,
        default=0.0,
        help="Target KL capacity C (nats). If >0, use capacity schedule instead of beta",
    )
    p.add_argument(
        "--capacity_warmup_epochs",
        type=int,
        default=0,
        help="Epochs to increase capacity from 0 to C_final",
    )
    p.add_argument(
        "--capacity_gamma",
        type=float,
        default=0.0,
        help="Gamma weight for capacity loss |KL-C| (recommend 50-200)",
    )
    return p


def _run_epoch(
    loader: DataLoader,
    train: bool,
    epoch_idx: int,
    args: argparse.Namespace,
    vae: torch.nn.Module,
    opt: torch.optim.Optimizer,
    device: torch.device,
    kh_const: int,
    kw_const: int,
) -> tuple[float, dict[str, float]]:
    vae.train(train)
    total = 0.0
    count = 0
    total_l1 = 0.0
    total_ssim = 0.0
    total_edge = 0.0
    total_kl = 0.0
    total_kl_eff = 0.0
    beta_t = float(args.beta) * min(1.0, max(0.0, epoch_idx / max(1, args.beta_warmup_epochs)))
    if args.capacity_C_final > 0.0 and args.capacity_gamma > 0.0:
        if args.capacity_warmup_epochs > 0:
            C_t = float(args.capacity_C_final) * min(
                1.0, max(0.0, epoch_idx / float(args.capacity_warmup_epochs))
            )
        else:
            C_t = float(args.capacity_C_final)
    else:
        C_t = 0.0
    for batch in loader:
        batch = batch.to(device)
        if train:
            opt.zero_grad(set_to_none=True)
        if args.collision_image:
            if args.pixel_dropout > 0.0:
                drop = torch.rand_like(batch).le(args.pixel_dropout).float()
            else:
                drop = torch.zeros_like(batch)
            if args.depth_noise_sigma > 0.0:
                sigma = args.depth_noise_sigma * batch
                noise = (
                    torch.normal(mean=0.0, std=1.0, size=batch.shape, device=batch.device) * sigma
                )
            else:
                noise = torch.zeros_like(batch)
            noisy = torch.clamp(batch + noise, 0.0, 1.0)
            target = collision_image(
                noisy,
                near=args.near,
                far=args.far,
                hfov_deg=args.hfov_deg,
                vfov_deg=args.vfov_deg,
                image_w=args.image_w,
                image_h=args.image_h,
                robot_width_m=args.robot_width_m,
                robot_height_m=args.robot_height_m,
                kh_const=kh_const,
                kw_const=kw_const,
                depth_bins=args.depth_bins,
                dilate3d=args.dilate3d,
                z_bins=args.z_bins,
            )
            mask = 1.0 - drop
        else:
            target = batch
            mask = torch.ones_like(batch)

        enc_in = noisy if args.collision_image else batch
        recon, mu, logvar, _ = vae.forward(enc_in)
        mse = (recon - target) ** 2
        denom = mask.mean(dim=[1, 2, 3]).clamp_min(1e-6)
        l1 = (mse * mask).mean(dim=[1, 2, 3]) / denom
        ssim_val = ssim_local(recon, target)
        l_ssim = 1.0 - ssim_val
        rx, ry = sobel_edges(recon)
        bx, by = sobel_edges(target)
        l_edge = (
            F.l1_loss(rx, bx, reduction="none").mean(dim=[1, 2, 3])
            + F.l1_loss(ry, by, reduction="none").mean(dim=[1, 2, 3])
        ) * 0.5
        if args.capacity_C_final > 0.0 and args.capacity_gamma > 0.0:
            kl_sum = kld(mu, logvar)
            kl_eff = args.capacity_gamma * torch.abs(kl_sum - C_t)
            loss = (l1 + args.lambda_ssim * l_ssim + args.lambda_edge * l_edge + kl_eff).mean()
            kl_for_log = kl_sum
            kl_eff_for_log = kl_eff
        else:
            if args.kl_free_bits > 0.0:
                kl_per_dim = 0.5 * (torch.exp(logvar) + mu * mu - 1.0 - logvar)
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


def _log_epoch(
    epoch: int,
    train_loss: float,
    train_comps: dict[str, float],
    val_loss: float,
    args: argparse.Namespace,
) -> None:
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


def main() -> None:
    args = _build_parser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    W, H = args.image_w, args.image_h

    train_rows = load_index(args.index_csv, "train")
    val_rows = load_index(args.index_csv, "val")
    test_rows = load_index(args.index_csv, "test")

    train_ds = StaticDepthDataset(train_rows, image_wh=(W, H), augment=args.augment)
    val_ds = StaticDepthDataset(val_rows, image_wh=(W, H), augment=False)
    test_ds = StaticDepthDataset(test_rows, image_wh=(W, H), augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    vae = VAE(input_dim=1, latent_dim=args.latent_dims).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=args.lr)

    best_val = float("inf")
    out_dir = Path(args.weights_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    kh_const, kw_const = compute_inflation_kernel(
        hfov_deg=args.hfov_deg,
        vfov_deg=args.vfov_deg,
        image_w=args.image_w,
        image_h=args.image_h,
        robot_width_m=args.robot_width_m,
        robot_height_m=args.robot_height_m,
        ref_dist_m=args.ref_dist_m,
    )

    history: dict[int, dict[str, float | dict[str, float]]] = {}
    for epoch in range(1, args.epochs + 1):
        train_loss, train_comps = _run_epoch(
            train_loader,
            train=True,
            epoch_idx=epoch,
            args=args,
            vae=vae,
            opt=opt,
            device=device,
            kh_const=kh_const,
            kw_const=kw_const,
        )
        if epoch % args.val_every == 0:
            val_loss, val_comps = _run_epoch(
                val_loader,
                train=False,
                epoch_idx=epoch,
                args=args,
                vae=vae,
                opt=opt,
                device=device,
                kh_const=kh_const,
                kw_const=kw_const,
            )
        else:
            val_loss, val_comps = float("nan"), {}
        _log_epoch(epoch, train_loss, train_comps, val_loss, args)
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

    test_loss, test_comps = _run_epoch(
        test_loader,
        train=False,
        epoch_idx=args.epochs,
        args=args,
        vae=vae,
        opt=opt,
        device=device,
        kh_const=kh_const,
        kw_const=kw_const,
    )
    report = {
        "best_val": best_val,
        "test_loss": test_loss,
        "epochs": args.epochs,
        "test_components": test_comps,
    }
    with open(out_dir / f"train_report_L{args.latent_dims}_beta{args.beta}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
