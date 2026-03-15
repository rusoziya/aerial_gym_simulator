from __future__ import annotations

import os

import torch
import torch.nn as nn

from sample_factory.model.encoder import Encoder, ObsSpace, create_mlp, calc_num_elements, nonlinearity
from sample_factory.utils.typing import Config


class DualFusionEncoder(Encoder):
    """Encoder that supports early concat or gated late fusion.
    It expects a single flat obs vector and slices out ego/static 64D latents.
    """
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        self.obs_space = obs_space
        # Read fusion config from cfg/env
        fusion_mode = cfg.fusion
        self.fusion_mode = fusion_mode
        self.gate_per_feature = bool(int(cfg.gate_per_feature))
        # Indices for 150D obs layout
        self.slice_drone_vae = (22, 86)
        self.slice_static_vae = (86, 150)
        # Detect ablation spec to optionally hard-detach specific latent branches (full-slice zeros)
        import os
        spec_str = os.environ.get('ABLATE_OBS_RANGES', '').strip()
        self._ablate_drone_zero = False
        self._ablate_static_zero = False
        if spec_str:
            for spec in [s.strip() for s in spec_str.split(',') if s.strip() and '=' in s]:
                lhs, rhs = spec.split('=', 1)
                lhs = lhs.strip(); rhs = rhs.strip()
                if ':' not in lhs:
                    continue
                try:
                    a, b = lhs.split(':', 1)
                    a = int(a); b = int(b)
                except (ValueError, TypeError):
                    continue
                if rhs in ('zero', 'zerograd'):
                    # Exact-match ablations
                    if a == self.slice_drone_vae[0] and b == self.slice_drone_vae[1]:
                        self._ablate_drone_zero = True
                    if a == self.slice_static_vae[0] and b == self.slice_static_vae[1]:
                        self._ablate_static_zero = True
        # Build adapters/gate
        D = 64
        if self.fusion_mode == 'gated':
            self.ego_proj    = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.ELU(), nn.LayerNorm(D))
            self.static_proj = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.ELU(), nn.LayerNorm(D))
            gate_out = D if self.gate_per_feature else 1
            self.gate = nn.Sequential(nn.Linear(2*D, D), nn.ELU(), nn.Linear(D, gate_out))
            fused_latent_dim = D
        else:
            # Early concat (baseline)
            fused_latent_dim = 128
        # Determine total encoder input after fusion: fused latent + rest of features
        total_obs_dim = obs_space['obs'].shape[0]
        # Remove original two 64D latents (128) and add fused_latent_dim
        base_dim = total_obs_dim - 128 + fused_latent_dim
        mlp_layers = cfg.encoder_mlp_layers
        self.mlp = create_mlp(mlp_layers, base_dim, nonlinearity(cfg))
        if len(mlp_layers) > 0:
            self.mlp = torch.jit.script(self.mlp)
        self.encoder_out_size = calc_num_elements(self.mlp, (base_dim,))

        # Debug counters/last stats for fusion monitoring
        self._fwd_count = 0
        self._last_gate_stats = None
        # Encoder tap counter for periodic debug printing
        self._tap_count = 0

        # One-time info (print only in evaluation/inference)
        print(f"[FUSION] Using fusion mode: {self.fusion_mode} (gate_per_feature={int(self.gate_per_feature)})")

    def _slice_latents(self, x: torch.Tensor) -> None:
        z_e = x[..., self.slice_drone_vae[0]:self.slice_drone_vae[1]]
        z_s = x[..., self.slice_static_vae[0]:self.slice_static_vae[1]]
        return z_e, z_s

    def _remove_latents(self, x: torch.Tensor) -> None:
        a, b = self.slice_drone_vae, self.slice_static_vae
        # Keep prefix, skip [a0:a1] and [b0:b1], keep suffix
        prefix = x[..., :a[0]]
        middle = x[..., a[1]:b[0]]
        suffix = x[..., b[1]:]
        return torch.cat([prefix, middle, suffix], dim=-1)

    def forward(self, obs_dict) -> None:
        x = obs_dict['obs']
        # One-shot diagnostics for non-finite inputs (before any sanitization)
        try:
            import os as _os
            _want_diag = _os.environ.get('NAN_DIAG', 'false').lower() == 'true'
            if (not hasattr(self, '_nan_diag_printed')):
                self._nan_diag_printed = False
            if (not self._nan_diag_printed) and (not torch.isfinite(x).all() or _want_diag):
                def _slice_stats(name, t, a, b) -> None:
                    s = t[..., a:b]
                    finite = torch.isfinite(s)
                    n = s.numel()
                    bad = int((~finite).sum().item())
                    mn = float(torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0).min().item()) if n > 0 else 0.0
                    mx = float(torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0).max().item()) if n > 0 else 0.0
                    return f"{name}[{a}:{b}]: bad={bad}/{n} min={mn:.3e} max={mx:.3e}"
                msg = ["[NAN_DIAG][encoder_in] non-finite detected or forced dump:" ]
                msg.append(_slice_stats('pos', x, 0, 3))
                msg.append(_slice_stats('static_pose', x, 3, 9))
                msg.append(_slice_stats('orient', x, 9, 12))
                msg.append(_slice_stats('vel', x, 12, 18))
                msg.append(_slice_stats('actions', x, 18, 22))
                msg.append(_slice_stats('ego_latent', x, 22, 86))
                msg.append(_slice_stats('static_latent', x, 86, 150))
                print(" | ".join(msg))
                if not _want_diag:
                    self._nan_diag_printed = True
        except (ValueError, TypeError):
            pass
        # Sanitize input to prevent NaN/Inf propagation into LayerNorm/gating
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        z_e, z_s = self._slice_latents(x)
        rest = self._remove_latents(x)
        # Encoder tap: print what the encoder actually receives (before any LayerNorm/projections)
        try:
            import os
            tap_on = os.environ.get('ENC_TAP_DEBUG', os.environ.get('ABLATE_DEBUG', 'false')).lower() == 'true'
        except (KeyError, TypeError):
            tap_on = False
        if tap_on:
            self._tap_count += 1
            if (self._tap_count % 50) == 0:
                try:
                    is_eval = False
                    try:
                        is_eval = bool(self.cfg is None or getattr(self.cfg, 'evaluation', True))
                    except Exception:
                        is_eval = True
                    if is_eval:
                        z_e_abs = float(z_e.abs().mean().item()) if hasattr(z_e, 'abs') else float('nan')
                        z_s_abs = float(z_s.abs().mean().item()) if hasattr(z_s, 'abs') else float('nan')
                except (ValueError, TypeError):
                    pass
        # Optional: training-time normalized env0 latents per-frame (one episode)
        import os as _os
        if _os.environ.get('TRAIN_ENV0_LATENTS_NORM', 'false').lower() == 'true':
            # Only print from the first env
            ze0 = z_e[0]
            zs0 = z_s[0]
            # After projection+LayerNorm to match branch inputs, but before gating
            e0 = self.ego_proj(ze0.unsqueeze(0))[0]
            s0 = self.static_proj(zs0.unsqueeze(0))[0]
            e_abs = float(e0.abs().mean().item())
            s_abs = float(s0.abs().mean().item())
            # Also compute raw (pre-normalization) abs-means for comparison
            e_raw_abs = float(ze0.abs().mean().item())
            s_raw_abs = float(zs0.abs().mean().item())
            if not hasattr(self, '_train_env0_norm_state'):
                self._train_env0_norm_state = 0
                self._train_env0_norm_step = 0
            if self._train_env0_norm_state == 0:
                # Arm -> activate on first nonzero
                if (e_abs > 1e-6) or (s_abs > 1e-6):
                    self._train_env0_norm_state = 1
                    self._train_env0_norm_step = 0
                    print(f"[TRAIN_ENV0_LATENTS_NORM] step={self._train_env0_norm_step} abs_mean: drone_norm={e_abs:.6f} static_norm={s_abs:.6f} | drone_raw={e_raw_abs:.6f} static_raw={s_raw_abs:.6f}")
                    self._train_env0_norm_step += 1
            elif self._train_env0_norm_state == 1:
                print(f"[TRAIN_ENV0_LATENTS_NORM] step={self._train_env0_norm_step} abs_mean: drone_norm={e_abs:.6f} static_norm={s_abs:.6f} | drone_raw={e_raw_abs:.6f} static_raw={s_raw_abs:.6f}")
                self._train_env0_norm_step += 1
        if self.fusion_mode == 'gated':
            # If a full-slice zero ablation was requested, detach those inputs so grads cannot route via gate
            if self._ablate_drone_zero:
                z_e = (z_e * 0.0).detach()
            if self._ablate_static_zero:
                z_s = (z_s * 0.0).detach()

            # NOTE: Do not auto-ablated based on runtime magnitudes in inference.
            # Only honor explicit ABLATE_OBS_RANGES to avoid false positives.

            # Short-circuit: when one branch is ablated, ignore it entirely in forward (training behavior retained)
            if self._ablate_static_zero and not self._ablate_drone_zero:
                # Use only ego branch
                e = self.ego_proj(z_e)
                z = e
                fused = torch.cat([rest, z], dim=-1)
                # Periodic debug for ablated-static case (evaluation only)
                self._fwd_count += 1
                if (self._fwd_count % 200) == 0:
                    try:
                        e_norm = float(e.norm(dim=1).mean().item())
                        z_norm = float(z.norm(dim=1).mean().item())
                        print(
                            f"[FUSION] gated(per_feature={int(self.gate_per_feature)}) static_disabled: true, drone_only: true | "
                            f"norms: e={e_norm:.3f} z={z_norm:.3f}"
                        )
                        # Log to W&B even when static is disabled (treat gate as 0 towards static)
                        import wandb  # noqa: F401
                        frames = int(self._last_step_logged) if hasattr(self, '_last_step_logged') else None
                        payload = {
                            'episode_extra_stats/fusion/gate_mean_pct': 0.0,  # g≈0 → drone-only
                            'episode_extra_stats/fusion/gate_std_pct': 0.0,
                            'episode_extra_stats/fusion/gate_frac_gt_0_7_pct': 0.0,
                            'episode_extra_stats/fusion/gate_frac_lt_0_3_pct': 100.0,
                            'episode_extra_stats/fusion/e_norm_mean': float(e_norm),
                            'episode_extra_stats/fusion/s_norm_mean': 0.0,
                            'episode_extra_stats/fusion/z_norm_mean': float(z_norm),
                        }
                        if frames is not None and frames > 0:
                            wandb.log(payload, step=frames)
                        else:
                            wandb.log(payload)
                    except RuntimeError:
                        pass
            elif self._ablate_drone_zero and not self._ablate_static_zero:
                # Use only static branch
                s = self.static_proj(z_s)
                z = s
                fused = torch.cat([rest, z], dim=-1)
                # Periodic debug for ablated-drone case (evaluation only)
                self._fwd_count += 1
                if (self._fwd_count % 200) == 0:
                    try:
                        s_norm = float(s.norm(dim=1).mean().item())
                        z_norm = float(z.norm(dim=1).mean().item())
                        print(
                            f"[FUSION] gated(per_feature={int(self.gate_per_feature)}) drone_disabled: true, static_only: true | "
                            f"norms: s={s_norm:.3f} z={z_norm:.3f}"
                        )
                        # Log to W&B even when drone is disabled (treat gate as 1 towards static)
                        import wandb  # noqa: F401
                        frames = int(self._last_step_logged) if hasattr(self, '_last_step_logged') else None
                        payload = {
                            'episode_extra_stats/fusion/gate_mean_pct': 100.0,  # g≈1 → static-only
                            'episode_extra_stats/fusion/gate_std_pct': 0.0,
                            'episode_extra_stats/fusion/gate_frac_gt_0_7_pct': 100.0,
                            'episode_extra_stats/fusion/gate_frac_lt_0_3_pct': 0.0,
                            'episode_extra_stats/fusion/e_norm_mean': 0.0,
                            'episode_extra_stats/fusion/s_norm_mean': float(s_norm),
                            'episode_extra_stats/fusion/z_norm_mean': float(z_norm),
                        }
                        if frames is not None and frames > 0:
                            wandb.log(payload, step=frames)
                        else:
                            wandb.log(payload)
                    except RuntimeError:
                        pass
            elif self._ablate_drone_zero and self._ablate_static_zero:
                # Both ablated: feed zeros latent
                z = torch.zeros_like(z_e)
                fused = torch.cat([rest, z], dim=-1)
                # Periodic debug for both ablated (evaluation only)
                self._fwd_count += 1
                if (self._fwd_count % 200) == 0:
                    try:
                        print(
                            f"[FUSION] gated(per_feature={int(self.gate_per_feature)}) both_cameras_disabled: true | norms: z=0.000"
                        )
                        # Log a minimal payload when both branches are disabled
                        import wandb  # noqa: F401
                        frames = int(self._last_step_logged) if hasattr(self, '_last_step_logged') else None
                        payload = {
                            'episode_extra_stats/fusion/gate_mean_pct': 50.0,  # undefined, show neutral
                            'episode_extra_stats/fusion/gate_std_pct': 0.0,
                            'episode_extra_stats/fusion/gate_frac_gt_0_7_pct': 0.0,
                            'episode_extra_stats/fusion/gate_frac_lt_0_3_pct': 0.0,
                            'episode_extra_stats/fusion/e_norm_mean': 0.0,
                            'episode_extra_stats/fusion/s_norm_mean': 0.0,
                            'episode_extra_stats/fusion/z_norm_mean': 0.0,
                        }
                        if frames is not None and frames > 0:
                            wandb.log(payload, step=frames)
                        else:
                            wandb.log(payload)
                    except RuntimeError:
                        pass
            else:
                # Normal gated fusion
                e = self.ego_proj(z_e)
                s = self.static_proj(z_s)
                # Sanitize branch activations before gating
                e = torch.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
                s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
                g = torch.sigmoid(self.gate(torch.cat([e, s], dim=-1)))
                g = torch.nan_to_num(g, nan=0.5, posinf=1.0, neginf=0.0)
                if g.shape[-1] == 1:
                    g = g.expand_as(e)
                z = g * s + (1 - g) * e
                z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                fused = torch.cat([rest, z], dim=-1)

            # Periodic debug print for gate stats
            # Periodic debug print for gate stats (only when gate was active)
            if not (self._ablate_static_zero or self._ablate_drone_zero):
                self._fwd_count += 1
                if (self._fwd_count % 200) == 0:
                    try:
                        gate_mean = float(g.mean().item())
                        gate_std = float(g.std().item())
                        frac_high = float((g > 0.7).float().mean().item())
                        frac_low = float((g < 0.3).float().mean().item())
                        e_norm = float(e.norm(dim=1).mean().item())
                        s_norm = float(s.norm(dim=1).mean().item())
                        z_norm = float(z.norm(dim=1).mean().item())
                        self._last_gate_stats = {
                            'mean': gate_mean,
                            'std': gate_std,
                            'frac_gt_0_7': frac_high,
                            'frac_lt_0_3': frac_low,
                            'e_norm_mean': e_norm,
                            's_norm_mean': s_norm,
                            'z_norm_mean': z_norm,
                        }
                        print(
                            f"[FUSION] gated(per_feature={int(self.gate_per_feature)}) "
                            f"gate_mean={gate_mean:.3f} gate_std={gate_std:.3f} "
                            f">0.7={frac_high:.2%} <0.3={frac_low:.2%} "
                            f"|| norms: e={e_norm:.3f} s={s_norm:.3f} z={z_norm:.3f}"
                        )
                        # Mirror to W&B payload under episode_extra_stats/fusion/* (only for gated and both cameras enabled)
                        import wandb  # ensure wandb available
                        frames = int(self._last_step_logged) if hasattr(self, '_last_step_logged') else None
                        payload = {
                            'episode_extra_stats/fusion/gate_mean_pct': float(gate_mean * 100.0),
                            'episode_extra_stats/fusion/gate_std_pct': float(gate_std * 100.0),
                            'episode_extra_stats/fusion/gate_frac_gt_0_7_pct': float(frac_high * 100.0),
                            'episode_extra_stats/fusion/gate_frac_lt_0_3_pct': float(frac_low * 100.0),
                            'episode_extra_stats/fusion/e_norm_mean': float(e_norm),
                            'episode_extra_stats/fusion/s_norm_mean': float(s_norm),
                            'episode_extra_stats/fusion/z_norm_mean': float(z_norm),
                        }
                        if frames is not None and frames > 0:
                            wandb.log(payload, step=frames)
                        else:
                            wandb.log(payload)
                    except RuntimeError:
                        pass
        else:
            # Early concat baseline (kept for easy revert)
            fused = torch.cat([rest, z_e, z_s], dim=-1)
            fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
            # Periodic debug print for concat stats
            self._fwd_count += 1
            if (self._fwd_count % 200) == 0:
                B = int(x.shape[0]) if torch.is_tensor(x) else 0
                D_e = int(z_e.shape[-1])
                D_s = int(z_s.shape[-1])
                e_norm = float(z_e.norm(dim=1).mean().item())
                s_norm = float(z_s.norm(dim=1).mean().item())
                cat_norm = float(fused[:, - (D_e + D_s):].norm(dim=1).mean().item())
                balance = float((s_norm / (e_norm + s_norm + 1e-8)))
                print(
                    f"[FUSION] mode=concat B={B} D_e={D_e} D_s={D_s} | "
                    f"L2 norms (mean): ego={e_norm:.3f} static={s_norm:.3f} cat={cat_norm:.3f} | "
                    f"static_balance≈{balance:.2%}"
                )
        return self.mlp(fused)

    def get_out_size(self) -> int:
        return self.encoder_out_size


def make_dual_fusion_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    return DualFusionEncoder(cfg, obs_space)
