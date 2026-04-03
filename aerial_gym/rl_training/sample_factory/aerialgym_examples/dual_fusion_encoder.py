from __future__ import annotations

import os

import torch
import torch.nn as nn
from sample_factory.model.encoder import (
    Encoder,
    ObsSpace,
    calc_num_elements,
    create_mlp,
    nonlinearity,
)
from sample_factory.utils.typing import Config

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.tensor_utils import sanitize_tensor

logger = CustomLogger(__name__)


class DualFusionEncoder(Encoder):
    """Encoder that supports early concat or gated late fusion.
    It expects a single flat obs vector and slices out ego/static 64D latents.
    """

    def __init__(self, cfg: Config, obs_space: ObsSpace) -> None:
        super().__init__(cfg)
        self.obs_space = obs_space
        self.fusion_mode: str = cfg.fusion
        self.gate_per_feature: bool = bool(int(cfg.gate_per_feature))
        self.slice_drone_vae: tuple[int, int] = (22, 86)
        self.slice_static_vae: tuple[int, int] = (86, 150)

        self._ablate_drone_zero: bool = False
        self._ablate_static_zero: bool = False
        self._parse_ablation_spec()

        D = 64
        if self.fusion_mode == "gated":
            self.ego_proj = nn.Sequential(
                nn.LayerNorm(D), nn.Linear(D, D), nn.ELU(), nn.LayerNorm(D)
            )
            self.static_proj = nn.Sequential(
                nn.LayerNorm(D), nn.Linear(D, D), nn.ELU(), nn.LayerNorm(D)
            )
            gate_out = D if self.gate_per_feature else 1
            self.gate = nn.Sequential(nn.Linear(2 * D, D), nn.ELU(), nn.Linear(D, gate_out))
            fused_latent_dim = D
        else:
            self.ego_proj = nn.Identity()
            self.static_proj = nn.Identity()
            self.gate = nn.Identity()
            fused_latent_dim = 128

        total_obs_dim: int = obs_space["obs"].shape[0]
        base_dim = total_obs_dim - 128 + fused_latent_dim
        mlp_layers = cfg.encoder_mlp_layers
        self.mlp = create_mlp(mlp_layers, base_dim, nonlinearity(cfg))
        if len(mlp_layers) > 0:
            self.mlp = torch.jit.script(self.mlp)
        self.encoder_out_size: int = calc_num_elements(self.mlp, (base_dim,))

        self._fwd_count: int = 0
        self._last_gate_stats: dict[str, float] | None = None
        self._tap_count: int = 0
        self._nan_diag_printed: bool = False
        self._train_env0_norm_state: int = 0
        self._train_env0_norm_step: int = 0
        self._last_step_logged: int = 0

        logger.info(
            f"[FUSION] Using fusion mode: {self.fusion_mode} "
            f"(gate_per_feature={int(self.gate_per_feature)})"
        )

    def _parse_ablation_spec(self) -> None:
        """Parse ABLATE_OBS_RANGES env var for hard-detach ablation flags."""
        spec_str = os.environ.get("ABLATE_OBS_RANGES", "").strip()
        if not spec_str:
            return
        for spec in [s.strip() for s in spec_str.split(",") if s.strip() and "=" in s]:
            lhs, rhs = spec.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            if ":" not in lhs:
                continue
            try:
                a_str, b_str = lhs.split(":", 1)
                a, b = int(a_str), int(b_str)
            except (ValueError, TypeError):
                continue
            if rhs in ("zero", "zerograd"):
                if a == self.slice_drone_vae[0] and b == self.slice_drone_vae[1]:
                    self._ablate_drone_zero = True
                if a == self.slice_static_vae[0] and b == self.slice_static_vae[1]:
                    self._ablate_static_zero = True

    def _slice_latents(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_e = x[..., self.slice_drone_vae[0] : self.slice_drone_vae[1]]
        z_s = x[..., self.slice_static_vae[0] : self.slice_static_vae[1]]
        return z_e, z_s

    def _remove_latents(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.slice_drone_vae, self.slice_static_vae
        prefix = x[..., : a[0]]
        middle = x[..., a[1] : b[0]]
        suffix = x[..., b[1] :]
        return torch.cat([prefix, middle, suffix], dim=-1)

    def _run_nan_diagnostics(self, x: torch.Tensor) -> None:
        """One-shot diagnostics for non-finite inputs."""
        want_diag = os.environ.get("NAN_DIAG", "false").lower() == "true"
        if self._nan_diag_printed and not want_diag:
            return
        if torch.isfinite(x).all() and not want_diag:
            return

        slices = [
            ("pos", 0, 3),
            ("static_pose", 3, 9),
            ("orient", 9, 12),
            ("vel", 12, 18),
            ("actions", 18, 22),
            ("ego_latent", 22, 86),
            ("static_latent", 86, 150),
        ]
        parts = ["[NAN_DIAG][encoder_in] non-finite detected or forced dump:"]
        for name, a, b in slices:
            s = x[..., a:b]
            safe = sanitize_tensor(s)
            bad = int((~torch.isfinite(s)).sum().item())
            parts.append(
                f"{name}[{a}:{b}]: bad={bad}/{s.numel()} "
                f"min={float(safe.min().item()):.3e} max={float(safe.max().item()):.3e}"
            )
        logger.warning(" | ".join(parts))
        if not want_diag:
            self._nan_diag_printed = True

    def _fuse_gated(self, z_e: torch.Tensor, z_s: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
        """Apply gated fusion with ablation support."""
        if self._ablate_drone_zero:
            z_e = (z_e * 0.0).detach()
        if self._ablate_static_zero:
            z_s = (z_s * 0.0).detach()

        if self._ablate_static_zero and not self._ablate_drone_zero:
            z = self.ego_proj(z_e)
            self._log_ablated_stats("static_disabled", e=z, s=None, z_fused=z)
            return torch.cat([rest, z], dim=-1)

        if self._ablate_drone_zero and not self._ablate_static_zero:
            z = self.static_proj(z_s)
            self._log_ablated_stats("drone_disabled", e=None, s=z, z_fused=z)
            return torch.cat([rest, z], dim=-1)

        if self._ablate_drone_zero and self._ablate_static_zero:
            z = torch.zeros_like(z_e)
            self._log_ablated_stats("both_disabled", e=None, s=None, z_fused=z)
            return torch.cat([rest, z], dim=-1)

        return self._fuse_gated_normal(z_e, z_s, rest)

    def _fuse_gated_normal(
        self, z_e: torch.Tensor, z_s: torch.Tensor, rest: torch.Tensor
    ) -> torch.Tensor:
        """Normal gated fusion (no ablations)."""
        e = sanitize_tensor(self.ego_proj(z_e))
        s = sanitize_tensor(self.static_proj(z_s))
        g = torch.sigmoid(self.gate(torch.cat([e, s], dim=-1)))
        g = torch.nan_to_num(g, nan=0.5, posinf=1.0, neginf=0.0)
        if g.shape[-1] == 1:
            g = g.expand_as(e)
        z = sanitize_tensor(g * s + (1 - g) * e)
        fused = torch.cat([rest, z], dim=-1)

        self._fwd_count += 1
        if (self._fwd_count % 200) == 0:
            self._log_gate_stats(g, e, s, z)

        return fused

    def _fuse_concat(
        self, z_e: torch.Tensor, z_s: torch.Tensor, rest: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Early concat baseline fusion."""
        fused = sanitize_tensor(torch.cat([rest, z_e, z_s], dim=-1))
        self._fwd_count += 1
        if (self._fwd_count % 200) == 0:
            B = int(x.shape[0])
            D_e = int(z_e.shape[-1])
            D_s = int(z_s.shape[-1])
            e_norm = float(z_e.norm(dim=1).mean().item())
            s_norm = float(z_s.norm(dim=1).mean().item())
            cat_norm = float(fused[:, -(D_e + D_s) :].norm(dim=1).mean().item())
            balance = float(s_norm / (e_norm + s_norm + 1e-8))
            logger.info(
                f"[FUSION] mode=concat B={B} D_e={D_e} D_s={D_s} | "
                f"L2 norms (mean): ego={e_norm:.3f} static={s_norm:.3f} cat={cat_norm:.3f} | "
                f"static_balance={balance:.2%}"
            )
        return fused

    def _log_ablated_stats(
        self,
        label: str,
        e: torch.Tensor | None,
        s: torch.Tensor | None,
        z_fused: torch.Tensor,
    ) -> None:
        """Periodic debug logging for ablated branches."""
        self._fwd_count += 1
        if (self._fwd_count % 200) != 0:
            return
        e_norm = float(e.norm(dim=1).mean().item()) if e is not None else 0.0
        s_norm = float(s.norm(dim=1).mean().item()) if s is not None else 0.0
        z_norm = float(z_fused.norm(dim=1).mean().item())
        logger.info(
            f"[FUSION] gated(per_feature={int(self.gate_per_feature)}) "
            f"{label}: true | norms: e={e_norm:.3f} s={s_norm:.3f} z={z_norm:.3f}"
        )
        self._wandb_log_fusion_stats(
            gate_mean=0.0 if "static" in label else (100.0 if "drone" in label else 50.0),
            gate_std=0.0,
            frac_high=0.0 if "static" in label else (100.0 if "drone" in label else 0.0),
            frac_low=100.0 if "static" in label else 0.0,
            e_norm=e_norm,
            s_norm=s_norm,
            z_norm=z_norm,
        )

    def _log_gate_stats(
        self,
        g: torch.Tensor,
        e: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
    ) -> None:
        """Log gate statistics and push to W&B."""
        try:
            gate_mean = float(g.mean().item())
            gate_std = float(g.std().item())
            frac_high = float((g > 0.7).float().mean().item())
            frac_low = float((g < 0.3).float().mean().item())
            e_norm = float(e.norm(dim=1).mean().item())
            s_norm = float(s.norm(dim=1).mean().item())
            z_norm = float(z.norm(dim=1).mean().item())
            self._last_gate_stats = {
                "mean": gate_mean,
                "std": gate_std,
                "frac_gt_0_7": frac_high,
                "frac_lt_0_3": frac_low,
                "e_norm_mean": e_norm,
                "s_norm_mean": s_norm,
                "z_norm_mean": z_norm,
            }
            logger.info(
                f"[FUSION] gated(per_feature={int(self.gate_per_feature)}) "
                f"gate_mean={gate_mean:.3f} gate_std={gate_std:.3f} "
                f">0.7={frac_high:.2%} <0.3={frac_low:.2%} "
                f"|| norms: e={e_norm:.3f} s={s_norm:.3f} z={z_norm:.3f}"
            )
            self._wandb_log_fusion_stats(
                gate_mean=gate_mean * 100.0,
                gate_std=gate_std * 100.0,
                frac_high=frac_high * 100.0,
                frac_low=frac_low * 100.0,
                e_norm=e_norm,
                s_norm=s_norm,
                z_norm=z_norm,
            )
        except RuntimeError:
            pass

    def _wandb_log_fusion_stats(
        self,
        gate_mean: float,
        gate_std: float,
        frac_high: float,
        frac_low: float,
        e_norm: float,
        s_norm: float,
        z_norm: float,
    ) -> None:
        """Push fusion stats to W&B."""
        try:
            import wandb

            frames: int | None = self._last_step_logged if self._last_step_logged > 0 else None
            payload = {
                "episode_extra_stats/fusion/gate_mean_pct": gate_mean,
                "episode_extra_stats/fusion/gate_std_pct": gate_std,
                "episode_extra_stats/fusion/gate_frac_gt_0_7_pct": frac_high,
                "episode_extra_stats/fusion/gate_frac_lt_0_3_pct": frac_low,
                "episode_extra_stats/fusion/e_norm_mean": e_norm,
                "episode_extra_stats/fusion/s_norm_mean": s_norm,
                "episode_extra_stats/fusion/z_norm_mean": z_norm,
            }
            if frames is not None:
                wandb.log(payload, step=frames)
            else:
                wandb.log(payload)
        except RuntimeError:
            pass

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        x = obs_dict["obs"]
        self._run_nan_diagnostics(x)
        x = sanitize_tensor(x)
        z_e, z_s = self._slice_latents(x)
        rest = self._remove_latents(x)

        if self.fusion_mode == "gated":
            fused = self._fuse_gated(z_e, z_s, rest)
        else:
            fused = self._fuse_concat(z_e, z_s, rest, x)

        return self.mlp(fused)

    def get_out_size(self) -> int:
        return self.encoder_out_size


def make_dual_fusion_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    return DualFusionEncoder(cfg, obs_space)
