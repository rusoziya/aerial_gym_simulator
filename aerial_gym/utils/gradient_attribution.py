from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class GradientAttributionTracker:
    """
    Tracks loss-aligned importance of observation components via gradient norms.

    Mechanics:
    - Attaches a forward hook at the earliest non-script module processing observations to
      (a) ensure requires_grad on the incoming observation tensor and
      (b) register a tensor-level grad hook to capture dLoss/dObs.
    - Also attaches a full backward hook as a fallback (may not trigger if inputs don't require grad).
    - Aggregates mean gradient magnitude per observation slice over time.
    """

    def __init__(self, model: torch.nn.Module, config: dict[str, object]):
        self.model = model
        self.config = config or {}
        self.enabled = True
        self.step_count = 0
        self.backward_pass_count = 0
        self.module_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.backward_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        # Observation component mapping must match the runtime observation layout (150D)
        # name -> (start, end, emoji, description)
        self.obs_components: Dict[str, Tuple[int, int, str, str]] = {
            'drone_position': (0, 3, "🎯", "Drone absolute position in world coordinates"),
            'static_camera_pos': (3, 6, "📍", "Static camera position relative to drone"),
            'static_camera_orient': (6, 9, "🧭", "Static camera orientation relative to drone"),
            'drone_orientation': (9, 12, "✈️", "Drone full orientation (roll, pitch, yaw)"),
            'drone_linear_vel': (12, 15, "⚡", "Drone linear velocity in body frame"),
            'drone_angular_vel': (15, 18, "🌀", "Drone angular velocity in body frame"),
            'drone_actions': (18, 22, "🎮", "Previous drone actions (action history)"),
            'drone_camera_vae': (22, 86, "📷", "Drone camera VAE encoded features"),
            'static_camera_vae': (86, 150, "📹", "Static camera VAE encoded features"),
        }

        # Storage for aggregated gradient magnitudes
        self.grad_history: Dict[str, list] = {name: [] for name in self.obs_components.keys()}

        # Try to attach hooks
        self._attach_hooks()

    # ---------------------------------------------------------------------
    # Hooking
    # ---------------------------------------------------------------------
    def _attach_hooks(self):
        """Attach forward+backward hooks to capture grads wrt observation tensor."""
        hook_targets = [
            ('encoder.encoders.obs', 'Obs Encoder (non-compiled)'),
            ('obs_normalizer', 'Observation Normalizer'),
            ('encoder', 'Multi Input Encoder'),
            ('core', 'RNN Core'),
        ]

        for target_path, target_name in hook_targets:
            try:
                module = self.model
                for attr in target_path.split('.'):
                    module = getattr(module, attr)

                # Skip TorchScript modules (no hooks)
                if hasattr(module, '_c') or 'ScriptModule' in str(type(module)):
                    logger.warning(f"⚠️ Skipping {target_name} for gradient hooks - ScriptModule")
                    continue

                # Attach forward hook (to enable requires_grad and add tensor-level grad hook)
                self.module_hook_handle = module.register_forward_hook(self._forward_hook)
                # Optional fallback: module-level backward hook
                self.backward_hook_handle = module.register_full_backward_hook(self._backward_hook)

                logger.warning(f"✅ Gradient attribution tracker attached to {target_name}")
                return
            except AttributeError as e:
                logger.warning(f"⚠️ Could not access {target_name}: {e}")
                continue
            except Exception as e:
                logger.warning(f"⚠️ Failed to attach gradient hooks to {target_name}: {e}")
                continue

        logger.warning("❌ Failed to attach gradient attribution hooks - no suitable non-ScriptModule target found; falling back to model pre-hook")
        try:
            self.module_hook_handle = self.model.register_forward_pre_hook(self._forward_pre_hook_model)
            self.backward_hook_handle = self.model.register_full_backward_hook(self._backward_hook)
            logger.warning("✅ Gradient attribution fallback hooks attached at model level")
        except Exception as e:
            logger.warning(f"⚠️ Model-level fallback failed: {e}")
            self.enabled = False

    # ---------------------------------------------------------------------
    # Hooks
    # ---------------------------------------------------------------------
    def _forward_hook(self, module, input, output):
        """Enable grad on observation tensor and register a tensor-level grad hook.
        Robust to dict/tuple/list inputs; extracts 'obs'/'observations' if present.
        """
        def _extract_obs_tensor(obj):
            try:
                if torch.is_tensor(obj):
                    return obj
                if isinstance(obj, (tuple, list)) and len(obj) > 0:
                    return _extract_obs_tensor(obj[0])
                if isinstance(obj, dict):
                    for k in ('obs', 'observations'):
                        t = obj.get(k, None)
                        if torch.is_tensor(t):
                            return t
                    for v in obj.values():
                        if torch.is_tensor(v):
                            return v
                return None
            except Exception:
                return None
        try:
            if not input:
                return
            x = _extract_obs_tensor(input)
            if not torch.is_tensor(x) or x.dim() != 2:
                return
            if x.shape[1] < 81:
                return
            # Ensure gradients will be computed wrt inputs
            if not x.requires_grad:
                x.requires_grad_(True)
            # Register a one-time hook to capture its gradient on backward
            # Note: this hook runs every backward for this forward pass
            def _tensor_grad_hook(grad: torch.Tensor):
                if grad is None or grad.dim() != 2 or grad.shape[1] != 150:
                    return
                self.backward_pass_count += 1
                # Aggregate per-slice mean L2 norm
                for name, (start, end, _emoji, _desc) in self.obs_components.items():
                    g_slice = grad[:, start:end]
                    slice_norm = torch.norm(g_slice, dim=1).mean().item()
                    self.grad_history[name].append(slice_norm)
            x.register_hook(_tensor_grad_hook)
        except Exception as e:
            # Only warn a couple of times to avoid spam
            if self.backward_pass_count < 2:
                logger.warning(f"🔧 Forward hook setup failed: {e}")

    def _forward_pre_hook_model(self, module, input):
        """Model-level pre-hook to wrap obs with requires_grad when encoder is scripted."""
        try:
            if not input:
                return None
            arg = input[0] if isinstance(input, tuple) and len(input) > 0 else input
            def _wrap(obj):
                if torch.is_tensor(obj) and obj.dim() == 2 and obj.shape[1] >= 81:
                    if not obj.requires_grad:
                        obj = obj.detach().requires_grad_(True)
                    module._obs_proxy = obj
                    return obj
                if isinstance(obj, dict):
                    out = dict(obj)
                    for k in ('obs', 'observations'):
                        if k in out and torch.is_tensor(out[k]) and out[k].dim() == 2 and out[k].shape[1] >= 81:
                            t = out[k]
                            if not t.requires_grad:
                                t = t.detach().requires_grad_(True)
                            module._obs_proxy = t
                            out[k] = t
                            return out
                    return obj
                if isinstance(obj, (tuple, list)) and len(obj) > 0:
                    seq = list(obj)
                    seq[0] = _wrap(seq[0])
                    return type(obj)(seq)
                return obj
            wrapped = _wrap(arg)
            if wrapped is arg:
                return None
            if isinstance(input, tuple):
                lst = list(input)
                lst[0] = wrapped
                return tuple(lst)
            else:
                return wrapped
        except Exception:
            return None

    def _backward_hook(self, module, grad_input, grad_output):
        """Fallback: capture grad_input if available (requires inputs with requires_grad=True)."""
        try:
            grads = None
            if grad_input and len(grad_input) > 0 and grad_input[0] is not None:
                grads = grad_input[0]
            elif grad_output and len(grad_output) > 0 and grad_output[0] is not None:
                grads = grad_output[0]
            if grads is None or grads.dim() != 2 or grads.shape[1] != 150:
                return
            self.backward_pass_count += 1
            for name, (start, end, _emoji, _desc) in self.obs_components.items():
                g_slice = grads[:, start:end]
                slice_norm = torch.norm(g_slice, dim=1).mean().item()
                self.grad_history[name].append(slice_norm)
        except Exception as e:
            if self.backward_pass_count < 2:
                logger.warning(f"🔧 Gradient hook error: {e}")

    # ---------------------------------------------------------------------
    # API
    # ---------------------------------------------------------------------
    def step(self):
        # External step sync (optional)
        pass

    def should_log(self) -> bool:
        return self.step_count > 0 and self.step_count % int(self.config.get('log_interval', 100)) == 0

    def get_logging_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for name, values in self.grad_history.items():
            if values:
                metrics[f'obs_grad/{name}_mean_norm_recent'] = float(np.mean(values[-10:]))
                metrics[f'obs_grad/{name}_mean_norm_overall'] = float(np.mean(values))
        # Aggregate totals
        recent_total = sum(metrics[k] for k in metrics.keys() if k.endswith('_mean_norm_recent'))
        overall_total = sum(metrics[k] for k in metrics.keys() if k.endswith('_mean_norm_overall'))
        metrics['obs_grad/total_recent'] = float(recent_total)
        metrics['obs_grad/total_overall'] = float(overall_total)
        metrics['obs_grad/backward_passes'] = float(self.backward_pass_count)
        return metrics

    def print_gradient_summary(self):
        comp_avgs = {k: (float(np.mean(v)) if v else 0.0) for k, v in self.grad_history.items()}
        if not any(v > 0 for v in comp_avgs.values()):
            logger.warning("📉 No gradient attribution data collected yet")
            return

        logger.warning("================================================================================")
        logger.warning("🧮 GRADIENT-BASED ATTRIBUTION (loss-aligned sensitivity)")
        logger.warning("================================================================================")
        logger.warning(f"   Backward passes observed (tensor hooks): {self.backward_pass_count}")

        sorted_items = sorted(comp_avgs.items(), key=lambda kv: kv[1], reverse=True)
        total = sum(v for _, v in sorted_items)
        for rank, (name, value) in enumerate(sorted_items, 1):
            pct = (100.0 * value / total) if total > 0 else 0.0
            start, end, emoji, desc = self.obs_components[name]
            dims = end - start
            importance = (
                "🔥 CRITICAL" if value > 1e-1 else
                ("📊 SIGNIFICANT" if value > 5e-2 else ("⚡ MODERATE" if value > 1e-2 else "❌ MINIMAL"))
            )
            logger.warning(f"{rank:2d}. {emoji} {name.upper():<20s} | GradNorm: {value:.4e} ({pct:5.1f}%) | Dims: {dims}")
            logger.warning(f"     {importance} | Description: {desc}")

        visual = comp_avgs.get('drone_camera_vae', 0.0) + comp_avgs.get('static_camera_vae', 0.0)
        kine = (comp_avgs.get('drone_linear_vel', 0.0) + comp_avgs.get('drone_angular_vel', 0.0) +
                comp_avgs.get('drone_actions', 0.0))
        spatial = (comp_avgs.get('drone_position', 0.0) + comp_avgs.get('static_camera_pos', 0.0) +
                   comp_avgs.get('static_camera_orient', 0.0) + comp_avgs.get('drone_orientation', 0.0))
        total = visual + kine + spatial
        if total > 0:
            logger.warning("")
            logger.warning(f"   📹 Visual:   {100.0*visual/total:5.1f}%")
            logger.warning(f"   ⚡ Kinematic:{100.0*kine/total:5.1f}%")
            logger.warning(f"   🧭 Spatial:  {100.0*spatial/total:5.1f}%")
        logger.warning("================================================================================")

    def cleanup(self):
        if self.module_hook_handle is not None:
            self.module_hook_handle.remove()
            self.module_hook_handle = None
        if self.backward_hook_handle is not None:
            self.backward_hook_handle.remove()
            self.backward_hook_handle = None
        logger.warning("🧹 Gradient attribution tracker cleaned up")

    def consume_grad(self, grad: torch.Tensor) -> None:
        try:
            if grad is None or not torch.is_tensor(grad) or grad.dim() != 2 or grad.shape[1] != 150:
                return
            self.backward_pass_count += 1
            for name, (start, end, _emoji, _desc) in self.obs_components.items():
                g_slice = grad[:, start:end]
                slice_norm = torch.norm(g_slice, dim=1).mean().item()
                self.grad_history[name].append(slice_norm)
        except Exception:
            pass


def create_gradient_tracker(model: torch.nn.Module, config: dict[str, object]) -> Optional[GradientAttributionTracker]:
    try:
        tracker = GradientAttributionTracker(model, config)
        if tracker.enabled:
            logger.warning("✅ Gradient attribution tracker created successfully")
            return tracker
        else:
            logger.warning("❌ Gradient attribution tracker creation failed")
            return None
    except Exception as e:
        logger.warning(f"❌ Error creating gradient attribution tracker: {e}")
        return None


GRAD_ATTR_AVAILABLE = True 