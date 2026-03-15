from __future__ import annotations

"""
Static Camera Influence Tracker for RL Training
Activation-based monitoring to verify if neural network uses static camera data.
"""

import logging

import numpy as np
import torch

from aerial_gym.utils.gradient_monitor_report import (
    print_analysis_summary as _print_analysis_summary,
)

logger = logging.getLogger(__name__)


def _resolve_module_path(model: torch.nn.Module, dotted_path: str) -> torch.nn.Module:
    """Traverse a dotted attribute path on a module (e.g. 'encoder.encoders.obs')."""
    module = model
    for part in dotted_path.split("."):
        module = module.__getattr__(part)
    return module


def _is_script_module(module: torch.nn.Module) -> bool:
    """Check whether *module* is a TorchScript module (which doesn't support hooks)."""
    type_name = type(module).__name__
    return "ScriptModule" in type_name or "RecursiveScriptModule" in type_name


class CompleteObservationInfluenceTracker:
    """
    Enhanced tracker that analyzes the influence of ALL observation components on the neural network's encoded features.

    Extends the previous camera-only analysis to cover the complete 150D observation space:
    - Drone position (3D)
    - Static camera pose (6D)
    - Drone orientation (3D)
    - Kinematic state (6D)
    - Action history (4D)
    - Drone camera VAE (64D)
    - Static camera VAE (64D)

    Provides correlation, magnitude, and variance analysis for each component.
    """

    def __init__(self, model, config: dict[str, object]):
        self.model = model
        self.config = config
        self.enabled = True
        self.step_count = 0
        self.forward_pass_count = 0  # Track forward passes separately
        self.hook_handles = []

        # Complete observation component mapping (150D total)
        self.obs_components = {
            "drone_position": (0, 3, "🎯", "Drone absolute position in world coordinates"),
            "static_camera_pos": (3, 6, "📍", "Static camera position relative to drone"),
            "static_camera_orient": (6, 9, "🧭", "Static camera orientation relative to drone"),
            "drone_orientation": (9, 12, "✈️", "Drone full orientation (roll, pitch, yaw)"),
            "drone_linear_vel": (12, 15, "⚡", "Drone linear velocity in body frame"),
            "drone_angular_vel": (15, 18, "🌀", "Drone angular velocity in body frame"),
            "drone_actions": (18, 22, "🎮", "Previous drone actions (action history)"),
            "drone_camera_vae": (22, 86, "📷", "Drone camera VAE encoded features"),
            "static_camera_vae": (86, 150, "📹", "Static camera VAE encoded features"),
        }

        # Storage for analysis data
        self.activation_history = {
            component: {"correlations": [], "magnitudes": [], "variances": []}
            for component in self.obs_components.keys()
        }

        # Try to attach hooks in order of preference
        self._attach_hooks()

    def _attach_hooks(self) -> None:
        """Attach forward hooks to capture activations at the encoder output."""
        hook_targets = [
            ("encoder.encoders.obs", "Obs Encoder (non-compiled)"),
            ("obs_normalizer", "Observation Normalizer"),
            ("encoder", "Multi Input Encoder"),
            ("core", "RNN Core"),
        ]

        for target_path, target_name in hook_targets:
            try:
                target_module = _resolve_module_path(self.model, target_path)

                if _is_script_module(target_module):
                    logger.warning(f"Skipping {target_name} - ScriptModule doesn't support hooks")
                    continue

                # Try to attach hook
                handle = target_module.register_forward_hook(self._activation_hook)
                self.hook_handles.append(handle)
                logger.warning(f"✅ Complete observation tracker attached to {target_name}")
                return  # Success - only need one hook

            except AttributeError as e:
                logger.warning(f"⚠️ Could not access {target_name}: {e}")
                continue
            except (RuntimeError, TypeError) as e:
                logger.warning(f"Failed to attach to {target_name}: {e}")
                continue

        logger.warning("Searching for non-ScriptModule components...")
        for name, module in self.model.named_modules():
            if _is_script_module(module):
                continue

            # Skip modules that are too basic
            if any(
                skip_type in str(type(module)) for skip_type in ["Linear", "ELU", "Identity", "GRU"]
            ):
                continue

            try:
                handle = module.register_forward_hook(self._activation_hook)
                self.hook_handles.append(handle)
                logger.warning(
                    f"Complete observation tracker attached to non-ScriptModule: {name} ({type(module)})"
                )
                return
            except (AttributeError, RuntimeError, TypeError):
                continue

        logger.warning(
            "❌ Failed to attach complete observation tracker - no suitable non-ScriptModule hook points found"
        )
        logger.warning("💡 ScriptModules don't support hooks - this is a PyTorch limitation")
        self.enabled = False

    def _activation_hook(self, module, input, output) -> None:
        """Hook that captures activations and analyzes complete observation influence.
        Robust to modules that accept/return dicts or tuples (SF encoders, custom fusion encoders).
        """
        if not self.enabled:
            return

        observations = self._extract_obs_tensor(input)
        encoded_features = self._extract_feat_tensor(output)

        if not isinstance(observations, torch.Tensor) or observations.dim() != 2:
            return
        if not isinstance(encoded_features, torch.Tensor) or encoded_features.dim() != 2:
            return
        if observations.shape[1] < 81:
            return

        try:
            self._analyze_complete_observation_influence(observations, encoded_features)
        except (RuntimeError, ValueError, IndexError) as e:
            if self.step_count < 5:
                logger.warning(f"Hook analysis error: {e}")

    @staticmethod
    def _extract_obs_tensor(obj: object) -> torch.Tensor | None:
        """Extract observation tensor from hook input (may be tensor, tuple, or dict)."""
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (tuple, list)) and len(obj) > 0:
            return CompleteObservationInfluenceTracker._extract_obs_tensor(obj[0])
        if isinstance(obj, dict):
            for key in ("obs", "observations"):
                val = obj.get(key)
                if isinstance(val, torch.Tensor):
                    return val
            for v in obj.values():
                if isinstance(v, torch.Tensor):
                    return v
        return None

    @staticmethod
    def _extract_feat_tensor(obj: object) -> torch.Tensor | None:
        """Extract feature tensor from hook output (may be tensor, tuple, or dict)."""
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (tuple, list)) and len(obj) > 0:
            return CompleteObservationInfluenceTracker._extract_feat_tensor(obj[0])
        if isinstance(obj, dict):
            for key in ("encoding", "x", "h", "features", "out"):
                val = obj.get(key)
                if isinstance(val, torch.Tensor):
                    return val
            for v in obj.values():
                if isinstance(v, torch.Tensor):
                    return v
        return None

    def _analyze_complete_observation_influence(
        self, observations: torch.Tensor, encoded_features: torch.Tensor
    ) -> None:
        """Analyze the influence of all observation components on encoded features."""
        batch_size = observations.shape[0]

        if batch_size < 2:  # Need at least 2 samples for correlation
            return

        # Increment forward pass counter
        self.forward_pass_count += 1

        step_data = {}

        # Analyze each observation component
        for component_name, (start, end, emoji, description) in self.obs_components.items():
            component_data = observations[:, start:end]

            # Calculate metrics for this component
            magnitude = torch.norm(component_data, dim=1).mean().item()
            variance = torch.var(component_data).item()

            # Calculate correlation with encoded features
            correlation = self._calculate_component_correlation(component_data, encoded_features)

            # Store metrics
            step_data[component_name] = {
                "correlation": correlation,
                "magnitude": magnitude,
                "variance": variance,
                "emoji": emoji,
                "description": description,
            }

            # Add to history
            self.activation_history[component_name]["correlations"].append(correlation)
            self.activation_history[component_name]["magnitudes"].append(magnitude)
            self.activation_history[component_name]["variances"].append(variance)

        # Debug output (reduced frequency) - only every 50 forward passes
        if self.forward_pass_count <= 3 or self.forward_pass_count % 50 == 0:
            self._log_step_debug(step_data)

    def _calculate_component_correlation(
        self, component_data: torch.Tensor, encoded_features: torch.Tensor
    ) -> float:
        """Calculate correlation between component data and encoded features."""
        if encoded_features.shape[0] < 2:
            return 0.0

        correlations = []

        # Average component features across dimensions for correlation calculation
        if component_data.shape[1] > 1:
            component_avg = component_data.mean(dim=1)
        else:
            component_avg = component_data.squeeze(1)

        # Sample subset of encoded features for efficiency
        max_features_to_check = min(encoded_features.shape[1], 32)

        for i in range(max_features_to_check):
            encoded_feature_i = encoded_features[:, i]

            # Check for sufficient variance
            if torch.var(component_avg) > 1e-6 and torch.var(encoded_feature_i) > 1e-6:
                try:
                    corr_matrix = torch.corrcoef(torch.stack([component_avg, encoded_feature_i]))
                    corr_value = corr_matrix[0, 1].item()

                    if not torch.isnan(torch.tensor(corr_value)):
                        correlations.append(abs(corr_value))

                except (RuntimeError, ValueError):
                    continue

        return np.mean(correlations) if correlations else 0.0

    def _log_step_debug(self, step_data: dict[str, dict[str, float]]) -> None:
        """Log debug information for current step (no-op unless verbose logging enabled)."""

    def step(self) -> None:
        """Update step counter (synced externally with Sample Factory)."""

    def should_log(self) -> bool:
        """Determine if metrics should be logged this step."""
        return self.step_count > 0 and self.step_count % self.config.get("log_interval", 100) == 0

    def get_logging_metrics(self) -> dict[str, float]:
        """Get metrics for logging to wandb/tensorboard."""
        if not self.activation_history:
            return {}

        metrics = {}

        for component_name in self.obs_components.keys():
            history = self.activation_history[component_name]

            if history["correlations"]:
                # Recent average (last 10 samples)
                recent_corr = np.mean(history["correlations"][-10:])
                recent_mag = np.mean(history["magnitudes"][-10:])
                recent_var = np.mean(history["variances"][-10:])

                metrics[f"obs_influence/{component_name}_correlation"] = recent_corr
                metrics[f"obs_influence/{component_name}_magnitude"] = recent_mag
                metrics[f"obs_influence/{component_name}_variance"] = recent_var

        return metrics

    def print_analysis_summary(self) -> None:
        """Print comprehensive analysis summary (delegates to gradient_monitor_report)."""
        _print_analysis_summary(
            self.obs_components,
            self.activation_history,
            self.step_count,
            self.forward_pass_count,
        )

    def cleanup(self) -> None:
        """Remove hooks and cleanup resources."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        logger.warning("🧹 Complete observation tracker cleaned up")


def create_influence_tracker(
    model: object, config: dict[str, object]
) -> CompleteObservationInfluenceTracker | None:
    """
    Factory function to create complete observation influence tracker.

    Args:
        model: The neural network model to monitor
        config: Configuration dictionary with tracking parameters

    Returns:
        CompleteObservationInfluenceTracker instance or None if creation fails
    """
    try:
        tracker = CompleteObservationInfluenceTracker(model, config)
        if tracker.enabled:
            logger.warning("✅ Complete observation influence tracker created successfully")
            return tracker
        else:
            logger.warning("❌ Complete observation influence tracker creation failed")
            return None
    except (AttributeError, RuntimeError, TypeError) as e:
        logger.warning(f"Error creating complete observation influence tracker: {e}")
        return None


INFLUENCE_MONITOR_AVAILABLE = True
