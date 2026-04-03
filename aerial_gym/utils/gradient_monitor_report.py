from __future__ import annotations

"""Reporting logic for the observation influence tracker."""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Type aliases — use typing generics for Python 3.8 runtime compatibility
ObsComponentMap = Dict[str, Tuple[int, int, str, str]]
ActivationHistory = Dict[str, Dict[str, List[float]]]


def print_analysis_summary(
    obs_components: ObsComponentMap,
    activation_history: ActivationHistory,
    step_count: int,
    forward_pass_count: int,
) -> None:
    """Print comprehensive analysis summary of observation influence data."""
    if not any(activation_history[comp]["correlations"] for comp in obs_components.keys()):
        logger.warning("📊 No analysis data available yet")
        return

    logger.warning(
        "================================================================================"
    )
    logger.warning("📊 COMPLETE OBSERVATION INFLUENCE ANALYSIS")
    logger.warning(
        "================================================================================"
    )
    logger.warning(
        f"📊 Analysis based on {step_count} Sample Factory training steps and {forward_pass_count} forward passes"
    )
    logger.warning("    (Note: Sample Factory processes multiple rollout steps per training step)")
    logger.warning("")

    component_stats = _compute_component_stats(obs_components, activation_history)

    sorted_stats = sorted(component_stats.items(), key=lambda x: x[1]["correlation"], reverse=True)

    total_correlation = sum(stats["correlation"] for _, stats in component_stats.items())

    _log_individual_components(sorted_stats, total_correlation)
    _log_detailed_breakdown(sorted_stats, total_correlation)
    _log_modality_groupings(component_stats, total_correlation)
    _log_optimization_insights(sorted_stats, component_stats, obs_components)

    logger.warning(
        "================================================================================"
    )


def _compute_component_stats(
    obs_components: ObsComponentMap,
    activation_history: ActivationHistory,
) -> dict[str, dict[str, float | int | str]]:
    """Compute average statistics for each observation component."""
    component_stats: dict[str, dict[str, float | int | str]] = {}
    for component_name, (start, end, emoji, description) in obs_components.items():
        history = activation_history[component_name]

        if history["correlations"]:
            avg_correlation = float(np.mean(history["correlations"]))
            avg_magnitude = float(np.mean(history["magnitudes"]))
            avg_variance = float(np.mean(history["variances"]))
            sample_count = len(history["correlations"])

            component_stats[component_name] = {
                "correlation": avg_correlation,
                "magnitude": avg_magnitude,
                "variance": avg_variance,
                "samples": sample_count,
                "emoji": emoji,
                "description": description,
                "dimensions": end - start,
            }

    return component_stats


def _log_individual_components(
    sorted_stats: list[tuple[str, dict[str, float | int | str]]],
    total_correlation: float,
) -> None:
    """Log each component's influence with ranking and visual bars."""
    logger.warning("📊 INDIVIDUAL OBSERVATION COMPONENT INFLUENCE:")
    logger.warning("")

    for rank, (component_name, stats) in enumerate(sorted_stats, 1):
        corr = stats["correlation"]
        mag = stats["magnitude"]
        var = stats["variance"]
        samples = stats["samples"]
        emoji = stats["emoji"]
        desc = stats["description"]
        dims = stats["dimensions"]

        influence_pct = (corr / total_correlation * 100) if total_correlation > 0 else 0

        if corr > 0.2:
            importance = "🔥 CRITICAL"
            bar = "████████████"
        elif corr > 0.1:
            importance = "📊 SIGNIFICANT"
            bar = "████████░░░░"
        elif corr > 0.05:
            importance = "⚡ MODERATE"
            bar = "████░░░░░░░░"
        else:
            importance = "❌ MINIMAL"
            bar = "██░░░░░░░░░░"

        logger.warning(
            f"{rank:2d}. {emoji} {component_name.upper():<20s} | Correlation: {corr:.4f} ({influence_pct:5.1f}%)"
        )
        logger.warning(f"     {importance} | {bar} | Dimensions: {dims}D | Samples: {samples}")
        logger.warning(f"     Magnitude: {mag:8.1f} | Variance: {var:8.3f}")
        logger.warning(f"     Description: {desc}")
        logger.warning("")


def _log_detailed_breakdown(
    sorted_stats: list[tuple[str, dict[str, float | int | str]]],
    total_correlation: float,
) -> None:
    """Log the detailed per-component breakdown section."""
    logger.warning("🎯 DETAILED BREAKDOWN BY COMPONENT:")
    logger.warning("")

    for component_name, stats in sorted_stats:
        corr = stats["correlation"]
        emoji = stats["emoji"]
        influence_pct = (corr / total_correlation * 100) if total_correlation > 0 else 0
        logger.warning(
            f"   {emoji} {component_name:<20s}: {corr:.4f} correlation ({influence_pct:5.1f}% of total influence)"
        )

    logger.warning("")


def _log_modality_groupings(
    component_stats: dict[str, dict[str, float | int | str]],
    total_correlation: float,
) -> None:
    """Log grouped modality statistics (visual, kinematic, spatial)."""
    logger.warning("📊 MODALITY GROUPINGS (for reference):")

    visual_components = ["drone_camera_vae", "static_camera_vae"]
    kinematic_components = ["drone_linear_vel", "drone_angular_vel", "drone_actions"]
    spatial_components = [
        "drone_position",
        "static_camera_pos",
        "static_camera_orient",
        "drone_orientation",
    ]

    visual_corr = sum(
        component_stats.get(comp, {}).get("correlation", 0) for comp in visual_components
    )
    kinematic_corr = sum(
        component_stats.get(comp, {}).get("correlation", 0) for comp in kinematic_components
    )
    spatial_corr = sum(
        component_stats.get(comp, {}).get("correlation", 0) for comp in spatial_components
    )

    if total_correlation > 0:
        visual_pct = (visual_corr / total_correlation) * 100
        kinematic_pct = (kinematic_corr / total_correlation) * 100
        spatial_pct = (spatial_corr / total_correlation) * 100

        logger.warning(
            f"   📹 Visual Information:   {visual_pct:5.1f}% ({visual_corr:.3f} total correlation)"
        )
        logger.warning(
            f"      └─ 📷 Drone Camera:    {component_stats.get('drone_camera_vae', {}).get('correlation', 0):.4f}"
        )
        logger.warning(
            f"      └─ 📹 Static Camera:   {component_stats.get('static_camera_vae', {}).get('correlation', 0):.4f}"
        )
        logger.warning("")
        logger.warning(
            f"   ⚡ Kinematic State:      {kinematic_pct:5.1f}% ({kinematic_corr:.3f} total correlation)"
        )
        logger.warning(
            f"      └─ ⚡ Linear Velocity:  {component_stats.get('drone_linear_vel', {}).get('correlation', 0):.4f}"
        )
        logger.warning(
            f"      └─ 🌀 Angular Velocity: {component_stats.get('drone_angular_vel', {}).get('correlation', 0):.4f}"
        )
        logger.warning(
            f"      └─ 🎮 Action History:   {component_stats.get('drone_actions', {}).get('correlation', 0):.4f}"
        )
        logger.warning("")
        logger.warning(
            f"   🧭 Spatial Awareness:    {spatial_pct:5.1f}% ({spatial_corr:.3f} total correlation)"
        )
        logger.warning(
            f"      └─ 🎯 Drone Position:   {component_stats.get('drone_position', {}).get('correlation', 0):.4f}"
        )
        logger.warning(
            f"      └─ 📍 Static Cam Pos:   {component_stats.get('static_camera_pos', {}).get('correlation', 0):.4f}"
        )
        logger.warning(
            f"      └─ 🧭 Static Cam Orient:{component_stats.get('static_camera_orient', {}).get('correlation', 0):.4f}"
        )
        logger.warning(
            f"      └─ ✈️ Drone Orientation:{component_stats.get('drone_orientation', {}).get('correlation', 0):.4f}"
        )

    logger.warning("")


def _log_optimization_insights(
    sorted_stats: list[tuple[str, dict[str, float | int | str]]],
    component_stats: dict[str, dict[str, float | int | str]],
    obs_components: ObsComponentMap,
) -> None:
    """Log optimization insights based on component influence analysis."""
    logger.warning("💡 OPTIMIZATION INSIGHTS:")

    highest = sorted_stats[0] if sorted_stats else None
    lowest = sorted_stats[-1] if sorted_stats else None

    if highest:
        h_name, h_stats = highest
        logger.warning(
            f"   🏆 Most influential: {h_stats['emoji']} {h_name} ({h_stats['correlation']:.4f} correlation)"
        )

    if lowest:
        l_name, l_stats = lowest
        if l_stats["correlation"] < 0.05:
            logger.warning(
                f"   ⚠️  Least influential: {l_stats['emoji']} {l_name} ({l_stats['correlation']:.4f} correlation)"
            )
            logger.warning("      💡 Consider removing low-influence components for efficiency")

    unused_components = [
        name for name, stats in component_stats.items() if stats["correlation"] < 0.05
    ]

    if unused_components:
        logger.warning(
            f"   🚫 Minimal influence components ({len(unused_components)}): {', '.join(unused_components)}"
        )
        total_unused_dims = sum(
            obs_components[comp][1] - obs_components[comp][0] for comp in unused_components
        )
        logger.warning(f"      💡 Could remove {total_unused_dims}D from observation space")

    critical_components = [
        name for name, stats in component_stats.items() if stats["correlation"] > 0.2
    ]

    if critical_components:
        logger.warning(
            f"   🔥 Critical components ({len(critical_components)}): {', '.join(critical_components)}"
        )
        logger.warning("      ✅ These are essential for decision-making")
