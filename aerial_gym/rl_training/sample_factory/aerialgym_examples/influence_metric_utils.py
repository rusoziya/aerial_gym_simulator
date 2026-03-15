from __future__ import annotations

import math
from dataclasses import dataclass, field

SLICE_NAMES_STATE: set[str] = {
    "drone_position",
    "static_camera_pos",
    "static_camera_orient",
    "drone_orientation",
    "linear_vel",
    "angular_vel",
    "actions",
    "drone_linear_vel",
    "drone_angular_vel",
    "drone_actions",
}
SLICE_NAMES_CAMERA: set[str] = {"drone_camera_vae", "static_camera_vae"}
SLICE_NAMES_VISUAL: set[str] = {"drone_camera_vae", "static_camera_vae"}
SLICE_NAMES_KINEMATIC: set[str] = {
    "drone_linear_vel",
    "drone_angular_vel",
    "drone_actions",
    "linear_vel",
    "angular_vel",
    "actions",
}
SLICE_NAMES_SPATIAL: set[str] = {
    "drone_position",
    "static_camera_pos",
    "static_camera_orient",
    "drone_orientation",
}

CURRICULUM_KEYS: list[str] = [
    "curriculum/total_timeouts",
    "curriculum/total_successes",
    "curriculum/total_crashes",
    "curriculum/total_resets",
    "curriculum/timeout_rate",
    "curriculum/success_rate",
    "curriculum/state_noise_static_pos_std_m",
    "curriculum/state_noise_static_orient_std_deg",
    "curriculum/state_noise_drone_pos_std_m",
    "curriculum/state_noise_drone_orient_std_deg",
    "curriculum/progress",
    "curriculum/obstacles_behind_gate",
    "curriculum/max_level_reached",
    "curriculum/level",
    "curriculum/crash_rate",
    "curriculum/camera_max_angle",
    "curriculum/camera_gaussian_std",
    "curriculum/camera_frame_freeze_static",
    "curriculum/camera_frame_freeze_drone",
    "curriculum/camera_frame_dropout_static_total",
    "curriculum/camera_frame_dropout_drone_total",
    "curriculum/camera_frame_blank_static",
    "curriculum/camera_frame_blank_drone",
    "curriculum/camera_dropout_rate",
    "curriculum/camera_current_angle",
]

OBS_GRAD_PREFIXES: tuple[str, ...] = (
    "obs_grad/",
    "influence/",
    "grad_attr/",
    "obs_influence/",
)

SUFFIX_POSTFIXES: list[str] = [
    "_mean_norm_recent",
    "_mean_norm_overall",
    "_recent",
    "_overall",
    "_mean_norm",
    "_mean",
    "_norm",
]


def strip_base_postfix(name: str) -> str:
    """Remove known measurement postfixes to get the base slice name."""
    for post in SUFFIX_POSTFIXES:
        if name.endswith(post):
            return name[: -len(post)]
    return name


def is_obs_grad_key(name: str) -> bool:
    return isinstance(name, str) and name.startswith(OBS_GRAD_PREFIXES)


def classify_base(base: str) -> tuple[bool, bool, bool, bool, bool]:
    """Return (is_camera, is_state, is_visual, is_kinematic, is_spatial) for a base name."""
    is_camera = base in SLICE_NAMES_CAMERA or "camera_vae" in base
    is_state = (
        base in SLICE_NAMES_STATE
        or base.startswith("drone_position")
        or base.startswith("drone_orientation")
        or base.startswith("drone_linear_vel")
        or base.startswith("drone_angular_vel")
        or base.startswith("drone_actions")
        or base.startswith("static_camera_pos")
        or base.startswith("static_camera_orient")
    ) and not is_camera
    is_visual = base in SLICE_NAMES_VISUAL or "camera_vae" in base
    is_kinematic = (
        base in SLICE_NAMES_KINEMATIC
        or base.startswith("drone_linear_vel")
        or base.startswith("drone_angular_vel")
        or base.startswith("drone_actions")
    )
    is_spatial = (
        base in SLICE_NAMES_SPATIAL
        or base.startswith("drone_position")
        or base.startswith("drone_orientation")
        or base.startswith("static_camera_pos")
        or base.startswith("static_camera_orient")
    )
    return is_camera, is_state, is_visual, is_kinematic, is_spatial


def compute_obs_grad_shares(
    source_metrics: dict[str, float],
) -> dict[str, float]:
    """Compute camera/state/modality shares and per-slice metrics from obs-grad data."""
    total_val = 0.0
    camera_val = 0.0
    state_val = 0.0
    visual_val = 0.0
    kinematic_val = 0.0
    spatial_val = 0.0
    slice_values: dict[str, float] = {}

    for name, val in source_metrics.items():
        if not is_obs_grad_key(name):
            continue
        parts = name.split("/")
        label = parts[-1]
        if "slice_pct" in parts:
            idx = parts.index("slice_pct")
            if idx + 1 < len(parts):
                label = parts[idx + 1]
        elif "slice_mag" in parts:
            idx = parts.index("slice_mag")
            if idx + 1 < len(parts):
                label = parts[idx + 1]
        suffix = parts[-1]
        if suffix.startswith("total_") or suffix == "backward_passes":
            continue
        try:
            scalar = float(val)
        except (ValueError, TypeError):
            continue
        total_val += scalar
        base = strip_base_postfix(label)
        slice_values[base] = scalar
        is_camera, is_state, is_visual, is_kinematic, is_spatial = classify_base(base)
        if is_camera:
            camera_val += scalar
        elif is_state:
            state_val += scalar
        if is_visual:
            visual_val += scalar
        if is_kinematic:
            kinematic_val += scalar
        if is_spatial:
            spatial_val += scalar

    result: dict[str, float] = {}
    if total_val <= 0.0:
        return result

    if state_val <= 0.0 and camera_val > 0.0:
        state_val = max(total_val - camera_val, 0.0)

    camera_share = camera_val / total_val
    state_share = state_val / total_val
    visual_share = visual_val / total_val
    kinematic_share = kinematic_val / total_val
    spatial_share = spatial_val / total_val

    result["camera_share"] = camera_share
    result["state_share"] = state_share
    result["camera_share_pct"] = camera_share * 100.0
    result["state_share_pct"] = state_share * 100.0
    result["visual_share"] = visual_share
    result["kinematic_share"] = kinematic_share
    result["spatial_share"] = spatial_share
    result["visual_share_pct"] = visual_share * 100.0
    result["kinematic_share_pct"] = kinematic_share * 100.0
    result["spatial_share_pct"] = spatial_share * 100.0

    for base, sval in slice_values.items():
        result[f"slice_mag/{base}"] = sval
        result[f"slice_pct/{base}"] = (sval / total_val) * 100.0

    return result


def compute_obs_pct_by_window(
    source_metrics: dict[str, float],
) -> dict[str, float]:
    """Compute per-observation recent/overall percentages with sum/residual guardrails."""
    base_recent: dict[str, float] = {}
    base_overall: dict[str, float] = {}

    for name, val in source_metrics.items():
        if not isinstance(name, str) or not name.startswith(
            ("obs_grad/", "influence/", "grad_attr/")
        ):
            continue
        parts = name.split("/")
        suffix = parts[-1]
        base = strip_base_postfix(suffix)
        try:
            scalar = float(val)
        except (ValueError, TypeError):
            continue
        if suffix.endswith("_mean_norm_recent") or suffix.endswith("_recent"):
            base_recent[base] = base_recent.get(base, 0.0) + scalar
        if suffix.endswith("_mean_norm_overall") or suffix.endswith("_overall"):
            base_overall[base] = base_overall.get(base, 0.0) + scalar

    result: dict[str, float] = {}
    tot_r = sum(base_recent.values())
    if tot_r > 0.0:
        sum_pct_r = 0.0
        for b, v in base_recent.items():
            p = 100.0 * v / tot_r
            result[f"obs_pct_recent/{b}"] = p
            sum_pct_r += p
        result["obs_pct_recent/_sum"] = sum_pct_r
        result["obs_pct_recent/_residual"] = 100.0 - sum_pct_r

    tot_o = sum(base_overall.values())
    if tot_o > 0.0:
        sum_pct_o = 0.0
        for b, v in base_overall.items():
            p = 100.0 * v / tot_o
            result[f"obs_pct_overall/{b}"] = p
            sum_pct_o += p
        result["obs_pct_overall/_sum"] = sum_pct_o
        result["obs_pct_overall/_residual"] = 100.0 - sum_pct_o

    return result


@dataclass
class _WindowedAccumulator:
    """Accumulates scalar values into total/camera/state/modality buckets by window."""

    total: float = 0.0
    camera: float = 0.0
    state: float = 0.0
    total_recent: float = 0.0
    total_overall: float = 0.0
    camera_recent: float = 0.0
    state_recent: float = 0.0
    camera_overall: float = 0.0
    state_overall: float = 0.0
    visual_recent: float = 0.0
    kinematic_recent: float = 0.0
    spatial_recent: float = 0.0
    visual_overall: float = 0.0
    kinematic_overall: float = 0.0
    spatial_overall: float = 0.0
    slice_vals: dict[str, float] = field(default_factory=dict)

    def accumulate(self, suffix: str, scalar: float) -> None:
        self.total += scalar
        self.slice_vals[suffix] = scalar
        base = strip_base_postfix(suffix)
        is_recent = suffix.endswith("_mean_norm_recent") or suffix.endswith("_recent")
        is_overall = suffix.endswith("_mean_norm_overall") or suffix.endswith("_overall")
        if is_recent:
            self.total_recent += scalar
        if is_overall:
            self.total_overall += scalar
        is_cam, is_st, is_vis, is_kin, is_spa = classify_base(base)
        if is_cam:
            self.camera += scalar
            if is_recent:
                self.camera_recent += scalar
            if is_overall:
                self.camera_overall += scalar
        elif is_st:
            self.state += scalar
            if is_recent:
                self.state_recent += scalar
            if is_overall:
                self.state_overall += scalar
        if is_vis:
            if is_recent:
                self.visual_recent += scalar
            if is_overall:
                self.visual_overall += scalar
        if is_kin:
            if is_recent:
                self.kinematic_recent += scalar
            if is_overall:
                self.kinematic_overall += scalar
        if is_spa:
            if is_recent:
                self.spatial_recent += scalar
            if is_overall:
                self.spatial_overall += scalar


def _get_window_values(
    acc: _WindowedAccumulator, window: str
) -> tuple[float, float, float, float, float]:
    """Return (camera, state, visual, kinematic, spatial) values for the given window."""
    if window == "recent":
        return (
            acc.camera_recent,
            acc.state_recent,
            acc.visual_recent,
            acc.kinematic_recent,
            acc.spatial_recent,
        )
    return (
        acc.camera_overall,
        acc.state_overall,
        acc.visual_overall,
        acc.kinematic_overall,
        acc.spatial_overall,
    )


def _emit_window_shares(
    acc: _WindowedAccumulator, window_total: float, suffix: str
) -> dict[str, float]:
    """Emit camera/state/modality share metrics for a specific window (recent/overall)."""
    if window_total <= 0.0:
        return {}
    cam_val, st_val, vis_val, kin_val, spa_val = _get_window_values(acc, suffix)
    cam = cam_val / window_total
    st = st_val / window_total
    vis = vis_val / window_total
    kin = kin_val / window_total
    spa = spa_val / window_total
    return {
        f"camera_share_{suffix}": cam,
        f"state_share_{suffix}": st,
        f"camera_share_pct_{suffix}": cam * 100.0,
        f"state_share_pct_{suffix}": st * 100.0,
        f"visual_share_{suffix}": vis,
        f"kinematic_share_{suffix}": kin,
        f"spatial_share_{suffix}": spa,
        f"visual_share_pct_{suffix}": vis * 100.0,
        f"kinematic_share_pct_{suffix}": kin * 100.0,
        f"spatial_share_pct_{suffix}": spa * 100.0,
    }


def compute_windowed_shares(merged: dict[str, float]) -> dict[str, float]:
    """Compute camera/state/modality shares split by recent vs overall windows."""
    acc = _WindowedAccumulator()

    for name, val in merged.items():
        if not isinstance(name, str) or not name.startswith(
            ("obs_grad/", "influence/", "grad_attr/")
        ):
            continue
        suffix = name.split("/")[-1]
        if suffix.startswith("total_") or suffix == "backward_passes":
            continue
        try:
            scalar = float(val)
        except (ValueError, TypeError):
            continue
        acc.accumulate(suffix, scalar)

    if acc.total <= 0.0:
        return {}

    if acc.state <= 0.0 and acc.camera > 0.0:
        acc.state = max(acc.total - acc.camera, 0.0)

    result: dict[str, float] = {
        "camera_share": acc.camera / acc.total,
        "state_share": acc.state / acc.total,
        "camera_share_pct": (acc.camera / acc.total) * 100.0,
        "state_share_pct": (acc.state / acc.total) * 100.0,
    }
    result.update(_emit_window_shares(acc, acc.total_recent, "recent"))
    result.update(_emit_window_shares(acc, acc.total_overall, "overall"))

    for sfx, sval in acc.slice_vals.items():
        result[f"slice_mag/{sfx}"] = sval
        is_r = sfx.endswith("_mean_norm_recent") or sfx.endswith("_recent")
        is_o = sfx.endswith("_mean_norm_overall") or sfx.endswith("_overall")
        denom = (
            acc.total_recent
            if is_r and acc.total_recent > 0.0
            else (acc.total_overall if is_o and acc.total_overall > 0.0 else acc.total)
        )
        result[f"slice_pct/{sfx}"] = (sval / denom) * 100.0

    return result


def sanitize_metrics(metrics: dict[str, object]) -> dict[str, float | int]:
    """Cast tensor/numpy values to Python scalars, dropping non-loggable entries."""
    for k in list(metrics.keys()):
        v = metrics[k]
        try:
            metrics[k] = float(v)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            try:
                metrics[k] = int(v)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                del metrics[k]
    return metrics  # type: ignore[return-value]


def drop_nonfinite_episode_stats(metrics: dict[str, object]) -> None:
    """Remove NaN/Inf entries under episode_extra_stats/ to prevent W&B propagation."""
    for k in list(metrics.keys()):
        if isinstance(k, str) and k.startswith("episode_extra_stats/"):
            v = metrics[k]
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                del metrics[k]


def metrics_to_float(raw: dict[str, object]) -> dict[str, float]:
    """Cast all values to float, dropping entries that fail conversion."""
    result: dict[str, float] = {}
    for k, v in raw.items():
        try:
            result[k] = float(v)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            pass
    return result


def get_last_stat(latest: dict[str, object], key_name: str) -> float | None:
    """Extract the last scalar value from an episodic stats dict entry."""
    if key_name not in latest:
        return None
    v = latest[key_name]
    if isinstance(v, (list, tuple)) and len(v) > 0:
        return float(v[-1])
    if isinstance(v, (int, float)):
        return float(v)
    return None


def get_last_any(latest: dict[str, object], names: list[str]) -> float | None:
    """Try multiple key names and return the first that yields a value."""
    for nm in names:
        val = get_last_stat(latest, nm)
        if val is not None:
            return val
    return None
