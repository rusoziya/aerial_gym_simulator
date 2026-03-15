from __future__ import annotations

import torch

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("gate_variant_selection")


def apply_gate_variant_selection(
    env_ids: torch.Tensor | None,
    global_tensor_dict: dict[str, object],
    num_envs: int,
    device: str,
    ige_env_prepared: bool,
) -> None:
    """Select exactly one gate variant per environment and hide the rest."""
    if env_ids is None:
        return
    if not ige_env_prepared:
        logger.debug("[GateVariant] Skipping selection (IGE_env not prepared yet)")
        return

    ids: list[int] = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
    env_asset_state = global_tensor_dict["unfolded_env_asset_state_tensor"].view(num_envs, -1, 13)

    for env_id in ids:
        gate_indices: list[int] = global_tensor_dict["gate_variant_indices_per_env"][env_id]
        gate_names: list[str] = global_tensor_dict["gate_variant_names_per_env"][env_id]
        if not gate_indices:
            continue

        disable_rand = bool(global_tensor_dict.get("gate_randomization/disabled", False))
        if disable_rand:
            chosen_idx = _select_fixed_gate(gate_names, global_tensor_dict)
        else:
            chosen_idx = _select_curriculum_gate(
                gate_names, global_tensor_dict, env_id, device
            )

        chosen_local_index = gate_indices[chosen_idx]
        global_tensor_dict["active_gate_variant_index"][env_id] = chosen_local_index
        global_tensor_dict["active_gate_variant_array_index"][env_id] = chosen_idx

        _place_gate_variants(
            env_asset_state, env_id, gate_indices, chosen_idx, device
        )

    global_tensor_dict["unfolded_env_asset_state_tensor"][:] = env_asset_state.view(-1, 13)


def _parse_gate_scales(gate_names: list[str]) -> list[tuple[int, int, str]]:
    """Parse (index, scale_percent, name) from gate variant names."""
    parsed: list[tuple[int, int, str]] = []
    for j, name in enumerate(gate_names):
        scale = 100
        if isinstance(name, str) and "gate_scale_" in name:
            try:
                scale = int(name.replace("gate_scale_", ""))
            except (ValueError, TypeError):
                scale = 100
        parsed.append((j, scale, name))
    return parsed


def _select_fixed_gate(
    gate_names: list[str],
    global_tensor_dict: dict[str, object],
) -> int:
    """Select a gate variant at a fixed scale (ablation mode)."""
    fixed_scale = int(global_tensor_dict.get("gate_randomization/fixed_scale_percent", 100))
    parsed = _parse_gate_scales(gate_names)
    if not parsed:
        return 0
    exact = [j for (j, s, _) in parsed if s == fixed_scale]
    if exact:
        return exact[0]
    nearest = min(parsed, key=lambda t: abs(t[1] - fixed_scale))
    return nearest[0]


def _compute_min_allowed_scale(
    cur_level: int,
    eval_stretch_enabled: bool,
    stretch_end_level: int,
) -> int:
    """Compute the minimum allowed gate scale based on curriculum level."""
    if cur_level <= 3:
        return 80
    if cur_level >= 23 and (not eval_stretch_enabled or stretch_end_level <= 23):
        return 60
    if cur_level < 23:
        frac = (cur_level - 3) / (23 - 3)
        raw = 80 - frac * (80 - 60)
    else:
        upper = max(23, stretch_end_level)
        span = max(1, upper - 23)
        frac = min(1.0, (cur_level - 23) / float(span))
        raw = 60 - frac * (60 - 50)
    min_allowed = int((int(raw) // 2) * 2)
    return max(50, min(100, min_allowed))


def _select_curriculum_gate(
    gate_names: list[str],
    global_tensor_dict: dict[str, object],
    env_id: int,
    device: str,
) -> int:
    """Select a gate variant based on curriculum-unlocked scales."""
    cur_level = global_tensor_dict.get("curriculum_level", 3)
    try:
        cur_level = int(cur_level.item()) if isinstance(cur_level, torch.Tensor) else int(cur_level)
    except (ValueError, TypeError):
        cur_level = 3

    eval_stretch_enabled = bool(global_tensor_dict.get("eval_stretch_enabled", False))
    stretch_end_level = int(global_tensor_dict.get("eval_stretch_end_level", 23))
    min_allowed_scale = _compute_min_allowed_scale(
        cur_level, eval_stretch_enabled, stretch_end_level
    )

    parsed = _parse_gate_scales(gate_names)
    allowed_js = [j for (j, scale, _) in parsed if scale >= min_allowed_scale]
    if not allowed_js:
        parsed.sort(key=lambda x: x[1], reverse=True)
        allowed_js = [j for (j, _, __) in parsed[:1]] if parsed else []

    if not allowed_js:
        return 0

    allowed_pairs = [(j, scale) for (j, scale, _) in parsed if j in allowed_js]
    unique_scales = sorted({scale for (_, scale) in allowed_pairs}, reverse=True)

    global_tensor_dict["gate_variant_counter"][env_id] += 1
    scale_idx = int(
        torch.randint(low=0, high=len(unique_scales), size=(1,), device=device).item()
    )
    chosen_scale = unique_scales[scale_idx]
    candidates = [j for (j, scale) in allowed_pairs if scale == chosen_scale]
    cand_idx = int(
        torch.randint(low=0, high=len(candidates), size=(1,), device=device).item()
    )
    return candidates[cand_idx]


def _place_gate_variants(
    env_asset_state: torch.Tensor,
    env_id: int,
    gate_indices: list[int],
    chosen_idx: int,
    device: str,
) -> None:
    """Place chosen gate at center, hide all others."""
    num_assets = env_asset_state.shape[1]
    for j, local_index in enumerate(gate_indices):
        if local_index < 0 or local_index >= num_assets:
            continue
        if j == chosen_idx:
            env_asset_state[env_id, local_index, 0:3] = torch.tensor(
                [0.0, 0.0, 0.0], device=device
            )
        else:
            env_asset_state[env_id, local_index, 0:3] = torch.tensor(
                [-1000.0, -1000.0, -1000.0], device=device
            )
