from __future__ import annotations

import isaacgym  # noqa: F401 (ensures gym is loaded)

import torch
import os
import time
import numpy as np
from aerial_gym.utils import get_args
from aerial_gym.registry.task_registry import task_registry

from aerial_gym.examples.dce_rl_navigation.dce_navigation_task_gate import (
    DCE_RL_Navigation_Task_Gate,
)
from aerial_gym.examples.dce_rl_navigation.sf_inference_class_gate import (
    NN_Inference_Class_Gate,
)
from aerial_gym.examples.dce_rl_navigation.inference_utils import apply_obs_ablation
from aerial_gym.examples.dce_rl_navigation.wandb_setup import init_wandb
from aerial_gym.examples.dce_rl_navigation.gif_recorder import GifRecorder
from aerial_gym.examples.dce_rl_navigation.episode_metrics import (
    EpisodeMetrics,
    VisibilityTracker,
)
from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net_gate import (
    parse_aerialgym_cfg,
)

_ADDITIONAL_CLI_PARAMS: list[dict[str, object]] = [
    {"name": "--run_name", "type": str, "default": "", "help": "W&B run name."},
    {
        "name": "--enable_static_camera_arc_follow",
        "type": lambda x: str(x).lower() == "true",
        "default": False,
        "help": "Enable arc-follow static camera mode.",
    },
    {
        "name": "--static_camera_arc_radius_m",
        "type": float,
        "default": 2.0,
        "help": "Arc-follow radius in meters.",
    },
    {
        "name": "--dynamic_camera_follow_y_offset_m",
        "type": float,
        "default": None,
        "help": "Override dynamic camera follow Y-offset in meters.",
    },
    {
        "name": "--disable_dynamic_follow_gate_blending",
        "type": lambda x: str(x).lower() == "true",
        "default": False,
        "help": "Disable blending toward gate in dynamic-follow.",
    },
    {"name": "--wandb_project", "type": str, "default": "", "help": "W&B project name."},
    {"name": "--wandb_entity", "type": str, "default": "", "help": "W&B entity."},
    {"name": "--wandb_dir", "type": str, "default": "", "help": "Local W&B directory."},
    {
        "name": "--rnn_warmup_steps",
        "type": int,
        "default": 50,
        "help": "Pre-logging warm-up steps to prime GRU hidden state.",
    },
]


def _parse_args_and_config() -> tuple:
    """Parse CLI args, build eval config, and apply overrides to a task config.

    Returns (args, cfg, base_task_config) ready for task registration.
    """
    args = get_args(additional_parameters=_ADDITIONAL_CLI_PARAMS)
    print(f"DCE Gate Inference - Headless mode: {args.headless}")

    cfg = parse_aerialgym_cfg(evaluation=True)
    task_config_class = task_registry.get_task_config("navigation_task_gate")
    base_task_config = task_config_class()
    _apply_config_overrides(cfg, args, base_task_config)

    task_registry.register_task(
        task_name="dce_navigation_task_gate",
        task_class=DCE_RL_Navigation_Task_Gate,
        task_config=base_task_config,
    )
    return args, cfg, base_task_config


def _apply_config_overrides(cfg: object, args: object, tc: object) -> None:
    """Apply CLI / SF config overrides to the base task config."""
    try:
        if cfg.env_agents:
            tc.num_envs = int(cfg.env_agents)
            os.environ["SF_ENV_AGENTS"] = str(int(cfg.env_agents))
        tc.headless = bool(args.headless)
        if cfg.disable_gate_size_randomization is not None:
            tc.disable_gate_size_randomization = bool(cfg.disable_gate_size_randomization)
        if cfg.fixed_gate_scale_percent is not None:
            tc.fixed_gate_scale_percent = int(cfg.fixed_gate_scale_percent)
        if cfg.disable_obstacle_randomization is not None:
            tc.disable_obstacle_randomization = bool(cfg.disable_obstacle_randomization)
        if cfg.fixed_obstacles_behind_gate is not None:
            tc.fixed_obstacles_behind_gate = int(cfg.fixed_obstacles_behind_gate)
        tc.static_camera_yaw_sweep_enabled = bool(cfg.enable_static_camera_yaw_sweep)
        tc.enable_static_camera_yaw_sweep = bool(cfg.enable_static_camera_yaw_sweep)
        tc.static_camera_yaw_sweep_speed_deg = float(cfg.static_camera_yaw_sweep_speed_deg)
        tc.static_camera_base_y = float(
            cfg.static_camera_base_y if cfg.static_camera_base_y is not None else -3.0
        )
        tc.static_camera_base_z = cfg.static_camera_base_z
        tc.disable_static_camera_orientation_randomization = bool(
            cfg.disable_static_camera_orientation_randomization
        )
        cur = tc.curriculum
        if cur is not None:
            if bool(cfg.disable_dynamic_camera_following):
                cur.enable_dynamic_camera_following = False
            if cfg.enable_dynamic_camera_following is not None:
                cur.enable_dynamic_camera_following = bool(cfg.enable_dynamic_camera_following)
        if bool(args.enable_static_camera_arc_follow):
            os.environ["SF_ENABLE_STATIC_CAMERA_ARC_FOLLOW"] = "true"
        if args.static_camera_arc_radius_m is not None:
            os.environ["SF_STATIC_CAMERA_ARC_RADIUS_M"] = str(
                float(args.static_camera_arc_radius_m)
            )
        if args.dynamic_camera_follow_y_offset_m is not None:
            os.environ["SF_DYNAMIC_CAMERA_FOLLOW_OFFSET_Y"] = str(
                float(args.dynamic_camera_follow_y_offset_m)
            )
        os.environ["SF_DISABLE_DYNAMIC_FOLLOW_GATE_BLENDING"] = (
            "true" if bool(args.disable_dynamic_follow_gate_blending) else "false"
        )
    except (ValueError, TypeError):
        pass


def _init_task_and_model(
    cfg: object,
    base_task_config: object,
    headless: bool,
) -> tuple[DCE_RL_Navigation_Task_Gate, NN_Inference_Class_Gate, int, int, bool, int]:
    """Create the RL task and inference model."""
    seed_val = None
    try:
        if cfg.seed is not None:
            seed_val = int(cfg.seed)
    except (ValueError, TypeError):
        seed_val = None
    rl_task = task_registry.make_task(
        "dce_navigation_task_gate", seed=seed_val, use_warp=True, headless=headless
    )
    print("Number of environments", rl_task.num_envs)

    dump_obs_parity = os.environ.get("DUMP_OBS_PARITY", "false").lower() == "true"
    parity_dump_steps = int(os.environ.get("OBS_PARITY_STEPS", "0"))

    rl_task.sf_cfg = cfg

    num_actions = rl_task.task_config.action_space_dim
    num_obs = rl_task.task_config.observation_space_dim

    nn_model = NN_Inference_Class_Gate(rl_task.num_envs, num_actions, num_obs, cfg)
    nn_model.eval()
    nn_model.reset(torch.arange(rl_task.num_envs))

    return rl_task, nn_model, num_actions, num_obs, dump_obs_parity, parity_dump_steps


def _run_rnn_warmup(
    rl_task: DCE_RL_Navigation_Task_Gate,
    nn_model: NN_Inference_Class_Gate,
    args: object,
    obs_dict: dict[str, torch.Tensor],
    nn_obs_key: str,
) -> dict[str, torch.Tensor]:
    """Run optional RNN warm-up steps to prime GRU hidden state."""
    try:
        warmup_steps = max(0, int(args.rnn_warmup_steps))
    except (ValueError, TypeError):
        warmup_steps = 0
    if warmup_steps == 0:
        try:
            warmup_steps = max(0, int(os.environ.get("RNN_WARMUP_STEPS", "0")))
        except (ValueError, TypeError):
            warmup_steps = 0
    if warmup_steps <= 0:
        return obs_dict
    try:
        print(f"[RNN_WARMUP] Running {warmup_steps} warm-up steps (not logged)")
        nn_model.eval()
        nn_model.actor_critic.eval()
        o_w = obs_dict
        for _ in range(warmup_steps):
            v_w = (
                o_w["observations"]
                if isinstance(o_w, dict) and "observations" in o_w
                else o_w.get(nn_obs_key, o_w["obs"])
            )
            a_w = nn_model.get_action({nn_obs_key: v_w})
            step_result_w = rl_task.step(a_w)
            if isinstance(step_result_w, tuple) and len(step_result_w) == 5:
                o_next, _, term_w, trunc_w, _ = step_result_w
            else:
                o_next = step_result_w[0] if isinstance(step_result_w, tuple) else step_result_w
                term_w = None
                trunc_w = None
            if term_w is not None and trunc_w is not None:
                done_ids = (term_w | trunc_w).nonzero(as_tuple=True)[0]
                if done_ids.numel() > 0:
                    nn_model.reset(done_ids)
            o_w = o_next
        obs_dict = o_w
    except (ValueError, TypeError):
        pass
    return obs_dict


def _log_wandb_resets(
    wandb_run: object,
    metrics: EpisodeMetrics,
    succ_mask: np.ndarray | None,
    ids_cpu: torch.Tensor,
) -> None:
    """Log per-batch success fraction and per-reset binary success to W&B."""
    import wandb

    if succ_mask is not None:
        batch_succ = float(succ_mask.sum() / max(1, succ_mask.size))
    else:
        batch_succ = float("nan")
    wandb.log(
        {
            "episodes_batch": float(metrics.episode_counter_total),
            "episodes/success_rate_batch": batch_succ,
            "frames": metrics.frames,
            "global_step": metrics.frames,
        },
        step=metrics.frames,
    )
    try:
        if ids_cpu.numel() > 0:
            base_ep = int(metrics.episode_counter_total)
            for j, _env_id in enumerate(ids_cpu.tolist()):
                succ_bit = float(1.0 if (succ_mask is not None and bool(succ_mask[j])) else 0.0)
                wandb.log(
                    {
                        "episodes": float(base_ep + j + 1),
                        "episodes/success_binary": succ_bit,
                        "frames": metrics.frames,
                        "global_step": metrics.frames,
                    },
                    step=metrics.frames,
                )
    except RuntimeError:
        pass


def _run_inference_loop(
    rl_task: DCE_RL_Navigation_Task_Gate,
    nn_model: NN_Inference_Class_Gate,
    cfg: object,
    args: object,
    wandb_run: object | None,
    use_wandb: bool,
    num_actions: int,
    num_obs: int,
    dump_obs_parity: bool,
    parity_dump_steps: int,
) -> None:
    """Run the main inference loop."""
    try:
        nn_obs_key = nn_model.cfg.obs_key
    except AttributeError:
        nn_obs_key = "obs"
    try:
        max_episodes = int(cfg.max_num_episodes)
    except (ValueError, TypeError):
        max_episodes = 0

    metrics = EpisodeMetrics(rl_task.num_envs, num_actions)
    vis_tracker = VisibilityTracker(rl_task.num_envs)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gif_recorder = GifRecorder(
        enabled=bool(cfg.save_gifs),
        output_dir=os.path.join(script_dir, "gif_episodes"),
    )

    reset_result = rl_task.reset()
    obs_dict = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs_dict = _run_rnn_warmup(rl_task, nn_model, args, obs_dict, nn_obs_key)

    while True:
        vec = (
            obs_dict["observations"]
            if isinstance(obs_dict, dict) and "observations" in obs_dict
            else obs_dict["obs"]
        )
        if dump_obs_parity and metrics.frames < parity_dump_steps:
            if isinstance(vec, torch.Tensor) and vec.ndim == 2 and vec.shape[1] >= 150:
                z_drone = vec[:, 22:86]
                z_static = vec[:, 86:150]
                print(
                    f"[OBS_PARITY] step={metrics.frames} abs_mean: "
                    f"drone={float(z_drone.abs().mean().item()):.6f} "
                    f"static={float(z_static.abs().mean().item()):.6f}"
                )

        vec = apply_obs_ablation(vec)
        if isinstance(vec, torch.Tensor):
            vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        actions = nn_model.get_action({nn_obs_key: vec})
        step_result = rl_task.step(actions)

        if isinstance(step_result, tuple) and len(step_result) == 5:
            obs_next, rew, terminated, truncated, infos = step_result
        else:
            obs_next = step_result[0] if isinstance(step_result, tuple) else step_result
            rew = None
            terminated = None
            truncated = None
            infos = {}

        action_stats = metrics.step(rew, actions)

        # VAE latent norms (periodic)
        vae_drone_norm = None
        vae_static_norm = None
        vae_ratio = None
        if (metrics.frames % 100) == 0:
            z_e = vec[:, 22:86]
            z_s = vec[:, 86:150]
            vae_drone_norm = float(torch.linalg.norm(z_e, dim=1).mean().item())
            vae_static_norm = float(torch.linalg.norm(z_s, dim=1).mean().item())
            vae_ratio = float(vae_static_norm / (vae_drone_norm + 1e-6))

        gif_recorder.collect_frames(rl_task.obs_dict)
        try:
            gif_recorder.collect_static_camera_frames(rl_task.static_camera_manager)
        except (ValueError, TypeError):
            pass

        vis_tracker.accumulate(infos)

        _handle_resets(
            rl_task,
            nn_model,
            terminated,
            truncated,
            infos,
            metrics,
            vis_tracker,
            gif_recorder,
            wandb_run,
            action_stats,
            vae_drone_norm,
            vae_static_norm,
            vae_ratio,
        )

        if max_episodes > 0 and metrics.episodes_done >= max_episodes:
            break
        obs_dict = obs_next


def _handle_resets(
    rl_task: DCE_RL_Navigation_Task_Gate,
    nn_model: NN_Inference_Class_Gate,
    terminated: torch.Tensor | None,
    truncated: torch.Tensor | None,
    infos: dict[str, object],
    metrics: EpisodeMetrics,
    vis_tracker: VisibilityTracker,
    gif_recorder: GifRecorder,
    wandb_run: object | None,
    action_stats: object | None,
    vae_drone_norm: float | None,
    vae_static_norm: float | None,
    vae_ratio: float | None,
) -> None:
    """Process episode resets: update metrics, log to W&B, save GIFs."""
    if terminated is None or truncated is None:
        return
    if not (torch.any(terminated) or torch.any(truncated)):
        return
    try:
        ids = (terminated | truncated).nonzero(as_tuple=True)[0]
        if ids.numel() == 0:
            return
    except RuntimeError:
        metrics.episodes_done += 1
        return

    ids_cpu = ids.cpu()
    succ_mask = _extract_success_mask(ids, infos)

    if wandb_run is not None:
        import wandb

        payload: dict[str, float] = {"frames": metrics.frames}
        payload["episode_extra_stats/curriculum/level"] = float(rl_task.curriculum_level)
        payload["episode_extra_stats/curriculum/progress"] = float(
            rl_task.curriculum_progress_fraction
        )
        _add_extra_stats(payload, infos)

        reset_payload = metrics.on_resets(ids, infos, nn_model)
        payload.update(reset_payload)
        metrics.add_action_stats(payload, action_stats)
        metrics.add_vae_and_throughput(payload, vae_drone_norm, vae_static_norm, vae_ratio)

        _log_wandb_resets(wandb_run, metrics, succ_mask, ids_cpu)

        wandb.log(payload, step=metrics.frames)

        vis_tracker.on_resets(ids)
        total_resets = (
            metrics.totals["successes"] + metrics.totals["crashes"] + metrics.totals["timeouts"]
        )
        vis_payload = vis_tracker.get_running_payload(metrics.frames, total_resets)
        wandb.log(vis_payload, step=metrics.frames)
    else:
        metrics.on_resets(ids, infos, nn_model)
        vis_tracker.on_resets(ids)

    if gif_recorder.enabled and 0 in ids_cpu.tolist():
        gif_recorder.on_episode_end()


def _add_extra_stats(payload: dict[str, float], infos: dict[str, object]) -> None:
    """Extract episode_extra_stats from infos into the payload."""
    if not isinstance(infos, dict):
        return
    extra = infos.get("episode_extra_stats")
    if not isinstance(extra, dict):
        return
    for k, v in list(extra.items()):
        try:
            payload[k] = float(v)
        except (ValueError, TypeError):
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                payload[k] = float(v.item())


def _extract_success_mask(ids: torch.Tensor, infos: dict[str, object]) -> np.ndarray | None:
    try:
        s = infos.get("successes", None)
        if isinstance(s, torch.Tensor):
            return s[ids].bool().cpu().numpy()
    except (KeyError, TypeError):
        pass
    return None


def main() -> None:
    args, cfg, base_task_config = _parse_args_and_config()

    rl_task, nn_model, num_actions, num_obs, dump_obs_parity, parity_dump_steps = (
        _init_task_and_model(cfg, base_task_config, args.headless)
    )

    with torch.no_grad():
        wandb_run, use_wandb = init_wandb(cfg, args)

        _run_inference_loop(
            rl_task=rl_task,
            nn_model=nn_model,
            cfg=cfg,
            args=args,
            wandb_run=wandb_run,
            use_wandb=use_wandb,
            num_actions=num_actions,
            num_obs=num_obs,
            dump_obs_parity=dump_obs_parity,
            parity_dump_steps=parity_dump_steps,
        )


if __name__ == "__main__":
    main()
