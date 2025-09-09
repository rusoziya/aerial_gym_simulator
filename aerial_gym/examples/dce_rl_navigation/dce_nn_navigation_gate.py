import isaacgym  # noqa: F401 (ensures gym is loaded)

import torch
import os
import time
import numpy as np
from math import sqrt
from aerial_gym.utils import get_args
from aerial_gym.registry.task_registry import task_registry

from aerial_gym.examples.dce_rl_navigation.dce_navigation_task_gate import (
    DCE_RL_Navigation_Task_Gate,
)
from aerial_gym.examples.dce_rl_navigation.sf_inference_class_gate import (
    NN_Inference_Class_Gate,
)
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net_gate import (
    parse_aerialgym_cfg,
)


def main():
    # Add optional flags to override W&B settings for inference runs
    args = get_args(additional_parameters=[
        {
            "name": "--run_name",
            "type": str,
            "default": "",
            "help": "W&B run name (overrides default eval_<experiment>).",
        },
        {
            "name": "--wandb_project",
            "type": str,
            "default": "",
            "help": "W&B project name for inference (overrides config/env).",
        },
        {
            "name": "--wandb_entity",
            "type": str,
            "default": "",
            "help": "W&B entity (team/user) for inference (overrides config/env).",
        },
        {
            "name": "--wandb_dir",
            "type": str,
            "default": "",
            "help": "Local W&B directory to store run files (optional).",
        },
    ])
    headless = getattr(args, "headless", False)
    print(f"DCE Gate Inference - Headless mode: {headless}")

    # Build eval cfg and apply static camera overrides to task config before registering
    cfg = parse_aerialgym_cfg(evaluation=True)
    # Instantiate task_config the same way as training does
    task_config_class = task_registry.get_task_config("navigation_task_gate")
    base_task_config = task_config_class()
    try:
        # Env count
        if hasattr(cfg, "env_agents") and cfg.env_agents:
            base_task_config.num_envs = int(cfg.env_agents)
            os.environ["SF_ENV_AGENTS"] = str(int(cfg.env_agents))
        # Apply viewer/headless explicitly (training script exports SF_HEADLESS; here set config directly too)
        if hasattr(base_task_config, "headless"):
            base_task_config.headless = bool(headless)
        # Gate/obstacle overrides (mirror training subprocess setup)
        if hasattr(cfg, "disable_gate_size_randomization"):
            base_task_config.disable_gate_size_randomization = bool(getattr(cfg, "disable_gate_size_randomization", False))
        if hasattr(cfg, "fixed_gate_scale_percent"):
            base_task_config.fixed_gate_scale_percent = int(getattr(cfg, "fixed_gate_scale_percent", 100))
        if hasattr(cfg, "disable_obstacle_randomization"):
            base_task_config.disable_obstacle_randomization = bool(getattr(cfg, "disable_obstacle_randomization", False))
        if hasattr(cfg, "fixed_obstacles_behind_gate"):
            base_task_config.fixed_obstacles_behind_gate = int(getattr(cfg, "fixed_obstacles_behind_gate", 0))
        # Static camera control flags
        if hasattr(base_task_config, "static_camera_yaw_sweep_enabled"):
            base_task_config.static_camera_yaw_sweep_enabled = bool(getattr(cfg, "enable_static_camera_yaw_sweep", False))
        if hasattr(base_task_config, "enable_static_camera_yaw_sweep"):
            base_task_config.enable_static_camera_yaw_sweep = bool(getattr(cfg, "enable_static_camera_yaw_sweep", False))
        if hasattr(base_task_config, "static_camera_yaw_sweep_speed_deg"):
            base_task_config.static_camera_yaw_sweep_speed_deg = float(getattr(cfg, "static_camera_yaw_sweep_speed_deg", 180.0))
        # Base position overrides
        if hasattr(base_task_config, "static_camera_base_y"):
            base_task_config.static_camera_base_y = float(getattr(cfg, "static_camera_base_y", -3.0))
        if hasattr(base_task_config, "static_camera_base_z"):
            base_task_config.static_camera_base_z = getattr(cfg, "static_camera_base_z", "fixed")
        # Orientation randomization toggle
        if hasattr(base_task_config, "disable_static_camera_orientation_randomization"):
            base_task_config.disable_static_camera_orientation_randomization = bool(getattr(cfg, "disable_static_camera_orientation_randomization", False))
        # Dynamic camera following toggles
        try:
            cur = getattr(base_task_config, 'curriculum', None)
            if cur is not None:
                if hasattr(cfg, 'disable_dynamic_camera_following') and bool(getattr(cfg, 'disable_dynamic_camera_following', False)):
                    setattr(cur, 'enable_dynamic_camera_following', False)
                if hasattr(cfg, 'enable_dynamic_camera_following') and getattr(cfg, 'enable_dynamic_camera_following') is not None:
                    setattr(cur, 'enable_dynamic_camera_following', bool(getattr(cfg, 'enable_dynamic_camera_following')))
        except Exception:
            pass
    except Exception:
        pass

    # Register gate task for inference with updated config
    task_registry.register_task(
        task_name="dce_navigation_task_gate",
        task_class=DCE_RL_Navigation_Task_Gate,
        task_config=base_task_config,
    )

    # Build env (respect CLI seed like training)
    seed_val = None
    try:
        if hasattr(cfg, "seed") and cfg.seed is not None:
            seed_val = int(cfg.seed)
    except Exception:
        seed_val = None
    rl_task = task_registry.make_task(
        "dce_navigation_task_gate", seed=seed_val, use_warp=True, headless=headless
    )
    print("Number of environments", rl_task.num_envs)

    # Set compat attribute on the task
    try:
        setattr(rl_task, "sf_cfg", cfg)
    except Exception:
        pass

    # obs/action dims for gate task
    num_actions = rl_task.task_config.action_space_dim
    num_obs = rl_task.task_config.observation_space_dim

    # Inference wrapper uses Sample Factory checkpoints/config
    nn_model = NN_Inference_Class_Gate(rl_task.num_envs, num_actions, num_obs, cfg)
    nn_model.eval()
    nn_model.reset(torch.arange(rl_task.num_envs))

    with torch.no_grad():
        # Optional W&B init for inference logging
        wandb_run = None
        try:
            import os as _os
            use_wandb = _os.environ.get("WANDB_DISABLED", "true").lower() == "false"
        except Exception:
            use_wandb = False
        if use_wandb:
            try:
                import wandb  # noqa: F401
                # Resolve project/entity with override order: CLI > ENV > CFG > default
                project = (
                    getattr(args, 'wandb_project', '') or
                    _os.environ.get('WANDB_PROJECT', '') or
                    getattr(cfg, 'wandb_project', '') or
                    'gate_eval_runs'
                )
                entity = (
                    getattr(args, 'wandb_entity', '') or
                    _os.environ.get('WANDB_ENTITY', '') or
                    getattr(cfg, 'wandb_user', None)
                ) or None
                # Determine run name: CLI flag > env var > default pattern
                cli_run_name = getattr(args, 'run_name', '') or ''
                env_run_name = _os.environ.get('WANDB_RUN_NAME', '')
                fallback_name = f"eval_{getattr(cfg, 'experiment', getattr(cfg, 'algo', 'sf'))}"
                run_name = cli_run_name if len(cli_run_name) > 0 else (env_run_name if len(env_run_name) > 0 else fallback_name)
                mode = _os.environ.get("WANDB_MODE", "online")
                # Optional custom local directory for run files
                dir_path = getattr(args, 'wandb_dir', '') or _os.environ.get('WANDB_DIR', '')
                if dir_path:
                    wandb_run = wandb.init(project=project, entity=entity, name=run_name, mode=mode, dir=dir_path)
                else:
                    wandb_run = wandb.init(project=project, entity=entity, name=run_name, mode=mode)
                # Define common step metric and map groups to frames
                try:
                    wandb.define_metric('frames')
                    for name in (
                        'curriculum/level','curriculum/progress',
                        'episode_extra_stats/path_efficiency','episode_extra_stats/time_to_gate_steps',
                        'episode_extra_stats/min_gate_distance','episode_extra_stats/center_offset_success',
                        'episode_extra_stats/height_offset_success','episode_extra_stats/gate_pass_rate',
                        'episode_extra_stats/successes','episode_extra_stats/crashes','episode_extra_stats/timeouts',
                        'path_efficiency_running_mean','time_to_gate_running_mean','min_gate_distance_running_mean',
                        'center_offset_running_mean','height_offset_running_mean',
                        # new: episode counters and curriculum totals/rates
                        'episode_extra_stats/episodes_total','episode_extra_stats/episodes_crossed',
                        'curriculum/total_successes','curriculum/total_crashes','curriculum/total_timeouts',
                        'curriculum/total_resets','curriculum/success_rate','curriculum/crash_rate','curriculum/timeout_rate',
                        # episode return/length
                        'episode_extra_stats/episode_return','episode_return_mean','episode_return_std',
                        'episode_length','episode_length_mean','success_only/episode_return_mean','success_only/episode_length_mean',
                        # running success/crash diagnostics
                        'success_rate_running','early_crash_rate',
                        # action statistics
                        'action_abs_mean/x','action_abs_mean/y','action_abs_mean/z','action_abs_mean/yaw',
                        'action_diff_mean/x','action_diff_mean/y','action_diff_mean/z','action_diff_mean/yaw',
                        'action_saturation_rate',
                        # episode-level action diff means per dimension
                        'episode_action_diff_mean/x','episode_action_diff_mean/y','episode_action_diff_mean/z','episode_action_diff_mean/yaw',
                        # VAE sanity
                        'vae/drone_norm_mean','vae/static_norm_mean','vae/static_to_drone_norm_ratio',
                        # throughput
                        'throughput/fps_env','throughput/episodes_per_min',
                        # spatial quantiles
                        'spatial/center_offset_p50','spatial/center_offset_p90',
                        'spatial/height_offset_p50','spatial/height_offset_p90',
                        'spatial/min_gate_distance_p50','spatial/min_gate_distance_p90'
                    ):
                        try:
                            wandb.define_metric(name, step_metric='frames')
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                wandb_run = None

        # Global step/episode counters for logging
        _frames = 0
        _episodes_done = 0
        try:
            _max_episodes = int(getattr(cfg, 'max_num_episodes', 0))
        except Exception:
            _max_episodes = 0

        # Running episode-aggregates to mirror training-side summaries
        _traj_running = {
            'path_efficiency_sum': 0.0,
            'path_efficiency_count': 0,
            'min_gate_distance_sum': 0.0,
            'min_gate_distance_count': 0,
            'time_to_gate_sum': 0.0,
            'time_to_gate_count': 0,
            'center_offset_sum': 0.0,
            'center_offset_count': 0,
            'height_offset_sum': 0.0,
            'height_offset_count': 0,
            'episodes_total': 0,
            'episodes_crossed': 0,
        }
        _totals = {'successes': 0, 'crashes': 0, 'timeouts': 0}

        # Episode-wise accumulators per env
        _ep_return = torch.zeros(rl_task.num_envs, dtype=torch.float32)
        _ep_length = torch.zeros(rl_task.num_envs, dtype=torch.int32)
        _hist_returns = []
        _hist_returns_success = []
        _hist_lengths = []
        _hist_lengths_success = []
        # Spatial history (for quantiles)
        _hist_center_offset = []
        _hist_height_offset = []
        _hist_min_gate_distance = []
        # Action tracking
        _prev_actions = torch.zeros(rl_task.num_envs, num_actions, dtype=torch.float32)
        # Episode accumulators for per-dimension action diffs
        _ep_action_diff_sum = torch.zeros(rl_task.num_envs, num_actions, dtype=torch.float32)
        _ep_action_diff_count = torch.zeros(rl_task.num_envs, dtype=torch.int32)
        # Success EMA
        _success_ema = None
        _ema_alpha = 0.1
        # Throughput
        _t0 = time.time()
        _episode_counter_total = 0

        # GIF saving (optional)
        save_gifs = bool(getattr(cfg, 'save_gifs', False))
        # Save GIFs alongside the inference script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gif_dir = os.path.join(script_dir, 'gif_episodes')
        if save_gifs:
            try:
                os.makedirs(gif_dir, exist_ok=True)
            except Exception:
                save_gifs = False
        _gif_drone_noised_frames = []
        _gif_static_noised_frames = []
        _gif_static_seg_frames = []
        _gif_drone_recon_frames = []
        _gif_static_recon_frames = []
        _gif_episode_counter_env0 = 0
        _gif_every_n = 1  # save every episode for env 0 (inference only)

        # One-time per-frame latent logging for env 0
        _print_env0_once = os.environ.get('PRINT_ENV0_LATENTS_ONCE', 'false').lower() == 'true'
        # state: 0=armed (wait for nonzero), 1=active (printing this episode), 2=done (printed one episode)
        _env0_log_state = 0 if _print_env0_once else 2
        _env0_step_in_ep = 0

        # One-time per-frame NORMALIZED latent logging for env 0
        _print_env0_norm_once = os.environ.get('PRINT_ENV0_LATENTS_ONCE_NORM', 'false').lower() == 'true'
        _env0_norm_log_state = 0 if _print_env0_norm_once else 2
        _env0_norm_step_in_ep = 0

        def _to_pil_gray(image_data):
            try:
                import numpy as _np
                from PIL import Image as _PILImage
                if isinstance(image_data, torch.Tensor):
                    img = image_data.detach().cpu().numpy()
                else:
                    img = _np.array(image_data)
                if img.ndim > 2:
                    img = _np.squeeze(img)
                img_u8 = (img.clip(0.0, 1.0) * 255.0).astype('uint8')
                return _PILImage.fromarray(img_u8)
            except Exception:
                return None

        def _save_gif(frames, filename):
            if not frames or len(frames) < 1:
                return
            try:
                frames[0].save(
                    os.path.join(gif_dir, filename),
                    save_all=True,
                    append_images=frames[1:],
                    duration=100,
                    loop=0,
                )
            except Exception:
                pass

        # Rolling windows / CI helpers
        _success_flags = []  # append 1 for success, 0 otherwise
        _time_to_gate_hist = []  # for successes
        _latency_ms_hist = []
        _crash_bins = {'<=5': 0, '6-20': 0, '>20': 0}
        _reset_hist = {'success': 0, 'crash': 0, 'timeout': 0}
        _per_level = {}  # level -> {'succ': int, 'tot': int, 'ttg_sum': float}

        def _wilson_ci(s, n, z=1.96):
            if n <= 0:
                return 0.0, 0.0
            p = s / n
            denom = 1.0 + z*z/n
            centre = p + z*z/(2*n)
            rad = z*sqrt((p*(1.0-p)+z*z/(4*n))/n)
            lo = (centre - rad)/denom
            hi = (centre + rad)/denom
            return max(0.0, lo), min(1.0, hi)

        reset_result = rl_task.reset()
        obs_dict = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        # Determine the observation key the model expects
        nn_obs_key = getattr(getattr(nn_model, "cfg", object()), "obs_key", "obs")
        # One-time evaluation config summary (curriculum/randomization/effective flags)
        try:
            print("=" * 80)
            print("EVAL CONFIG SUMMARY (Gate Navigation)")
            # Curriculum level
            try:
                lvl = int(getattr(rl_task, 'curriculum_level', -1))
            except Exception:
                lvl = -1
            print(f"  Curriculum level: {lvl}")
            # Curriculum-derived camera noise and frame dropout at this level (if available)
            try:
                cur = getattr(rl_task.task_config, 'curriculum', None)
                if cur is not None:
                    try:
                        cstd, cdrop = cur.get_camera_noise(lvl)
                        print(f"  Camera noise: gaussian_std={float(cstd):.4f}, dropout={float(cdrop):.4f}")
                    except Exception:
                        pass
                    try:
                        fd = cur.get_camera_frame_dropout(lvl)
                        if isinstance(fd, dict):
                            dt = fd.get('drone_total', 0.0); st = fd.get('static_total', 0.0)
                            print(f"  Frame dropout: drone_total={float(dt):.4f}, static_total={float(st):.4f}")
                    except Exception:
                        pass
            except Exception:
                pass
            # Effective toggles from global tensor dict (authoritative during runtime)
            try:
                gtd = getattr(rl_task, 'sim_env', None)
                g = gtd.global_tensor_dict if (gtd is not None and hasattr(gtd, 'global_tensor_dict')) else {}
                def _p(key, label=None):
                    if key in g:
                        print(f"  {(label or key)}: {g[key]}")
                _p('gate_randomization/disabled', 'Gate size randomization disabled')
                _p('obstacles_randomization/disabled', 'Obstacle randomization disabled')
                _p('obstacles_randomization/fixed_count', 'Fixed obstacles behind gate')
                _p('state_randomization/noise_disabled', 'State noise disabled')
                _p('camera_randomization/noise_disabled', 'Camera noise disabled (global)')
                _p('camera_randomization/drone_noise_disabled', 'Drone noise disabled')
                _p('camera_randomization/static_noise_disabled', 'Static noise disabled')
                _p('camera_randomization/frame_dropout_disabled', 'Camera frame dropout disabled (global)')
                _p('camera_randomization/drone_frame_dropout_disabled', 'Drone frame dropout disabled')
                _p('camera_randomization/static_frame_dropout_disabled', 'Static frame dropout disabled')
                _p('dynamic_camera_following/disabled', 'Dynamic camera following disabled')
                _p('static_camera/yaw_sweep_enabled', 'Static camera yaw sweep enabled')
                _p('static_camera/yaw_sweep_speed_deg', 'Static camera yaw sweep speed (deg/s)')
                _p('static_camera/base_y', 'Static camera base Y')
                _p('static_camera/base_z', 'Static camera base Z')
            except Exception:
                pass
            # Observation ablation spec
            try:
                import os as _os
                print(f"  ABLATE_OBS_RANGES: {_os.environ.get('ABLATE_OBS_RANGES', '')}")
            except Exception:
                pass
            print("=" * 80)
        except Exception:
            pass
        # Apply the same observation ablation used during training if requested via env vars
        # Global cap for ablation debug prints across the whole run
        _ablate_debug_total = 0

        def _apply_obs_ablation(obs_tensor: torch.Tensor) -> torch.Tensor:
            import os as _os
            if obs_tensor is None:
                return obs_tensor
            # Work on a private copy to avoid aliasing/view issues
            obs_tensor = obs_tensor.clone()
            debug = _os.environ.get("ABLATE_DEBUG", "false").lower() == "true"
            nonlocal _ablate_debug_total
            # Simple switch for drone position (parity with training wrapper)
            if _os.environ.get("ABLATE_DRONE_POS", "false").lower() == "true":
                start, end = 0, 3
                obs_tensor[:, start:end] = 0.0
                if debug and _ablate_debug_total < 10:
                    v = obs_tensor[:, start:end]
                    print(f"[ABLATE_DEBUG] applied: {start}:{end}=zero | min={v.min().item():.3e} max={v.max().item():.3e} mean={v.mean().item():.3e} nonzero={int((v!=0).sum().item())}")
                    _ablate_debug_total += 1
            # General ranges
            spec_str = _os.environ.get("ABLATE_OBS_RANGES", "").strip()
            if not spec_str:
                return obs_tensor
            grad_mask = None
            zero_ranges = []
            zerograd_ranges = []
            for spec in spec_str.split(","):
                spec = spec.strip()
                if not spec:
                    continue
                if "=" not in spec:
                    continue
                lhs, rhs = spec.split("=", 1)
                lhs = lhs.strip(); rhs = rhs.strip()
                if ":" not in lhs:
                    continue
                try:
                    start_s, end_s = lhs.split(":", 1)
                    start = int(start_s); end = int(end_s)
                except Exception:
                    continue
                op = rhs
                if op == "zero":
                    if grad_mask is None:
                        grad_mask = torch.ones_like(obs_tensor)
                    grad_mask[:, start:end] = 0.0
                    zero_ranges.append((start, end))
                elif op == "zerograd":
                    zerograd_ranges.append((start, end))
                elif op == "shuffle":
                    if obs_tensor.shape[0] > 1:
                        perm = torch.randperm(obs_tensor.shape[0], device=obs_tensor.device)
                        obs_tensor[:, start:end] = obs_tensor[perm, start:end]
                    if debug and _ablate_debug_total < 10:
                        v = obs_tensor[:, start:end]
                        print(f"[ABLATE_DEBUG] applied: {start}:{end}=shuffle | sample_env0={v[0].detach().cpu().numpy()} sample_env1={v[1].detach().cpu().numpy() if v.shape[0]>1 else 'NA'}")
                        _ablate_debug_total += 1
                elif op.startswith("noise:"):
                    try:
                        std = float(op.split(":", 1)[1])
                    except Exception:
                        std = 0.0
                    if std > 0.0:
                        obs_tensor[:, start:end] = obs_tensor[:, start:end] + torch.randn_like(obs_tensor[:, start:end]) * std
                    if debug and _ablate_debug_total < 10:
                        v = obs_tensor[:, start:end]
                        print(f"[ABLATE_DEBUG] applied: {start}:{end}=noise:{std} | std_est={v.std().item():.3e} mean={v.mean().item():.3e}")
                        _ablate_debug_total += 1
            if grad_mask is not None:
                obs_tensor = obs_tensor * grad_mask
                if debug:
                    for (zs, ze) in zero_ranges:
                        if _ablate_debug_total >= 10:
                            break
                        v = obs_tensor[:, zs:ze]
                        print(f"[ABLATE_DEBUG] applied: {zs}:{ze}=zero | min={v.min().item():.3e} max={v.max().item():.3e} mean={v.mean().item():.3e} nonzero={int((v!=0).sum().item())}")
                        _ablate_debug_total += 1
            for (start, end) in zerograd_ranges:
                zero_slice = torch.zeros_like(obs_tensor[:, start:end])
                left = obs_tensor[:, :start]
                right = obs_tensor[:, end:]
                obs_tensor = torch.cat([left, zero_slice, right], dim=-1)
                if debug and _ablate_debug_total < 10:
                    v = obs_tensor[:, start:end]
                    print(f"[ABLATE_DEBUG] applied: {start}:{end}=zerograd | min={v.min().item():.3e} max={v.max().item():.3e} mean={v.mean().item():.3e} nonzero={int((v!=0).sum().item())}")
                    _ablate_debug_total += 1
            return obs_tensor
        while True:
            vec = obs_dict["observations"] if isinstance(obs_dict, dict) and "observations" in obs_dict else obs_dict["obs"]
            # Apply ablation if configured
            vec = _apply_obs_ablation(vec)
            # Optional: print env0 latent stats every frame for one episode only
            if _env0_log_state != 2:
                try:
                    if isinstance(vec, torch.Tensor) and vec.ndim == 2 and vec.shape[0] > 0 and vec.shape[1] >= 150:
                        ze0 = vec[0, 22:86]
                        zs0 = vec[0, 86:150]
                        ze_abs = float(torch.mean(torch.abs(ze0)).item())
                        zs_abs = float(torch.mean(torch.abs(zs0)).item())
                        if _env0_log_state == 0:
                            # arm -> activate only when latents become nonzero to avoid printing a 0-only reset frame
                            if (ze_abs > 1e-6) or (zs_abs > 1e-6):
                                _env0_log_state = 1
                                _env0_step_in_ep = 0
                                print(f"[ENV0_LATENTS] step={_env0_step_in_ep} abs_mean: drone={ze_abs:.6f} static={zs_abs:.6f}")
                                # Also print normalized latents for the same step
                                try:
                                    norm = prepare_and_normalize_obs(nn_model.actor_critic, {nn_obs_key: vec})
                                    pvec = norm.get(nn_obs_key, None)
                                    if isinstance(pvec, torch.Tensor) and pvec.ndim == 2 and pvec.shape[0] > 0 and pvec.shape[1] >= 150:
                                        ze0n = pvec[0, 22:86]
                                        zs0n = pvec[0, 86:150]
                                        ze_nabs = float(torch.mean(torch.abs(ze0n)).item())
                                        zs_nabs = float(torch.mean(torch.abs(zs0n)).item())
                                        print(f"[ENV0_LATENTS_NORM] step={_env0_step_in_ep} abs_mean: drone={ze_nabs:.6f} static={zs_nabs:.6f}")
                                except Exception:
                                    pass
                                _env0_step_in_ep += 1
                        elif _env0_log_state == 1:
                            print(f"[ENV0_LATENTS] step={_env0_step_in_ep} abs_mean: drone={ze_abs:.6f} static={zs_abs:.6f}")
                            # Also print normalized latents for the same step
                            try:
                                norm = prepare_and_normalize_obs(nn_model.actor_critic, {nn_obs_key: vec})
                                pvec = norm.get(nn_obs_key, None)
                                if isinstance(pvec, torch.Tensor) and pvec.ndim == 2 and pvec.shape[0] > 0 and pvec.shape[1] >= 150:
                                    ze0n = pvec[0, 22:86]
                                    zs0n = pvec[0, 86:150]
                                    ze_nabs = float(torch.mean(torch.abs(ze0n)).item())
                                    zs_nabs = float(torch.mean(torch.abs(zs0n)).item())
                                    print(f"[ENV0_LATENTS_NORM] step={_env0_step_in_ep} abs_mean: drone={ze_nabs:.6f} static={zs_nabs:.6f}")
                            except Exception:
                                pass
                            _env0_step_in_ep += 1
                except Exception:
                    pass

            # Optional: print env0 NORMALIZED latent stats every frame for one episode only
            if _env0_norm_log_state != 2:
                try:
                    mo = {nn_obs_key: vec}
                    norm = prepare_and_normalize_obs(nn_model.actor_critic, mo)
                    pvec = norm.get(nn_obs_key, None)
                    if isinstance(pvec, torch.Tensor) and pvec.ndim == 2 and pvec.shape[0] > 0 and pvec.shape[1] >= 150:
                        ze0n = pvec[0, 22:86]
                        zs0n = pvec[0, 86:150]
                        ze_nabs = float(torch.mean(torch.abs(ze0n)).item())
                        zs_nabs = float(torch.mean(torch.abs(zs0n)).item())
                        if _env0_norm_log_state == 0:
                            if (ze_nabs > 1e-6) or (zs_nabs > 1e-6):
                                _env0_norm_log_state = 1
                                _env0_norm_step_in_ep = 0
                                print(f"[ENV0_LATENTS_NORM] step={_env0_norm_step_in_ep} abs_mean: drone={ze_nabs:.6f} static={zs_nabs:.6f}")
                                _env0_norm_step_in_ep += 1
                        elif _env0_norm_log_state == 1:
                            print(f"[ENV0_LATENTS_NORM] step={_env0_norm_step_in_ep} abs_mean: drone={ze_nabs:.6f} static={zs_nabs:.6f}")
                            _env0_norm_step_in_ep += 1
                except Exception:
                    pass
            model_obs = {nn_obs_key: vec}
            actions = nn_model.get_action(model_obs)
            step_result = rl_task.step(actions)
            # Unpack step result (env returns 5-tuple)
            if isinstance(step_result, tuple) and len(step_result) == 5:
                obs_next, rew, terminated, truncated, infos = step_result
            else:
                # Fallback to previous behavior
                obs_next = step_result[0] if isinstance(step_result, tuple) else step_result
                rew = None; terminated = None; truncated = None; infos = {}

            _frames += 1

            # Accumulate returns and lengths
            try:
                if isinstance(rew, torch.Tensor):
                    _ep_return += rew.float().cpu()
                _ep_length += 1
            except Exception:
                pass

            # Action stats
            try:
                a = actions.detach().float().cpu()
                adiff = torch.abs(a - _prev_actions)
                abs_mean = torch.mean(torch.abs(a), dim=0)
                diff_mean = torch.mean(adiff, dim=0)
                sat = torch.mean((torch.abs(a) > 0.95).float()).item()
                # Store prev
                _prev_actions = a
                # Update episode accumulators
                try:
                    _ep_action_diff_sum += adiff
                    _ep_action_diff_count += 1
                except Exception:
                    pass
            except Exception:
                abs_mean = None; diff_mean = None; sat = None

            # Collect GIF frames (env 0 only, noised versions)
            if save_gifs:
                try:
                    d = getattr(rl_task, 'obs_dict', {})
                    if isinstance(d, dict) and 'depth_range_pixels_noised' in d:
                        dd = d['depth_range_pixels_noised']
                        if isinstance(dd, torch.Tensor) and dd.ndim >= 3:
                            pil = _to_pil_gray(dd[0, 0])
                            if pil is not None:
                                _gif_drone_noised_frames.append(pil)
                            # Reconstruct from latents if available on the task
                            try:
                                vae = getattr(rl_task, 'shared_vae_model', None)
                                lat = getattr(rl_task, 'image_latents', None)
                                if vae is not None and lat is not None and isinstance(lat, torch.Tensor) and lat.shape[0] > 0:
                                    rec = vae.decode(lat[0:1])  # (1,1,H,W)
                                    if isinstance(rec, torch.Tensor):
                                        rec_img = rec[0, 0]
                                        pil_rec = _to_pil_gray(rec_img)
                                        if pil_rec is not None:
                                            _gif_drone_recon_frames.append(pil_rec)
                            except Exception:
                                pass
                    if isinstance(d, dict) and 'static_depth_noised' in d:
                        sd = d['static_depth_noised']
                        # static_depth_noised may be tensor (H,W) or (N,H,W); pick env 0
                        if isinstance(sd, torch.Tensor):
                            if sd.ndim == 3:
                                pil = _to_pil_gray(sd[0])
                            else:
                                pil = _to_pil_gray(sd)
                        else:
                            pil = _to_pil_gray(sd)
                        if pil is not None:
                            _gif_static_noised_frames.append(pil)
                        # Reconstruct from static latents if available
                        try:
                            vae = getattr(rl_task, 'shared_vae_model', None)
                            slat = getattr(rl_task, 'static_image_latents', None)
                            if vae is not None and slat is not None and isinstance(slat, torch.Tensor) and slat.shape[0] > 0:
                                srec = vae.decode(slat[0:1])  # (1,1,H,W)
                                if isinstance(srec, torch.Tensor):
                                    srec_img = srec[0, 0]
                                    pil_srec = _to_pil_gray(srec_img)
                                    if pil_srec is not None:
                                        _gif_static_recon_frames.append(pil_srec)
                        except Exception:
                            pass
                    # Also collect static segmentation directly from StaticCameraManager (env 0)
                    try:
                        scm = getattr(rl_task, 'static_camera_manager', None)
                        if scm is not None and hasattr(scm, 'capture_images'):
                            _d, _seg = scm.capture_images(batched=False)
                            if _seg is not None:
                                # Normalize to [0,1] for visualization
                                import numpy as _np
                                seg = _seg
                                if isinstance(seg, _np.ndarray) and seg.ndim > 2:
                                    seg = _np.squeeze(seg)
                                seg_min = _np.min(seg)
                                seg_max = _np.max(seg)
                                if (seg_max - seg_min) > 0:
                                    seg_norm = (seg - seg_min) / float(seg_max - seg_min)
                                else:
                                    seg_norm = _np.zeros_like(seg, dtype=_np.float32)
                                pil_seg = _to_pil_gray(seg_norm)
                                if pil_seg is not None:
                                    _gif_static_seg_frames.append(pil_seg)
                    except Exception:
                        pass
                except Exception:
                    pass

            # VAE latent norms (periodic)
            vae_drone_norm = None; vae_static_norm = None; vae_ratio = None
            try:
                if (_frames % 100) == 0:
                    z_e = vec[:, 22:86]
                    z_s = vec[:, 86:150]
                    vae_drone_norm = float(torch.linalg.norm(z_e, dim=1).mean().item())
                    vae_static_norm = float(torch.linalg.norm(z_s, dim=1).mean().item())
                    vae_ratio = float(vae_static_norm / (vae_drone_norm + 1e-6))
            except Exception:
                pass

            # Per-reset logging similar to training
            try:
                import torch as _torch
                if (terminated is not None and truncated is not None and
                        (_torch.any(terminated) or _torch.any(truncated))):
                    if wandb_run is not None:
                        payload = {}
                        payload['frames'] = _frames
                        # Curriculum level/progress
                        try:
                            lvl = float(getattr(rl_task, 'curriculum_level', -1))
                            payload['curriculum/level'] = lvl
                        except Exception:
                            pass
                        try:
                            prog = float(getattr(rl_task, 'curriculum_progress_fraction', 0.0))
                            payload['curriculum/progress'] = prog
                        except Exception:
                            pass
                        # Episode-level metrics provided by env
                        if isinstance(infos, dict):
                            extra = infos.get('episode_extra_stats', None)
                            if isinstance(extra, dict):
                                for k, v in list(extra.items()):
                                    try:
                                        payload[k] = float(v)
                                    except Exception:
                                        try:
                                            import torch as __t
                                            if isinstance(v, __t.Tensor):
                                                payload[k] = float(v.item()) if v.numel() == 1 else None
                                        except Exception:
                                            pass
                                # Update running aggregates from episode_extra_stats
                                def _get_float(name):
                                    val = extra.get(name, None)
                                    try:
                                        return float(val)
                                    except Exception:
                                        try:
                                            import torch as __tt
                                            if isinstance(val, __tt.Tensor):
                                                return float(val.item()) if val.numel() == 1 else None
                                        except Exception:
                                            return None
                                        return None
                                pe = _get_float('path_efficiency')
                                ttg = _get_float('time_to_gate_steps')
                                mgd = _get_float('min_gate_distance')
                                co  = _get_float('center_offset_success')
                                ho  = _get_float('height_offset_success')
                                crossed = _get_float('episodes_crossed')
                                total   = _get_float('episodes_total')
                                if pe is not None:
                                    _traj_running['path_efficiency_sum'] += pe
                                    _traj_running['path_efficiency_count'] += 1
                                if mgd is not None:
                                    _traj_running['min_gate_distance_sum'] += mgd
                                    _traj_running['min_gate_distance_count'] += 1
                                if ttg is not None:
                                    _traj_running['time_to_gate_sum'] += ttg
                                    _traj_running['time_to_gate_count'] += 1
                                if co is not None:
                                    _traj_running['center_offset_sum'] += co
                                    _traj_running['center_offset_count'] += 1
                                if ho is not None:
                                    _traj_running['height_offset_sum'] += ho
                                    _traj_running['height_offset_count'] += 1
                                # Append to spatial histories for quantiles
                                if co is not None:
                                    _hist_center_offset.append(co)
                                if ho is not None:
                                    _hist_height_offset.append(ho)
                                if mgd is not None:
                                    _hist_min_gate_distance.append(mgd)
                                if total is not None:
                                    _traj_running['episodes_total'] += int(total)
                                elif crossed is not None:
                                    _traj_running['episodes_total'] += 1
                                if crossed is not None and crossed > 0:
                                    _traj_running['episodes_crossed'] += int(crossed)
                            # successes/crashes/timeouts if present this step
                            for key in ('successes', 'crashes', 'timeouts'):
                                val = infos.get(key, None)
                                try:
                                    if isinstance(val, _torch.Tensor):
                                        c = float(val.sum().item())
                                        payload[f'episode_extra_stats/{key}'] = c
                                        _totals[key] += int(c)
                                    elif isinstance(val, (int, float)):
                                        c = float(val)
                                        payload[f'episode_extra_stats/{key}'] = c
                                        _totals[key] += int(c)
                                except Exception:
                                    pass
                        # Episode return/length for envs that reset now
                        try:
                            ids = (terminated | truncated).nonzero(as_tuple=True)[0]
                            if ids.numel() > 0:
                                ids_cpu = ids.cpu()
                                ret = _ep_return[ids_cpu].numpy()
                                length = _ep_length[ids_cpu].numpy()
                                # Stop env0 one-episode logging when env 0 resets
                                if _env0_log_state == 1:
                                    if 0 in ids_cpu.tolist():
                                        _env0_log_state = 2
                                        print(f"[ENV0_LATENTS] episode_end steps={_env0_step_in_ep}")
                                if _env0_norm_log_state == 1:
                                    if 0 in ids_cpu.tolist():
                                        _env0_norm_log_state = 2
                                        print(f"[ENV0_LATENTS_NORM] episode_end steps={_env0_norm_step_in_ep}")
                                # Reset accumulators for those envs
                                _ep_return[ids_cpu] = 0.0
                                _ep_length[ids_cpu] = 0
                                # Episode action diff means per dimension (average over steps, then average over resetting envs)
                                try:
                                    counts = _ep_action_diff_count[ids].float().clamp_min(1.0)
                                    per_env_means = _ep_action_diff_sum[ids] / counts.unsqueeze(1)
                                    diff_ep_mean = per_env_means.mean(dim=0)
                                    dims = ['x','y','z','yaw']
                                    for i, d in enumerate(dims[:diff_ep_mean.numel()]):
                                        payload[f'episode_action_diff_mean/{d}'] = float(diff_ep_mean[i].item())
                                except Exception:
                                    pass
                                # Clear episode accumulators for these envs
                                try:
                                    _ep_action_diff_sum[ids] = 0.0
                                    _ep_action_diff_count[ids] = 0
                                except Exception:
                                    pass
                                # Identify success/crash via infos tensors when available
                                succ_mask = None
                                try:
                                    s = infos.get('successes', None)
                                    if isinstance(s, _torch.Tensor):
                                        succ_mask = s[ids].bool().cpu().numpy()
                                except Exception:
                                    succ_mask = None
                                # Update histories
                                _hist_returns.extend(ret.tolist())
                                _hist_lengths.extend(length.tolist())
                                if succ_mask is not None and succ_mask.size == ret.size:
                                    _hist_returns_success.extend(ret[succ_mask].tolist())
                                    _hist_lengths_success.extend(length[succ_mask].tolist())
                                # Log per-reset means
                                if len(_hist_returns) > 0:
                                    payload['episode_return_mean'] = float(np.mean(_hist_returns))
                                    payload['episode_return_std'] = float(np.std(_hist_returns))
                                if len(_hist_lengths) > 0:
                                    payload['episode_length_mean'] = float(np.mean(_hist_lengths))
                                if len(_hist_returns_success) > 0:
                                    payload['success_only/episode_return_mean'] = float(np.mean(_hist_returns_success))
                                if len(_hist_lengths_success) > 0:
                                    payload['success_only/episode_length_mean'] = float(np.mean(_hist_lengths_success))
                                # Early crash rate among current resets (<=5 steps and crash)
                                try:
                                    c = infos.get('crashes', None)
                                    if isinstance(c, _torch.Tensor):
                                        crashed = c[ids].bool().cpu().numpy()
                                        early = (length <= 5)
                                        denom = max(1, crashed.size)
                                        payload['early_crash_rate'] = float((crashed & early).sum() / denom)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # Running means (mirroring training names)
                        def _safe_mean(sum_key, count_key):
                            s = _traj_running.get(sum_key, 0.0)
                            c = max(1, _traj_running.get(count_key, 0))
                            return float(s / c)
                        payload['path_efficiency_running_mean'] = _safe_mean('path_efficiency_sum','path_efficiency_count')
                        payload['min_gate_distance_running_mean'] = _safe_mean('min_gate_distance_sum','min_gate_distance_count')
                        payload['time_to_gate_running_mean'] = _safe_mean('time_to_gate_sum','time_to_gate_count')
                        payload['center_offset_running_mean'] = _safe_mean('center_offset_sum','center_offset_count')
                        payload['height_offset_running_mean'] = _safe_mean('height_offset_sum','height_offset_count')
                        if _traj_running['episodes_total'] > 0:
                            payload['episode_extra_stats/gate_pass_rate'] = float(_traj_running['episodes_crossed']) / float(_traj_running['episodes_total'])
                        # Derive cumulative totals/rates (aligned with training naming)
                        total_resets = _totals['successes'] + _totals['crashes'] + _totals['timeouts']
                        payload['curriculum/total_successes'] = float(_totals['successes'])
                        payload['curriculum/total_crashes'] = float(_totals['crashes'])
                        payload['curriculum/total_timeouts'] = float(_totals['timeouts'])
                        payload['curriculum/total_resets'] = float(total_resets)
                        if total_resets > 0:
                            payload['curriculum/success_rate'] = float(_totals['successes']) / float(total_resets)
                            payload['curriculum/crash_rate'] = float(_totals['crashes']) / float(total_resets)
                            payload['curriculum/timeout_rate'] = float(_totals['timeouts']) / float(total_resets)
                        # Success EMA
                        try:
                            if _success_ema is None and total_resets > 0:
                                _success_ema = payload.get('curriculum/success_rate', None)
                            elif total_resets > 0 and _success_ema is not None:
                                _success_ema = (1 - _ema_alpha) * _success_ema + _ema_alpha * payload.get('curriculum/success_rate', 0.0)
                            if _success_ema is not None:
                                payload['success_rate_running'] = float(_success_ema)
                                # Also expose as target success running mean for clearer naming in dashboards
                                payload['target_success_running_mean'] = float(_success_ema)
                        except Exception:
                            pass
                        # Spatial quantiles
                        try:
                            if len(_hist_center_offset) > 0:
                                payload['spatial/center_offset_p50'] = float(np.percentile(_hist_center_offset, 50))
                                payload['spatial/center_offset_p90'] = float(np.percentile(_hist_center_offset, 90))
                            if len(_hist_height_offset) > 0:
                                payload['spatial/height_offset_p50'] = float(np.percentile(_hist_height_offset, 50))
                                payload['spatial/height_offset_p90'] = float(np.percentile(_hist_height_offset, 90))
                            if len(_hist_min_gate_distance) > 0:
                                payload['spatial/min_gate_distance_p50'] = float(np.percentile(_hist_min_gate_distance, 50))
                                payload['spatial/min_gate_distance_p90'] = float(np.percentile(_hist_min_gate_distance, 90))
                        except Exception:
                            pass
                        # Action stats (latest step)
                        try:
                            if abs_mean is not None:
                                dims = ['x','y','z','yaw']
                                for i, d in enumerate(dims[:abs_mean.numel()]):
                                    payload[f'action_abs_mean/{d}'] = float(abs_mean[i].item())
                            if diff_mean is not None:
                                dims = ['x','y','z','yaw']
                                for i, d in enumerate(dims[:diff_mean.numel()]):
                                    payload[f'action_diff_mean/{d}'] = float(diff_mean[i].item())
                            if sat is not None:
                                payload['action_saturation_rate'] = float(sat)
                        except Exception:
                            pass
                        # VAE norms (periodic)
                        if vae_drone_norm is not None:
                            payload['vae/drone_norm_mean'] = vae_drone_norm
                            payload['vae/static_norm_mean'] = vae_static_norm if vae_static_norm is not None else 0.0
                            payload['vae/static_to_drone_norm_ratio'] = vae_ratio if vae_ratio is not None else 0.0
                        # Throughput
                        try:
                            dt = max(1e-6, time.time() - _t0)
                            fps = _frames / dt
                            epm = (_episode_counter_total / dt) * 60.0
                            payload['throughput/fps_env'] = float(fps)
                            payload['throughput/episodes_per_min'] = float(epm)
                        except Exception:
                            pass
                        # Log
                        try:
                            import wandb  # noqa: F401
                            wandb.log(payload, step=_frames)
                        except Exception:
                            pass
                    # Update episode counter by number of resets this step
                    try:
                        ids = (terminated | truncated).nonzero(as_tuple=True)[0]
                        _episodes_done += int(ids.numel())
                        _episode_counter_total += int(ids.numel())
                        # Save GIFs for env 0 occasionally
                        if save_gifs and (ids.numel() > 0) and (0 in ids.cpu().tolist()):
                            _gif_episode_counter_env0 += 1
                            if (_gif_episode_counter_env0 % _gif_every_n) == 0:
                                _save_gif(_gif_drone_noised_frames, f"episode_{_gif_episode_counter_env0:04d}_drone_depth_D455_NOISED.gif")
                                _save_gif(_gif_static_noised_frames, f"episode_{_gif_episode_counter_env0:04d}_static_depth_D455_NOISED.gif")
                                _save_gif(_gif_static_seg_frames, f"episode_{_gif_episode_counter_env0:04d}_static_seg.gif")
                                _save_gif(_gif_drone_recon_frames, f"episode_{_gif_episode_counter_env0:04d}_drone_depth_VAE_RECON.gif")
                                _save_gif(_gif_static_recon_frames, f"episode_{_gif_episode_counter_env0:04d}_static_depth_VAE_RECON.gif")
                            # Clear buffers for env 0 episode
                            _gif_drone_noised_frames = []
                            _gif_static_noised_frames = []
                            _gif_static_seg_frames = []
                            _gif_drone_recon_frames = []
                            _gif_static_recon_frames = []
                    except Exception:
                        _episodes_done += 1
            except Exception:
                pass

            # Termination condition by episodes if configured
            if _max_episodes > 0 and _episodes_done >= _max_episodes:
                break

            # Continue loop with next obs dict
            obs_dict = obs_next


if __name__ == "__main__":
    main()


