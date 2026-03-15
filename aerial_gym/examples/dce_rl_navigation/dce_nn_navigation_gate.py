from __future__ import annotations

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


def main() -> None:
    # Add optional flags to override W&B settings for inference runs
    args = get_args(additional_parameters=[
        {
            "name": "--run_name",
            "type": str,
            "default": "",
            "help": "W&B run name (overrides default eval_<experiment>).",
        },
        {
            "name": "--enable_static_camera_arc_follow",
            "type": lambda x: str(x).lower() == 'true',
            "default": False,
            "help": "Enable arc-follow static camera mode (fixed-radius arc around gate)",
        },
        {
            "name": "--static_camera_arc_radius_m",
            "type": float,
            "default": 2.0,
            "help": "Arc-follow radius in meters (default 2.0)",
        },
        {
            "name": "--dynamic_camera_follow_y_offset_m",
            "type": float,
            "default": None,
            "help": "Override dynamic camera follow Y-offset in meters (requires dynamic follow to be enabled)",
        },
        {
            "name": "--disable_dynamic_follow_gate_blending",
            "type": lambda x: str(x).lower() == 'true',
            "default": False,
            "help": "Disable blending toward gate in dynamic-follow; always look at drone",
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
        {
            "name": "--rnn_warmup_steps",
            "type": int,
            "default": 50,
            "help": "Pre-logging warm-up steps to prime GRU hidden state (0 disables).",
        },
    ])
    headless = args.headless
    print(f"DCE Gate Inference - Headless mode: {headless}")

    # Build eval cfg and apply static camera overrides to task config before registering
    cfg = parse_aerialgym_cfg(evaluation=True)
    # Instantiate task_config the same way as training does
    task_config_class = task_registry.get_task_config("navigation_task_gate")
    base_task_config = task_config_class()
    try:
        # Env count
        if cfg.env_agents:
            base_task_config.num_envs = int(cfg.env_agents)
            os.environ["SF_ENV_AGENTS"] = str(int(cfg.env_agents))
        # Apply viewer/headless explicitly (training script exports SF_HEADLESS; here set config directly too)
        base_task_config.headless = bool(headless)
        # Gate/obstacle overrides (mirror training subprocess setup)
        if cfg.disable_gate_size_randomization is not None:
            base_task_config.disable_gate_size_randomization = bool(cfg.disable_gate_size_randomization)
        if cfg.fixed_gate_scale_percent is not None:
            base_task_config.fixed_gate_scale_percent = int(cfg.fixed_gate_scale_percent)
        if cfg.disable_obstacle_randomization is not None:
            base_task_config.disable_obstacle_randomization = bool(cfg.disable_obstacle_randomization)
        if cfg.fixed_obstacles_behind_gate is not None:
            base_task_config.fixed_obstacles_behind_gate = int(cfg.fixed_obstacles_behind_gate)
        # Static camera control flags
        base_task_config.static_camera_yaw_sweep_enabled = bool(cfg.enable_static_camera_yaw_sweep)
        base_task_config.enable_static_camera_yaw_sweep = bool(cfg.enable_static_camera_yaw_sweep)
        base_task_config.static_camera_yaw_sweep_speed_deg = float(cfg.static_camera_yaw_sweep_speed_deg)
        # Base position overrides
        base_task_config.static_camera_base_y = float(getattr(cfg, "static_camera_base_y", -3.0))
        base_task_config.static_camera_base_z = cfg.static_camera_base_z
        # Orientation randomization toggle
        base_task_config.disable_static_camera_orientation_randomization = bool(cfg.disable_static_camera_orientation_randomization)
        # Dynamic camera following toggles
        cur = base_task_config.curriculum
        if cur is not None:
            if hasattr(cfg, 'disable_dynamic_camera_following') and bool(cfg.disable_dynamic_camera_following):
                cur.enable_dynamic_camera_following = False
            if cfg.enable_dynamic_camera_following is not None:
                cur.enable_dynamic_camera_following = bool(cfg.enable_dynamic_camera_following)
        # Arc-follow overrides to env variables
        if bool(args.enable_static_camera_arc_follow):
            os.environ['SF_ENABLE_STATIC_CAMERA_ARC_FOLLOW'] = 'true'
        if hasattr(args, 'static_camera_arc_radius_m') and args.static_camera_arc_radius_m is not None:
            os.environ['SF_STATIC_CAMERA_ARC_RADIUS_M'] = str(float(args.static_camera_arc_radius_m))
        # Dynamic follow distance override (honored only if dynamic-follow is active)
        if hasattr(args, 'dynamic_camera_follow_y_offset_m') and args.dynamic_camera_follow_y_offset_m is not None:
            os.environ['SF_DYNAMIC_CAMERA_FOLLOW_OFFSET_Y'] = str(float(args.dynamic_camera_follow_y_offset_m))
        # Disable dynamic follow gate blending (optional)
        if hasattr(args, 'disable_dynamic_follow_gate_blending'):
            os.environ['SF_DISABLE_DYNAMIC_FOLLOW_GATE_BLENDING'] = 'true' if bool(getattr(args, 'disable_dynamic_follow_gate_blending')) else 'false'
    except (ValueError, TypeError):
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
        if cfg.seed is not None:
            seed_val = int(cfg.seed)
    except (ValueError, TypeError):
        seed_val = None
    rl_task = task_registry.make_task(
        "dce_navigation_task_gate", seed=seed_val, use_warp=True, headless=headless
    )
    print("Number of environments", rl_task.num_envs)

    # Optional: strict obs tensor parity dump for first few steps (disabled by default)
    dump_obs_parity = os.environ.get('DUMP_OBS_PARITY', 'false').lower() == 'true'
    parity_dump_steps = int(os.environ.get('OBS_PARITY_STEPS', '0'))

    # Set compat attribute on the task
    rl_task.sf_cfg = cfg

    # obs/action dims for gate task
    num_actions = rl_task.task_config.action_space_dim
    num_obs = rl_task.task_config.observation_space_dim

    # Inference wrapper uses Sample Factory checkpoints/config
    nn_model = NN_Inference_Class_Gate(rl_task.num_envs, num_actions, num_obs, cfg)
    nn_model.eval()
    nn_model.reset(torch.arange(rl_task.num_envs))

    with torch.no_grad():
        # Optional RNN warm-up (burn-in) before counting/logging episodes.
        # We keep the same eval_deterministic mode, run with no_grad + eval mode,
        # and DO NOT reset the env after warm-up. We reset RNN state only for
        # envs that finish during warm-up to avoid stale memory.
        try:
            warmup_steps = max(0, int(args.rnn_warmup_steps))
        except (ValueError, TypeError):
            warmup_steps = 0
        # Fallback to environment variable if CLI flag not provided
        if warmup_steps == 0:
            try:
                warmup_steps = max(0, int(os.environ.get('RNN_WARMUP_STEPS', '0')))
            except (ValueError, TypeError):
                warmup_steps = 0
        if warmup_steps > 0:
            try:
                print(f"[RNN_WARMUP] Running {warmup_steps} warm-up steps (not logged)")
                # Ensure model stays in eval mode
                nn_model.eval(); nn_model.actor_critic.eval()
                o_w = obs_dict
                for _ in range(warmup_steps):
                    v_w = o_w["observations"] if isinstance(o_w, dict) and "observations" in o_w else o_w.get(nn_obs_key, o_w["obs"])  # prefer nn_obs_key if present
                    a_w = nn_model.get_action({nn_obs_key: v_w})
                    step_result_w = rl_task.step(a_w)
                    if isinstance(step_result_w, tuple) and len(step_result_w) == 5:
                        o_next, _, term_w, trunc_w, _ = step_result_w
                    else:
                        o_next = step_result_w[0] if isinstance(step_result_w, tuple) else step_result_w
                        term_w = None; trunc_w = None
                    # Per-env done handling: reset RNN hidden state for finished envs
                    if term_w is not None and trunc_w is not None:
                        done_ids = (term_w | trunc_w).nonzero(as_tuple=True)[0]
                        if done_ids.numel() > 0:
                            nn_model.reset(done_ids)
                    o_w = o_next
                # Do not reset env; proceed with current obs_dict so RNN state is primed.
            except (ValueError, TypeError):
                pass

        # Optional W&B init for inference logging
        wandb_run = None
        try:
            import os as _os
            use_wandb = _os.environ.get("WANDB_DISABLED", "true").lower() == "false"
        except RuntimeError:
            use_wandb = False
        if use_wandb:
            try:
                import wandb  # noqa: F401
                # Resolve project/entity with override order: CLI > ENV > CFG > default
                project = (
                    args.wandb_project or
                    _os.environ.get('WANDB_PROJECT', '') or
                    cfg.wandb_project or
                    'gate_eval_runs'
                )
                entity = (
                    args.wandb_entity or
                    _os.environ.get('WANDB_ENTITY', '') or
                    cfg.wandb_user
                ) or None
                # Determine run name: CLI flag > env var > default pattern
                cli_run_name = args.run_name or ''
                env_run_name = _os.environ.get('WANDB_RUN_NAME', '')
                fallback_name = f"eval_{getattr(cfg, 'experiment', cfg.algo)}"
                run_name = cli_run_name if len(cli_run_name) > 0 else (env_run_name if len(env_run_name) > 0 else fallback_name)
                mode = _os.environ.get("WANDB_MODE", "online")
                # Optional custom local directory for run files
                dir_path = args.wandb_dir or _os.environ.get('WANDB_DIR', '')
                if dir_path:
                    wandb_run = wandb.init(project=project, entity=entity, name=run_name, mode=mode, dir=dir_path)
                else:
                    wandb_run = wandb.init(project=project, entity=entity, name=run_name, mode=mode)
                # Define common step metric and map groups to frames
                try:
                    # Use episodes as primary x-axis for cross-run alignment; also expose global_step/frames
                    wandb.define_metric('episodes')
                    wandb.define_metric('episodes_batch')  # per-batch reset index (1 tick per group of resetting envs)
                    wandb.define_metric('global_step')
                    wandb.define_metric('frames')
                    for name in (
                        'curriculum/level','curriculum/progress',
                        'episode_extra_stats/path_efficiency','episode_extra_stats/time_to_gate_steps',
                        'episode_extra_stats/min_gate_distance','episode_extra_stats/center_offset_success',
                        'episode_extra_stats/height_offset_success','episode_extra_stats/gate_pass_rate','gate_pass_rate_running_mean',
                        'episode_extra_stats/target_success_rate',
                        'episode_extra_stats/successes','episode_extra_stats/crashes','episode_extra_stats/timeouts',
                        'path_efficiency_running_mean','time_to_gate_running_mean','min_gate_distance_running_mean',
                        'center_offset_running_mean','height_offset_running_mean','target_success_running_mean',
                        # new: episode counters and curriculum totals/rates
                        'episode_extra_stats/episodes_total','episode_extra_stats/episodes_crossed',
                        'curriculum/total_successes','curriculum/total_crashes','curriculum/total_timeouts',
                        'curriculum/total_resets','curriculum/success_rate','curriculum/crash_rate','curriculum/timeout_rate',
                        # per-episode binary success (0/1) logged on each reset (separate series)
                        'episodes/success_binary',
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
                        wandb.define_metric(name, step_metric='episodes')
                    # Visibility/FOV running means (inference-only metrics)
                    for name in (
                        'visibility/abs_running_mean','visibility/frustum_running_mean','visibility/eff_running_mean',
                        'fov/score_running_mean','fov/visible_rate_running_mean',
                        'fov/horiz_angle_rad_running_mean','fov/vert_angle_rad_running_mean'
                    ):
                        wandb.define_metric(name, step_metric='episodes')
                    # Per-batch success fraction (averaged across envs that reset this tick)
                    wandb.define_metric('episodes/success_rate_batch', step_metric='episodes_batch')
                except RuntimeError:
                    pass
            except RuntimeError:
                wandb_run = None

        # Global step/episode counters for logging
        _frames = 0
        _episodes_done = 0
        try:
            _max_episodes = int(cfg.max_num_episodes)
        except (ValueError, TypeError):
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
            'target_success_sum': 0.0,
            'target_success_count': 0,
            'gate_pass_rate_sum': 0.0,
            'gate_pass_rate_count': 0,
            'episodes_total': 0,
            'episodes_crossed': 0,
        }
        _totals = {'successes': 0, 'crashes': 0, 'timeouts': 0}

        # Per-env episode accumulators for visibility/FOV (reset per episode)
        _vis_ep = {
            'abs_sum_env': torch.zeros(rl_task.num_envs, dtype=torch.float32),
            'frustum_sum_env': torch.zeros(rl_task.num_envs, dtype=torch.float32),
            'eff_sum_env': torch.zeros(rl_task.num_envs, dtype=torch.float32),
            'fov_score_sum_env': torch.zeros(rl_task.num_envs, dtype=torch.float32),
            'fov_visible_sum_env': torch.zeros(rl_task.num_envs, dtype=torch.float32),
            'fov_horiz_sum_env': torch.zeros(rl_task.num_envs, dtype=torch.float32),
            'fov_vert_sum_env': torch.zeros(rl_task.num_envs, dtype=torch.float32),
            'abs_count_env': torch.zeros(rl_task.num_envs, dtype=torch.int32),
            'frustum_count_env': torch.zeros(rl_task.num_envs, dtype=torch.int32),
            'eff_count_env': torch.zeros(rl_task.num_envs, dtype=torch.int32),
            'fov_score_count_env': torch.zeros(rl_task.num_envs, dtype=torch.int32),
            'fov_visible_count_env': torch.zeros(rl_task.num_envs, dtype=torch.int32),
            'fov_horiz_count_env': torch.zeros(rl_task.num_envs, dtype=torch.int32),
            'fov_vert_count_env': torch.zeros(rl_task.num_envs, dtype=torch.int32),
        }
        # Running aggregates across all completed episodes (for W&B running means)
        _vis_running = {
            'abs_sum': 0.0, 'abs_count': 0,
            'frustum_sum': 0.0, 'frustum_count': 0,
            'eff_sum': 0.0, 'eff_count': 0,
            'fov_score_sum': 0.0, 'fov_score_count': 0,
            'fov_visible_sum': 0.0, 'fov_visible_count': 0,
            'fov_horiz_sum': 0.0, 'fov_horiz_count': 0,
            'fov_vert_sum': 0.0, 'fov_vert_count': 0,
        }

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
        save_gifs = bool(cfg.save_gifs)
        # Save GIFs alongside the inference script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gif_dir = os.path.join(script_dir, 'gif_episodes')
        if save_gifs:
            try:
                os.makedirs(gif_dir, exist_ok=True)
            except Exception:
                save_gifs = False
        _gif_drone_noised_frames = []
        _gif_drone_clean_frames = []
        _gif_drone_seg_frames = []
        _gif_static_noised_frames = []
        _gif_static_clean_frames = []
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

        def _to_pil_gray(image_data) -> None:
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
            except ImportError:
                return None

        def _save_gif(frames, filename) -> None:
            if not frames or len(frames) < 1:
                return
            frames[0].save(
                os.path.join(gif_dir, filename),
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
            )

        # Rolling windows / CI helpers
        _success_flags = []  # append 1 for success, 0 otherwise
        _time_to_gate_hist = []  # for successes
        _latency_ms_hist = []
        _crash_bins = {'<=5': 0, '6-20': 0, '>20': 0}
        _reset_hist = {'success': 0, 'crash': 0, 'timeout': 0}
        _per_level = {}  # level -> {'succ': int, 'tot': int, 'ttg_sum': float}

        def _wilson_ci(s, n, z=1.96) -> None:
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
        nn_obs_key = getattr(nn_model.cfg, "obs_key", "obs")
        # One-time evaluation config summary (curriculum/randomization/effective flags)
        try:
            print("=" * 80)
            print("EVAL CONFIG SUMMARY (Gate Navigation)")
            # Curriculum level
            try:
                lvl = int(rl_task.curriculum_level)
            except (ValueError, TypeError):
                lvl = -1
            print(f"  Curriculum level: {lvl}")
            # Curriculum-derived camera noise and frame dropout at this level (if available)
            try:
                cur = rl_task.task_config.curriculum
                if cur is not None:
                    cstd, cdrop = cur.get_camera_noise(lvl)
                    print(f"  Camera noise: gaussian_std={float(cstd):.4f}, dropout={float(cdrop):.4f}")
                    fd = cur.get_camera_frame_dropout(lvl)
                    if isinstance(fd, dict):
                        dt = fd.get('drone_total', 0.0); st = fd.get('static_total', 0.0)
                        print(f"  Frame dropout: drone_total={float(dt):.4f}, static_total={float(st):.4f}")
            except (ValueError, TypeError):
                pass
            # Effective toggles from global tensor dict (authoritative during runtime)
            gtd = rl_task.sim_env
            g = gtd.global_tensor_dict if (gtd is not None and hasattr(gtd, 'global_tensor_dict')) else {}
            def _p(key, label=None) -> None:
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
            # Observation ablation spec
            import os as _os
            print(f"  ABLATE_OBS_RANGES: {_os.environ.get('ABLATE_OBS_RANGES', '')}")
            print("=" * 80)
        except (ValueError, TypeError):
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
                except (ValueError, TypeError):
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
                    except (ValueError, TypeError):
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
        step_counter_total = 0
        while True:
            vec = obs_dict["observations"] if isinstance(obs_dict, dict) and "observations" in obs_dict else obs_dict["obs"]
            if dump_obs_parity and step_counter_total < parity_dump_steps:
                if isinstance(vec, torch.Tensor) and vec.ndim == 2 and vec.shape[1] >= 150:
                    z_drone = vec[:, 22:86]
                    z_static = vec[:, 86:150]
                    print(f"[OBS_PARITY] step={step_counter_total} abs_mean: drone={float(z_drone.abs().mean().item()):.6f} static={float(z_static.abs().mean().item()):.6f}")
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
                                norm = prepare_and_normalize_obs(nn_model.actor_critic, {nn_obs_key: vec})
                                pvec = norm.get(nn_obs_key, None)
                                if isinstance(pvec, torch.Tensor) and pvec.ndim == 2 and pvec.shape[0] > 0 and pvec.shape[1] >= 150:
                                    ze0n = pvec[0, 22:86]
                                    zs0n = pvec[0, 86:150]
                                    ze_nabs = float(torch.mean(torch.abs(ze0n)).item())
                                    zs_nabs = float(torch.mean(torch.abs(zs0n)).item())
                                    print(f"[ENV0_LATENTS_NORM] step={_env0_step_in_ep} abs_mean: drone={ze_nabs:.6f} static={zs_nabs:.6f}")
                                _env0_step_in_ep += 1
                        elif _env0_log_state == 1:
                            print(f"[ENV0_LATENTS] step={_env0_step_in_ep} abs_mean: drone={ze_abs:.6f} static={zs_abs:.6f}")
                            # Also print normalized latents for the same step
                            norm = prepare_and_normalize_obs(nn_model.actor_critic, {nn_obs_key: vec})
                            pvec = norm.get(nn_obs_key, None)
                            if isinstance(pvec, torch.Tensor) and pvec.ndim == 2 and pvec.shape[0] > 0 and pvec.shape[1] >= 150:
                                ze0n = pvec[0, 22:86]
                                zs0n = pvec[0, 86:150]
                                ze_nabs = float(torch.mean(torch.abs(ze0n)).item())
                                zs_nabs = float(torch.mean(torch.abs(zs0n)).item())
                                print(f"[ENV0_LATENTS_NORM] step={_env0_step_in_ep} abs_mean: drone={ze_nabs:.6f} static={zs_nabs:.6f}")
                            _env0_step_in_ep += 1
                except (ValueError, TypeError):
                    pass

            # Optional: print env0 NORMALIZED latent stats every frame for one episode only
            if _env0_norm_log_state != 2:
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
            # Sanitize obs before passing to model to avoid NaN/Inf in normalization/encoder
            if isinstance(vec, torch.Tensor):
                vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            model_obs = {nn_obs_key: vec}
            actions = nn_model.get_action(model_obs)
            step_result = rl_task.step(actions)
            step_counter_total += 1
            # Unpack step result (env returns 5-tuple)
            if isinstance(step_result, tuple) and len(step_result) == 5:
                obs_next, rew, terminated, truncated, infos = step_result
            else:
                # Fallback to previous behavior
                obs_next = step_result[0] if isinstance(step_result, tuple) else step_result
                rew = None; terminated = None; truncated = None; infos = {}

            _frames += 1

            # Accumulate returns and lengths
            if isinstance(rew, torch.Tensor):
                _ep_return += rew.float().cpu()
            _ep_length += 1

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
                _ep_action_diff_sum += adiff
                _ep_action_diff_count += 1
            except (ValueError, TypeError):
                abs_mean = None; diff_mean = None; sat = None

            # Collect GIF frames (env 0 only, noised versions)
            if save_gifs:
                try:
                    d = rl_task.obs_dict
                    if isinstance(d, dict) and 'depth_range_pixels_noised' in d:
                        dd = d['depth_range_pixels_noised']
                        if isinstance(dd, torch.Tensor) and dd.ndim >= 3:
                            pil = _to_pil_gray(dd[0, 0])
                            if pil is not None:
                                _gif_drone_noised_frames.append(pil)
                        # Also capture clean (non-noised) drone depth if available
                        dc = d.get('depth_range_pixels', None)
                        pil_clean = None
                        if isinstance(dc, torch.Tensor):
                            if dc.ndim == 4:  # (N,C,H,W)
                                pil_clean = _to_pil_gray(dc[0, 0])
                            elif dc.ndim == 3:  # (N,H,W)
                                pil_clean = _to_pil_gray(dc[0])
                            elif dc.ndim == 2:  # (H,W)
                                pil_clean = _to_pil_gray(dc)
                        elif dc is not None:
                            pil_clean = _to_pil_gray(dc)
                        if pil_clean is not None:
                            _gif_drone_clean_frames.append(pil_clean)
                    # Also collect drone segmentation if available
                    ds = d.get('segmentation_pixels', None)
                    if isinstance(ds, torch.Tensor):
                        if ds.ndim == 4:
                            ds0 = ds[0, 0]
                        elif ds.ndim == 3:
                            ds0 = ds[0]
                        else:
                            ds0 = ds
                        import numpy as _np
                        seg = ds0.detach().cpu().numpy()
                        # Normalize to [0,1] for visualization
                        seg_min = _np.min(seg)
                        seg_max = _np.max(seg)
                        if (seg_max - seg_min) > 0:
                            seg_norm = (seg - seg_min) / float(seg_max - seg_min)
                        else:
                            seg_norm = _np.zeros_like(seg, dtype=_np.float32)
                        pil_dseg = _to_pil_gray(seg_norm)
                        if pil_dseg is not None:
                            _gif_drone_seg_frames.append(pil_dseg)
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
                        # Also capture clean (non-noised) static depth if available
                        sc = d.get('static_depth', None)
                        pil_sclean = None
                        if isinstance(sc, torch.Tensor):
                            if sc.ndim == 3:
                                pil_sclean = _to_pil_gray(sc[0])
                            elif sc.ndim == 2:
                                pil_sclean = _to_pil_gray(sc)
                        elif sc is not None:
                            pil_sclean = _to_pil_gray(sc)
                        if pil_sclean is not None:
                            _gif_static_clean_frames.append(pil_sclean)
                    # Also collect static segmentation directly from StaticCameraManager (env 0)
                    try:
                        scm = rl_task.static_camera_manager
                        if scm is not None and hasattr(scm, 'capture_images'):
                            _d, _seg = scm.capture_images(batched=False)
                            # Save clean static depth from camera manager if available
                            import numpy as _np
                            if _d is not None:
                                if isinstance(_d, _np.ndarray):
                                    if _d.ndim == 3:  # (N,H,W)
                                        pil_d = _to_pil_gray(_d[0])
                                    else:  # (H,W)
                                        pil_d = _to_pil_gray(_d)
                                else:
                                    pil_d = _to_pil_gray(_d)
                                if pil_d is not None:
                                    _gif_static_clean_frames.append(pil_d)
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
                    except (ValueError, TypeError):
                        pass
                except (ValueError, TypeError):
                    pass

            # VAE latent norms (periodic)
            vae_drone_norm = None; vae_static_norm = None; vae_ratio = None
            if (_frames % 100) == 0:
                z_e = vec[:, 22:86]
                z_s = vec[:, 86:150]
                vae_drone_norm = float(torch.linalg.norm(z_e, dim=1).mean().item())
                vae_static_norm = float(torch.linalg.norm(z_s, dim=1).mean().item())
                vae_ratio = float(vae_static_norm / (vae_drone_norm + 1e-6))

            # Per-reset logging similar to training
            try:
                import torch as _torch
                # Update per-env episode accumulators for visibility/FOV metrics from infos
                if isinstance(infos, dict):
                    vabs = infos.get('static_visibility/abs', None)
                    vfru = infos.get('static_visibility/frustum', None)
                    veff = infos.get('static_visibility/eff', None)
                    fvis = infos.get('static_fov/visible', None)
                    fscore = infos.get('static_fov/score', None)
                    fh = infos.get('static_fov/horiz_angle_rad', None)
                    fv = infos.get('static_fov/vert_angle_rad', None)
                    if isinstance(vabs, _torch.Tensor):
                        _vis_ep['abs_sum_env'] += vabs.detach().float().cpu()
                        _vis_ep['abs_count_env'] += 1
                    if isinstance(vfru, _torch.Tensor):
                        _vis_ep['frustum_sum_env'] += vfru.detach().float().cpu()
                        _vis_ep['frustum_count_env'] += 1
                    if isinstance(veff, _torch.Tensor):
                        _vis_ep['eff_sum_env'] += veff.detach().float().cpu()
                        _vis_ep['eff_count_env'] += 1
                    if isinstance(fvis, _torch.Tensor):
                        _vis_ep['fov_visible_sum_env'] += fvis.detach().float().cpu()
                        _vis_ep['fov_visible_count_env'] += 1
                    if isinstance(fscore, _torch.Tensor):
                        _vis_ep['fov_score_sum_env'] += fscore.detach().float().cpu()
                        _vis_ep['fov_score_count_env'] += 1
                    if isinstance(fh, _torch.Tensor):
                        _vis_ep['fov_horiz_sum_env'] += fh.detach().float().cpu()
                        _vis_ep['fov_horiz_count_env'] += 1
                    if isinstance(fv, _torch.Tensor):
                        _vis_ep['fov_vert_sum_env'] += fv.detach().float().cpu()
                        _vis_ep['fov_vert_count_env'] += 1
                if (terminated is not None and truncated is not None and
                        (_torch.any(terminated) or _torch.any(truncated))):
                    if wandb_run is not None:
                        payload = {}
                        payload['frames'] = _frames
                        # Curriculum level/progress (log only under episode namespace)
                        lvl = float(rl_task.curriculum_level)
                        payload['episode_extra_stats/curriculum/level'] = lvl
                        prog = float(rl_task.curriculum_progress_fraction)
                        payload['episode_extra_stats/curriculum/progress'] = prog
                        # Episode-level metrics provided by env
                        if isinstance(infos, dict):
                            extra = infos.get('episode_extra_stats', None)
                            if isinstance(extra, dict):
                                for k, v in list(extra.items()):
                                    try:
                                        payload[k] = float(v)
                                    except (ValueError, TypeError):
                                        import torch as __t
                                        if isinstance(v, __t.Tensor):
                                            payload[k] = float(v.item()) if v.numel() == 1 else None
                                # Update running aggregates from episode_extra_stats
                                def _get_float(name) -> None:
                                    val = extra.get(name, None)
                                    try:
                                        return float(val)
                                    except (ValueError, TypeError):
                                        try:
                                            import torch as __tt
                                            if isinstance(val, __tt.Tensor):
                                                return float(val.item()) if val.numel() == 1 else None
                                        except (ValueError, TypeError):
                                            return None
                                        return None
                                pe = _get_float('path_efficiency')
                                ttg = _get_float('time_to_gate_steps')
                                mgd = _get_float('min_gate_distance')
                                co  = _get_float('center_offset_success')
                                ho  = _get_float('height_offset_success')
                                tsr = _get_float('target_success_rate')
                                gpr = _get_float('gate_pass_rate')
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
                                if tsr is not None:
                                    _traj_running['target_success_sum'] += tsr
                                    _traj_running['target_success_count'] += 1
                                if gpr is not None:
                                    _traj_running['gate_pass_rate_sum'] += gpr
                                    _traj_running['gate_pass_rate_count'] += 1
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
                                if isinstance(val, _torch.Tensor):
                                    c = float(val.sum().item())
                                    payload[f'episode_extra_stats/{key}'] = c
                                    _totals[key] += int(c)
                                elif isinstance(val, (int, float)):
                                    c = float(val)
                                    payload[f'episode_extra_stats/{key}'] = c
                                    _totals[key] += int(c)
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
                                # Reset accumulators and RNN state for those envs (parity with training masking)
                                _ep_return[ids_cpu] = 0.0
                                _ep_length[ids_cpu] = 0
                                nn_model.reset(ids)
                                # Episode action diff means per dimension (average over steps, then average over resetting envs)
                                counts = _ep_action_diff_count[ids].float().clamp_min(1.0)
                                per_env_means = _ep_action_diff_sum[ids] / counts.unsqueeze(1)
                                diff_ep_mean = per_env_means.mean(dim=0)
                                dims = ['x','y','z','yaw']
                                for i, d in enumerate(dims[:diff_ep_mean.numel()]):
                                    payload[f'episode_action_diff_mean/{d}'] = float(diff_ep_mean[i].item())
                                # Clear episode accumulators for these envs
                                _ep_action_diff_sum[ids] = 0.0
                                _ep_action_diff_count[ids] = 0
                                # Identify success/crash via infos tensors when available
                                succ_mask = None
                                try:
                                    s = infos.get('successes', None)
                                    if isinstance(s, _torch.Tensor):
                                        succ_mask = s[ids].bool().cpu().numpy()
                                except (KeyError, TypeError):
                                    succ_mask = None
                                # Per-batch success fraction across resetting envs this tick
                                if wandb_run is not None:
                                    if succ_mask is not None:
                                        batch_succ = float(succ_mask.sum() / max(1, succ_mask.size))
                                    else:
                                        # fallback: derive successes from episode_extra_stats deltas if needed
                                        batch_succ = float('nan')
                                    # Maintain an independent batch counter (episodes_batch)
                                    _episodes_batch = _episode_counter_total
                                    wandb.log({'episodes_batch': float(_episodes_batch),
                                               'episodes/success_rate_batch': batch_succ,
                                               'frames': _frames,
                                               'global_step': _frames}, step=_frames)
                                # Per-reset binary success logging (0/1) under 'episodes/success_binary'
                                try:
                                    if wandb_run is not None and ids_cpu.numel() > 0:
                                        base_ep = int(_episode_counter_total)
                                        for j, _env_id in enumerate(ids_cpu.tolist()):
                                            succ_bit = float(1.0 if (succ_mask is not None and bool(succ_mask[j])) else 0.0)
                                            single = {
                                                'episodes': float(base_ep + j + 1),
                                                'episodes/success_binary': succ_bit,
                                                'frames': _frames,
                                                'global_step': _frames,
                                            }
                                            import wandb  # noqa: F401
                                            wandb.log(single, step=_frames)
                                except RuntimeError:
                                    pass
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
                                c = infos.get('crashes', None)
                                if isinstance(c, _torch.Tensor):
                                    crashed = c[ids].bool().cpu().numpy()
                                    early = (length <= 5)
                                    denom = max(1, crashed.size)
                                    payload['early_crash_rate'] = float((crashed & early).sum() / denom)
                        except RuntimeError:
                            pass
                        # Running means (mirroring training names)
                        def _safe_mean(sum_key, count_key, none_if_zero=False) -> None:
                            s = _traj_running.get(sum_key, 0.0)
                            c = _traj_running.get(count_key, 0)
                            if c <= 0:
                                return None if none_if_zero else float('nan')
                            return float(s / c)
                        payload['path_efficiency_running_mean'] = _safe_mean('path_efficiency_sum','path_efficiency_count')
                        payload['min_gate_distance_running_mean'] = _safe_mean('min_gate_distance_sum','min_gate_distance_count')
                        payload['time_to_gate_running_mean'] = _safe_mean('time_to_gate_sum','time_to_gate_count', none_if_zero=True)
                        payload['center_offset_running_mean'] = _safe_mean('center_offset_sum','center_offset_count', none_if_zero=True)
                        payload['height_offset_running_mean'] = _safe_mean('height_offset_sum','height_offset_count', none_if_zero=True)
                        payload['target_success_running_mean'] = _safe_mean('target_success_sum','target_success_count')
                        payload['gate_pass_rate_running_mean'] = _safe_mean('gate_pass_rate_sum','gate_pass_rate_count')
                        if _traj_running['episodes_total'] > 0:
                            payload['episode_extra_stats/gate_pass_rate'] = float(_traj_running['episodes_crossed']) / float(_traj_running['episodes_total'])
                        # Derive cumulative totals/rates (aligned with training naming)
                        total_resets = _totals['successes'] + _totals['crashes'] + _totals['timeouts']
                        payload['episode_extra_stats/curriculum/total_successes'] = float(_totals['successes'])
                        payload['episode_extra_stats/curriculum/total_crashes'] = float(_totals['crashes'])
                        payload['episode_extra_stats/curriculum/total_timeouts'] = float(_totals['timeouts'])
                        payload['episode_extra_stats/curriculum/total_resets'] = float(total_resets)
                        # Provide a generic 'episodes' counter for step alignment across runs
                        payload['episodes'] = float(total_resets)
                        if total_resets > 0:
                            payload['episode_extra_stats/curriculum/success_rate'] = float(_totals['successes']) / float(total_resets)
                            payload['episode_extra_stats/curriculum/crash_rate'] = float(_totals['crashes']) / float(total_resets)
                            payload['episode_extra_stats/curriculum/timeout_rate'] = float(_totals['timeouts']) / float(total_resets)
                        # Success EMA
                        if _success_ema is None and total_resets > 0:
                            _success_ema = payload.get('episode_extra_stats/curriculum/success_rate', None)
                        elif total_resets > 0 and _success_ema is not None:
                            _success_ema = (1 - _ema_alpha) * _success_ema + _ema_alpha * payload.get('episode_extra_stats/curriculum/success_rate', 0.0)
                        if _success_ema is not None:
                            payload['success_rate_running'] = float(_success_ema)
                        # Spatial quantiles
                        if len(_hist_center_offset) > 0:
                            payload['spatial/center_offset_p50'] = float(np.percentile(_hist_center_offset, 50))
                            payload['spatial/center_offset_p90'] = float(np.percentile(_hist_center_offset, 90))
                        if len(_hist_height_offset) > 0:
                            payload['spatial/height_offset_p50'] = float(np.percentile(_hist_height_offset, 50))
                            payload['spatial/height_offset_p90'] = float(np.percentile(_hist_height_offset, 90))
                        if len(_hist_min_gate_distance) > 0:
                            payload['spatial/min_gate_distance_p50'] = float(np.percentile(_hist_min_gate_distance, 50))
                            payload['spatial/min_gate_distance_p90'] = float(np.percentile(_hist_min_gate_distance, 90))
                        # Action stats (latest step)
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
                        # VAE norms (periodic)
                        if vae_drone_norm is not None:
                            payload['vae/drone_norm_mean'] = vae_drone_norm
                            payload['vae/static_norm_mean'] = vae_static_norm if vae_static_norm is not None else 0.0
                            payload['vae/static_to_drone_norm_ratio'] = vae_ratio if vae_ratio is not None else 0.0
                        # Throughput
                        dt = max(1e-6, time.time() - _t0)
                        fps = _frames / dt
                        epm = (_episode_counter_total / dt) * 60.0
                        payload['throughput/fps_env'] = float(fps)
                        payload['throughput/episodes_per_min'] = float(epm)
                        # Keep auxiliary step aliases in payload as well
                        payload['global_step'] = _frames
                        # Log
                        import wandb  # noqa: F401
                        payload['global_step'] = _frames
                        wandb.log(payload, step=_frames)
                    # Update episode counter by number of resets this step
                    try:
                        ids = (terminated | truncated).nonzero(as_tuple=True)[0]
                        # Compute per-episode means for visibility/FOV for envs that reset now
                        if ids.numel() > 0:
                            ids_cpu = ids.cpu()
                            def _update_running(sum_key_env, count_key_env, sum_key_run, count_key_run) -> None:
                                s = _vis_ep[sum_key_env][ids_cpu]
                                c = _vis_ep[count_key_env][ids_cpu].float().clamp_min(1.0)
                                means = (s / c).numpy()
                                _vis_running[sum_key_run] += float(means.sum())
                                _vis_running[count_key_run] += int(means.size)
                                # reset per-env episode accumulators
                                _vis_ep[sum_key_env][ids_cpu] = 0.0
                                _vis_ep[count_key_env][ids_cpu] = 0
                            _update_running('abs_sum_env','abs_count_env','abs_sum','abs_count')
                            _update_running('frustum_sum_env','frustum_count_env','frustum_sum','frustum_count')
                            _update_running('eff_sum_env','eff_count_env','eff_sum','eff_count')
                            _update_running('fov_score_sum_env','fov_score_count_env','fov_score_sum','fov_score_count')
                            _update_running('fov_visible_sum_env','fov_visible_count_env','fov_visible_sum','fov_visible_count')
                            _update_running('fov_horiz_sum_env','fov_horiz_count_env','fov_horiz_sum','fov_horiz_count')
                            _update_running('fov_vert_sum_env','fov_vert_count_env','fov_vert_sum','fov_vert_count')
                            # Compose running means into payload and log
                            if wandb_run is not None:
                                vis_payload = {
                                    'frames': _frames,
                                    'global_step': _frames,
                                    'episodes': float(total_resets),
                                }
                                def _safe_run_mean(s_key, c_key) -> None:
                                    c = max(1, _vis_running[c_key])
                                    return float(_vis_running[s_key] / c)
                                vis_payload['visibility/abs_running_mean'] = _safe_run_mean('abs_sum','abs_count')
                                vis_payload['visibility/frustum_running_mean'] = _safe_run_mean('frustum_sum','frustum_count')
                                vis_payload['visibility/eff_running_mean'] = _safe_run_mean('eff_sum','eff_count')
                                vis_payload['fov/score_running_mean'] = _safe_run_mean('fov_score_sum','fov_score_count')
                                vis_payload['fov/visible_rate_running_mean'] = _safe_run_mean('fov_visible_sum','fov_visible_count')
                                vis_payload['fov/horiz_angle_rad_running_mean'] = _safe_run_mean('fov_horiz_sum','fov_horiz_count')
                                vis_payload['fov/vert_angle_rad_running_mean'] = _safe_run_mean('fov_vert_sum','fov_vert_count')
                                import wandb  # noqa: F401
                                wandb.log(vis_payload, step=_frames)
                        _episodes_done += int(ids.numel())
                        _episode_counter_total += int(ids.numel())
                        # Save GIFs for env 0 occasionally
                        if save_gifs and (ids.numel() > 0) and (0 in ids.cpu().tolist()):
                            _gif_episode_counter_env0 += 1
                            if (_gif_episode_counter_env0 % _gif_every_n) == 0:
                                _save_gif(_gif_drone_noised_frames, f"episode_{_gif_episode_counter_env0:04d}_drone_depth_D455_NOISED.gif")
                                _save_gif(_gif_drone_clean_frames, f"episode_{_gif_episode_counter_env0:04d}_drone_depth_D455_CLEAN.gif")
                                _save_gif(_gif_drone_seg_frames, f"episode_{_gif_episode_counter_env0:04d}_drone_seg.gif")
                                _save_gif(_gif_static_noised_frames, f"episode_{_gif_episode_counter_env0:04d}_static_depth_D455_NOISED.gif")
                                _save_gif(_gif_static_clean_frames, f"episode_{_gif_episode_counter_env0:04d}_static_depth_D455_CLEAN.gif")
                                _save_gif(_gif_static_seg_frames, f"episode_{_gif_episode_counter_env0:04d}_static_seg.gif")
                                # _save_gif(_gif_drone_recon_frames, f"episode_{_gif_episode_counter_env0:04d}_drone_depth_VAE_RECON.gif")  # disabled per request
                                # _save_gif(_gif_static_recon_frames, f"episode_{_gif_episode_counter_env0:04d}_static_depth_VAE_RECON.gif")  # disabled per request
                            # Clear buffers for env 0 episode
                            _gif_drone_noised_frames = []
                            _gif_drone_clean_frames = []
                            _gif_drone_seg_frames = []
                            _gif_static_noised_frames = []
                            _gif_static_clean_frames = []
                            _gif_static_seg_frames = []
                            _gif_drone_recon_frames = []
                            _gif_static_recon_frames = []
                    except RuntimeError:
                        _episodes_done += 1
            except RuntimeError:
                pass

            # Termination condition by episodes if configured
            if _max_episodes > 0 and _episodes_done >= _max_episodes:
                break

            # Continue loop with next obs dict
            obs_dict = obs_next


if __name__ == "__main__":
    main()


