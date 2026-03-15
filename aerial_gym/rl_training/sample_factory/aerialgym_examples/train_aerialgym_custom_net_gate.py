from __future__ import annotations

import sys
import os
from typing import Optional

import isaacgym  # noqa: F401 — must import before PyTorch
import gymnasium as gym
import torch
import numpy as np

from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.utils.typing import Config, Env
from sample_factory.utils.utils import str2bool

from aerial_gym.registry.task_registry import task_registry
from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_common import (
    clear_sf_cache,
    setup_env_agents,
)
from aerial_gym.rl_training.sample_factory.aerialgym_examples.env_wrapper_gate import (
    AerialGymVecEnvGate,
)
from aerial_gym.rl_training.sample_factory.aerialgym_examples.gate_env_configs import env_configs
from aerial_gym.rl_training.sample_factory.aerialgym_examples.dual_fusion_encoder import (
    DualFusionEncoder,
    make_dual_fusion_encoder,
)
from aerial_gym.rl_training.sample_factory.aerialgym_examples.task_registration import (
    register_gate_task_in_subprocess,
    register_main_process_tasks,
)
from aerial_gym.rl_training.sample_factory.aerialgym_examples.cfg_env_bridge import (
    bridge_cfg_to_env_vars,
)
from aerial_gym.rl_training.sample_factory.aerialgym_examples.influence_tracking import (
    run_with_influence_tracking,
)

# Deterministic mode disabled — incompatible with Isaac Gym CuBLAS operations
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def make_aerialgym_env(
    full_task_name: str,
    cfg: Config,
    _env_config: object = None,
    render_mode: Optional[str] = None,
) -> Env:
    register_gate_task_in_subprocess(full_task_name, cfg)

    seed_val: int | None = cfg.seed
    env = AerialGymVecEnvGate(
        task_registry.make_task(task_name=full_task_name, seed=seed_val),
        "obs",
        save_gifs=cfg.save_gifs,
    )

    forced_action_space = _build_action_space(full_task_name)
    env.action_space = convert_space(forced_action_space)

    return env


def _build_action_space(full_task_name: str) -> gym.spaces.Box:
    if full_task_name == "quad_with_obstacles_gate":
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)


def add_extra_params_func(parser: object) -> None:
    """Register CLI arguments for gate navigation ablation and feature flags."""
    p = parser
    p.add_argument(
        "--env_agents", default=None, type=int, help="Num agents in env (multi-agent only)"
    )
    p.add_argument(
        "--headless",
        type=lambda x: x.lower() == "true",
        default=None,
        help="Force headless mode (True/False)",
    )
    p.add_argument(
        "--save_gifs",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Save episode GIFs for both cameras (True/False)",
    )

    _add_gate_ablation_params(p)
    _add_camera_noise_params(p)
    _add_camera_position_params(p)
    _add_spawn_curriculum_params(p)
    _add_fusion_params(p)
    _add_influence_params(p)
    _add_sf_standard_params(p)


def _add_gate_ablation_params(p: object) -> None:
    _bool_arg = lambda x: x.lower() == "true"
    p.add_argument(
        "--disable_gate_size_randomization",
        type=_bool_arg,
        default=False,
        help="Disable gate size randomization and use a fixed scale percent",
    )
    p.add_argument(
        "--fixed_gate_scale_percent",
        type=int,
        default=100,
        help="Fixed gate scale percent when randomization is disabled (40..100, step 2)",
    )
    p.add_argument(
        "--disable_obstacle_randomization",
        type=_bool_arg,
        default=False,
        help="Disable obstacle randomization behind the gate",
    )
    p.add_argument(
        "--fixed_obstacles_behind_gate",
        type=int,
        default=0,
        help="Fixed number of obstacles behind the gate when randomization is disabled",
    )


def _add_camera_noise_params(p: object) -> None:
    _bool_arg = lambda x: x.lower() == "true"
    p.add_argument(
        "--disable_static_camera_orientation_randomization",
        type=_bool_arg,
        default=False,
        help="Disable static camera orientation randomization, fix angle to 0.0",
    )
    p.add_argument(
        "--disable_camera_noise_randomization",
        type=_bool_arg,
        default=False,
        help="Disable camera noise randomization for both drone and static",
    )
    p.add_argument(
        "--disable_camera_frame_dropout_randomization",
        type=_bool_arg,
        default=False,
        help="Disable entire-frame dropout randomization for both cameras",
    )
    p.add_argument(
        "--disable_drone_camera_noise_randomization",
        type=_bool_arg,
        default=None,
        help="Disable noise randomization for DRONE camera only",
    )
    p.add_argument(
        "--disable_static_camera_noise_randomization",
        type=_bool_arg,
        default=None,
        help="Disable noise randomization for STATIC camera only",
    )
    p.add_argument(
        "--disable_drone_camera_frame_dropout",
        type=_bool_arg,
        default=None,
        help="Disable frame dropout for DRONE camera only",
    )
    p.add_argument(
        "--disable_static_camera_frame_dropout",
        type=_bool_arg,
        default=None,
        help="Disable frame dropout for STATIC camera only",
    )
    p.add_argument(
        "--disable_state_noise_randomization",
        type=_bool_arg,
        default=False,
        help="Disable pose state noise randomization for drone and static camera",
    )


def _add_camera_position_params(p: object) -> None:
    _bool_arg = lambda x: x.lower() == "true"
    p.add_argument(
        "--enable_static_camera_yaw_sweep",
        type=_bool_arg,
        default=False,
        help="Enable constant yaw oscillation for static camera",
    )
    p.add_argument(
        "--enable_static_camera_locked",
        type=_bool_arg,
        default=False,
        help="Lock static camera position and rotate to center the drone",
    )
    p.add_argument(
        "--static_camera_yaw_sweep_speed_deg",
        type=float,
        default=10.0,
        help="Yaw sweep speed in deg/s",
    )
    p.add_argument(
        "--static_camera_base_y",
        type=float,
        default=None,
        help="Override static camera base Y (meters; negative is behind gate)",
    )

    def _parse_base_z(val: str) -> str | float:
        v = str(val).strip().lower()
        if v == "adaptive":
            return "adaptive"
        try:
            return float(val)
        except (ValueError, TypeError):
            raise ValueError("--static_camera_base_z must be a float or 'adaptive'")

    p.add_argument(
        "--static_camera_base_z",
        type=_parse_base_z,
        default=None,
        help="Static cam Z (meters) or 'adaptive' to follow gate center height",
    )
    p.add_argument(
        "--disable_dynamic_camera_following",
        type=_bool_arg,
        default=False,
        help="Disable dynamic camera following mode",
    )
    p.add_argument(
        "--enable_dynamic_camera_following",
        type=_bool_arg,
        default=None,
        help="Enable dynamic camera following mode (overrides config)",
    )
    p.add_argument(
        "--enable_static_camera_arc_follow",
        type=_bool_arg,
        default=False,
        help="Enable arc-follow static camera mode",
    )
    p.add_argument(
        "--static_camera_arc_radius_m",
        type=float,
        default=2.0,
        help="Arc-follow radius in meters",
    )
    p.add_argument(
        "--dynamic_camera_follow_y_offset_m",
        type=float,
        default=None,
        help="Override dynamic camera follow Y-offset in meters",
    )
    p.add_argument(
        "--disable_dynamic_follow_gate_blending",
        type=_bool_arg,
        default=False,
        help="Disable blending toward gate in dynamic-follow; always look at drone",
    )


def _add_spawn_curriculum_params(p: object) -> None:
    _bool_arg = lambda x: x.lower() == "true"
    p.add_argument(
        "--disable_spawn_position_randomization",
        type=_bool_arg,
        default=False,
        help="Disable robot spawn POSITION randomization",
    )
    p.add_argument(
        "--disable_spawn_orientation_randomization",
        type=_bool_arg,
        default=False,
        help="Disable robot spawn ORIENTATION randomization",
    )
    p.add_argument(
        "--disable_curriculum_multiplier",
        type=_bool_arg,
        default=False,
        help="Disable curriculum reward multiplier (sets multiplier to 1.0)",
    )
    p.add_argument(
        "--force_curriculum_level",
        type=str,
        default=None,
        help="Force a specific curriculum level (disables auto progression). 'none' to disable.",
    )
    p.add_argument(
        "--max_curriculum_level",
        type=int,
        default=None,
        help="Maximum curriculum level cap for progression",
    )
    p.add_argument(
        "--min_curriculum_level",
        type=int,
        default=None,
        help="Minimum curriculum level to start from during TRAINING",
    )


def _add_fusion_params(p: object) -> None:
    p.add_argument(
        "--fusion",
        type=str,
        default="gated",
        choices=["concat", "gated"],
        help="Fusion strategy: concat (early concat) or gated (dual gated late fusion)",
    )
    p.add_argument(
        "--gate_per_feature",
        type=int,
        default=1,
        help="Use per-feature gate (1) or scalar gate (0)",
    )


def _add_influence_params(p: object) -> None:
    _inf_env = os.getenv("SF_ENABLE_INFLUENCE_TRACKER")
    _inf_default = str(_inf_env).lower() == "true" if _inf_env is not None else False
    _grad_env = os.getenv("SF_ENABLE_GRAD_ATTR")
    _grad_default = str(_grad_env).lower() == "true" if _grad_env is not None else True

    p.add_argument(
        "--enable_gradient_monitoring",
        type=lambda x: x.lower() == "true",
        default=_inf_default,
        help="Enable complete observation influence tracking",
    )
    p.add_argument("--gradient_log_interval", default=100, type=int)
    p.add_argument("--gradient_print_interval", default=100, type=int)
    p.add_argument(
        "--enable_grad_attribution",
        type=lambda x: x.lower() == "true",
        default=_grad_default,
        help="Enable gradient-based attribution alongside correlation analysis",
    )


def _add_sf_standard_params(p: object) -> None:
    p.add_argument(
        "--obs_key",
        default="obs",
        type=str,
        help="Observation dict key to use ('obs' or 'states').",
    )
    p.add_argument(
        "--subtask",
        default=None,
        type=str,
        help="Subtask for envs that support it.",
    )
    p.add_argument(
        "--ige_api_version",
        default="preview4",
        type=str,
        choices=["preview3", "preview4"],
        help="IsaacGymEnvs API version.",
    )
    p.add_argument("--eval_stats", default=False, type=str2bool)


def override_default_params_func(env: str, parser: object) -> None:
    """Set default training hyperparameters for gate navigation."""
    parser.set_defaults(
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        actor_worker_gpus=[0],
        train_for_env_steps=10000000000,
        use_rnn=False,
        adaptive_stddev=True,
        policy_initialization="torch_default",
        env_gpu_actions=True,
        env_gpu_observations=True,
        reward_scale=0.1,
        rollout=32,
        max_grad_norm=1.0,
        batch_size=1024,
        num_batches_per_epoch=4,
        num_epochs=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=2.0,
        exploration_loss_coeff=0.001,
        nonlinearity="elu",
        learning_rate=3e-4,
        lr_schedule="kl_adaptive_epoch",
        lr_schedule_kl_threshold=0.016,
        shuffle_minibatches=False,
        gamma=0.98,
        gae_lambda=0.95,
        with_vtrace=False,
        value_bootstrap=True,
        normalize_input=True,
        normalize_returns=True,
        save_best_after=int(1e6),
        serial_mode=True,
        async_rl=False,
        use_env_info_cache=False,
        kl_loss_coeff=0.1,
        restart_behavior="resume",
    )

    if env in env_configs:
        parser.set_defaults(**env_configs[env])


def register_aerialgym_custom_components() -> None:
    clear_sf_cache()
    setup_env_agents()

    if "SF_TRAIN_DIR" not in os.environ:
        os.environ["SF_TRAIN_DIR"] = "./train_dir"

    register_main_process_tasks()

    for env_name in env_configs:
        register_env(env_name, make_aerialgym_env)

    try:
        global_model_factory().register_encoder_factory(make_dual_fusion_encoder)
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not register DualFusionEncoder: {e}")


def parse_aerialgym_cfg(evaluation: bool = False) -> Config:
    parser, partial_cfg = parse_sf_args(evaluation=evaluation)
    add_extra_params_func(parser)
    override_default_params_func(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser)

    try:
        bridge_cfg_to_env_vars(final_cfg)
    except (ValueError, TypeError):
        pass

    return final_cfg


def main() -> int | None:
    """Script entry point."""
    register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg()
    return run_with_influence_tracking(cfg)


if __name__ == "__main__":
    sys.exit(main())
