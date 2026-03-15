from __future__ import annotations

# Training script for DCE navigation task - FIXED for inference compatibility
# This script has been fixed to use 4D action space matching existing inference scripts
# 
# ARCHITECTURE CHANGES (to match inference expectations):
# - Single input processing: Uses only "obs" (145D) instead of multi-input ("image_obs" + "observations")
# - Disabled ConvNet encoder: VAE latents are pre-computed by DCE task, no raw image processing in Sample Factory
# - 4D action space: Trains with 4D actions directly matching inference script expectations
# - Simplified pipeline: obs (145D) -> MLP encoder -> 128D -> RNN -> 4D actions (compatible with sf_inference_class_gate.py)
#
# DCE Gate Navigation Task Specifics:
# - Action space: 4D Sample Factory output directly matching DCE task input (x_vel, y_vel, z_vel, yaw_rate)
# - SOLUTION FOR INFERENCE COMPATIBILITY: Train with 4D actions directly
#   * Training and inference both use 4D action space to avoid shape mismatch
#   * This ensures trained models have 4D action output compatible with inference scripts
# - Observation space: 150D total = 3D drone position + 6D static camera pose + 3D full orientation + 9D state + 64D drone VAE + 64D static camera VAE
#   * 0-2: drone absolute position (x, y, z in world coordinates)
#   * 3-5: static camera position relative to drone (x, y, z in drone's reference frame)
#   * 6-8: static camera orientation relative to drone (roll, pitch, yaw in drone's reference frame)
#   * 9-11: drone full orientation including yaw (roll, pitch, yaw)
#   * 12-14: drone linear velocity in body frame
#   * 15-17: drone angular velocity in body frame
#   * 18-21: drone actions (4D for velocity controller)
#   * 22-85: drone camera VAE latents (64D)
#   * 86-149: static camera VAE latents (64D)
# - Curriculum: starts at level 3 and goes up to level 20 (custom range for progressive difficulty)
# - 128 parallel environments (1 agent per environment) for maximum parallelization
# - Uses LMF2 robot with VELOCITY CONTROL for direct responsive control
# - Compatible with existing inference scripts: sf_inference_class_gate.py, dce_nn_navigation_gate.py
#
# Environment is registered as "quad_with_obstacles_gate" for gate navigation

# this is here just to guarantee that isaacgym is imported before PyTorch
# isort: off
# noinspection PyUnresolvedReferences

# isort: on

import sys
from typing import Dict, Optional, Tuple


import isaacgym
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.cm
from PIL import Image
import os
import math

VERBOSE = os.environ.get('TRAIN_VERBOSE', 'false').lower() == 'true'


from torch import Tensor
from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.encoder import Encoder, ObsSpace, create_mlp, calc_num_elements, nonlinearity
import gymnasium.spaces as spaces
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.utils.typing import Config, Env
from sample_factory.utils.utils import str2bool

from aerial_gym.registry.task_registry import task_registry

from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_common import (
    AerialGymVecEnv as AerialGymVecEnvBase,
    BASE_ENV_CONFIGS,
    override_default_params,
    clear_sf_cache,
    setup_env_agents,
    parse_cfg,
)

import numpy as np

# Enforce deterministic backends for reproducibility
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from aerial_gym.rl_training.sample_factory.aerialgym_examples.env_wrapper_gate import AerialGymVecEnvGate



def make_aerialgym_env(
    full_task_name: str,
    cfg: Config,
    _env_config=None,
    render_mode: Optional[str] = None,
) -> Env:
    
    # Import task_registry for this function
    from aerial_gym.registry.task_registry import task_registry

    # Ensure DCE navigation task is registered in this subprocess
    if full_task_name == "quad_with_obstacles" or full_task_name == "quad_with_obstacles_gate":
        try:
            # Check if task is already registered
            task_registry.get_task_class(full_task_name)
        except KeyError:
            # Task not registered, register it now
            try:
                if full_task_name == "quad_with_obstacles_gate":
                    # Register gate navigation task
                    from aerial_gym.examples.dce_rl_navigation.dce_navigation_task_gate import DCE_RL_Navigation_Task_Gate
                    from aerial_gym.config.task_config.navigation_task_config_gate import task_config
                    
                    gate_config = task_config()
                    # Apply ablation flags from CLI cfg so the task can propagate to EnvManager
                    try:
                        gate_config.disable_gate_size_randomization = bool(cfg.disable_gate_size_randomization)
                    except Exception:
                        gate_config.disable_gate_size_randomization = False
                    try:
                        gate_config.fixed_gate_scale_percent = int(cfg.fixed_gate_scale_percent)
                    except (ValueError, TypeError):
                        gate_config.fixed_gate_scale_percent = 100
                    # Obstacle ablation flags
                    try:
                        gate_config.disable_obstacle_randomization = bool(cfg.disable_obstacle_randomization)
                    except Exception:
                        gate_config.disable_obstacle_randomization = False
                    try:
                        gate_config.fixed_obstacles_behind_gate = int(cfg.fixed_obstacles_behind_gate)
                    except (ValueError, TypeError):
                        gate_config.fixed_obstacles_behind_gate = 0
                    # Handle headless and environment settings for gate task
                    TaskClass = DCE_RL_Navigation_Task_Gate
                    config = gate_config
                    register_name = "quad_with_obstacles_gate"
                    backup_name = "dce_navigation_task_gate"
                else:
                    # Register standard DCE navigation task  
                    from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import DCE_RL_Navigation_Task
                    from aerial_gym.config.task_config.navigation_task_config import task_config
                    
                    # Get config the same way as original DCE script  
                    base_config = task_registry.get_task_config("navigation_task")
                    config = base_config()
                    # Apply DCE-specific configuration changes
                    config.action_space_dim = 3  # DCE uses 3D actions (not 4D)
                    config.curriculum.min_level = 3  # Gate curriculum starts from level 3 (matches environment obstacles)
                    config.curriculum.max_level = 23  # Gate curriculum goes up to level 23 (full difficulty range)
                    TaskClass = DCE_RL_Navigation_Task
                    register_name = "quad_with_obstacles"
                    backup_name = "dce_navigation_task"
                
                # CRITICAL FIX: Force headless mode for all Sample Factory training to avoid Isaac Gym conflicts
                config.headless = True
                print(f"[SUBPROCESS] FORCED headless mode for all Sample Factory training: headless={config.headless}")
                print(f"[SUBPROCESS] This prevents Isaac Gym viewer conflicts across all processes")
                # Propagate optional max curriculum cap from CLI
                try:
                    cap = cfg.max_curriculum_level
                except Exception:
                    cap = None
                if cap is not None:
                    try:
                        config.max_curriculum_level = int(cap)
                    except (ValueError, TypeError):
                        config.max_curriculum_level = None
                
                # CRITICAL FIX: Override action space to match inference expectations
                if hasattr(config, 'sample_factory_action_space_dim'):
                    if full_task_name == "quad_with_obstacles_gate":
                        config.sample_factory_action_space_dim = 4  # 4D for gate navigation
                    else:
                        config.sample_factory_action_space_dim = 3  # 3D for standard navigation
                print(f"[SUBPROCESS] Task action_space_dim: {config.action_space_dim}")
                print(f"[SUBPROCESS] Target Sample Factory action space: {config.action_space_dim}D")
                
                # CRITICAL: Set environment count in subprocess based on env_agents
                if hasattr(cfg, 'env_agents') and cfg.env_agents > 0:
                    config.num_envs = cfg.env_agents
                    # Set environment variable so task can detect env count
                    import os
                    os.environ['SF_ENV_AGENTS'] = str(cfg.env_agents)
                    print(f"[SUBPROCESS] Setting num_envs to {cfg.env_agents} based on env_agents={cfg.env_agents}")
                    print(f"[SUBPROCESS] Set SF_ENV_AGENTS={cfg.env_agents} environment variable")
                    print(f"[SUBPROCESS] Config batch_size: {cfg.batch_size}")
                    if cfg.env_agents == 128:
                        print(f"[SUBPROCESS] Using MAXIMUM PARALLELIZATION CONFIG (128 environments)")
                    elif cfg.env_agents == 32:
                        print(f"[SUBPROCESS] Using HIGH PARALLELIZATION CONFIG (32 environments)")
                    elif cfg.env_agents == 16:
                        print(f"[SUBPROCESS] Using STANDARD CONFIG (16 environments)")
                    elif cfg.env_agents == 6:
                        print(f"[SUBPROCESS] Using MEDIUM CONFIG (6 environments)")
                    elif cfg.env_agents == 4:
                        print(f"[SUBPROCESS] Using MEDIUM CONFIG (4 environments)")
                    elif cfg.env_agents == 1:
                        print(f"[SUBPROCESS] Using LOW CONFIG (1 environment)")
                    else:
                        print(f"[SUBPROCESS] Using CUSTOM CONFIG ({cfg.env_agents} environments)")
                else:
                    print(f"[SUBPROCESS] env_agents={cfg.env_agents}, using default num_envs")
                
                task_registry.register_task(register_name, TaskClass, config)
                # Also register backup name for backward compatibility
                task_registry.register_task(backup_name, TaskClass, config)
                print(f"Registered {register_name} and {backup_name} in subprocess")
            except (ValueError, TypeError) as e:
                print(f"Failed to register quad_with_obstacles in subprocess: {e}")

    # Get save_gifs parameter from config
    save_gifs = cfg.save_gifs

    # Create the environment and force correct action space for inference compatibility
    # Forward seed from cfg if provided, else None
    seed_val = cfg.seed
    env = AerialGymVecEnvGate(
        task_registry.make_task(task_name=full_task_name, seed=seed_val),
        "obs",
        save_gifs=save_gifs,
    )
    
    # Debug: list available gate variants if present
    try:
        gd = getattr(env.env, 'global_tensor_dict', {})
        names0 = gd.get('gate_variant_names_per_env', [])
        if names0 and len(names0) > 0:
            print(f"[GateVariant] Available gate variants for env0: {names0[0]}")
        active = gd.get('active_gate_variant_index', None)
        if active is not None:
            print(f"[GateVariant] Active gate variant index tensor: {active}")
    except (ValueError, TypeError) as e:
        print(f"[GateVariant] Debug listing failed: {e}")
    
    # CRITICAL FIX: Force action space to exactly match inference expectations
    # Override action space after environment creation to ensure it sticks
    import gymnasium as gym
    import numpy as np
    if full_task_name == "quad_with_obstacles_gate":
        forced_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)  # 4D for gate navigation
        expected_dims = "4D"
    else:
        forced_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)  # 3D for standard navigation
        expected_dims = "3D"
    env.action_space = convert_space(forced_action_space)
    
    # Debug: Verify action space dimensions
    print(f"[make_aerialgym_env] Final action space shape: {env.action_space.shape}")
    print(f"[make_aerialgym_env] Expected {expected_dims} action space: {env.action_space}")
    
    return env


def add_extra_params_func(parser) -> None:
    """
    Specify extra arguments for this family of environments.
    """
    
    parser.add_argument("--env_agents", default=None, type=int, help="Num agents in env (multi-agent only)")
    parser.add_argument("--headless", type=lambda x: x.lower() == 'true', default=None, help="Force headless mode (True/False)")
    parser.add_argument("--save_gifs", type=lambda x: x.lower() == 'true', default=False, help="Save episode GIFs for both cameras (True/False)")
    # Gate size ablation flags
    parser.add_argument("--disable_gate_size_randomization", type=lambda x: x.lower() == 'true', default=False, help="Disable gate size randomization and use a fixed scale percent")
    parser.add_argument("--fixed_gate_scale_percent", type=int, default=100, help="Fixed gate scale percent to use when randomization is disabled (40..100, step 2)")
    # Obstacle ablation flags (behind-gate objects)
    parser.add_argument("--disable_obstacle_randomization", type=lambda x: x.lower() == 'true', default=False, help="Disable obstacle randomization behind the gate (spawns zero obstacles)")
    parser.add_argument("--fixed_obstacles_behind_gate", type=int, default=0, help="Fixed number of obstacles behind the gate when randomization is disabled (default 0)")
    # Static camera orientation randomization ablation flag
    parser.add_argument("--disable_static_camera_orientation_randomization", type=lambda x: x.lower() == 'true', default=False, help="Disable static camera orientation randomization, fix angle to 0.0°")
    # Camera noise randomization ablation flag (drone & static)
    parser.add_argument("--disable_camera_noise_randomization", type=lambda x: x.lower() == 'true', default=False, help="Disable camera noise randomization (Gaussian STD=0, Dropout=0) for both drone & static")
    # Camera frame dropout randomization ablation flag (drone & static)
    parser.add_argument("--disable_camera_frame_dropout_randomization", type=lambda x: x.lower() == 'true', default=False, help="Disable entire-frame dropout randomization for both drone & static cameras")
    # NEW: Per-camera noise/dropout controls
    parser.add_argument("--disable_drone_camera_noise_randomization", type=lambda x: x.lower() == 'true', default=None, help="Disable noise randomization for DRONE camera only (overrides global when set)")
    parser.add_argument("--disable_static_camera_noise_randomization", type=lambda x: x.lower() == 'true', default=None, help="Disable noise randomization for STATIC camera only (overrides global when set)")
    parser.add_argument("--disable_drone_camera_frame_dropout", type=lambda x: x.lower() == 'true', default=None, help="Disable frame dropout for DRONE camera only (overrides global when set)")
    parser.add_argument("--disable_static_camera_frame_dropout", type=lambda x: x.lower() == 'true', default=None, help="Disable frame dropout for STATIC camera only (overrides global when set)")
    # Static camera yaw sweep (constant oscillation)
    parser.add_argument("--enable_static_camera_yaw_sweep", type=lambda x: x.lower() == 'true', default=False, help="Enable constant yaw oscillation for static camera (±30°)")
    parser.add_argument("--enable_static_camera_locked", type=lambda x: x.lower() == 'true', default=False, help="Lock static camera position and rotate to center the drone")
    parser.add_argument("--static_camera_yaw_sweep_speed_deg", type=float, default=10.0, help="Yaw sweep speed in deg/s (default 10)")
    # Static camera base position overrides (Y back distance, Z height)
    parser.add_argument("--static_camera_base_y", type=float, default=None, help="Override static camera base Y (meters; negative is behind gate). Default -3.0 if not set")
    # Accept float or the literal string 'adaptive'
    def parse_base_z(val) -> None:
        v = str(val).strip().lower()
        if v == 'adaptive':
            return 'adaptive'
        try:
            return float(val)
        except (ValueError, TypeError):
            raise ValueError("--static_camera_base_z must be a float or 'adaptive'")
    parser.add_argument("--static_camera_base_z", type=parse_base_z, default=None, help="Static cam Z (meters) or 'adaptive' to follow gate center height")
    # State noise randomization ablation flag (drone & static pose noise)
    parser.add_argument("--disable_state_noise_randomization", type=lambda x: x.lower() == 'true', default=False, help="Disable pose state noise randomization for drone and static camera")
    # Dynamic camera following ablation flag
    parser.add_argument("--disable_dynamic_camera_following", type=lambda x: x.lower() == 'true', default=False, help="Disable dynamic camera following mode (forces static camera even if enabled in config)")
    # Dynamic camera following enable flag (override config setting)
    parser.add_argument("--enable_dynamic_camera_following", type=lambda x: x.lower() == 'true', default=None, help="Enable dynamic camera following mode (overrides config setting when specified)")
    # Arc-follow (separate static-camera mode)
    parser.add_argument("--enable_static_camera_arc_follow", type=lambda x: x.lower() == 'true', default=False, help="Enable arc-follow static camera mode (camera moves on a fixed-radius arc around the gate)")
    parser.add_argument("--static_camera_arc_radius_m", type=float, default=2.0, help="Arc-follow radius in meters (default 2.0)")
    # Dynamic follow distance override (Y offset in meters; negative is behind gate)
    parser.add_argument("--dynamic_camera_follow_y_offset_m", type=float, default=None, help="Override dynamic camera follow Y-offset in meters (requires dynamic follow to be enabled)")
    # Disable gate blending in dynamic follow (pure drone-follow look target)
    parser.add_argument("--disable_dynamic_follow_gate_blending", type=lambda x: x.lower() == 'true', default=False, help="Disable blending toward gate in dynamic-follow; always look at drone")
    # Spawn randomization ablations (position vs orientation independently)
    parser.add_argument("--disable_spawn_position_randomization", type=lambda x: x.lower() == 'true', default=False, help="Disable robot spawn POSITION randomization (lock to baseline level)")
    parser.add_argument("--disable_spawn_orientation_randomization", type=lambda x: x.lower() == 'true', default=False, help="Disable robot spawn ORIENTATION randomization (lock yaw to baseline level)")
    # Curriculum multiplier ablation
    parser.add_argument("--disable_curriculum_multiplier", type=lambda x: x.lower() == 'true', default=False, help="Disable curriculum reward multiplier (sets multiplier to 1.0)")
    # Force fixed curriculum level (disables auto progression)
    parser.add_argument(
        "--force_curriculum_level",
        type=str,
        default=None,
        help="Force a specific curriculum level for the entire run (disables auto curriculum progression). Use 'none' to disable forcing.")
    # Optional maximum curriculum level cap (progression will not exceed this level). Does not affect scaling.
    parser.add_argument("--max_curriculum_level", type=int, default=None, help="Maximum curriculum level cap for progression (e.g., 13). Scaling at each level remains unchanged.")
    # NEW: Minimum curriculum level to START training from (training only; inference unaffected)
    parser.add_argument("--min_curriculum_level", type=int, default=None, help="Minimum curriculum level to start from during TRAINING (e.g., 13). Auto-progression proceeds up to max_curriculum_level or config max. Inference ignores this flag.")
    
    # Fusion mode flags
    parser.add_argument("--fusion", type=str, default="gated", choices=["concat", "gated"], help="Fusion strategy: concat (early concat) or gated (dual gated late fusion)")
    parser.add_argument("--gate_per_feature", type=int, default=1, help="Use per-feature gate (1) or scalar gate (0)")

    # Complete observation influence tracking arguments
    # Allow env var overrides: SF_ENABLE_INFLUENCE_TRACKER and SF_ENABLE_GRAD_ATTR
    import os as _os
    _inf_env = _os.getenv('SF_ENABLE_INFLUENCE_TRACKER')
    _inf_default = (str(_inf_env).lower() == 'true') if _inf_env is not None else False
    parser.add_argument("--enable_gradient_monitoring", type=lambda x: x.lower() == 'true', default=_inf_default, help="Enable complete observation influence tracking (overridable via SF_ENABLE_INFLUENCE_TRACKER)")
    parser.add_argument("--gradient_log_interval", default=100, type=int, help="Log influence metrics every N steps")
    parser.add_argument("--gradient_print_interval", default=100, type=int, help="Print analysis summary every N steps")
    _grad_env = _os.getenv('SF_ENABLE_GRAD_ATTR')
    _grad_default = (str(_grad_env).lower() == 'true') if _grad_env is not None else True
    parser.add_argument("--enable_grad_attribution", type=lambda x: x.lower() == 'true', default=_grad_default, help="Enable gradient-based attribution alongside correlation analysis (overridable via SF_ENABLE_GRAD_ATTR)")
    
    p = parser
    p.add_argument(
        "--obs_key",
        default="obs",
        type=str,
        help='IsaacGym envs return dicts, some envs return just "obs", and some return "obs" and "states".'
        "States key denotes the full state of the environment, and obs key corresponds to limited observations "
        'available in real world deployment. If we use "states" here we can train will full information '
        "(although the original idea was to use asymmetric training - critic sees full state and policy only sees obs).",
    )
    p.add_argument(
        "--subtask",
        default=None,
        type=str,
        help="Subtask for envs that support it (i.e. AllegroKuka regrasping or manipulation or throw).",
    )
    p.add_argument(
        "--ige_api_version",
        default="preview4",
        type=str,
        choices=["preview3", "preview4"],
        help="We can switch between different versions of IsaacGymEnvs API using this parameter.",
    )
    p.add_argument(
        "--eval_stats",
        default=False,
        type=str2bool,
        help="Whether to collect env stats during evaluation.",
    )


def override_default_params_func(env, parser) -> None:
    """Most of these parameters are taken from IsaacGymEnvs default config files."""

    # Default parameters for medium configuration (4 environments)
    default_batch_size = 1024
    default_num_batches_per_epoch = 4

    parser.set_defaults(
        # we're using a single very vectorized env, no need to parallelize it further
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,  # CRITICAL: Only 1 environment per worker (but 128 agents inside it)
        worker_num_splits=1,
        actor_worker_gpus=[0],  # obviously need a GPU
        train_for_env_steps=10000000000,
        use_rnn=False,
        adaptive_stddev=True,  # Default for other environments
        policy_initialization="torch_default",
        env_gpu_actions=True,
        env_gpu_observations=True,  # Critical: Tell Sample Factory we're providing GPU tensors
        reward_scale=0.1,
        rollout=32,  # REVERTED: Issue was tensor reference bug, not rollout frequency
        max_grad_norm=1.0,  # changed to match DCE config
        # batch_size=2048,
        # num_batches_per_epoch=2,
        batch_size=default_batch_size,  # Adjusted based on environment
        num_batches_per_epoch=default_num_batches_per_epoch,  # Adjusted based on environment
        num_epochs=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=2.0,
        exploration_loss_coeff=0.001,  # changed to match DCE config
        nonlinearity="elu",
        learning_rate=3e-4,
        lr_schedule="kl_adaptive_epoch",
        lr_schedule_kl_threshold=0.016,
        shuffle_minibatches=False,  # changed to match DCE config
        gamma=0.98,
        gae_lambda=0.95,
        with_vtrace=False,
        value_bootstrap=True,  # assuming reward from the last step in the episode can generally be ignored
        normalize_input=True,
        normalize_returns=True,  # does not improve results on all envs, but with return normalization we don't need to tune reward scale
        save_best_after=int(1e6),
        serial_mode=True,  # it makes sense to run isaacgym envs in serial mode since most of the parallelism comes from the env itself (although async mode works!)
        async_rl=False,
        use_env_info_cache=False,  # speeds up startup
        kl_loss_coeff=0.1,
        restart_behavior="resume",  # changed to match DCE config
    )

    # override default config parameters for specific envs
    if env in env_configs:
        parser.set_defaults(**env_configs[env])


from aerial_gym.rl_training.sample_factory.aerialgym_examples.gate_env_configs import env_configs

from aerial_gym.rl_training.sample_factory.aerialgym_examples.dual_fusion_encoder import (
    DualFusionEncoder,
    make_dual_fusion_encoder,
)



def register_aerialgym_custom_components() -> None:
    clear_sf_cache()
    setup_env_agents()
    
    # Set train_dir for curriculum logging
    import os
    if 'SF_TRAIN_DIR' not in os.environ:
        os.environ['SF_TRAIN_DIR'] = './train_dir'  # Default train directory
    if current_env_agents == '256':
        print(f"Set SF_ENV_AGENTS={current_env_agents} environment variable for all processes (HIGH PARALLELIZATION CONFIG)")
    elif current_env_agents == '32':
        print(f"Set SF_ENV_AGENTS={current_env_agents} environment variable for all processes (UPDATED DCE CONFIG - high parallelization)")
    elif current_env_agents == '16':
        print(f"Set SF_ENV_AGENTS={current_env_agents} environment variable for all processes (ORIGINAL DCE CONFIG - high performance)")
    elif current_env_agents == '6':
        print(f"Set SF_ENV_AGENTS={current_env_agents} environment variable for all processes (MEDIUM CONFIG - reduced memory)")
    elif current_env_agents == '4':
        print(f"Set SF_ENV_AGENTS={current_env_agents} environment variable for all processes (MEDIUM CONFIG - reduced memory)")
    elif current_env_agents == '1':
        print(f"Set SF_ENV_AGENTS={current_env_agents} environment variable for all processes (LOW CONFIG - minimum memory)")
    else:
        print(f"Set SF_ENV_AGENTS={current_env_agents} environment variable for all processes (CUSTOM CONFIG)")
    
    # Register DCE navigation task as "quad_with_obstacles" to match original config
    try:
        from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import DCE_RL_Navigation_Task
        from aerial_gym.config.task_config.navigation_task_config import task_config
        from aerial_gym.registry.task_registry import task_registry

        # Use navigation task config as base for DCE navigation with DCE-specific overrides
        # Get config the same way as original DCE script
        base_config = task_registry.get_task_config("navigation_task")
        dce_config = base_config()
        # Apply DCE-specific configuration changes
        dce_config.action_space_dim = 3  # DCE uses 3D actions (not 4D)
        dce_config.curriculum.min_level = 3  # Gate curriculum starts from level 3 (matches environment obstacles)
        dce_config.curriculum.max_level = 23  # Gate curriculum goes up to level 23 (full difficulty range)
        
        # FORCE headless setting - let DCE task handle the default, no override here
        # The headless setting will be properly handled in make_aerialgym_env function
        print(f"[MAIN] DCE task will handle headless setting based on command line parameters")
        
        # CRITICAL FIX: Override action space to match inference expectations
        # Force environment to report 3D action space for inference compatibility
        if hasattr(dce_config, 'sample_factory_action_space_dim'):
            dce_config.sample_factory_action_space_dim = 3
        print(f"[MAIN] DCE task action_space_dim: {dce_config.action_space_dim}")
        print(f"[MAIN] Target Sample Factory action space: 3D")
        # Note: num_envs will be set based on env_agents parameter during env creation
        # Register as "quad_with_obstacles" to match original config.json
        task_registry.register_task("quad_with_obstacles", DCE_RL_Navigation_Task, dce_config)
        print("Successfully registered quad_with_obstacles (DCE navigation task)")
        
        # Also register as "dce_navigation_task" for backward compatibility with inference scripts
        task_registry.register_task("dce_navigation_task", DCE_RL_Navigation_Task, dce_config)
        print("Successfully registered dce_navigation_task for backward compatibility")
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not register quad_with_obstacles: {e}")
    
    # Register Gate Navigation task as "quad_with_obstacles_gate"
    try:
        from aerial_gym.examples.dce_rl_navigation.dce_navigation_task_gate import DCE_RL_Navigation_Task_Gate
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config as gate_task_config
        
        # Use gate navigation task config
        gate_config = gate_task_config()
        
        # Gate navigation task settings
        print(f"[MAIN] Gate navigation task will handle headless setting based on command line parameters")
        print(f"[MAIN] Gate navigation task action_space_dim: {gate_config.action_space_dim}")
        print(f"[MAIN] Gate navigation observation_space_dim: {gate_config.observation_space_dim}")
        print(f"[MAIN] Target Sample Factory action space: 3D")
        print(f"[MAIN] Gate environment: {gate_config.env_name}")
        print(f"[MAIN] Gate robot: {gate_config.robot_name}")
        
        # The headless setting will be properly handled by the DCE gate task itself
        print(f"[MAIN] Gate navigation task will handle headless setting via DCE task logic")
        
        # Register gate navigation task
        task_registry.register_task("quad_with_obstacles_gate", DCE_RL_Navigation_Task_Gate, gate_config)
        print("Successfully registered quad_with_obstacles_gate (Gate navigation task)")
        
        # Also register as "dce_navigation_task_gate" for backward compatibility with inference scripts
        task_registry.register_task("dce_navigation_task_gate", DCE_RL_Navigation_Task_Gate, gate_config)
        print("Successfully registered dce_navigation_task_gate for backward compatibility")
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not register quad_with_obstacles_gate: {e}")
    
    for env_name in env_configs:
        register_env(env_name, make_aerialgym_env)

    # Register custom encoder to perform fusion inside SF model
    try:
        global_model_factory().register_encoder_factory(make_dual_fusion_encoder)
        print("Registered DualFusionEncoder with fusion/ gating options")
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not register DualFusionEncoder: {e}")


def parse_aerialgym_cfg(evaluation=False) -> None:
    parser, partial_cfg = parse_sf_args(evaluation=evaluation)
    add_extra_params_func(parser)
    override_default_params_func(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser)
    # Bridge CLI flag to environment variable so worker processes can read it reliably
    try:
        # Fusion flags to env for workers
        if True:
            os.environ['SF_FUSION_MODE'] = str(final_cfg.fusion)
            print(f"[CFG] fusion mode: {final_cfg.fusion}")
        if True:
            os.environ['SF_GATE_PER_FEATURE'] = '1' if int(final_cfg.gate_per_feature) != 0 else '0'
            print(f"[CFG] gate_per_feature: {final_cfg.gate_per_feature}")
        if True:
            os.environ['SF_DISABLE_STATIC_CAMERA_ORIENT_RANDOMIZATION'] = 'true' if final_cfg.disable_static_camera_orientation_randomization else 'false'
            print(f"[CFG] static camera orientation randomization disabled: {final_cfg.disable_static_camera_orientation_randomization}")
        if True:
            os.environ['SF_DISABLE_CAMERA_NOISE_RANDOMIZATION'] = 'true' if final_cfg.disable_camera_noise_randomization else 'false'
            print(f"[CFG] camera noise randomization disabled: {final_cfg.disable_camera_noise_randomization}")
        # Per-camera noise/dropout overrides
        if final_cfg.disable_drone_camera_noise_randomization is not None:
            os.environ['SF_DISABLE_DRONE_CAMERA_NOISE_RANDOMIZATION'] = 'true' if final_cfg.disable_drone_camera_noise_randomization else 'false'
            print(f"[CFG] DRONE camera noise disabled override: {final_cfg.disable_drone_camera_noise_randomization}")
        if final_cfg.disable_static_camera_noise_randomization is not None:
            os.environ['SF_DISABLE_STATIC_CAMERA_NOISE_RANDOMIZATION'] = 'true' if final_cfg.disable_static_camera_noise_randomization else 'false'
            print(f"[CFG] STATIC camera noise disabled override: {final_cfg.disable_static_camera_noise_randomization}")
        if final_cfg.disable_drone_camera_frame_dropout is not None:
            os.environ['SF_DISABLE_DRONE_CAMERA_FRAME_DROPOUT'] = 'true' if final_cfg.disable_drone_camera_frame_dropout else 'false'
            print(f"[CFG] DRONE camera frame-drop disabled override: {final_cfg.disable_drone_camera_frame_dropout}")
        if final_cfg.disable_static_camera_frame_dropout is not None:
            os.environ['SF_DISABLE_STATIC_CAMERA_FRAME_DROPOUT'] = 'true' if final_cfg.disable_static_camera_frame_dropout else 'false'
            print(f"[CFG] STATIC camera frame-drop disabled override: {final_cfg.disable_static_camera_frame_dropout}")
        # Static camera yaw sweep (const ±30°, curriculum-independent for now)
        if True:
            os.environ['SF_ENABLE_STATIC_CAMERA_YAW_SWEEP'] = 'true' if final_cfg.enable_static_camera_yaw_sweep else 'false'
            print(f"[CFG] Static camera yaw sweep enabled: {final_cfg.enable_static_camera_yaw_sweep}")
        if True:
            os.environ['SF_STATIC_CAMERA_YAW_SWEEP_SPEED_DEG'] = str(float(final_cfg.static_camera_yaw_sweep_speed_deg))
        if True:
            os.environ['SF_STATIC_CAMERA_LOCKED_FOLLOW'] = 'true' if final_cfg.enable_static_camera_locked else 'false'
            print(f"[CFG] Static camera locked-follow enabled: {final_cfg.enable_static_camera_locked}")
            print(f"[CFG] Static camera yaw sweep speed: {final_cfg.static_camera_yaw_sweep_speed_deg} deg/s")
        # Static camera base position overrides to env for workers
        if final_cfg.static_camera_base_y is not None:
            os.environ['SF_STATIC_CAMERA_BASE_Y'] = str(float(final_cfg.static_camera_base_y))
            print(f"[CFG] Static camera base Y: {final_cfg.static_camera_base_y}")
        if final_cfg.static_camera_base_z is not None:
            if isinstance(final_cfg.static_camera_base_z, str) and str(final_cfg.static_camera_base_z).lower() == 'adaptive':
                os.environ['SF_STATIC_CAMERA_BASE_Z'] = 'adaptive'
                print(f"[CFG] Static camera base Z: adaptive")
            else:
                os.environ['SF_STATIC_CAMERA_BASE_Z'] = str(float(final_cfg.static_camera_base_z))
                print(f"[CFG] Static camera base Z: {final_cfg.static_camera_base_z}")
        if True:
            os.environ['SF_DISABLE_CAMERA_FRAME_DROPOUT_RANDOMIZATION'] = 'true' if final_cfg.disable_camera_frame_dropout_randomization else 'false'
            print(f"[CFG] camera frame dropout randomization disabled: {final_cfg.disable_camera_frame_dropout_randomization}")
        if True:
            os.environ['SF_DISABLE_STATE_NOISE_RANDOMIZATION'] = 'true' if final_cfg.disable_state_noise_randomization else 'false'
            print(f"[CFG] state noise randomization disabled: {final_cfg.disable_state_noise_randomization}")
        if True:
            os.environ['disable_dynamic_camera_following'] = 'true' if final_cfg.disable_dynamic_camera_following else 'false'
            print(f"[CFG] dynamic camera following disabled: {final_cfg.disable_dynamic_camera_following}")
        if final_cfg.enable_dynamic_camera_following is not None:
            os.environ['enable_dynamic_camera_following'] = 'true' if final_cfg.enable_dynamic_camera_following else 'false'
            print(f"[CFG] dynamic camera following enabled (override): {final_cfg.enable_dynamic_camera_following}")
        # Arc-follow flags → env
        if True:
            os.environ['SF_ENABLE_STATIC_CAMERA_ARC_FOLLOW'] = 'true' if final_cfg.enable_static_camera_arc_follow else 'false'
            print(f"[CFG] static camera arc-follow enabled: {final_cfg.enable_static_camera_arc_follow}")
        if final_cfg.static_camera_arc_radius_m is not None:
            os.environ['SF_STATIC_CAMERA_ARC_RADIUS_M'] = str(float(final_cfg.static_camera_arc_radius_m))
            print(f"[CFG] static camera arc radius: {final_cfg.static_camera_arc_radius_m} m")
        if final_cfg.dynamic_camera_follow_y_offset_m is not None:
            os.environ['SF_DYNAMIC_CAMERA_FOLLOW_OFFSET_Y'] = str(float(final_cfg.dynamic_camera_follow_y_offset_m))
            print(f"[CFG] dynamic camera follow Y-offset: {final_cfg.dynamic_camera_follow_y_offset_m} m")
        if True:
            os.environ['SF_DISABLE_DYNAMIC_FOLLOW_GATE_BLENDING'] = 'true' if final_cfg.disable_dynamic_follow_gate_blending else 'false'
            print(f"[CFG] dynamic follow gate blending disabled: {final_cfg.disable_dynamic_follow_gate_blending}")
        if True:
            os.environ['SF_DISABLE_SPAWN_POSITION_RANDOMIZATION'] = 'true' if final_cfg.disable_spawn_position_randomization else 'false'
            print(f"[CFG] spawn position randomization disabled: {final_cfg.disable_spawn_position_randomization}")
        if True:
            os.environ['SF_DISABLE_SPAWN_ORIENTATION_RANDOMIZATION'] = 'true' if final_cfg.disable_spawn_orientation_randomization else 'false'
            print(f"[CFG] spawn orientation randomization disabled: {final_cfg.disable_spawn_orientation_randomization}")
        if True:
            os.environ['SF_DISABLE_CURRICULUM_MULTIPLIER'] = 'true' if final_cfg.disable_curriculum_multiplier else 'false'
            print(f"[CFG] curriculum multiplier disabled: {final_cfg.disable_curriculum_multiplier}")
        if (final_cfg.force_curriculum_level is not None):
            lvl_str = str(final_cfg.force_curriculum_level).strip().lower()
            if lvl_str and lvl_str != 'none':
                os.environ['SF_FORCE_CURRICULUM_LEVEL'] = str(int(lvl_str))
                print(f"[CFG] forcing curriculum level: {lvl_str}")
            else:
                # ensure any previous env var is cleared
                os.environ.pop('SF_FORCE_CURRICULUM_LEVEL', None)
                print("[CFG] force curriculum level: none (disabled)")
        # Apply min_curriculum_level ONLY during training; do not affect evaluation/inference
        try:
            if not final_cfg.evaluation:
                min_lvl_override = final_cfg.min_curriculum_level
                if min_lvl_override is not None:
                    min_lvl = int(min_lvl_override)
                    # Respect any explicit max cap if provided
                    max_cap = final_cfg.max_curriculum_level
                    if max_cap is not None:
                        os.environ['SF_MAX_CURRICULUM_LEVEL'] = str(int(max_cap))
                    os.environ['SF_MIN_CURRICULUM_LEVEL'] = str(min_lvl)
                    print(f"[CFG] Curriculum start level (training): min_level={min_lvl}, max_level={max_cap if max_cap is not None else 'config default'}")
        except (ValueError, TypeError):
            pass
    except (ValueError, TypeError):
        pass
    return final_cfg


def main() -> None:
    """Script entry point."""
    register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg()
    
    # Check if complete observation influence tracking is enabled
    # Always attempt enhanced run; it falls back internally if trackers are unavailable
    return run_with_influence_tracking(cfg)


from aerial_gym.rl_training.sample_factory.aerialgym_examples.influence_tracking import (
    run_with_influence_tracking,
)




if __name__ == "__main__":
    sys.exit(main())
