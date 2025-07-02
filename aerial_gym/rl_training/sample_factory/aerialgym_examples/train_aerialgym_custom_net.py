# Training script for DCE navigation task - FIXED for inference compatibility
# This script has been fixed to use 3D action space matching existing inference scripts
# 
# ARCHITECTURE CHANGES (to match inference expectations):
# - Single input processing: Uses only "obs" (81D) instead of multi-input ("image_obs" + "observations")
# - Disabled ConvNet encoder: VAE latents are pre-computed by DCE task, no raw image processing in Sample Factory
# - 3D action space: Trains with 3D actions directly matching inference script expectations
# - Simplified pipeline: obs (81D) -> MLP encoder -> 64D -> RNN -> 3D actions (compatible with sf_inference_class.py)
#
# DCE Navigation Task Specifics:
# - Action space: 3D Sample Factory output directly matching DCE task input (x_vel, y_vel, yaw_rate)
# - SOLUTION FOR INFERENCE COMPATIBILITY: Train with 3D actions directly
#   * Training and inference both use 3D action space to avoid shape mismatch
#   * This ensures trained models have 3D action output compatible with inference scripts
# - Observation space: 81D total = 17D basic state + 64D VAE-encoded image latents
#   * 0-3: normalized vector to target + distance to target / 5.0
#   * 4-6: euler angles (roll, pitch, 0.0)  
#   * 7-9: robot body linear velocity
#   * 10-12: robot body angular velocity
#   * 13-16: robot actions (last action taken)
#   * 17-80: VAE-encoded depth image latents (64 dimensions, pre-computed by DCE task)
# - Curriculum: starts at level 30 and goes up to level 50 (custom range for progressive difficulty)
# - 32 parallel environments (1 agent per environment) for maximum parallelization
# - Uses LMF2 robot with velocity control
# - Compatible with existing inference scripts: sf_inference_class.py, dce_nn_navigation.py
#
# Environment is registered as "quad_with_obstacles" to match original config.json

# this is here just to guarantee that isaacgym is imported before PyTorch
# isort: off
# noinspection PyUnresolvedReferences

# isort: on

import sys
from typing import Dict, Optional, Tuple


import isaacgym
import gymnasium as gym
import torch
import numpy as np


from torch import Tensor
from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.utils.typing import Config, Env
from sample_factory.utils.utils import str2bool

from aerial_gym.registry.task_registry import task_registry

import numpy as np


class AerialGymVecEnv(gym.Env):
    """
    Wrapper for isaacgym environments to make them compatible with the sample factory.
    Modified to match old 1333 model architecture - single input processing.
    """

    def __init__(self, aerialgym_env, obs_key):
        self.env = aerialgym_env
        self.num_agents = self.env.num_envs
        self.is_multiagent = True
        # CRITICAL FIX: Force action space to exactly match inference expectations (3D)
        # The inference script expects 3D actions, so train with 3D to avoid shape mismatch
        import numpy as np
        base_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = convert_space(base_action_space)

        # Debug: Print action space info to verify it's 3D
        print(f"[AerialGymVecEnv] Forced action space shape: {self.action_space.shape}")
        print(f"[AerialGymVecEnv] is_multiagent: {self.is_multiagent}, num_agents: {self.num_agents}")

        # MODIFIED: Single input processing to match old 1333 model
        # Use only "obs" key containing 81D observations (17D state + 64D VAE latents)
        # This avoids multi-input processing that caused 576D RNN input
        if obs_key == "obs":
            # Create single input observation space with 81D observations
            self.observation_space = gym.spaces.Dict({
                "obs": convert_space(gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(81,), dtype=np.float32
                ))
            })
        else:
            raise ValueError(f"Unknown observation key: {obs_key}")

        self._truncated: Tensor = torch.zeros(self.num_agents, dtype=torch.bool)

    def reset(self, *args, **kwargs) -> Tuple[Dict[str, Tensor], Dict]:
        # some IGE envs return all zeros on the first timestep, but this is probably okay
        obs, rew, terminated, truncated, infos = self.env.reset()
        # MODIFIED: Transform multi-input to single input to match old 1333 model
        # DCE task provides both "observations" (81D) and "image_obs" (raw images)
        # We only want "observations" (81D = 17D state + 64D VAE latents) as "obs"
        transformed_obs = {"obs": obs["observations"]}
        return transformed_obs, infos

    def step(self, action) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Dict]:
        # FIXED: Direct 3D action pass-through for DCE task
        # Sample Factory now provides 3D actions directly matching DCE task expectations (x_vel, y_vel, yaw_rate)
        dce_action = action
            
        obs, rew, terminated, truncated, infos = self.env.step(dce_action)
        # MODIFIED: Transform multi-input to single input to match old 1333 model
        # DCE task provides both "observations" (81D) and "image_obs" (raw images)  
        # We only want "observations" (81D = 17D state + 64D VAE latents) as "obs"
        transformed_obs = {"obs": obs["observations"]}
        return transformed_obs, rew, terminated, truncated, infos

    def render(self):
        pass


def make_aerialgym_env(
    full_task_name: str,
    cfg: Config,
    _env_config=None,
    render_mode: Optional[str] = None,
) -> Env:
    
    # Import task_registry for this function
    from aerial_gym.registry.task_registry import task_registry
    
    # Ensure DCE navigation task is registered in this subprocess
    if full_task_name == "quad_with_obstacles":
        try:
            # Check if task is already registered
            task_registry.get_task_class(full_task_name)
        except KeyError:
            # Task not registered, register it now
            try:
                from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import DCE_RL_Navigation_Task
                from aerial_gym.config.task_config.navigation_task_config import task_config
                
                # Get config the same way as original DCE script  
                base_config = task_registry.get_task_config("navigation_task")
                dce_config = base_config()
                # Apply DCE-specific configuration changes
                dce_config.action_space_dim = 3  # DCE uses 3D actions (not 4D)
                dce_config.curriculum.min_level = 30  # DCE curriculum starts from level 30
                dce_config.curriculum.max_level = 50  # DCE curriculum goes up to level 50
                
                # Handle headless parameter - force viewer for rollout workers when requested
                if hasattr(cfg, 'headless') and cfg.headless is not None:
                    # If headless=False is explicitly requested, override for rollout workers too
                    if cfg.headless == False:
                        dce_config.headless = False
                        print(f"[SUBPROCESS] FORCING viewer mode for rollout worker: headless={dce_config.headless}")
                    else:
                        dce_config.headless = cfg.headless
                        print(f"[SUBPROCESS] Setting headless mode to: {dce_config.headless}")
                else:
                    # Use DCE default (disabled headless for visualization)
                    print(f"[SUBPROCESS] Using DCE default headless setting: {dce_config.headless}")
                
                # CRITICAL FIX: Override action space to match inference expectations
                # Force environment to report 3D action space for inference compatibility
                if hasattr(dce_config, 'sample_factory_action_space_dim'):
                    dce_config.sample_factory_action_space_dim = 3
                print(f"[SUBPROCESS] DCE task action_space_dim: {dce_config.action_space_dim}")
                print(f"[SUBPROCESS] Target Sample Factory action space: 3D")
                # CRITICAL: Set environment count in subprocess based on env_agents
                # This ensures rollout workers use the correct number of environments
                if hasattr(cfg, 'env_agents') and cfg.env_agents > 0:
                    dce_config.num_envs = cfg.env_agents
                    # Set environment variable so DCE task can detect env count
                    import os
                    os.environ['SF_ENV_AGENTS'] = str(cfg.env_agents)
                    print(f"[SUBPROCESS] Setting num_envs to {cfg.env_agents} based on env_agents={cfg.env_agents}")
                    print(f"[SUBPROCESS] Set SF_ENV_AGENTS={cfg.env_agents} environment variable")
                    print(f"[SUBPROCESS] DCE config batch_size: {getattr(cfg, 'batch_size', 'not set')}")
                    if cfg.env_agents == 128:
                        print(f"[SUBPROCESS] Using MAXIMUM PARALLELIZATION DCE CONFIG (128 environments)")
                    elif cfg.env_agents == 32:
                        print(f"[SUBPROCESS] Using UPDATED DCE CONFIG (32 environments - high parallelization)")
                    elif cfg.env_agents == 16:
                        print(f"[SUBPROCESS] Using ORIGINAL DCE CONFIG (16 environments - high performance)")
                    elif cfg.env_agents == 6:
                        print(f"[SUBPROCESS] Using MEDIUM CONFIG (6 environments - reduced memory)")
                    elif cfg.env_agents == 4:
                        print(f"[SUBPROCESS] Using MEDIUM CONFIG (4 environments - reduced memory)")
                    elif cfg.env_agents == 1:
                        print(f"[SUBPROCESS] Using LOW CONFIG (1 environment - minimum memory)")
                    else:
                        print(f"[SUBPROCESS] Using CUSTOM CONFIG ({cfg.env_agents} environments)")
                else:
                    print(f"[SUBPROCESS] env_agents={getattr(cfg, 'env_agents', 'not set')}, using default num_envs")
                
                # CRITICAL FIX: Force headless setting for rollout workers based on Sample Factory configuration
                if hasattr(cfg, 'headless') and cfg.headless is not None:
                    dce_config.headless = cfg.headless
                    print(f"[SUBPROCESS] FORCING headless={cfg.headless} for rollout worker")
                    if not cfg.headless:
                        print(f"[SUBPROCESS] FORCING viewer mode for rollout worker")
                else:
                    # Keep the default behavior from DCE task
                    print(f"[SUBPROCESS] Using DCE task default headless setting")
                
                task_registry.register_task("quad_with_obstacles", DCE_RL_Navigation_Task, dce_config)
                # Also register as "dce_navigation_task" for backward compatibility
                task_registry.register_task("dce_navigation_task", DCE_RL_Navigation_Task, dce_config)
                print(f"Registered quad_with_obstacles and dce_navigation_task in subprocess")
            except Exception as e:
                print(f"Failed to register quad_with_obstacles in subprocess: {e}")

    # Create the environment and force 3D action space for inference compatibility
    env = AerialGymVecEnv(task_registry.make_task(task_name=full_task_name), "obs")
    
    # CRITICAL FIX: Force action space to exactly 3D to match inference expectations
    # Override action space again after environment creation to ensure it sticks
    import gymnasium as gym
    import numpy as np
    forced_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    env.action_space = convert_space(forced_action_space)
    
    # Debug: Verify action space is 3D
    print(f"[make_aerialgym_env] Final action space shape: {env.action_space.shape}")
    print(f"[make_aerialgym_env] Action space: {env.action_space}")
    
    return env


def add_extra_params_func(parser):
    """
    Specify extra arguments for this family of environments.
    """
    parser.add_argument("--env_agents", default=None, type=int, help="Num agents in env (multi-agent only)")
    parser.add_argument("--headless", type=lambda x: x.lower() == 'true', default=None, help="Force headless mode (True/False)")
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


def override_default_params_func(env, parser):
    """Most of these parameters are taken from IsaacGymEnvs default config files."""

    # Default parameters for medium configuration (4 environments)
    default_batch_size = 1024
    default_num_batches_per_epoch = 4

    parser.set_defaults(
        # we're using a single very vectorized env, no need to parallelize it further
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        actor_worker_gpus=[0],  # obviously need a GPU
        train_for_env_steps=10000000,
        use_rnn=False,
        adaptive_stddev=True,  # Default for other environments
        policy_initialization="torch_default",
        env_gpu_actions=True,
        env_gpu_observations=True,  # Critical: Tell Sample Factory we're providing GPU tensors
        reward_scale=0.1,
        rollout=32,  # changed to match DCE config
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
        async_rl=True,
        use_env_info_cache=False,  # speeds up startup
        kl_loss_coeff=0.1,
        restart_behavior="resume",  # changed to match DCE config
    )

    # override default config parameters for specific envs
    if env in env_configs:
        parser.set_defaults(**env_configs[env])


# custom default configuration parameters for specific envs
# add more envs here analogously (env names should match config file names in IGE)
env_configs = dict(
    position_setpoint_task=dict(
        train_for_env_steps=131000000000,
        encoder_mlp_layers=[256, 128, 64],
        gamma=0.99,
        rollout=16,
        learning_rate=1e-4,
        lr_schedule_kl_threshold=0.016,
        batch_size=16384,
        num_epochs=4,
        max_grad_norm=1.0,
        num_batches_per_epoch=4,
        exploration_loss_coeff=0.0,
        with_wandb=False,
        wandb_project="quad",
        wandb_user="mihirkulkarni",
    ),
    navigation_task=dict(
        train_for_env_steps=131000000000,
        encoder_mlp_layers=[256, 128, 64],
        use_rnn=True,
        encoder_conv_architecture="convnet_simple",  # "resnet_impala_mihirk",
        rnn_num_layers=1,
        rnn_size=64,
        rnn_type="gru",
        gamma=0.98,
        rollout=32,
        learning_rate=1e-4,
        lr_schedule_kl_threshold=0.016,
        batch_size=1024,
        num_epochs=4,
        max_grad_norm=1.0,
        num_batches_per_epoch=4,
        exploration_loss_coeff=0.0,
        with_wandb=False,
        wandb_project="quad",
        wandb_user="mihirkulkarni",
    ),
    quad_with_obstacles=dict(
        # CRITICAL FIX: Configure for 3D action space to match inference expectations
        # The inference script expects 3D actions, so train with 3D to avoid shape mismatch
        adaptive_stddev=True,  # Can use adaptive_stddev with 3D actions
        action_space_dim=3,  # FORCE 3D action space for inference compatibility
        
        # MODIFIED CONFIGURATION - Single input processing to match old 1333 model
        train_for_env_steps=100000000,  # 100M steps to match original config
        encoder_mlp_layers=[512, 256, 64],  # Match original config
        # REMOVED ConvNet processing - VAE latents handled by DCE task, single input only
        encoder_conv_mlp_layers=[],  # Disable ConvNet encoder for single input processing
        use_rnn=True,
        rnn_num_layers=1,
        rnn_size=64,
        rnn_type="gru",
        recurrence=32,  # Match original config
        gamma=0.98,
        reward_scale=0.1,  # Match original config
        rollout=32,
        learning_rate=0.0003,  # Match original config (3e-4)
        lr_schedule="kl_adaptive_epoch",
        lr_schedule_kl_threshold=0.016,
        # UPDATED CONFIG - 128 environments with optimized batch accumulation
        batch_size=16384,  # 8x batch size for 128 environments (was 2048 for 16 envs)
        num_batches_to_accumulate=1,  # Reduced accumulation for memory optimization
        num_batches_per_epoch=8,  # Keep batches per epoch the same
        num_epochs=4,
        max_grad_norm=1.0,
        exploration_loss_coeff=0.001,  # Match original config
        value_loss_coeff=2.0,  # Match original config
        kl_loss_coeff=0.1,  # Match original config
        normalize_input=True,  # Match original config
        normalize_returns=True,  # Match original config
        async_rl=True,  # Match original config
        serial_mode=True,  # Match original config
        batched_sampling=True,  # Match original config
        num_workers=1,  # Match original config
        num_envs_per_worker=1,  # Match original config
        # UPDATED CONFIG - 128 environments (3D action space for inference compatibility)
        env_agents=128,  # Increased to 128 environments for maximum parallelization
        worker_num_splits=1,  # Match original config
        policy_workers_per_policy=1,  # Match original config
        nonlinearity="elu",  # Match original config
        shuffle_minibatches=False,  # Match original config (was True in defaults)
        gae_lambda=0.95,  # Match original config
        ppo_clip_ratio=0.2,  # Match original config
        ppo_clip_value=1.0,  # Match original config
        with_vtrace=False,  # Match original config
        value_bootstrap=True,  # Match original config
        reward_clip=1000.0,  # Match original config
        obs_subtract_mean=0.0,  # Match original config
        obs_scale=1.0,  # Match original config
        decorrelate_experience_max_seconds=0,  # Match original config
        decorrelate_envs_on_one_worker=True,  # Match original config
        max_policy_lag=1000,  # Match original config  
        vtrace_rho=1.0,  # Match original config
        vtrace_c=1.0,  # Match original config
        lr_adaptive_min=1e-06,  # Match original config
        lr_adaptive_max=0.01,  # Match original config
        save_every_sec=120,  # Regular checkpoint every 2 minutes
        keep_checkpoints=5,  # Keep 5 regular checkpoints (increased for safety)
        save_milestones_sec=-1,  # No milestone saving
        save_best_every_sec=5,  # Check for best model every 5 seconds
        save_best_metric="reward",  # Use reward to determine best model
        save_best_after=100000,  # Save best models after 100K steps (much more reasonable)
        policy_initialization="torch_default",  # Match original config
        policy_init_gain=1.0,  # Match original config
        actor_critic_share_weights=True,  # Match original config
        # adaptive_stddev=False set above to prevent 12D action space doubling
        continuous_tanh_scale=0.0,  # Match original config
        initial_stddev=1.0,  # Match original config
        restart_behavior="resume",  # Match original config (not "overwrite")
        optimizer="adam",  # Match original config
        adam_eps=1e-06,  # Match original config
        adam_beta1=0.9,  # Match original config
        adam_beta2=0.999,  # Match original config
        exploration_loss="entropy",  # Match original config
        decoder_mlp_layers=[],  # Match original config
        env_frameskip=1,  # Match original config
        env_framestack=1,  # Match original config
        pixel_format="CHW",  # Match original config
        use_record_episode_statistics=False,  # Match original config
        normalize_input_keys=None,  # Match original config
        set_workers_cpu_affinity=True,  # Match original config
        force_envs_single_thread=False,  # Match original config
        default_niceness=0,  # Match original config
        log_to_file=True,  # Match original config
        experiment_summaries_interval=10,  # Match original config
        flush_summaries_interval=30,  # Match original config
        stats_avg=100,  # Match original config
        summaries_use_frameskip=True,  # Match original config
        heartbeat_interval=20,  # Match original config
        heartbeat_reporting_interval=180,  # Match original config
        train_for_seconds=10000000000,  # Match original config
        load_checkpoint_kind="latest",  # Match original config
        benchmark=False,  # Match original config
        with_wandb=True,  # Enable Weights & Biases logging
        wandb_project="vae_rl_navigation",  # Match original project name
        wandb_user="ziya-ruso-ucl",  # Your team entity name
        wandb_group="dce_navigation_training",
        wandb_tags=["aerial_gym", "dce", "navigation", "sample_factory"],
        wandb_job_type="SF",  # Match original config
        with_pbt=False,  # Match original config
        pbt_mix_policies_in_one_env=True,  # Match original config
        pbt_period_env_steps=5000000,  # Match original config
        pbt_start_mutation=20000000,  # Match original config
        pbt_replace_fraction=0.3,  # Match original config
        pbt_mutation_rate=0.15,  # Match original config
        pbt_replace_reward_gap=0.1,  # Match original config
        pbt_replace_reward_gap_absolute=1e-06,  # Match original config
        pbt_optimize_gamma=False,  # Match original config
        pbt_target_objective="true_objective",  # Match original config
        pbt_perturb_min=1.1,  # Match original config
        pbt_perturb_max=1.5,  # Match original config
        help=False,  # Match original config
        algo="APPO",  # Match original config  
        device="gpu",  # Match original config
        seed=None,  # Match original config
        num_policies=1,  # Match original config
        actor_worker_gpus=[0],  # Match original config (ensure this is set)
        obs_key="obs",  # Match original config
        subtask=None,  # Match original config
        ige_api_version="preview4",  # Match original config
        eval_stats=False,  # Match original config
    ),
)

# =============================================================================
# DCE CONFIGURATION SCALING COMPARISON
# Current config above uses MAXIMUM PARALLELIZATION DCE CONFIG (128 environments)
# 
# CONFIGURATION COMPARISON TABLE:
# Config Name              | Envs | Batch Size | Accumulate | Effective Batch | Memory
# -------------------------|------|------------|------------|-----------------|--------
# ORIGINAL DCE (1333.322) | 16   | 2048       | 2          | 4096           | Low
# UPDATED DCE              | 32   | 4096       | 2          | 8192           | Medium  
# MAXIMUM DCE (Current)    | 128  | 16384      | 1          | 16384          | High
# 
# All configurations maintain the same core training parameters (3D actions, 81D obs, etc.)
# =============================================================================
#
# CURRENT ACTIVE CONFIG (128 environments) - MAXIMUM PARALLELIZATION DCE CONFIG:
# env_agents=128             # 128 environments (8x original for maximum parallelization)
# batch_size=16384           # 8x batch size for 128 environments
# num_batches_to_accumulate=1 # Reduced accumulation for memory optimization
# num_batches_per_epoch=8    # Keep batches per epoch the same
# Effective Batch Size=16384  # 16384 * 1 = 16384 (4x original 4096, memory optimized)
#
# ORIGINAL CONFIG (16 environments) - ORIGINAL DCE CONFIG (1333.322 reward):
# env_agents=16              # 16 environments (original successful model)
# batch_size=2048            # Original batch size
# num_batches_to_accumulate=2 # Original accumulation  
# num_batches_per_epoch=8    # Original batches per epoch
# Effective Batch Size=4096   # 2048 * 2 = 4096 (original)
# curriculum.min_level=36    # Original curriculum level
# curriculum.max_level=50    # Original max level
# action_space_dim=3         # 3D actions (x_vel, y_vel, yaw_rate)
# observation_space_dim=81   # 17D basic state + 64D VAE latents
# environment="quad_with_obstacles" # Forest environment with obstacles
# robot="lmf2"              # LMF2 quadrotor with velocity control
# controller="lmf2_velocity_control" # Velocity control
#
# PREVIOUS CONFIG (32 environments) - UPDATED DCE CONFIG:
# env_agents=32              # 32 environments (2x original)
# batch_size=4096            # 2x batch size for 32 environments
# num_batches_to_accumulate=2 # Same accumulation as original
# num_batches_per_epoch=8    # Same batches per epoch
# Effective Batch Size=8192   # 4096 * 2 = 8192 (2x original 4096)
#
# MEDIUM CONFIG (6 environments) - Reduced Memory Usage:
# env_agents=6
# batch_size=1536
# num_batches_to_accumulate=2
# num_batches_per_epoch=4
#
# LOW CONFIG (1 environment) - Minimum Memory Usage:
# env_agents=1  
# batch_size=512
# num_batches_to_accumulate=4
# num_batches_per_epoch=2


# CustomEncoder removed - DCE task handles VAE encoding internally and provides 81-dimensional observations


def register_aerialgym_custom_components():
    # Clear cached environment info for single agent mode to prevent mismatch
    import os
    import glob
    cache_files = glob.glob("/tmp/sf2_*/env_info_quad_with_obstacles*")
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
            print(f"Cleared cache file: {cache_file}")
        except:
            pass
    
    # Use environment variable from shell script if set, otherwise default to 128 (maximum parallelization config)
    # This will be updated based on the actual env_agents parameter when env is created
    current_env_agents = os.environ.get('SF_ENV_AGENTS', '128')  # Default to maximum parallelization DCE configuration
    os.environ['SF_ENV_AGENTS'] = current_env_agents
    if current_env_agents == '128':
        print(f"Set SF_ENV_AGENTS={current_env_agents} environment variable for all processes (MAXIMUM PARALLELIZATION DCE CONFIG)")
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
        dce_config.curriculum.min_level = 30  # DCE curriculum starts from level 30
        dce_config.curriculum.max_level = 50  # DCE curriculum goes up to level 50
        
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
    except Exception as e:
        print(f"Warning: Could not register quad_with_obstacles: {e}")
    
    for env_name in env_configs:
        register_env(env_name, make_aerialgym_env)

    # Don't register custom encoder since DCE task handles VAE encoding internally


def parse_aerialgym_cfg(evaluation=False):
    parser, partial_cfg = parse_sf_args(evaluation=evaluation)
    add_extra_params_func(parser)
    override_default_params_func(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser)
    return final_cfg


def main():
    """Script entry point."""
    register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
