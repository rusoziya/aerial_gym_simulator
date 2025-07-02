# Training script for DCE navigation task - OPTIMIZED configuration
# This script uses memory-optimized configuration with 6 environments for ~90% VRAM usage
# 
# DCE Navigation Task Specifics:
# - Action space: 3D (x_vel, y_vel, yaw_rate) - reduced from base 4D navigation task
# - Observation space: 81D total = 17D basic state + 64D VAE-encoded image latents
#   * 0-3: normalized vector to target + distance to target / 5.0
#   * 4-6: euler angles (roll, pitch, 0.0)  
#   * 7-9: robot body linear velocity
#   * 10-12: robot body angular velocity
#   * 13-16: robot actions (last action taken)
#   * 17-80: VAE-encoded depth image latents (64 dimensions)
# - Curriculum: starts at level 36 (higher than base navigation task)
# - 6 parallel environments (1 agent per environment) in 2x3 grid layout
# - Uses LMF2 robot with velocity control
# - Batch size: 1024, 4 batches per epoch, 3 batches to accumulate
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
    """

    def __init__(self, aerialgym_env, obs_key):
        self.env = aerialgym_env
        self.num_agents = self.env.num_envs
        self.is_multiagent = True  # Fix: bypass BatchedMultiAgentWrapper to avoid tensor conversion error
        self.action_space = convert_space(self.env.action_space)

        # Aerial Gym examples environments actually return dicts
        if obs_key == "obs":
            self.observation_space = gym.spaces.Dict(convert_space(self.env.observation_space))
        else:
            raise ValueError(f"Unknown observation key: {obs_key}")

        self._truncated: Tensor = torch.zeros(self.num_agents, dtype=torch.bool)

    def reset(self, *args, **kwargs) -> Tuple[Dict[str, Tensor], Dict]:
        # some IGE envs return all zeros on the first timestep, but this is probably okay
        obs, rew, terminated, truncated, infos = self.env.reset()
        return obs, infos

    def step(self, action) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Dict]:
        obs, rew, terminated, truncated, infos = self.env.step(action)
        return obs, rew, terminated, truncated, infos

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
                dce_config.curriculum.min_level = 36  # DCE uses higher min curriculum level
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
                    if cfg.env_agents == 1:
                        print(f"[SUBPROCESS] Using single environment (low memory) configuration")
                    elif cfg.env_agents == 4:
                        print(f"[SUBPROCESS] Using medium configuration (4 environments)")
                    elif cfg.env_agents == 6:
                        print(f"[SUBPROCESS] Using OPTIMIZED configuration (6 environments)")
                    elif cfg.env_agents == 8:
                        print(f"[SUBPROCESS] Using high performance configuration (8 environments)")
                    elif cfg.env_agents == 16:
                        print(f"[SUBPROCESS] Using original DCE configuration (16 environments)")
                    else:
                        print(f"[SUBPROCESS] Using custom {cfg.env_agents} environment configuration")
                else:
                    print(f"[SUBPROCESS] env_agents={getattr(cfg, 'env_agents', 'not set')}, using default num_envs")
                task_registry.register_task("quad_with_obstacles", DCE_RL_Navigation_Task, dce_config)
                # Also register as "dce_navigation_task" for backward compatibility
                task_registry.register_task("dce_navigation_task", DCE_RL_Navigation_Task, dce_config)
                print(f"Registered quad_with_obstacles and dce_navigation_task in subprocess")
            except Exception as e:
                print(f"Failed to register quad_with_obstacles in subprocess: {e}")

    return AerialGymVecEnv(task_registry.make_task(task_name=full_task_name), "obs")


def add_extra_params_func(parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument(
        "--env_agents",
        default=-1,
        type=int,
        help="Num agents in each env (default: -1, means use default value from isaacgymenvs env yaml config file)",
    )
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
        adaptive_stddev=True,
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
        # OPTIMIZED CONFIGURATION - Memory-optimized for ~90% VRAM usage, 6 environments
        train_for_env_steps=100000000,  # 100M steps to match original config
        encoder_mlp_layers=[512, 256, 64],  # Match original config
        encoder_conv_architecture="convnet_simple",
        encoder_conv_mlp_layers=[512],  # Match original config
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
        # OPTIMIZED CONFIG (6 environments) - Memory-optimized for ~90% VRAM usage
        batch_size=1536,
        num_batches_to_accumulate=2,
        num_batches_per_epoch=4,
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
        # OPTIMIZED CONFIG - 6 environments (memory-optimized for ~90% VRAM usage)
        env_agents=6,
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
        save_every_sec=120,  # Match original config
        keep_checkpoints=2,  # Match original config
        save_milestones_sec=-1,  # Match original config
        save_best_every_sec=5,  # Match original config
        save_best_metric="reward",  # Match original config
        save_best_after=5000000,  # Match original config
        policy_initialization="torch_default",  # Match original config
        policy_init_gain=1.0,  # Match original config
        actor_critic_share_weights=True,  # Match original config
        adaptive_stddev=True,  # Match original config
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
# ALTERNATIVE CONFIGURATIONS (commented out)
# Uncomment and modify the env_agents and batch config above if needed
# =============================================================================
#
# HIGH CONFIG (8 environments) - High Performance (requires more GPU memory):
# env_agents=8
# batch_size=2048
# num_batches_to_accumulate=2
# num_batches_per_epoch=8
#
# ORIGINAL DCE CONFIG (16 environments) - Maximum Performance:
# env_agents=16
# batch_size=2048
# num_batches_to_accumulate=2
# num_batches_per_epoch=8
#
# LOW CONFIG (1 environment) - Low Memory Usage:
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
    
    # Use environment variable from shell script if set, otherwise default to 6 (optimized config)
    # This will be updated based on the actual env_agents parameter when env is created
    current_env_agents = os.environ.get('SF_ENV_AGENTS', '6')  # Default to optimized configuration
    os.environ['SF_ENV_AGENTS'] = current_env_agents
    if current_env_agents == '6':
        print(f"Set SF_ENV_AGENTS={current_env_agents} environment variable for all processes (OPTIMIZED config)")
    elif current_env_agents == '4':
        print(f"Set SF_ENV_AGENTS={current_env_agents} environment variable for all processes (MEDIUM config)")
    else:
        print(f"Set SF_ENV_AGENTS={current_env_agents} environment variable for all processes (CUSTOM config)")
    
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
        dce_config.curriculum.min_level = 36  # DCE uses higher min curriculum level
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
