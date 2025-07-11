#!/usr/bin/env python3

"""
Training script for DCE RL Navigation Task

This script trains the DCE navigation task using Sample Factory with image observations (depth + segmentation).
Based on train_aerialgym_custom_net.py but specifically configured for DCE navigation.
"""

# Import isaacgym first to avoid conflicts
import isaacgym
import sys
from typing import Dict, Optional, Tuple

import gymnasium as gym
import torch
from torch import Tensor

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.encoder import *
from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.utils.typing import Config, Env
from sample_factory.utils.utils import str2bool

# Import aerial gym components
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import DCE_RL_Navigation_Task

import numpy as np
import torch.nn as nn
from gymnasium import spaces
from typing import List


class AerialGymVecEnv(gym.Env):
    """
    Wrapper for isaacgym environments to make them compatible with sample factory.
    """

    def __init__(self, aerialgym_env, obs_key):
        self.env = aerialgym_env
        self.num_agents = self.env.num_envs
        self.action_space = convert_space(self.env.action_space)

        # isaacgym_examples environments actually return dicts
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
    # Register the DCE navigation task in each subprocess
    if "dce_navigation_task" not in task_registry.task_class_registry:
        task_registry.register_task(
            task_name="dce_navigation_task",
            task_class=DCE_RL_Navigation_Task,
            task_config=task_registry.get_task_config("navigation_task"),
        )
    
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
    """Configure default parameters specifically for DCE navigation training."""
    parser.set_defaults(
        # Basic training parameters
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        actor_worker_gpus=[0],
        train_for_env_steps=10000000,  # 10M steps for DCE navigation
        
        # Network architecture for image processing
        use_rnn=True,
        encoder_conv_architecture="convnet_simple",  # Enable conv encoder for images
        encoder_mlp_layers=[256, 128, 64],  # Match working navigation task config
        rnn_num_layers=1,
        rnn_size=64,
        rnn_type="gru",
        
        # Training hyperparameters optimized for DCE navigation
        adaptive_stddev=True,
        policy_initialization="torch_default",
        env_gpu_actions=True,
        reward_scale=0.1,
        rollout=32,  # Longer rollouts for better temporal understanding
        max_grad_norm=1.0,
        batch_size=1024,
        num_batches_per_epoch=4,
        num_epochs=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=2.0,
        exploration_loss_coeff=0.0,
        nonlinearity="elu",
        learning_rate=1e-4,  # Lower learning rate for stable image training
        lr_schedule="kl_adaptive_epoch",
        lr_schedule_kl_threshold=0.016,
        shuffle_minibatches=True,
        gamma=0.98,
        gae_lambda=0.95,
        with_vtrace=False,
        value_bootstrap=True,
        normalize_input=True,
        normalize_returns=True,
        save_best_after=int(1e6),
        serial_mode=True,
        async_rl=True,
        use_env_info_cache=False,
        kl_loss_coeff=0.1,
        restart_behavior="overwrite",
        
        # Logging and experiment tracking  
        with_wandb=False,
        wandb_project="dce_navigation",
        wandb_user="aerial_gym",
    )


class CustomEncoder(Encoder):
    """Custom encoder for DCE navigation task with image observations."""

    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        out_size = 0
        self.encoders = nn.ModuleDict()
        out_size += obs_space["observations"].shape[0]

        encoder_fn_image = make_img_encoder
        self.encoders["image_obs"] = encoder_fn_image(cfg, obs_space["image_obs"])
        out_size += self.encoders["image_obs"].get_out_size()

        obs_space_custom = spaces.Box(np.ones(out_size) * -np.Inf, np.ones(out_size) * np.Inf)
        mlp_layers: List[int] = cfg.encoder_mlp_layers
        self.mlp_head_custom = create_mlp(mlp_layers, obs_space_custom.shape[0], nonlinearity(cfg))
        if len(mlp_layers) > 0:
            self.mlp_head_custom = torch.jit.script(self.mlp_head_custom)
        self.encoder_out_size = calc_num_elements(self.mlp_head_custom, obs_space_custom.shape)

    def forward(self, obs_dict):
        x_image_encoding = self.encoders["image_obs"](obs_dict["image_obs"])
        encoding = self.mlp_head_custom(torch.cat((obs_dict["observations"], x_image_encoding), 1))
        return encoding

    def get_out_size(self) -> int:
        return self.encoder_out_size


def make_custom_encoder(cfg: Config, obs_space) -> Encoder:
    """Factory function as required by the API."""
    return CustomEncoder(cfg, obs_space)


def register_aerialgym_custom_components():
    register_env("dce_navigation_task", make_aerialgym_env)
    global_model_factory().register_encoder_factory(make_custom_encoder)


def parse_aerialgym_cfg(evaluation=False):
    parser, _ = parse_sf_args(evaluation=evaluation)
    add_extra_params_func(parser)
    override_default_params_func("dce_navigation_task", parser)
    cfg = parse_full_cfg(parser)
    
    # Set experiment directory
    cfg.train_dir = "./train_dir/dce_navigation"
    cfg.experiment = "dce_navigation_experiment"
    
    return cfg


def main():
    """Main entry point for DCE navigation training."""
    
    # Register the DCE navigation task
    task_registry.register_task(
        task_name="dce_navigation_task",
        task_class=DCE_RL_Navigation_Task,
        task_config=task_registry.get_task_config("navigation_task"),
    )
    
    # Register custom components
    register_aerialgym_custom_components()
    
    # Parse configuration
    cfg = parse_aerialgym_cfg()
    
    # Set the environment name
    cfg.env = "dce_navigation_task"
    
    # Start training
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main()) 