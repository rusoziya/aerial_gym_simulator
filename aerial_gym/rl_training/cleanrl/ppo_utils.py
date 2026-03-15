from __future__ import annotations

# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os

import gym
import numpy as np
import torch
import torch.nn as nn

from aerial_gym.utils.helpers import parse_arguments


def get_args() -> argparse.Namespace:
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "position_setpoint_task",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "default": "ppo_continuous_action",
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": str,
            "default": None,
            "help": "Saved model checkpoint number.",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": 2048,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--seed",
            "type": int,
            "default": 1,
            "help": "Random seed. Overrides config file if provided.",
        },
        {
            "name": "--play",
            "required": False,
            "help": "only run network",
            "action": "store_true",
        },
        {
            "name": "--torch-deterministic-off",
            "action": "store_true",
            "default": False,
            "help": "if toggled, `torch.backends.cudnn.deterministic=False`",
        },
        {
            "name": "--track",
            "action": "store_true",
            "default": False,
            "help": "if toggled, this experiment will be tracked with Weights and Biases",
        },
        {
            "name": "--wandb-project-name",
            "type": str,
            "default": "cleanRL",
            "help": "the wandb's project name",
        },
        {
            "name": "--wandb-entity",
            "type": str,
            "default": None,
            "help": "the entity (team) of wandb's project",
        },
        {
            "name": "--total-timesteps",
            "type": int,
            "default": 3000000000,
            "help": "total timesteps of the experiments",
        },
        {
            "name": "--learning-rate",
            "type": float,
            "default": 0.0026,
            "help": "the learning rate of the optimizer",
        },
        {
            "name": "--num-steps",
            "type": int,
            "default": 32,
            "help": "the number of steps to run in each environment per policy rollout",
        },
        {
            "name": "--anneal-lr",
            "action": "store_true",
            "default": False,
            "help": "Toggle learning rate annealing for policy and value networks",
        },
        {
            "name": "--gamma",
            "type": float,
            "default": 0.99,
            "help": "the discount factor gamma",
        },
        {
            "name": "--gae-lambda",
            "type": float,
            "default": 0.95,
            "help": "the lambda for the general advantage estimation",
        },
        {
            "name": "--num-minibatches",
            "type": int,
            "default": 2,
            "help": "the number of mini-batches",
        },
        {
            "name": "--update-epochs",
            "type": int,
            "default": 4,
            "help": "the K epochs to update the policy",
        },
        {
            "name": "--norm-adv-off",
            "action": "store_true",
            "default": False,
            "help": "Toggles advantages normalization",
        },
        {
            "name": "--clip-coef",
            "type": float,
            "default": 0.2,
            "help": "the surrogate clipping coefficient",
        },
        {
            "name": "--clip-vloss",
            "action": "store_true",
            "default": False,
            "help": "Toggles whether or not to use a clipped loss for the value function, as per the paper.",
        },
        {
            "name": "--ent-coef",
            "type": float,
            "default": 0.0,
            "help": "coefficient of the entropy",
        },
        {
            "name": "--vf-coef",
            "type": float,
            "default": 2,
            "help": "coefficient of the value function",
        },
        {
            "name": "--max-grad-norm",
            "type": float,
            "default": 1,
            "help": "the maximum norm for the gradient clipping",
        },
        {
            "name": "--target-kl",
            "type": float,
            "default": None,
            "help": "the target KL divergence threshold",
        },
    ]

    args = parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    args.torch_deterministic = not args.torch_deterministic_off
    args.norm_adv = not args.norm_adv_off

    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env: gym.Env, device: str) -> None:
        super().__init__(env)
        self.num_envs: int = env.num_envs
        self.device: str = device
        self.episode_returns: torch.Tensor | None = None
        self.episode_lengths: torch.Tensor | None = None
        self.returned_episode_returns: torch.Tensor = torch.zeros(0)
        self.returned_episode_lengths: torch.Tensor = torch.zeros(0)

    def reset(self, **kwargs: object) -> torch.Tensor:
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.returned_episode_lengths = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        return observations

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        observations, rewards, terminations, truncations, infos = super().step(action)

        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        dones = torch.where(terminations | truncations, 1, 0).to(self.device)
        self.episode_returns *= 1 - dones
        self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
