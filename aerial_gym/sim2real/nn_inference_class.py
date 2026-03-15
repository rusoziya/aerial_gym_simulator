from __future__ import annotations

import time
from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from gymnasium import spaces

from torch import nn


class Sim2RealInferenceClass(nn.Module):
    def __init__(self, num_envs, num_actions, num_obs, cfg: Config) -> None:
        super().__init__()
        self.cfg = load_from_checkpoint(cfg)
        self.cfg.num_envs = num_envs
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.num_agents = num_envs
        self.observation_space = spaces.Dict(
            dict(
                observations=convert_space(
                    spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
                )
            )
        )
        self.action_space = convert_space(
            spaces.Box(np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0)
        )
        self.init_env_info()
        self.actor_critic = create_actor_critic(self.cfg, self.observation_space, self.action_space)
        self.actor_critic.eval()
        self.device = torch.device("cpu" if self.cfg.device == "cpu" else "cuda")
        self.actor_critic.model_to_device(self.device)
        print("Model:\n\n", self.actor_critic)
        # Load policy into model
        policy_id = self.cfg.policy_index
        name_prefix = dict(latest="checkpoint", best="best")[self.cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(
            Learner.checkpoint_dir(self.cfg, policy_id), f"{name_prefix}_*"
        )
        checkpoint_dict = Learner.load_checkpoint(checkpoints, self.device)
        self.actor_critic.load_state_dict(checkpoint_dict["model"])
        self.rnn_states = torch.zeros(
            [self.num_agents, get_rnn_size(self.cfg)],
            dtype=torch.float32,
            device=self.device,
        )

    def init_env_info(self) -> None:
        self.env_info = EnvInfo(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_agents=self.num_agents,
            gpu_actions=self.cfg.env_gpu_actions,
            gpu_observations=self.cfg.env_gpu_observations,
            action_splits=None,
            all_discrete=None,
            frameskip=self.cfg.env_frameskip,
        )

    def reset(self, env_ids) -> None:
        self.rnn_states[env_ids] = 0.0

    def get_action(self, obs, get_np=False, get_robot_zero=False) -> None:
        with torch.no_grad():
            # put obs to device
            import os
            disable_norm = os.environ.get("DISABLE_NORM", "false").lower() == "true"
            norm_tap = os.environ.get("NORM_TAP_DEBUG", "false").lower() == "true"
            if norm_tap:
                try:
                    vec = obs.get('observations', None)
                    if isinstance(vec, torch.Tensor) and vec.ndim == 2 and vec.shape[1] >= 150:
                        z_e = vec[:, 22:86]; z_s = vec[:, 86:150]
                        print(f"[NORM_TAP] pre abs_mean: drone(22:86)={float(z_e.abs().mean().item()):.6e} static(86:150)={float(z_s.abs().mean().item()):.6e}")
                except (RuntimeError, TypeError, KeyError, IndexError):
                    pass
            if disable_norm:
                processed_obs = obs
                if norm_tap:
                    print("[NORM_TAP] normalization disabled: post == pre (bypassed)")
            else:
                processed_obs = prepare_and_normalize_obs(self.actor_critic, obs)
                if norm_tap:
                    try:
                        pvec = processed_obs.get('observations', None)
                        if isinstance(pvec, torch.Tensor) and pvec.ndim == 2 and pvec.shape[1] >= 150:
                            z_e2 = pvec[:, 22:86]; z_s2 = pvec[:, 86:150]
                            print(f"[NORM_TAP] post abs_mean: drone(22:86)={float(z_e2.abs().mean().item()):.6e} static(86:150)={float(z_s2.abs().mean().item()):.6e}")
                    except (RuntimeError, TypeError, KeyError, IndexError):
                        pass
            policy_outputs = self.actor_critic(processed_obs, self.rnn_states)
            # sample actions from the distribution by default
            actions = policy_outputs["actions"]
            if self.cfg.eval_deterministic:
                action_distribution = self.actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(self.env_info, actions)

            self.rnn_states = policy_outputs["new_rnn_states"]
        if get_robot_zero:
            actions = actions[0]
        if get_np:
            return actions.cpu().numpy()
        return actions
