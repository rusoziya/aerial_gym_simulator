from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import os
from sample_factory.algo.utils.context import global_model_factory
from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net_gate import (
    make_dual_fusion_encoder,
)

from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from gymnasium import spaces


class NN_Inference_Class_Gate(nn.Module):
    def __init__(self, num_envs: int, num_actions: int, num_obs: int, cfg: Config) -> None:
        super().__init__()
        # Prefer direct file from DCE_MODEL; otherwise load config from train_dir/experiment
        dce_model_path = os.environ.get("DCE_MODEL", None)
        self._load_from_file = bool(dce_model_path and os.path.exists(dce_model_path))
        self._dce_model_path = dce_model_path if self._load_from_file else None
        # Load config from train_dir/experiment when available (ensures 1:1 arch/hparams with training),
        # even if we load weights from DCE_MODEL. Fallback to provided cfg if not available.
        try:
            has_exp = bool(cfg.train_dir) and bool(cfg.experiment)
        except Exception:
            has_exp = False
        if has_exp:
            try:
                self.cfg = load_from_checkpoint(cfg)
            except Exception:
                self.cfg = cfg
        else:
            self.cfg = cfg if self._load_from_file else load_from_checkpoint(cfg)
        # Unify observation key expected by the model/normalizer
        self.cfg.obs_key = "obs"
        # Ensure evaluation toggles from CLI are honored after loading checkpoint config
        if True:
            self.cfg.eval_deterministic = bool(cfg.eval_deterministic)
        if True and cfg.fusion is not None:
            self.cfg.fusion = cfg.fusion
        if True and cfg.gate_per_feature is not None:
            self.cfg.gate_per_feature = cfg.gate_per_feature
        # One-time report of evaluation mode for clarity
        print(f"[EVAL_MODE] eval_deterministic={bool(self.cfg.eval_deterministic)}"
              f" | deterministic=greedy argmax (no sampling), stochastic=samples from policy distribution")
        self.cfg.num_envs = num_envs
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.num_agents = num_envs

        # Minimal dict observation space with flat vector "obs"
        self.observation_space = spaces.Dict(
            dict(
                obs=convert_space(
                    spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
                )
            )
        )
        self.action_space = convert_space(
            spaces.Box(np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0)
        )

        # Ensure the gate encoder architecture is registered to match checkpoint keys
        global_model_factory().register_encoder_factory(make_dual_fusion_encoder)

        # Observation/action spaces
        self.init_env_info()
        self.actor_critic = create_actor_critic(self.cfg, self.observation_space, self.action_space)
        self.actor_critic.eval()
        self.device = torch.device("cpu" if str(self.cfg.device).lower() == "cpu" else "cuda")
        self.actor_critic.model_to_device(self.device)

        # Load policy weights
        if self._load_from_file:
            # Ensure obs key matches our constructed space
            self.cfg.obs_key = "obs"
            checkpoint_dict = torch.load(self._dce_model_path, map_location=self.device)
            # Strict key/shape check before loading
            try:
                model_state = self.actor_critic.state_dict()
                ckpt_state = checkpoint_dict.get("model", {})
                missing = [k for k in model_state.keys() if k not in ckpt_state]
                unexpected = [k for k in ckpt_state.keys() if k not in model_state]
                shape_mismatch = [
                    (k, tuple(ckpt_state[k].shape), tuple(model_state[k].shape))
                    for k in model_state.keys() if k in ckpt_state and ckpt_state[k].shape != model_state[k].shape
                ]
                if missing or unexpected or shape_mismatch:
                    print("[STRICT_LOAD] State dict mismatch detected:")
                    if missing:
                        print(f"  Missing keys in checkpoint: {missing}")
                    if unexpected:
                        print(f"  Unexpected keys in checkpoint: {unexpected}")
                    if shape_mismatch:
                        print(f"  Shape mismatches: {shape_mismatch}")
                    raise RuntimeError("Strict state_dict check failed for loaded checkpoint.")
                self.actor_critic.load_state_dict(ckpt_state, strict=True)
            except Exception:
                # Re-raise after an attempt to load to provide default error
                self.actor_critic.load_state_dict(checkpoint_dict["model"], strict=True)
        else:
            policy_id = self.cfg.policy_index
            name_prefix = dict(latest="checkpoint", best="best")[self.cfg.load_checkpoint_kind]
            checkpoints = Learner.get_checkpoints(
                Learner.checkpoint_dir(self.cfg, policy_id), f"{name_prefix}_*"
            )
            checkpoint_dict = Learner.load_checkpoint(checkpoints, self.device)
            # Strict key/shape check before loading
            model_state = self.actor_critic.state_dict()
            ckpt_state = checkpoint_dict.get("model", {})
            missing = [k for k in model_state.keys() if k not in ckpt_state]
            unexpected = [k for k in ckpt_state.keys() if k not in model_state]
            shape_mismatch = [
                (k, tuple(ckpt_state[k].shape), tuple(model_state[k].shape))
                for k in model_state.keys() if k in ckpt_state and ckpt_state[k].shape != model_state[k].shape
            ]
            if missing or unexpected or shape_mismatch:
                print("[STRICT_LOAD] State dict mismatch detected:")
                if missing:
                    print(f"  Missing keys in checkpoint: {missing}")
                if unexpected:
                    print(f"  Unexpected keys in checkpoint: {unexpected}")
                if shape_mismatch:
                    print(f"  Shape mismatches: {shape_mismatch}")
                raise RuntimeError("Strict state_dict check failed for loaded checkpoint.")
            self.actor_critic.load_state_dict(ckpt_state, strict=True)

        self.rnn_states = torch.zeros(
            [self.num_agents, get_rnn_size(self.cfg)], dtype=torch.float32, device=self.device
        )
        # Notice flag for ABLATE_ZERO_RNN to avoid spamming
        self._abl_zero_rnn_notice_shown = False

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
        if env_ids is None:
            self.rnn_states.zero_()
            return
        idx = torch.as_tensor(env_ids, dtype=torch.long, device=self.rnn_states.device).view(-1)
        # Robustly zero only the requested environments along the first dim
        self.rnn_states.index_fill_(0, idx, 0.0)

    def get_action(self, obs: Dict, get_np: bool = False, get_robot_zero: bool = False) -> None:
        with torch.no_grad():
            # Optional pre/post-normalization latent taps for debugging
            norm_tap = os.environ.get("NORM_TAP_DEBUG", "false").lower() == "true"
            disable_norm = os.environ.get("DISABLE_NORM", "false").lower() == "true"
            if norm_tap:
                vec = obs.get(self.cfg.obs_key, None)
                if isinstance(vec, torch.Tensor) and vec.ndim == 2 and vec.shape[1] >= 150:
                    z_e = vec[:, 22:86]
                    z_s = vec[:, 86:150]
                    print(f"[NORM_TAP] pre abs_mean: drone(22:86)={float(z_e.abs().mean().item()):.6e} static(86:150)={float(z_s.abs().mean().item()):.6e}")

            if disable_norm:
                processed_obs = obs
                if norm_tap:
                    print("[NORM_TAP] normalization disabled: post == pre (bypassed)")
            else:
                processed_obs = prepare_and_normalize_obs(self.actor_critic, obs)
                if norm_tap:
                    pvec = processed_obs.get(self.cfg.obs_key, None)
                    if isinstance(pvec, torch.Tensor) and pvec.ndim == 2 and pvec.shape[1] >= 150:
                        z_e2 = pvec[:, 22:86]
                        z_s2 = pvec[:, 86:150]
                        print(f"[NORM_TAP] post abs_mean: drone(22:86)={float(z_e2.abs().mean().item()):.6e} static(86:150)={float(z_s2.abs().mean().item()):.6e}")
            policy_outputs = self.actor_critic(processed_obs, self.rnn_states)
            actions = policy_outputs["actions"]
            if self.cfg.eval_deterministic:
                action_distribution = self.actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)

            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(self.env_info, actions)

            self.rnn_states = policy_outputs["new_rnn_states"]

        if get_robot_zero:
            actions = actions[0]
        if get_np:
            return actions.cpu().numpy()
        return actions


