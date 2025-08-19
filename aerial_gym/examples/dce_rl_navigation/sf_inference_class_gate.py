"""
Sample Factory inference class for DCE navigation with gate environment - POSITION-AWARE NAVIGATION
This class provides trained model inference for DCE navigation tasks with gate navigation using velocity control.

ENHANCED: Position-aware navigation with drone absolute position and full yaw sensing
- Includes drone absolute position in world coordinates (3D)
- Includes static camera position and orientation relative to drone (6D)
- Includes full drone orientation with yaw sensing (3D instead of 2D)
- Tests navigation to gate center using complete spatial awareness
- Observation space: 150D (3D drone position + 6D static camera pose + 3D full orientation + 9D state + 64D drone VAE + 64D static camera VAE)

The class is specifically designed to interface with trained Sample Factory models and:
- Uses 4D action space matching the training configuration [x_vel, y_vel, z_vel, yaw_rate]
- Processes 150D observations with complete drone state and spatial awareness  
- Outputs 4D actions directly compatible with DCE gate navigation task

OBSERVATION STRUCTURE (150D):
- [0:3] = Drone absolute position (x, y, z in world coordinates)
- [3:6] = Static camera position relative to drone (x, y, z in drone's reference frame)
- [6:9] = Static camera orientation relative to drone (roll, pitch, yaw in drone's reference frame)
- [9:12] = Drone full orientation including yaw (roll, pitch, yaw)
- [12:15] = Drone linear velocity in body frame
- [15:18] = Drone angular velocity in body frame
- [18:22] = Drone actions (4D for velocity controller)
- [22:86] = Drone camera VAE latents (64D)
- [86:150] = Static camera VAE latents (64D)

TRAINING COMPATIBILITY:
- Compatible with models trained using train_aerialgym_custom_net_gate.py with 150D observations
- Requires models trained after the position-aware navigation upgrade

Usage:
    inference = SampleFactoryInferenceGateNew(num_envs=1, action_space_dim=4, obs_space_dim=150, cfg=config)
    inference.load_model("/path/to/model.pth")
    action = inference.get_action_deterministic(observation)
"""

import time
import copy
from typing import Dict, Any, Union
import torch
import torch.nn as nn
import numpy as np

from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
# Version-compatible AttrDict import across SF versions
try:
    from sample_factory.utils.attr_dict import AttrDict  # SF2+
except Exception:
    try:
        from sample_factory.utils.utils import AttrDict  # older SF
    except Exception:
        class AttrDict(dict):
            def __getattr__(self, k):
                return self[k]
            __setattr__ = dict.__setitem__


class NN_Inference_Class:
    """
    Sample Factory inference class for gate navigation with static camera pose observations.
    
    Handles models trained with 150D observation space for gate navigation with position awareness.
    """

    def __init__(self, num_envs, action_dim, obs_dim, cfg):
        """
        Initialize inference class for gate navigation.
        
        Args:
            num_envs: Number of parallel environments
            action_dim: Action space dimension (should be 4 for gate navigation)
            obs_dim: Observation space dimension (should be 150 for position-aware gate navigation)
            cfg: Sample Factory configuration
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_envs = int(num_envs)
        
        # Store Sample Factory config
        self.cfg = AttrDict(cfg) if isinstance(cfg, dict) else cfg
        
        # CRITICAL: Action space configuration for gate navigation (x, y, z, yaw_rate)
        self.action_space_dim = int(action_dim)
        print(f"[NN_Inference_Class] Configured for {self.action_space_dim}D action space (gate navigation)")
        
        # Observation space configuration (150D for gate navigation with position awareness)
        self.obs_space_dim = int(obs_dim)
        print(f"[NN_Inference_Class] Configured for {self.obs_space_dim}D observation space (position-aware gate navigation)")
        
        # Model / state placeholders
        self.model = None
        self.rnn_states = None  # shape: [num_envs, rnn_size]
        self.is_model_loaded = False
        
        print(f"[NN_Inference_Class] Initialized for inference with device: {self.device}")

    def load_model(self, model_path: str):
        """
        Load a trained Sample Factory model for inference.
        
        Args:
            model_path: Path to the trained model checkpoint file
        """
        try:
            print(f"[NN_Inference_Class] Loading model from: {model_path}")

            # Load checkpoint first to infer architecture if needed
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Infer dimensions from checkpoint or env overrides
            inferred_rnn = None
            inferred_head = None
            try:
                # GRU gate stacking: weight_ih_l0 has shape [3*rnn_size, input_size]
                w_ih = state_dict.get('core.core.weight_ih_l0', None)
                if w_ih is not None and hasattr(w_ih, 'shape') and len(w_ih.shape) == 2:
                    inferred_rnn = int(w_ih.shape[0] // 3)
                # Encoder head last linear: mlp_head.<idx>.weight has shape [head_dim, prev]
                # Try a few common indices
                for idx in (4, 2, 0):
                    key = f'encoder.encoders.obs.mlp_head.{idx}.weight'
                    if key in state_dict:
                        inferred_head = int(state_dict[key].shape[0])
                        break
            except Exception:
                pass

            # Env var overrides
            import os
            env_rnn = os.environ.get('DCE_RNN_SIZE', '').strip()
            env_head = os.environ.get('DCE_HEAD_DIM', '').strip()
            if env_rnn.isdigit():
                inferred_rnn = int(env_rnn)
            if env_head.isdigit():
                inferred_head = int(env_head)

            # Prepare cfg copy and override if we inferred something
            cfg = self.cfg
            # Some checkpoints were trained with rnn_size=64 and head=64
            if inferred_rnn is not None and hasattr(cfg, 'rnn_size'):
                try:
                    print(f"[NN_Inference_Class] Overriding rnn_size: {getattr(cfg, 'rnn_size', None)} -> {inferred_rnn}")
                except Exception:
                    pass
                setattr(cfg, 'rnn_size', inferred_rnn)
                setattr(cfg, 'use_rnn', True)
            if inferred_head is not None and hasattr(cfg, 'encoder_mlp_layers'):
                try:
                    layers = list(getattr(cfg, 'encoder_mlp_layers', [512, 256, inferred_head]))
                except Exception:
                    layers = [512, 256, inferred_head]
                # Force last layer to inferred_head, keep earlier as-is when possible
                if len(layers) >= 1:
                    if len(layers) == 1:
                        layers = [inferred_head]
                    elif len(layers) == 2:
                        layers = [layers[0], inferred_head]
                    else:
                        layers[-1] = inferred_head
                setattr(cfg, 'encoder_mlp_layers', layers)
                print(f"[NN_Inference_Class] Using encoder_mlp_layers={layers}")

            # Define action and observation spaces for model creation
            import gymnasium as gym
            action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_space_dim,), dtype=np.float32)
            obs_space = gym.spaces.Dict({
                'obs': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_space_dim,), dtype=np.float32)
            })

            # Create the actor-critic model with possibly adjusted cfg
            self.model = create_actor_critic(cfg, obs_space, action_space)

            # Load the state dict into the model (strict by default)
            self.model.load_state_dict(state_dict)

            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()

            # Initialize RNN states for all envs
            rnn_size = get_rnn_size(cfg)
            self.rnn_states = torch.zeros(self.num_envs, rnn_size, dtype=torch.float32, device=self.device)

            self.is_model_loaded = True
            print(f"[NN_Inference_Class] Model loaded successfully")
            print(f"[NN_Inference_Class] Model action output dimension: {self.action_space_dim}D")
            print(f"[NN_Inference_Class] Model observation input dimension: {self.obs_space_dim}D")

        except Exception as e:
            print(f"[NN_Inference_Class] Error loading model: {e}")
            self.is_model_loaded = False
            raise

    def eval(self):
        if self.model is not None:
            self.model.eval()

    def reset(self, reset_ids: Union[torch.Tensor, tuple]):
        """Reset RNN states for environments that just terminated/truncated."""
        if not self.is_model_loaded or self.rnn_states is None:
            return
        if isinstance(reset_ids, tuple):
            # Nonzero(as_tuple=True) returns a tuple; first element is indices
            if len(reset_ids) > 0:
                reset_ids = reset_ids[0]
        if reset_ids is None:
            return
        if isinstance(reset_ids, torch.Tensor) and reset_ids.numel() > 0:
            self.rnn_states[reset_ids] = 0.0

    def _to_obs_tensor(self, observation: Union[np.ndarray, torch.Tensor, Dict[str, Any]]):
        """Convert various observation inputs into a torch tensor [N, obs_dim] on device."""
        if isinstance(observation, dict):
            # Prefer 'observations' then 'obs'
            arr = observation.get('observations', observation.get('obs', None))
            observation = arr
        if isinstance(observation, torch.Tensor):
            obs_tensor = observation.to(self.device)
        else:
            obs_np = np.asarray(observation, dtype=np.float32)
            obs_tensor = torch.from_numpy(obs_np).to(self.device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action from the loaded model given a single observation (legacy API).
        """
        actions = self.get_action_batched(observation)
        return actions[0]

    def get_action_batched(self, observations: Union[np.ndarray, torch.Tensor, Dict[str, Any]]) -> np.ndarray:
        """
        Batched inference: accepts [N, obs_dim] or an obs dict and returns [N, action_dim] actions.
        """
        if not self.is_model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            obs_tensor = self._to_obs_tensor(observations)
            # Create observation dictionary format expected by Sample Factory
            obs_dict = {'obs': obs_tensor}
            
            with torch.no_grad():
                model_output = self.model(obs_dict, self.rnn_states)
                action_logits = model_output['action_logits']
                # Handle both cases:
                # 1) action_logits are actor features -> pass through distribution_linear
                # 2) action_logits already contain distribution parameters (mu||logstd) of size 2*action_dim
                feat_in = None
                try:
                    feat_in = int(self.model.action_parameterization.distribution_linear.in_features)
                except Exception:
                    feat_in = None

                if feat_in is None or action_logits.shape[-1] == feat_in:
                    action_distribution = self.model.action_parameterization(action_logits)
                    actions = action_distribution.sample()
                elif action_logits.shape[-1] == self.action_space_dim * 2:
                    # Interpret as concatenated [mu, log_std]
                    mu, log_std = torch.split(action_logits, self.action_space_dim, dim=-1)
                    std = torch.exp(log_std).clamp_min(1e-6)
                    actions = mu + std * torch.randn_like(mu)
                else:
                    raise RuntimeError(
                        f"Unexpected action_logits shape {tuple(action_logits.shape)}; "
                        f"expected features {feat_in} or params {self.action_space_dim*2}"
                    )
                # Update RNN states for next step
                self.rnn_states = model_output['new_rnn_states']
            
            actions_np = actions.detach().cpu().numpy()
            return actions_np
        except Exception as e:
            print(f"[NN_Inference_Class] Error during batched inference: {e}")
            raise

    def get_action_deterministic(self, observation: np.ndarray) -> np.ndarray:
        """
        Get deterministic action from the loaded model (using mean/mode of distribution).
        """
        if not self.is_model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            obs_tensor = self._to_obs_tensor(observation)
            obs_dict = {'obs': obs_tensor}
            
            with torch.no_grad():
                model_output = self.model(obs_dict, self.rnn_states)
                action_logits = model_output['action_logits']
                feat_in = None
                try:
                    feat_in = int(self.model.action_parameterization.distribution_linear.in_features)
                except Exception:
                    feat_in = None
                if feat_in is None or action_logits.shape[-1] == feat_in:
                    action_distribution = self.model.action_parameterization(action_logits)
                    action = action_distribution.mode() if hasattr(action_distribution, 'mode') else action_distribution.mean
                elif action_logits.shape[-1] == self.action_space_dim * 2:
                    mu, log_std = torch.split(action_logits, self.action_space_dim, dim=-1)
                    action = mu
                else:
                    raise RuntimeError(
                        f"Unexpected action_logits shape {tuple(action_logits.shape)}; expected features {feat_in} or params {self.action_space_dim*2}"
                    )
                # Update RNN states for next step
                self.rnn_states = model_output['new_rnn_states']
            
            action_np = action.detach().cpu().numpy()
            if action_np.ndim == 2 and action_np.shape[0] == 1:
                action_np = action_np.squeeze(0)
            return action_np
        except Exception as e:
            print(f"[NN_Inference_Class] Error during deterministic inference: {e}")
            raise

    def reset_rnn_states(self):
        """Reset RNN states for all environments (useful when starting a new episode)."""
        if self.is_model_loaded:
            rnn_size = get_rnn_size(self.cfg)
            self.rnn_states = torch.zeros(self.num_envs, rnn_size, dtype=torch.float32, device=self.device)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_model_loaded:
            return {"model_loaded": False}
        
        return {
            "model_loaded": True,
            "action_space_dim": self.action_space_dim,
            "obs_space_dim": self.obs_space_dim,
            "device": str(self.device),
            "rnn_size": get_rnn_size(self.cfg),
        } 