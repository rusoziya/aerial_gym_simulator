"""
Sample Factory inference class for DCE navigation with gate environment - FIXED for 4D action compatibility
This class provides trained model inference for DCE navigation tasks with gate navigation.

The class is specifically designed to interface with trained Sample Factory models and:
- Uses 4D action space matching the training configuration [x_vel, y_vel, z_vel, yaw_rate]
- Processes 145D observations (17D basic state + 64D drone VAE + 64D static camera VAE)
- Directly interfaces with Sample Factory trained models for gate navigation
- Supports both inference and evaluation with dual camera setup

Architecture compatibility:
- Inference action output: 4D Sample Factory model output [x_vel, y_vel, z_vel, yaw_rate]
- DCE Task input: 4D action space directly compatible
- No action transformation needed - direct pass-through
"""

import time
import copy
from typing import Dict, Any
import torch
import torch.nn as nn
import numpy as np

from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.utils import AttrDict


class NN_Inference_Class:
    def __init__(self, cfg: Dict[str, Any], device: str):
        """
        Initialize the inference class with Sample Factory configuration.
        
        Args:
            cfg: Sample Factory configuration dictionary 
            device: Device to run inference on (e.g., 'cuda:0', 'cpu')
        """
        self.device = device
        
        # Store Sample Factory config
        self.cfg = AttrDict(cfg) if isinstance(cfg, dict) else cfg
        
        # CRITICAL: Action space configuration for gate navigation
        # The trained model expects 4D action output to match DCE task input
        # Action space: [x_vel, y_vel, z_vel, yaw_rate] ∈ [-1, 1]^4
        self.action_space_dim = 4
        print(f"[NN_Inference_Class] Configured for 4D action space (gate navigation with Z-axis control)")
        
        # Observation space configuration (145D for gate navigation)
        # 17D basic state + 64D drone VAE + 64D static camera VAE = 145D total
        self.obs_space_dim = 145
        print(f"[NN_Inference_Class] Configured for {self.obs_space_dim}D observation space (dual camera gate navigation)")
        
        # Model initialization placeholder
        self.model = None
        self.rnn_states = None
        self.is_model_loaded = False
        
        print(f"[NN_Inference_Class] Initialized for inference with device: {device}")

    def load_model(self, model_path: str):
        """
        Load a trained Sample Factory model for inference.
        
        Args:
            model_path: Path to the trained model checkpoint file
        """
        try:
            print(f"[NN_Inference_Class] Loading model from: {model_path}")
            
            # Define action and observation spaces for model creation
            import gymnasium as gym
            action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_space_dim,), dtype=np.float32)
            obs_space = gym.spaces.Dict({
                'obs': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_space_dim,), dtype=np.float32)
            })
            
            # Create the actor-critic model 
            self.model = create_actor_critic(self.cfg, obs_space, action_space)
            
            # Load the trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load the state dict into the model
            self.model.load_state_dict(state_dict)
            
            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize RNN states
            rnn_size = get_rnn_size(self.cfg)
            self.rnn_states = torch.zeros(1, rnn_size, dtype=torch.float32, device=self.device)
            
            self.is_model_loaded = True
            print(f"[NN_Inference_Class] Model loaded successfully")
            print(f"[NN_Inference_Class] Model action output dimension: {self.action_space_dim}D")
            print(f"[NN_Inference_Class] Model observation input dimension: {self.obs_space_dim}D")
            
        except Exception as e:
            print(f"[NN_Inference_Class] Error loading model: {e}")
            self.is_model_loaded = False
            raise

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action from the loaded model given an observation.
        
        Args:
            observation: Observation array of shape (obs_space_dim,)
            
        Returns:
            Action array of shape (action_space_dim,)
        """
        if not self.is_model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Convert observation to tensor and add batch dimension
            obs_tensor = torch.from_numpy(observation).float().to(self.device).unsqueeze(0)
            
            # Create observation dictionary format expected by Sample Factory
            obs_dict = {'obs': obs_tensor}
            
            # Get action from model (no gradient computation needed for inference)
            with torch.no_grad():
                model_output = self.model(obs_dict, self.rnn_states)
                action_logits = model_output['action_logits']
                
                # Sample action from the policy distribution
                action_distribution = self.model.action_parameterization(action_logits)
                action = action_distribution.sample()
                
                # Update RNN states for next step
                self.rnn_states = model_output['new_rnn_states']
            
            # Convert action tensor to numpy array and remove batch dimension
            action_np = action.cpu().numpy().squeeze(0)
            
            # Ensure correct action dimensionality
            assert action_np.shape == (self.action_space_dim,), f"Expected {self.action_space_dim}D action, got {action_np.shape}"
            
            return action_np
            
        except Exception as e:
            print(f"[NN_Inference_Class] Error during inference: {e}")
            raise

    def get_action_deterministic(self, observation: np.ndarray) -> np.ndarray:
        """
        Get deterministic action from the loaded model (using mean of distribution).
        
        Args:
            observation: Observation array of shape (obs_space_dim,)
            
        Returns:
            Action array of shape (action_space_dim,)
        """
        if not self.is_model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Convert observation to tensor and add batch dimension
            obs_tensor = torch.from_numpy(observation).float().to(self.device).unsqueeze(0)
            
            # Create observation dictionary format expected by Sample Factory
            obs_dict = {'obs': obs_tensor}
            
            # Get deterministic action from model
        with torch.no_grad():
                model_output = self.model(obs_dict, self.rnn_states)
                action_logits = model_output['action_logits']
                
                # Get deterministic action (mean of distribution)
                action_distribution = self.model.action_parameterization(action_logits)
                if hasattr(action_distribution, 'mode'):
                    action = action_distribution.mode()
                else:
                    # For distributions without mode, use mean
                    action = action_distribution.mean
                
                # Update RNN states for next step
                self.rnn_states = model_output['new_rnn_states']
            
            # Convert action tensor to numpy array and remove batch dimension
            action_np = action.cpu().numpy().squeeze(0)
            
            # Ensure correct action dimensionality
            assert action_np.shape == (self.action_space_dim,), f"Expected {self.action_space_dim}D action, got {action_np.shape}"
            
            return action_np
            
        except Exception as e:
            print(f"[NN_Inference_Class] Error during deterministic inference: {e}")
            raise

    def reset_rnn_states(self):
        """Reset RNN states (useful when starting a new episode)."""
        if self.is_model_loaded:
            rnn_size = get_rnn_size(self.cfg)
            self.rnn_states = torch.zeros(1, rnn_size, dtype=torch.float32, device=self.device)

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