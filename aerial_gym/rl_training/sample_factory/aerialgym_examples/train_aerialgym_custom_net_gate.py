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


class AerialGymVecEnvGate(AerialGymVecEnvBase):
    """
    Wrapper for isaacgym environments to make them compatible with the sample factory.
    Modified to match old 1333 model architecture - single input processing.
    Enhanced with dual camera GIF saving functionality and 4D action space for gate navigation.
    """

    def __init__(self, aerialgym_env, obs_key, save_gifs=False):
        # Initialize base wrapper (sets self.env, action_space, observation_space, etc.)
        obs_dim = 150
        action_dim = 4
        super().__init__(aerialgym_env, obs_key, action_dim=action_dim, obs_dim=obs_dim)
        
        # GIF saving functionality
        self.save_gifs = save_gifs
        if self.save_gifs:
            if VERBOSE:
                print(f"[AerialGymVecEnv] GIF saving ENABLED for dual cameras (drone + static)")
            self.gif_episode_counter = 0
            self.gif_output_dir = "./gif_episodes"
            os.makedirs(self.gif_output_dir, exist_ok=True)
            
            # Frame storage for each environment - Clean versions
            self.drone_depth_frames = [[] for _ in range(self.num_agents)]
            self.drone_seg_frames = [[] for _ in range(self.num_agents)]
            self.static_depth_frames = [[] for _ in range(self.num_agents)]
            self.static_seg_frames = [[] for _ in range(self.num_agents)]
            self.merged_frames = [[] for _ in range(self.num_agents)]
            
            # NEW: Frame storage for D455 noised versions
            self.drone_depth_noised_frames = [[] for _ in range(self.num_agents)]
            self.static_depth_noised_frames = [[] for _ in range(self.num_agents)]
            self.merged_noised_frames = [[] for _ in range(self.num_agents)]
        else:
            if VERBOSE:
                print(f"[AerialGymVecEnv] GIF saving DISABLED")
        
        self.step_count = 0
        # CRITICAL FIX: Force action space to exactly match inference expectations (4D for gate navigation)
        # The inference script expects 4D actions [x_vel, y_vel, z_vel, yaw_rate], so train with 4D to avoid shape mismatch
        import numpy as np
        base_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)  # UPDATED: 4D action space
        self.action_space = convert_space(base_action_space)

        # Debug: Print action space info to verify it's 4D
        if VERBOSE:
            print(f"[AerialGymVecEnv] Forced action space shape: {self.action_space.shape}")
            print(f"[AerialGymVecEnv] is_multiagent: {self.is_multiagent}, num_agents: {self.num_agents}")

        # Fusion controls
        self.fusion_mode = os.environ.get('SF_FUSION_MODE', 'concat')
        self.gate_per_feature = os.environ.get('SF_GATE_PER_FEATURE', '1') == '1'

        # DYNAMIC OBSERVATION SPACE: Detect observation space dimension from task config
        # This handles both standard DCE navigation (81D) and gate navigation (147D)
        if obs_key == "obs":
            # Get the actual observation space dimension from the task configuration
            task_obs_dim = self.env.task_config.observation_space_dim  # Default to 150D for gate navigation
            if VERBOSE:
                print(f"[AerialGymVecEnv] Detected observation space: {task_obs_dim}D")
            
            if VERBOSE:
                if task_obs_dim == 150:
                    print(f"[AerialGymVecEnv] Using GATE NAVIGATION configuration (150D = 3D drone position + 6D static camera pose + 3D full orientation + 9D state + 64D drone VAE + 64D static camera VAE)")
                elif task_obs_dim == 143:
                    print(f"[AerialGymVecEnv] Using ALTERNATIVE GATE NAVIGATION configuration (143D)")
                elif task_obs_dim == 145:
                    print(f"[AerialGymVecEnv] Using OLDER GATE NAVIGATION configuration (145D)")
                elif task_obs_dim == 81:
                    print(f"[AerialGymVecEnv] Using STANDARD DCE configuration (81D = 17D basic + 64D drone VAE)")
                else:
                    print(f"[AerialGymVecEnv] Using CUSTOM configuration ({task_obs_dim}D observations)")
            
            # Create dynamic observation space based on actual task requirements
            self.observation_space = gym.spaces.Dict({
                "obs": convert_space(gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(task_obs_dim,), dtype=np.float32
                ))
            })
        else:
            raise ValueError(f"Unknown observation key: {obs_key}")

        self._truncated: Tensor = torch.zeros(self.num_agents, dtype=torch.bool)
        self.episode_count = 0  # Track episode count for GIF saving
        
        # Observation slice map for 150D configuration to support ablation
        # Names mirror task documentation
        self.obs_slices = {
            "drone_position": (0, 3),
            "static_camera_pos": (3, 6),
            "static_camera_orient": (6, 9),
            "drone_orientation": (9, 12),
            "drone_linear_vel": (12, 15),
            "drone_angular_vel": (15, 18),
            "drone_actions": (18, 22),
            "drone_camera_vae": (22, 86),
            "static_camera_vae": (86, 150),
        }

        # Running episode-level aggregates for trajectory quality metrics
        # We keep sums and counts (and success-specific counts) across all finished episodes in this run
        self._traj_running = {
            'path_efficiency_sum': 0.0,
            'path_efficiency_count': 0,
            'min_gate_distance_sum': 0.0,
            'min_gate_distance_count': 0,
            'time_to_gate_sum': 0.0,
            'time_to_gate_count': 0,  # only when crossed
            'center_offset_sum': 0.0,
            'center_offset_count': 0,  # only when crossed
            'height_offset_sum': 0.0,
            'height_offset_count': 0,  # only when crossed
            'episodes_total': 0,
            'episodes_crossed': 0,
        }

        # Running totals for curriculum counters (for episode_extra_stats)
        self._curriculum_totals = {
            'total_successes': 0,
            'total_crashes': 0,
            'total_timeouts': 0,
        }

        # Build fusion module for gated mode (drop-in; actor-critic trunk remains in SF)
        if self.fusion_mode == 'gated':
            class DualGatedLateFusion(nn.Module):
                def __init__(self, latent_dim=64, kin_dim=22, last_act_dim=0, trunk_dims=(512,256,128), gate_per_feature=True):
                    super().__init__()
                    D = latent_dim
                    self.ego_proj    = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.ELU(), nn.LayerNorm(D))
                    self.static_proj = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.ELU(), nn.LayerNorm(D))
                    gate_out = D if gate_per_feature else 1
                    self.gate = nn.Sequential(nn.Linear(2*D, D), nn.ELU(), nn.Linear(D, gate_out))
                def forward(self, ego_latent, static_latent) -> None:
                    e = self.ego_proj(ego_latent)
                    s = self.static_proj(static_latent)
                    g = torch.sigmoid(self.gate(torch.cat([e, s], dim=-1)))
                    if g.shape[-1] == 1:
                        g = g.expand_as(e)
                    z = g * s + (1 - g) * e
                    return z, g
            self._gated_fuser = DualGatedLateFusion(latent_dim=64, gate_per_feature=self.gate_per_feature).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def _process_camera_image(self, image_data, camera_type="depth") -> None:
        """Process camera image for GIF saving."""
        if camera_type == "depth":
            # Process depth image (similar to dce_nn_navigation.py)
            image = (255.0 * image_data.cpu().numpy()).astype(np.uint8)
            return Image.fromarray(image)
        elif camera_type == "segmentation":
            # Process segmentation image with colormap
            seg_image = image_data.cpu().numpy()
            # Preserve zero IDs (e.g., background) so we can see if any actor uses 0
            # If all values are the same, avoid divide-by-zero
            seg_min, seg_max = seg_image.min(), seg_image.max()
            if seg_max - seg_min < 1e-8:
                seg_norm = np.zeros_like(seg_image, dtype=np.float32)
            else:
                seg_norm = (seg_image - seg_min) / (seg_max - seg_min)
            seg_image_plasma = matplotlib.cm.plasma(seg_norm)
            return Image.fromarray((seg_image_plasma * 255.0).astype(np.uint8))
        return None

    def _apply_obs_ablation(self, obs_tensor: Tensor) -> Tensor:
        """
        Return-drop ablation controlled by environment variables.
        Supported env vars:
          - ABLATE_DRONE_POS=true           -> zero [0:3]
          - ABLATE_OBS_RANGES="0:3=zero,22:86=shuffle,86:150=noise:0.1,0:22=zerograd"
        Ops:
          - zero     : set slice to 0
          - zerograd : set slice to constant zeros detached from graph (no grad influence)
          - shuffle  : permute values across envs for this slice
          - noise:std: add Gaussian noise with given std
        """
        import os
        if obs_tensor is None:
            return obs_tensor
        debug = os.environ.get("ABLATE_DEBUG", "false").lower() == "true"
        if not hasattr(self, "_ablate_debug_count"):
            self._ablate_debug_count = 0
        # Simple switch for drone position
        if os.environ.get("ABLATE_DRONE_POS", "false").lower() == "true":
            start, end = self.obs_slices.get("drone_position", (0, 3))
            obs_tensor[:, start:end] = 0.0
            if debug and self._ablate_debug_count < 10:
                v = obs_tensor[:, start:end]
                print(f"[ABLATE_DEBUG] applied: {start}:{end}=zero | min={v.min().item():.3e} max={v.max().item():.3e} mean={v.mean().item():.3e} nonzero={int((v!=0).sum().item())}")
                self._ablate_debug_count += 1
        # General ranges
        spec_str = os.environ.get("ABLATE_OBS_RANGES", "").strip()
        if not spec_str:
            return obs_tensor
        grad_mask = None
        zero_ranges = []
        zerograd_ranges = []
        for spec in spec_str.split(","):
            spec = spec.strip()
            if not spec:
                continue
            if "=" not in spec:
                continue
            lhs, rhs = spec.split("=", 1)
            lhs = lhs.strip(); rhs = rhs.strip()
            if ":" not in lhs:
                continue
            try:
                start_s, end_s = lhs.split(":", 1)
                start = int(start_s); end = int(end_s)
            except (ValueError, TypeError):
                continue
            op = rhs
            if op == "zero":
                # Defer zeroing via a constant mask so gradients are zeroed on these dims too
                if grad_mask is None:
                    grad_mask = torch.ones_like(obs_tensor)
                grad_mask[:, start:end] = 0.0
                zero_ranges.append((start, end))
            elif op == "zerograd":
                # Mark for replacement with constant zeros detached from the current graph
                zerograd_ranges.append((start, end))
            elif op == "shuffle":
                if obs_tensor.shape[0] > 1:
                    perm = torch.randperm(obs_tensor.shape[0], device=obs_tensor.device)
                    obs_tensor[:, start:end] = obs_tensor[perm, start:end]
                if debug and self._ablate_debug_count < 10:
                    v = obs_tensor[:, start:end]
                    # Report per-env differences quickly with norm
                    print(f"[ABLATE_DEBUG] applied: {start}:{end}=shuffle | sample_env0={v[0].detach().cpu().numpy()} sample_env1={v[1].detach().cpu().numpy() if v.shape[0]>1 else 'NA'}")
                    self._ablate_debug_count += 1
            elif op.startswith("noise:"):
                try:
                    std = float(op.split(":", 1)[1])
                except (ValueError, TypeError):
                    std = 0.0
                if std > 0.0:
                    obs_tensor[:, start:end] = obs_tensor[:, start:end] + torch.randn_like(obs_tensor[:, start:end]) * std
                if debug and self._ablate_debug_count < 10:
                    v = obs_tensor[:, start:end]
                    print(f"[ABLATE_DEBUG] applied: {start}:{end}=noise:{std} | std_est={v.std().item():.3e} mean={v.mean().item():.3e}")
                    self._ablate_debug_count += 1
        # Apply accumulated mask at once (for `zero` ops)
        if grad_mask is not None:
            obs_tensor = obs_tensor * grad_mask
            if debug:
                for (start, end) in zero_ranges:
                    if self._ablate_debug_count >= 10:
                        break
                    v = obs_tensor[:, start:end]
                    print(f"[ABLATE_DEBUG] applied: {start}:{end}=zero | min={v.min().item():.3e} max={v.max().item():.3e} mean={v.mean().item():.3e} nonzero={int((v!=0).sum().item())}")
                    self._ablate_debug_count += 1

        # Apply zerograd replacements last: replace ranges with constant zero tensors detached from the graph
        for (start, end) in zerograd_ranges:
            # Build zero slice that is not connected to original obs_tensor graph
            zero_slice = torch.zeros_like(obs_tensor[:, start:end])
            # Concatenate to avoid any multiply-by-zero path
            left = obs_tensor[:, :start]
            right = obs_tensor[:, end:]
            obs_tensor = torch.cat([left, zero_slice, right], dim=-1)
            if debug and self._ablate_debug_count < 10:
                v = obs_tensor[:, start:end]
                print(f"[ABLATE_DEBUG] applied: {start}:{end}=zerograd | min={v.min().item():.3e} max={v.max().item():.3e} mean={v.mean().item():.3e} nonzero={int((v!=0).sum().item())}")
                self._ablate_debug_count += 1
        return obs_tensor

    def _collect_frames(self, obs_dict) -> None:
        """Collect frames from both drone and static cameras for GIF generation (clean + noised versions)."""
        if not self.save_gifs:
            return
        
        try:
            # Access camera data directly from the underlying task
            task = self.env
            
            # Get drone camera depth image from task's obs_dict (original clean version)
            if "depth_range_pixels" in task.obs_dict:
                drone_depth = task.obs_dict["depth_range_pixels"][0, 0]  # First env, first camera
                drone_depth_img = self._process_camera_image(drone_depth, "depth")
                self.drone_depth_frames[0].append(drone_depth_img)
            
            # Get drone camera segmentation image from task's obs_dict
            if "segmentation_pixels" in task.obs_dict:
                drone_seg = task.obs_dict["segmentation_pixels"][0, 0]  # First env, first camera  
                drone_seg_img = self._process_camera_image(drone_seg, "segmentation")
                self.drone_seg_frames[0].append(drone_seg_img)
            
            # Get noised drone camera depth image (with D455 noise applied)
            if "depth_range_pixels_noised" in task.obs_dict:
                drone_depth_noised = task.obs_dict["depth_range_pixels_noised"][0, 0]  # First env, first camera
                drone_depth_noised_img = self._process_camera_image(drone_depth_noised, "depth")
                self.drone_depth_noised_frames[0].append(drone_depth_noised_img)
            
            # Get static camera images from stored clean versions
            if "static_depth_clean" in task.obs_dict:
                static_depth = task.obs_dict["static_depth_clean"]
                
                # Process static depth
                if static_depth is not None:
                    # Convert to tensor if numpy array
                    if isinstance(static_depth, np.ndarray):
                        static_depth_tensor = torch.from_numpy(static_depth)
                    else:
                        static_depth_tensor = static_depth
                    
                    # Ensure 2D array for image processing
                    if static_depth_tensor.dim() > 2:
                        static_depth_tensor = static_depth_tensor.squeeze()
                    
                    static_depth_img = self._process_camera_image(static_depth_tensor, "depth")
                    self.static_depth_frames[0].append(static_depth_img)
                
                # Process static segmentation
            if "static_seg" in task.obs_dict:
                static_seg = task.obs_dict["static_seg"]
                if static_seg is not None:
                    # Convert to tensor if numpy array
                    if isinstance(static_seg, np.ndarray):
                        static_seg_tensor = torch.from_numpy(static_seg.astype(np.float32))
                    else:
                        static_seg_tensor = static_seg.float()
                    
                    # Ensure 2D array for image processing
                    if static_seg_tensor.dim() > 2:
                        static_seg_tensor = static_seg_tensor.squeeze()
                    
                    static_seg_img = self._process_camera_image(static_seg_tensor, "segmentation")
                    self.static_seg_frames[0].append(static_seg_img)
                
            # Get noised static camera depth image (with D455 noise applied)
            if "static_depth_noised" in task.obs_dict:
                static_depth_noised = task.obs_dict["static_depth_noised"]
                
                # Process static depth
                if static_depth_noised is not None:
                    # Convert to tensor if numpy array
                    if isinstance(static_depth_noised, np.ndarray):
                        static_depth_noised_tensor = torch.from_numpy(static_depth_noised)
                    else:
                        static_depth_noised_tensor = static_depth_noised
                    
                    # Ensure 2D array for image processing
                    if static_depth_noised_tensor.dim() > 2:
                        static_depth_noised_tensor = static_depth_noised_tensor.squeeze()
                    
                    static_depth_noised_img = self._process_camera_image(static_depth_noised_tensor, "depth")
                    self.static_depth_noised_frames[0].append(static_depth_noised_img)
            
            if (len(self.drone_depth_frames[0]) > 0 and len(self.static_depth_frames[0]) > 0 and 
                len(self.drone_depth_frames[0]) == len(self.static_depth_frames[0])):
                drone_img = self.drone_depth_frames[0][-1]
                static_img = self.static_depth_frames[0][-1]
                
                # Convert to numpy arrays for concatenation
                drone_array = np.array(drone_img)
                static_array = np.array(static_img)
                
                # Resize static image to match drone image dimensions if needed
                if drone_array.shape != static_array.shape:
                    from PIL import Image as PILImage
                    static_img_resized = PILImage.fromarray(static_array).resize((drone_array.shape[1], drone_array.shape[0]))
                    static_array = np.array(static_img_resized)
                
                # Concatenate horizontally (side by side)
                merged_array = np.concatenate((drone_array, static_array), axis=1)
                merged_img = Image.fromarray(merged_array)
                self.merged_frames[0].append(merged_img)
            
            if (len(self.drone_depth_noised_frames[0]) > 0 and len(self.static_depth_noised_frames[0]) > 0 and 
                len(self.drone_depth_noised_frames[0]) == len(self.static_depth_noised_frames[0])):
                drone_noised_img = self.drone_depth_noised_frames[0][-1]
                static_noised_img = self.static_depth_noised_frames[0][-1]
                
                # Convert to numpy arrays for concatenation
                drone_noised_array = np.array(drone_noised_img)
                static_noised_array = np.array(static_noised_img)
                
                # Resize static image to match drone image dimensions if needed
                if drone_noised_array.shape != static_noised_array.shape:
                    from PIL import Image as PILImage
                    static_noised_img_resized = PILImage.fromarray(static_noised_array).resize((drone_noised_array.shape[1], drone_noised_array.shape[0]))
                    static_noised_array = np.array(static_noised_img_resized)
                
                # Concatenate horizontally (side by side)
                merged_noised_array = np.concatenate((drone_noised_array, static_noised_array), axis=1)
                merged_noised_img = Image.fromarray(merged_noised_array)
                self.merged_noised_frames[0].append(merged_noised_img)
                
        except (ValueError, TypeError) as e:
            if VERBOSE:
                print(f"[GIF] Warning: Failed to collect frames: {e}")
                import traceback
                print(f"[GIF] Traceback: {traceback.format_exc()}")

    def _save_episode_gifs(self, env_id=0) -> None:
        """Save collected frames as GIFs for the specified environment."""
        if not self.save_gifs or env_id >= self.num_agents:
            return
        
        try:
            episode_num = self.gif_episode_counter
            self.gif_episode_counter += 1
            
            # Determine curriculum level suffix if available
            level_suffix = ""
            try:
                if hasattr(self.env, 'curriculum_level'):
                    level = int(self.env.curriculum_level)
                    level_suffix = f"_L{level:02d}"
                elif hasattr(self.env, 'task_config') and hasattr(self.env.task_config, 'curriculum'):
                    # fallback to min level if needed
                    level = int(getattr(self.env.task_config.curriculum, 'min_level', 0))
                    level_suffix = f"_L{level:02d}"
            except (ValueError, TypeError):
                level_suffix = ""
            
            
            
                print(f"[GIF] Saved static depth: {gif_path}")
            
            if len(self.static_seg_frames[env_id]) > 0:
                gif_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_static_seg{level_suffix}.gif")
                self.static_seg_frames[env_id][0].save(
                    gif_path,
                    save_all=True,
                    append_images=self.static_seg_frames[env_id][1:],
                    duration=100,
                    loop=0
                )
                if VERBOSE:
                    print(f"[GIF] Saved static segmentation: {gif_path}")
            
            
            # Save drone camera noised GIFs
            if len(self.drone_depth_noised_frames[env_id]) > 0:
                gif_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_drone_depth_D455_NOISED{level_suffix}.gif")
                self.drone_depth_noised_frames[env_id][0].save(
                    gif_path,
                    save_all=True,
                    append_images=self.drone_depth_noised_frames[env_id][1:],
                    duration=100,
                    loop=0
                )
                if VERBOSE:
                    print(f"[GIF] Saved drone depth (D455 NOISED): {gif_path}")
            
            # Save static camera noised GIFs
            if len(self.static_depth_noised_frames[env_id]) > 0:
                gif_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_static_depth_D455_NOISED{level_suffix}.gif")
                self.static_depth_noised_frames[env_id][0].save(
                    gif_path,
                    save_all=True,
                    append_images=self.static_depth_noised_frames[env_id][1:],
                    duration=100,
                    loop=0
                )
                if VERBOSE:
                    print(f"[GIF] Saved static depth (D455 NOISED): {gif_path}")
            
        
        except OSError as e:
            if VERBOSE:
                print(f"[GIF] Warning: Failed to save GIFs for episode {episode_num}: {e}")


    def _clear_frames(self, env_id=0) -> None:
        """Clear collected frames for the specified environment (clean + noised versions)."""
        if self.save_gifs and env_id < self.num_agents:
            # Clear clean frames
            self.drone_depth_frames[env_id] = []
            self.drone_seg_frames[env_id] = []
            self.static_depth_frames[env_id] = []
            self.static_seg_frames[env_id] = []
            self.merged_frames[env_id] = []
            
            # Clear D455 noised frames
            self.drone_depth_noised_frames[env_id] = []
            self.static_depth_noised_frames[env_id] = []
            self.merged_noised_frames[env_id] = []

    def reset(self, *args, **kwargs) -> Tuple[Dict[str, Tensor], Dict]:
        # some IGE envs return all zeros on the first timestep, but this is probably okay
        obs, rew, terminated, truncated, infos = self.env.reset()
        
        # Clear frames for new episode
        if self.save_gifs:
            for env_id in range(self.num_agents):
                self._clear_frames(env_id)
        
        # DYNAMIC OBSERVATION PROCESSING: Handle both standard DCE (81D) and gate navigation (145D)
        # Task provides "observations" with correct dimensionality (81D or 145D) based on task type
        # We pass this through as "obs" for Sample Factory
        transformed_obs = {"obs": obs["observations"]}
        # Apply return-drop ablation if requested
        transformed_obs["obs"] = self._apply_obs_ablation(transformed_obs["obs"]) 
        # Sanitize non-finite values BEFORE Sample Factory normalization
        try:
            vec = transformed_obs.get('obs', None)
            if isinstance(vec, torch.Tensor):
                if not torch.isfinite(vec).all():
                    import os as _os
                    if _os.environ.get('ABLATE_DEBUG', 'false').lower() == 'true':
                        n_bad = int((~torch.isfinite(vec)).sum().item())
                        print(f"[SANITIZE][reset] replacing {n_bad} non-finite obs values with 0")
                transformed_obs['obs'] = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        except (ValueError, TypeError):
            pass
        # Optional: print the full obs vector for env0 each step for one episode
        if os.environ.get('PRINT_ENV0_OBS_ONCE', 'false').lower() == 'true':
            if not hasattr(self, '_train_env0_obs_state'):
                self._train_env0_obs_state = 0  # 0=armed, 1=active, 2=done
                self._train_env0_obs_step = 0
            vec = transformed_obs.get('obs', None)
            if isinstance(vec, torch.Tensor) and vec.ndim == 2 and vec.shape[0] > 0:
                obs0 = vec[0]
                # Activate on first nonzero norm
                if self._train_env0_obs_state == 0:
                    if float(torch.norm(obs0).item()) > 0.0:
                        self._train_env0_obs_state = 1
                        self._train_env0_obs_step = 0
                        if VERBOSE:
                            print(f"[TRAIN_ENV0_OBS] step={self._train_env0_obs_step} obs0={obs0.detach().cpu().numpy()}")
                        self._train_env0_obs_step += 1
                elif self._train_env0_obs_state == 1:
                    if VERBOSE:
                        print(f"[TRAIN_ENV0_OBS] step={self._train_env0_obs_step} obs0={obs0.detach().cpu().numpy()}")
                    self._train_env0_obs_step += 1
        # Optional: one-episode env0 latent per-frame logging during training
        if os.environ.get('PRINT_ENV0_LATENTS_ONCE', 'false').lower() == 'true':
            if not hasattr(self, '_train_env0_log_state'):
                self._train_env0_log_state = 0  # 0=armed, 1=active, 2=done
                self._train_env0_step = 0
            vec = transformed_obs.get('obs', None)
            if isinstance(vec, torch.Tensor) and vec.ndim == 2 and vec.shape[0] > 0 and vec.shape[1] >= 150:
                ze0 = vec[0, 22:86]; zs0 = vec[0, 86:150]
                ze_abs = float(torch.mean(torch.abs(ze0)).item()); zs_abs = float(torch.mean(torch.abs(zs0)).item())
                if self._train_env0_log_state == 0:
                    if (ze_abs > 1e-6) or (zs_abs > 1e-6):
                        self._train_env0_log_state = 1
                        self._train_env0_step = 0
                        if VERBOSE:
                            print(f"[TRAIN_ENV0_LATENTS] step={self._train_env0_step} abs_mean: drone={ze_abs:.6f} static={zs_abs:.6f}")
                        self._train_env0_step += 1
                elif self._train_env0_log_state == 1:
                    if VERBOSE:
                        print(f"[TRAIN_ENV0_LATENTS] step={self._train_env0_step} abs_mean: drone={ze_abs:.6f} static={zs_abs:.6f}")
                    self._train_env0_step += 1
        
        # Collect first frame if GIF saving is enabled
        if self.save_gifs:
            self._collect_frames(obs)
        
        return transformed_obs, infos

    def step(self, action) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Dict]:
        # FIXED: Direct 4D action pass-through for DCE gate navigation task
        # Sample Factory now provides 4D actions directly matching DCE task expectations (x_vel, y_vel, z_vel, yaw_rate)
        dce_action = action
        # Sanitize action to avoid NaN/Inf entering physics
        try:
            if isinstance(dce_action, torch.Tensor):
                if not torch.isfinite(dce_action).all():
                    import os as _os
                    if _os.environ.get('ABLATE_DEBUG', 'false').lower() == 'true':
                        n_bad = int((~torch.isfinite(dce_action)).sum().item())
                        print(f"[SANITIZE][action] replacing {n_bad} non-finite action values with 0 and clamping")
                dce_action = torch.nan_to_num(dce_action, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1.0, 1.0)
        except (ValueError, TypeError):
            pass
            
        obs, rew, terminated, truncated, infos = self.env.step(dce_action)
        
        # Collect frames for GIF generation
        if self.save_gifs:
            self._collect_frames(obs)
        
        # Save GIFs when episodes terminate
        if self.save_gifs and (torch.any(terminated) or torch.any(truncated)):
            # Save GIFs for terminated/truncated environments (only save for env 0 to avoid spam)
            reset_ids = (terminated + truncated).nonzero(as_tuple=True)[0]
            if len(reset_ids) > 0 and 0 in reset_ids:  # Only save for first environment
                # Only save if env_agents is 16
                try:
                    env_agents = int(os.environ.get('SF_ENV_AGENTS', '0'))
                except (ValueError, TypeError):
                    env_agents = 0
                if env_agents != 16:
                    # Clear frames and skip saving to control output volume
                    self._clear_frames(env_id=0)
                    self.episode_count += 1
                    return transformed_obs, rew, terminated, truncated, infos
                # End training latent logging when env0 resets once
                if os.environ.get('PRINT_ENV0_LATENTS_ONCE', 'false').lower() == 'true' and self._train_env0_log_state == 1:
                    reset_ids = (terminated + truncated).nonzero(as_tuple=True)[0]
                    if len(reset_ids) > 0 and 0 in reset_ids.tolist():
                        self._train_env0_log_state = 2
                        if VERBOSE:
                            print(f"[TRAIN_ENV0_LATENTS] episode_end steps={self._train_env0_step}")
                if os.environ.get('PRINT_ENV0_OBS_ONCE', 'false').lower() == 'true' and self._train_env0_obs_state == 1:
                    reset_ids = (terminated + truncated).nonzero(as_tuple=True)[0]
                    if len(reset_ids) > 0 and 0 in reset_ids.tolist():
                        self._train_env0_obs_state = 2
                        if VERBOSE:
                            print(f"[TRAIN_ENV0_OBS] episode_end steps={self._train_env0_obs_step}")
                # Save every 5 episodes for env 0
                if self.episode_count % 5 == 0:
                    if VERBOSE:
                        if terminated[0]:
                            print(f"[GIF] Episode {self.episode_count} terminated - saving GIFs (every 5 episodes)")
                        elif truncated[0]:
                            print(f"[GIF] Episode {self.episode_count} truncated - saving GIFs (every 5 episodes)")
                    self._save_episode_gifs(env_id=0)
                self._clear_frames(env_id=0)
                
                # Increment episode counter when first environment resets
                self.episode_count += 1
        
        # Inject curriculum level into infos for learner-side W&B logging
        try:
            if isinstance(infos, dict):
                extra = infos.get('episode_extra_stats', {})
                if not isinstance(extra, dict):
                    extra = {}
                ids = (terminated + truncated).nonzero(as_tuple=True)[0]
                if ids.numel() > 0:
                    # Access underlying task to read curriculum level
                    task = self.env
                    curr_level = None
                    if task is not None:
                        curr_level = task.curriculum_level
                        # Also pull step-averaged traj metrics directly if the task stashed them
                        traj_avg = getattr(task, '_last_traj_metrics_avg', None)
                        # NEW: Update running aggregates using per-env episode metrics when resets happen
                        per_env = getattr(task, '_last_traj_metrics_per_env', None)
                        if isinstance(per_env, dict):
                            # Limit to environments that actually reset this step
                            reset_ids = ids.detach().cpu().tolist()
                            crossed_mask = per_env.get('crossed', None)
                            def _to_list(t) -> None:
                                return t.detach().cpu().tolist() if torch.is_tensor(t) else t
                            if reset_ids is not None:
                                for eid in reset_ids:
                                    pe = float(per_env['path_efficiency'][eid].item())
                                    mgd = float(per_env['min_gate_distance'][eid].item())
                                    ttg = float(per_env['time_to_gate_steps'][eid].item())
                                    co = float(per_env['center_offset_success'][eid].item())
                                    ho = float(per_env['height_offset_success'][eid].item())
                                    # Update totals
                                    if math.isfinite(pe):
                                        self._traj_running['path_efficiency_sum'] += pe
                                        self._traj_running['path_efficiency_count'] += 1
                                    if math.isfinite(mgd):
                                        self._traj_running['min_gate_distance_sum'] += mgd
                                        self._traj_running['min_gate_distance_count'] += 1
                                    crossed = False
                                    if isinstance(crossed_mask, torch.Tensor):
                                        crossed = bool(crossed_mask[eid].item())
                                    # Only update these when crossed (finite ttg/offsets expected)
                                    if crossed and math.isfinite(ttg):
                                        self._traj_running['time_to_gate_sum'] += ttg
                                        self._traj_running['time_to_gate_count'] += 1
                                    if crossed and math.isfinite(co):
                                        self._traj_running['center_offset_sum'] += co
                                        self._traj_running['center_offset_count'] += 1
                                    if crossed and math.isfinite(ho):
                                        self._traj_running['height_offset_sum'] += ho
                                        self._traj_running['height_offset_count'] += 1
                                    # Episode counters
                                    self._traj_running['episodes_total'] += 1
                                    if crossed:
                                        self._traj_running['episodes_crossed'] += 1
                    # Log a run-level stat (aggregated by SF) without per-env nesting
                    extra['curriculum_level'] = float(curr_level) if curr_level is not None else -1.0
                    # Also provide curriculum level - 1 for plotting convenience
                    if curr_level is not None:
                        extra['curriculum_level_minus_1'] = float(curr_level - 1)
                    else:
                        extra['curriculum_level_minus_1'] = -1.0
                    # Inject traj metrics into episode_extra_stats following the same pattern
                    if isinstance(traj_avg, dict):
                        for k, v in traj_avg.items():
                            extra[k] = float(v)
                        # Also mirror last-position metrics when present
                        for k in ('last_position_x','last_position_y','last_position_z','last_center_distance'):
                            if k in traj_avg:
                                extra[k] = float(traj_avg[k])
                    # Pass-through any episode-level trajectory metrics already stored by env
                    # They will be aggregated by SF and picked up by the learner later
                    for k in ('path_efficiency','time_to_gate_steps','min_gate_distance','center_offset_success','height_offset_success','target_success_rate'):
                        if k in extra:
                            # ensure float cast
                            extra[k] = float(extra[k])
                    # Include last-position series if present (already floats)
                    for k in ('last_position_x','last_position_y','last_position_z','last_center_distance'):
                        if k in extra:
                            extra[k] = float(extra[k])
                    # Add running-mean episode-level metrics
                    # For success-conditioned metrics (time_to_gate/offsets), return None when count==0
                    def _safe_mean(sum_key, count_key, none_if_zero=False) -> None:
                        s = self._traj_running.get(sum_key, 0.0)
                        c = self._traj_running.get(count_key, 0)
                        if c <= 0:
                            return None if none_if_zero else float('nan')
                        return float(s / c)
                    extra['path_efficiency_running_mean'] = _safe_mean('path_efficiency_sum', 'path_efficiency_count')
                    extra['min_gate_distance_running_mean'] = _safe_mean('min_gate_distance_sum', 'min_gate_distance_count')
                    # Means conditioned on success: drop when no successes
                    extra['time_to_gate_running_mean'] = _safe_mean('time_to_gate_sum', 'time_to_gate_count', none_if_zero=True)
                    extra['center_offset_running_mean'] = _safe_mean('center_offset_sum', 'center_offset_count', none_if_zero=True)
                    extra['height_offset_running_mean'] = _safe_mean('height_offset_sum', 'height_offset_count', none_if_zero=True)
                    # Helpful counts
                    extra['gate_pass_rate'] = float(self._traj_running['episodes_crossed']) / float(max(1, self._traj_running['episodes_total']))
                    extra['episodes_total'] = float(self._traj_running['episodes_total'])
                    extra['episodes_crossed'] = float(self._traj_running['episodes_crossed'])

                    # Curriculum (generic task) counters derived from infos flags
                    # Sum over the envs that reset this step
                    if isinstance(infos, dict) and 'successes' in infos and 'crashes' in infos and 'timeouts' in infos:
                        ids = (terminated + truncated).nonzero(as_tuple=True)[0]
                        if ids.numel() > 0:
                            step_successes = int(infos['successes'][ids].sum().item())
                            step_crashes = int(infos['crashes'][ids].sum().item())
                            step_timeouts = int(infos['timeouts'][ids].sum().item())
                            # Update running totals
                            self._curriculum_totals['total_successes'] += step_successes
                            self._curriculum_totals['total_crashes'] += step_crashes
                            self._curriculum_totals['total_timeouts'] += step_timeouts
                            # Expose per-episode counts for this step (averages not needed)
                            extra['successes'] = float(step_successes)
                            extra['crashes'] = float(step_crashes)
                            extra['timeouts'] = float(step_timeouts)
                            # Expose cumulative totals (curriculum namespace)
                            extra['curriculum/total_successes'] = float(self._curriculum_totals['total_successes'])
                            extra['curriculum/total_crashes'] = float(self._curriculum_totals['total_crashes'])
                            extra['curriculum/total_timeouts'] = float(self._curriculum_totals['total_timeouts'])

                    # Mirror curriculum/current_* using task attributes (fallback) so they always show up
                    try:
                        # Prefer values emitted by env in infos; otherwise fall back to task attributes
                        # Move any current_* tensors into episode_extra_stats namespace
                        cur_lvl_tensor = infos.get('curriculum/current_level', None)
                        cur_prog_tensor = infos.get('curriculum/current_progress', None)
                        if cur_lvl_tensor is not None:
                            extra['episode_extra_stats/curriculum/current_level'] = float(cur_lvl_tensor.mean().item()) if hasattr(cur_lvl_tensor, 'mean') else float(cur_lvl_tensor)
                            if 'curriculum/current_level' in infos:
                                del infos['curriculum/current_level']
                        else:
                            if task is not None and hasattr(task, 'curriculum_level'):
                                extra['episode_extra_stats/curriculum/current_level'] = float(task.curriculum_level)
                        if cur_prog_tensor is not None:
                            extra['episode_extra_stats/curriculum/current_progress'] = float(cur_prog_tensor.mean().item()) if hasattr(cur_prog_tensor, 'mean') else float(cur_prog_tensor)
                            if 'curriculum/current_progress' in infos:
                                del infos['curriculum/current_progress']
                        else:
                            if task is not None and hasattr(task, 'curriculum_progress_fraction'):
                                extra['episode_extra_stats/curriculum/current_progress'] = float(task.curriculum_progress_fraction)
                    except (ValueError, TypeError):
                        pass

                    # Gate/task-specific + camera alignment — mean across envs if present in infos
                    gate_keys = (
                        'gate/passed','gate/distance','gate/alignment',
                        'camera/facing_alignment','camera/alignment_angle_deg','camera/alignment_category',
                    )
                    for key in gate_keys:
                        val = infos.get(key, None)
                        if val is not None:
                            extra[key] = float(val.mean().item()) if hasattr(val, 'mean') else float(val)

                    # Curriculum snapshot & progression (gate task)
                    # 1) Mirror when the task provides them in infos; 2) Fallback to task attributes so they always appear
                    try:
                        # Mirror block (when present)
                        snapshot_keys = (
                            'curriculum/level','curriculum/progress','curriculum/success_rate',
                            'curriculum/crash_rate','curriculum/timeout_rate',
                            'curriculum/obstacles_behind_gate','curriculum/total_assets','curriculum/max_level_reached',
                            'curriculum/camera_gaussian_std','curriculum/camera_dropout_rate',
                            'curriculum/camera_frame_dropout_drone_total','curriculum/camera_frame_dropout_static_total',
                            'curriculum/camera_frame_freeze_drone','curriculum/camera_frame_blank_drone',
                            'curriculum/camera_frame_freeze_static','curriculum/camera_frame_blank_static',
                            'curriculum/camera_max_angle','curriculum/camera_current_angle',
                            'curriculum/state_noise_drone_pos_std_m','curriculum/state_noise_drone_orient_std_deg',
                            'curriculum/state_noise_static_pos_std_m','curriculum/state_noise_static_orient_std_deg',
                        )
                        mirrored = set()
                        for key in snapshot_keys:
                            val = infos.get(key, None)
                            if val is not None:
                                extra[key] = float(val.mean().item()) if hasattr(val, 'mean') else float(val)
                                mirrored.add(key)
                        # Fallback compute block (only for those not mirrored)
                        if task is not None:
                            # Current level/progress
                            if 'curriculum/level' not in mirrored:
                                extra['curriculum/level'] = float(task.curriculum_level)
                            if 'curriculum/progress' not in mirrored:
                                extra['curriculum/progress'] = float(task.curriculum_progress_fraction)
                            # Environment totals
                            try:
                                cur_lvl_val = int(task.curriculum_level)
                                curri = task.task_config.curriculum
                                if curri is not None and hasattr(curri, 'get_obstacle_count_behind_gate'):
                                    obg = int(curri.get_obstacle_count_behind_gate(cur_lvl_val))
                                else:
                                    obg = 0
                            except (ValueError, TypeError):
                                obg = 0
                            if 'curriculum/obstacles_behind_gate' not in mirrored:
                                extra['curriculum/obstacles_behind_gate'] = float(obg)
                            # Visible fixed assets (1 gate + 6 walls)
                            fixed_assets_visible = 1 + 6
                            total_assets = fixed_assets_visible + obg
                            if 'curriculum/total_assets' not in mirrored:
                                extra['curriculum/total_assets'] = float(total_assets)
                            if 'curriculum/max_level_reached' not in mirrored:
                                extra['curriculum/max_level_reached'] = float(getattr(task, 'max_curriculum_level_reached', cur_lvl_val))
                            # Camera noise
                            try:
                                if curri is not None and hasattr(curri, 'get_camera_noise'):
                                    cstd, cdrop = curri.get_camera_noise(cur_lvl_val)
                                else:
                                    cstd, cdrop = 0.0, 0.0
                            except Exception:
                                cstd, cdrop = 0.0, 0.0
                            if 'curriculum/camera_gaussian_std' not in mirrored:
                                extra['curriculum/camera_gaussian_std'] = float(cstd)
                            if 'curriculum/camera_dropout_rate' not in mirrored:
                                extra['curriculum/camera_dropout_rate'] = float(cdrop)
                            # Frame dropout
                            try:
                                if curri is not None and hasattr(curri, 'get_camera_frame_dropout'):
                                    fd = curri.get_camera_frame_dropout(cur_lvl_val)
                                else:
                                    fd = {'drone_total':0.0,'static_total':0.0,'drone_freeze':0.0,'drone_blank':0.0,'static_freeze':0.0,'static_blank':0.0}
                            except Exception:
                                fd = {'drone_total':0.0,'static_total':0.0,'drone_freeze':0.0,'drone_blank':0.0,'static_freeze':0.0,'static_blank':0.0}
                            def _put_if_missing(k) -> None:
                                if k not in mirrored:
                                    extra[k] = float(fd.get(k.split('/')[-1], 0.0)) if 'curriculum/' in k else float(fd.get(k, 0.0))
                            _put_if_missing('curriculum/camera_frame_dropout_drone_total')
                            _put_if_missing('curriculum/camera_frame_dropout_static_total')
                            _put_if_missing('curriculum/camera_frame_freeze_drone')
                            _put_if_missing('curriculum/camera_frame_blank_drone')
                            _put_if_missing('curriculum/camera_frame_freeze_static')
                            _put_if_missing('curriculum/camera_frame_blank_static')
                            # Camera angles
                            try:
                                max_angle = getattr(task, 'max_camera_angle', None)
                                if max_angle is None and curri is not None and hasattr(curri, 'get_static_camera_difficulty'):
                                    max_angle, _, _ = curri.get_static_camera_difficulty(cur_lvl_val)
                            except Exception:
                                max_angle = 0.0
                            if 'curriculum/camera_max_angle' not in mirrored:
                                extra['curriculum/camera_max_angle'] = float(max_angle if max_angle is not None else 0.0)
                            try:
                                cur_angle = 0.0
                                scm = getattr(task, 'static_camera_manager', None)
                                if scm is not None and hasattr(scm, 'current_camera_angles') and scm.current_camera_angles:
                                    cur_angle = float(scm.current_camera_angles[0])
                            except (ValueError, TypeError):
                                cur_angle = 0.0
                            if 'curriculum/camera_current_angle' not in mirrored:
                                extra['curriculum/camera_current_angle'] = float(cur_angle)
                            # State noise
                            sn = None
                            if curri is not None and getattr(curri, 'enable_state_noise', False) and hasattr(curri, 'get_state_noise'):
                                sn = curri.get_state_noise(cur_lvl_val)
                            if sn is not None:
                                extra.setdefault('curriculum/state_noise_drone_pos_std_m', float(sn.get('drone_pos_std_m', 0.0)))
                                extra.setdefault('curriculum/state_noise_drone_orient_std_deg', float(sn.get('drone_orient_std_rad', 0.0) * 57.2958))
                                extra.setdefault('curriculum/state_noise_static_pos_std_m', float(sn.get('static_pos_std_m', 0.0)))
                                extra.setdefault('curriculum/state_noise_static_orient_std_deg', float(sn.get('static_orient_std_rad', 0.0) * 57.2958))
                            # Success/Crash/Timeout rates (run-level, from running totals)
                            tot_s = float(self._curriculum_totals['total_successes'])
                            tot_c = float(self._curriculum_totals['total_crashes'])
                            tot_t = float(self._curriculum_totals['total_timeouts'])
                            total = max(1.0, tot_s + tot_c + tot_t)
                            extra.setdefault('curriculum/success_rate', tot_s / total)
                            extra.setdefault('curriculum/crash_rate', tot_c / total)
                            extra.setdefault('curriculum/timeout_rate', tot_t / total)
                    except (ValueError, TypeError):
                        pass
                    infos['episode_extra_stats'] = extra
                    debug_groups = {
                        'curriculum_current': [
                            'curriculum/current_level','curriculum/current_progress'
                        ],
                        'curriculum_totals': [
                            'curriculum/total_successes','curriculum/total_crashes','curriculum/total_timeouts'
                        ],
                        'per_episode_counts': [
                            'successes','crashes','timeouts'
                        ],
                        'gate_camera': [
                            'gate/passed','gate/distance','gate/alignment',
                            'camera/facing_alignment','camera/alignment_angle_deg','camera/alignment_category'
                        ],
                        'curriculum_snapshot_core': [
                            'curriculum/level','curriculum/progress','curriculum/success_rate',
                            'curriculum/crash_rate','curriculum/timeout_rate'
                        ],
                        'curriculum_snapshot_env': [
                            'curriculum/obstacles_behind_gate','curriculum/total_assets','curriculum/max_level_reached'
                        ],
                        'curriculum_snapshot_camera': [
                            'curriculum/camera_gaussian_std','curriculum/camera_dropout_rate',
                            'curriculum/camera_frame_dropout_drone_total','curriculum/camera_frame_dropout_static_total',
                            'curriculum/camera_frame_freeze_drone','curriculum/camera_frame_blank_drone',
                            'curriculum/camera_frame_freeze_static','curriculum/camera_frame_blank_static',
                            'curriculum/camera_max_angle','curriculum/camera_current_angle'
                        ],
                        'curriculum_snapshot_state_noise': [
                            'curriculum/state_noise_drone_pos_std_m','curriculum/state_noise_drone_orient_std_deg',
                            'curriculum/state_noise_static_pos_std_m','curriculum/state_noise_static_orient_std_deg'
                        ],
                    }
                    for group_name, keys in debug_groups.items():
                        present = [k for k in keys if k in extra]
                        if present:
                            preview = {k: extra[k] for k in present}
        except (ValueError, TypeError):
            pass
        
        # DYNAMIC OBSERVATION PROCESSING: Handle both standard DCE (81D) and gate navigation (145D)
        # Task provides "observations" with correct dimensionality (81D or 145D) based on task type
        # We pass this through as "obs" for Sample Factory
        transformed_obs = {"obs": obs["observations"]}
        # Apply return-drop ablation if requested
        transformed_obs["obs"] = self._apply_obs_ablation(transformed_obs["obs"]) 
        # Sanitize non-finite values BEFORE Sample Factory normalization
        try:
            vec = transformed_obs.get('obs', None)
            if isinstance(vec, torch.Tensor):
                if not torch.isfinite(vec).all():
                    import os as _os
                    if _os.environ.get('ABLATE_DEBUG', 'false').lower() == 'true':
                        n_bad = int((~torch.isfinite(vec)).sum().item())
                        print(f"[SANITIZE][step] replacing {n_bad} non-finite obs values with 0")
                transformed_obs['obs'] = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        except (ValueError, TypeError):
            pass
        
        self.step_count += 1
        # Removed step-based GIF debugging - now using episode-based saving every 50 episodes
        
        return transformed_obs, rew, terminated, truncated, infos

    def render(self) -> None:
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
    
    # Use environment variable from shell script if set, otherwise default to 256 (high parallelization config)
    # This will be updated based on the actual env_agents parameter when env is created
    current_env_agents = os.environ.get('SF_ENV_AGENTS', '256')  # Default to high parallelization configuration
    os.environ['SF_ENV_AGENTS'] = current_env_agents
    
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
