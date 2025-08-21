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
import numpy as np
import matplotlib.cm
from PIL import Image
import os
import math


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
    Enhanced with dual camera GIF saving functionality and 4D action space for gate navigation.
    """

    def __init__(self, aerialgym_env, obs_key, save_gifs=False):
        self.env = aerialgym_env
        self.num_agents = self.env.num_envs
        self.is_multiagent = True
        
        # GIF saving functionality
        self.save_gifs = save_gifs
        if self.save_gifs:
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
            print(f"[AerialGymVecEnv] GIF saving DISABLED")
        
        self.step_count = 0
        # CRITICAL FIX: Force action space to exactly match inference expectations (4D for gate navigation)
        # The inference script expects 4D actions [x_vel, y_vel, z_vel, yaw_rate], so train with 4D to avoid shape mismatch
        import numpy as np
        base_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)  # UPDATED: 4D action space
        self.action_space = convert_space(base_action_space)

        # Debug: Print action space info to verify it's 4D
        print(f"[AerialGymVecEnv] Forced action space shape: {self.action_space.shape}")
        print(f"[AerialGymVecEnv] is_multiagent: {self.is_multiagent}, num_agents: {self.num_agents}")

        # DYNAMIC OBSERVATION SPACE: Detect observation space dimension from task config
        # This handles both standard DCE navigation (81D) and gate navigation (147D)
        if obs_key == "obs":
            # Get the actual observation space dimension from the task configuration
            task_obs_dim = getattr(self.env.task_config, 'observation_space_dim', 150)  # Default to 150D for gate navigation
            print(f"[AerialGymVecEnv] Detected observation space: {task_obs_dim}D")
            
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

    def _process_camera_image(self, image_data, camera_type="depth"):
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
          - ABLATE_OBS_RANGES="0:3=zero,22:86=shuffle,86:150=noise:0.1"
        Ops:
          - zero     : set slice to 0
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
            except Exception:
                continue
            op = rhs
            if op == "zero":
                obs_tensor[:, start:end] = 0.0
                if debug and self._ablate_debug_count < 10:
                    v = obs_tensor[:, start:end]
                    print(f"[ABLATE_DEBUG] applied: {start}:{end}=zero | min={v.min().item():.3e} max={v.max().item():.3e} mean={v.mean().item():.3e} nonzero={int((v!=0).sum().item())}")
                    self._ablate_debug_count += 1
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
                except Exception:
                    std = 0.0
                if std > 0.0:
                    obs_tensor[:, start:end] = obs_tensor[:, start:end] + torch.randn_like(obs_tensor[:, start:end]) * std
                if debug and self._ablate_debug_count < 10:
                    v = obs_tensor[:, start:end]
                    print(f"[ABLATE_DEBUG] applied: {start}:{end}=noise:{std} | std_est={v.std().item():.3e} mean={v.mean().item():.3e}")
                    self._ablate_debug_count += 1
        return obs_tensor

    def _collect_frames(self, obs_dict):
        """Collect frames from both drone and static cameras for GIF generation (clean + noised versions)."""
        if not self.save_gifs:
            return
        
        try:
            # Access camera data directly from the underlying task
            task = self.env
            
            # === CLEAN DRONE CAMERA IMAGES ===
            # Get drone camera depth image from task's obs_dict (original clean version)
            if hasattr(task, 'obs_dict') and "depth_range_pixels" in task.obs_dict:
                drone_depth = task.obs_dict["depth_range_pixels"][0, 0]  # First env, first camera
                drone_depth_img = self._process_camera_image(drone_depth, "depth")
                self.drone_depth_frames[0].append(drone_depth_img)
            
            # Get drone camera segmentation image from task's obs_dict
            if hasattr(task, 'obs_dict') and "segmentation_pixels" in task.obs_dict:
                drone_seg = task.obs_dict["segmentation_pixels"][0, 0]  # First env, first camera  
                drone_seg_img = self._process_camera_image(drone_seg, "segmentation")
                self.drone_seg_frames[0].append(drone_seg_img)
            
            # === D455 NOISED DRONE CAMERA IMAGES ===
            # Get noised drone camera depth image (with D455 noise applied)
            if hasattr(task, 'obs_dict') and "depth_range_pixels_noised" in task.obs_dict:
                drone_depth_noised = task.obs_dict["depth_range_pixels_noised"][0, 0]  # First env, first camera
                drone_depth_noised_img = self._process_camera_image(drone_depth_noised, "depth")
                self.drone_depth_noised_frames[0].append(drone_depth_noised_img)
            
            # === CLEAN STATIC CAMERA IMAGES ===
            # Get static camera images from stored clean versions
            if hasattr(task, 'obs_dict') and "static_depth_clean" in task.obs_dict:
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
            if hasattr(task, 'obs_dict') and "static_seg" in task.obs_dict:
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
                
            # === D455 NOISED STATIC CAMERA IMAGES ===
            # Get noised static camera depth image (with D455 noise applied)
            if hasattr(task, 'obs_dict') and "static_depth_noised" in task.obs_dict:
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
            
            # === MERGED CLEAN IMAGES (drone + static side by side) ===
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
            
            # === MERGED NOISED IMAGES (drone + static side by side) ===
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
                
        except Exception as e:
            print(f"[GIF] Warning: Failed to collect frames: {e}")
            import traceback
            print(f"[GIF] Traceback: {traceback.format_exc()}")

    def _save_episode_gifs(self, env_id=0):
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
            except Exception:
                level_suffix = ""
            
            # Save drone camera GIFs
            if len(self.drone_depth_frames[env_id]) > 0:
                gif_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_drone_depth{level_suffix}.gif")
                self.drone_depth_frames[env_id][0].save(
                    gif_path,
                    save_all=True,
                    append_images=self.drone_depth_frames[env_id][1:],
                    duration=100,
                    loop=0
                )
                print(f"[GIF] Saved drone depth: {gif_path}")
            
            if len(self.drone_seg_frames[env_id]) > 0:
                gif_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_drone_seg{level_suffix}.gif")
                self.drone_seg_frames[env_id][0].save(
                    gif_path,
                    save_all=True,
                    append_images=self.drone_seg_frames[env_id][1:],
                    duration=100,
                    loop=0
                )
                print(f"[GIF] Saved drone segmentation: {gif_path}")
            
            # Save static camera GIFs
            if len(self.static_depth_frames[env_id]) > 0:
                gif_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_static_depth{level_suffix}.gif")
                self.static_depth_frames[env_id][0].save(
                    gif_path,
                    save_all=True,
                    append_images=self.static_depth_frames[env_id][1:],
                    duration=100,
                    loop=0
                )
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
                print(f"[GIF] Saved static segmentation: {gif_path}")
            
            # Save merged GIF (drone + static side by side - CLEAN versions)
            if len(self.merged_frames[env_id]) > 0:
                gif_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_merged_dual_camera_CLEAN{level_suffix}.gif")
                self.merged_frames[env_id][0].save(
                    gif_path,
                    save_all=True,
                    append_images=self.merged_frames[env_id][1:],
                    duration=100,
                    loop=0
                )
                print(f"[GIF] Saved merged dual camera (CLEAN): {gif_path}")
            
            # === D455 NOISED CAMERA GIFS ===
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
                print(f"[GIF] Saved static depth (D455 NOISED): {gif_path}")
            
            # Save merged noised GIF (drone + static side by side - NOISED versions)
            if len(self.merged_noised_frames[env_id]) > 0:
                gif_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_merged_dual_camera_D455_NOISED{level_suffix}.gif")
                self.merged_noised_frames[env_id][0].save(
                    gif_path,
                    save_all=True,
                    append_images=self.merged_noised_frames[env_id][1:],
                    duration=100,
                    loop=0
                )
                print(f"[GIF] Saved merged dual camera (D455 NOISED): {gif_path}")
        
        except Exception as e:
            print(f"[GIF] Warning: Failed to save GIFs for episode {episode_num}: {e}")

        # === OPTIONAL: Save originals vs VAE reconstructions (depth) ===
        # try:
        #     task = self.env
        #     import numpy as np
        #     import torch
        #     from PIL import Image as PILImage
        #
        #     def _decode_to_pil(vae, latent_id, target_size_wh):
        #         z = latent_id.detach().unsqueeze(0)  # (1, L)
        #         dec = vae.decode(z)
        #         img = dec[0].clamp(0, 1).cpu().numpy()
        #         img_u8 = (img * 255.0).astype(np.uint8)
        #         pil = PILImage.fromarray(img_u8, mode='L')
        #         if target_size_wh is not None and pil.size != target_size_wh:
        #             pil = pil.resize(target_size_wh)
        #         return pil
        #
        #     def _stack_horiz(pil_left, pil_right):
        #         w, h = pil_left.size
        #         canvas = PILImage.new('L', (w * 2, h))
        #         canvas.paste(pil_left, (0, 0))
        #         canvas.paste(pil_right, (w, 0))
        #         return canvas
        #
        #     # Drone recon grid
        #     if hasattr(task, 'shared_vae_model') and hasattr(task, 'image_latents') and self.drone_depth_frames[env_id]:
        #         orig_drone_pil = self.drone_depth_frames[env_id][-1].convert('L')
        #         recon_drone_pil = _decode_to_pil(task.shared_vae_model, task.image_latents[0], orig_drone_pil.size)
        #         grid = _stack_horiz(orig_drone_pil, recon_drone_pil)
        #         out_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_recon_grid_drone{level_suffix}.png")
        #         grid.save(out_path)
        #         print(f"[GIF] Saved drone recon grid: {out_path}")
        #
        #     # Static recon grid
        #     if hasattr(task, 'shared_vae_model') and hasattr(task, 'static_image_latents') and self.static_depth_frames[env_id]:
        #         orig_static_pil = self.static_depth_frames[env_id][-1].convert('L')
        #         recon_static_pil = _decode_to_pil(task.shared_vae_model, task.static_image_latents[0], orig_static_pil.size)
        #         grid = _stack_horiz(orig_static_pil, recon_static_pil)
        #         out_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_recon_grid_static{level_suffix}.png")
        #         grid.save(out_path)
        #         print(f"[GIF] Saved static recon grid: {out_path}")
        # except Exception as e:
        #     print(f"[GIF] Warning: Failed to save recon grids: {e}")

    def _clear_frames(self, env_id=0):
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
        
        # Collect first frame if GIF saving is enabled
        if self.save_gifs:
            self._collect_frames(obs)
        
        return transformed_obs, infos

    def step(self, action) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Dict]:
        # FIXED: Direct 4D action pass-through for DCE gate navigation task
        # Sample Factory now provides 4D actions directly matching DCE task expectations (x_vel, y_vel, z_vel, yaw_rate)
        dce_action = action
            
        obs, rew, terminated, truncated, infos = self.env.step(dce_action)
        
        # Collect frames for GIF generation
        if self.save_gifs:
            self._collect_frames(obs)
        
        # Save GIFs when episodes terminate
        if self.save_gifs and (torch.any(terminated) or torch.any(truncated)):
            # Save GIFs for terminated/truncated environments (only save for env 0 to avoid spam)
            reset_ids = (terminated + truncated).nonzero(as_tuple=True)[0]
            if len(reset_ids) > 0 and 0 in reset_ids:  # Only save for first environment
                # Check if this episode should be saved (every 12 episodes including episode 0) - 4x more frequent than before
                save_this_episode = (self.episode_count % 15 == 0)
                
                if save_this_episode:
                    if terminated[0]:
                        print(f"[GIF] Episode {self.episode_count} terminated - saving GIFs (every 15 episodes)")
                    elif truncated[0]:
                        print(f"[GIF] Episode {self.episode_count} truncated - saving GIFs (every 15 episodes)")
                    self._save_episode_gifs(env_id=0)
                    self._clear_frames(env_id=0)
                else:
                    # Don't save but still clear frames to free memory
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
                    task = getattr(self, 'env', None)
                    curr_level = None
                    if task is not None:
                        curr_level = getattr(task, 'curriculum_level', None)
                        # Also pull step-averaged traj metrics directly if the task stashed them
                        traj_avg = getattr(task, '_last_traj_metrics_avg', None)
                        # NEW: Update running aggregates using per-env episode metrics when resets happen
                        per_env = getattr(task, '_last_traj_metrics_per_env', None)
                        if isinstance(per_env, dict):
                            # Limit to environments that actually reset this step
                            reset_ids = ids.detach().cpu().tolist()
                            crossed_mask = per_env.get('crossed', None)
                            def _to_list(t):
                                return t.detach().cpu().tolist() if torch.is_tensor(t) else t
                            if reset_ids is not None:
                                for eid in reset_ids:
                                    try:
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
                                    except Exception:
                                        pass
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
                            try:
                                extra[k] = float(v)
                            except Exception:
                                pass
                        # Also mirror last-position metrics when present
                        for k in ('last_position_x','last_position_y','last_position_z','last_center_distance'):
                            if k in traj_avg:
                                try:
                                    extra[k] = float(traj_avg[k])
                                except Exception:
                                    pass
                    # Pass-through any episode-level trajectory metrics already stored by env
                    # They will be aggregated by SF and picked up by the learner later
                    for k in ('path_efficiency','time_to_gate_steps','min_gate_distance','center_offset_success','height_offset_success'):
                        if k in extra:
                            # ensure float cast
                            try:
                                extra[k] = float(extra[k])
                            except Exception:
                                pass
                    # Include last-position series if present (already floats)
                    for k in ('last_position_x','last_position_y','last_position_z','last_center_distance'):
                        if k in extra:
                            try:
                                extra[k] = float(extra[k])
                            except Exception:
                                pass
                    # Add running-mean episode-level metrics (finite by construction; NaNs avoided by counts)
                    def _safe_mean(sum_key, count_key):
                        s = self._traj_running.get(sum_key, 0.0)
                        c = max(1, self._traj_running.get(count_key, 0))
                        return float(s / c)
                    extra['path_efficiency_running_mean'] = _safe_mean('path_efficiency_sum', 'path_efficiency_count')
                    extra['min_gate_distance_running_mean'] = _safe_mean('min_gate_distance_sum', 'min_gate_distance_count')
                    # Means conditioned on success
                    extra['time_to_gate_running_mean'] = _safe_mean('time_to_gate_sum', 'time_to_gate_count')
                    extra['center_offset_running_mean'] = _safe_mean('center_offset_sum', 'center_offset_count')
                    extra['height_offset_running_mean'] = _safe_mean('height_offset_sum', 'height_offset_count')
                    # Helpful counts
                    extra['gate_pass_rate'] = float(self._traj_running['episodes_crossed']) / float(max(1, self._traj_running['episodes_total']))
                    extra['episodes_total'] = float(self._traj_running['episodes_total'])
                    extra['episodes_crossed'] = float(self._traj_running['episodes_crossed'])

                    # Curriculum (generic task) counters derived from infos flags
                    # Sum over the envs that reset this step
                    try:
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
                    except Exception:
                        pass

                    # Mirror curriculum/current_* using task attributes (fallback) so they always show up
                    try:
                        # Prefer values emitted by env in infos; otherwise fall back to task attributes
                        cur_lvl_tensor = infos.get('curriculum/current_level', None)
                        cur_prog_tensor = infos.get('curriculum/current_progress', None)
                        if cur_lvl_tensor is not None:
                            extra['curriculum/current_level'] = float(cur_lvl_tensor.mean().item()) if hasattr(cur_lvl_tensor, 'mean') else float(cur_lvl_tensor)
                        else:
                            if task is not None and hasattr(task, 'curriculum_level'):
                                extra['curriculum/current_level'] = float(getattr(task, 'curriculum_level'))
                        if cur_prog_tensor is not None:
                            extra['curriculum/current_progress'] = float(cur_prog_tensor.mean().item()) if hasattr(cur_prog_tensor, 'mean') else float(cur_prog_tensor)
                        else:
                            if task is not None and hasattr(task, 'curriculum_progress_fraction'):
                                extra['curriculum/current_progress'] = float(getattr(task, 'curriculum_progress_fraction'))
                    except Exception:
                        pass

                    # Gate/task-specific + camera alignment — mean across envs if present in infos
                    try:
                        gate_keys = (
                            'gate/passed','gate/distance','gate/alignment',
                            'camera/facing_alignment','camera/alignment_angle_deg','camera/alignment_category',
                        )
                        for key in gate_keys:
                            val = infos.get(key, None)
                            if val is not None:
                                extra[key] = float(val.mean().item()) if hasattr(val, 'mean') else float(val)
                    except Exception:
                        pass

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
                                extra['curriculum/level'] = float(getattr(task, 'curriculum_level', curr_level if curr_level is not None else -1))
                            if 'curriculum/progress' not in mirrored:
                                extra['curriculum/progress'] = float(getattr(task, 'curriculum_progress_fraction', 0.0))
                            # Environment totals
                            try:
                                cur_lvl_val = int(getattr(task, 'curriculum_level', curr_level if curr_level is not None else 0))
                                curri = getattr(task.task_config, 'curriculum', None)
                                if curri is not None and hasattr(curri, 'get_obstacle_count_behind_gate'):
                                    obg = int(curri.get_obstacle_count_behind_gate(cur_lvl_val))
                                else:
                                    obg = 0
                            except Exception:
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
                            def _put_if_missing(k):
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
                            except Exception:
                                cur_angle = 0.0
                            if 'curriculum/camera_current_angle' not in mirrored:
                                extra['curriculum/camera_current_angle'] = float(cur_angle)
                            # State noise
                            try:
                                sn = None
                                if curri is not None and getattr(curri, 'enable_state_noise', False) and hasattr(curri, 'get_state_noise'):
                                    sn = curri.get_state_noise(cur_lvl_val)
                                if sn is not None:
                                    extra.setdefault('curriculum/state_noise_drone_pos_std_m', float(sn.get('drone_pos_std_m', 0.0)))
                                    extra.setdefault('curriculum/state_noise_drone_orient_std_deg', float(sn.get('drone_orient_std_rad', 0.0) * 57.2958))
                                    extra.setdefault('curriculum/state_noise_static_pos_std_m', float(sn.get('static_pos_std_m', 0.0)))
                                    extra.setdefault('curriculum/state_noise_static_orient_std_deg', float(sn.get('static_orient_std_rad', 0.0) * 57.2958))
                            except Exception:
                                pass
                            # Success/Crash/Timeout rates (run-level, from running totals)
                            try:
                                tot_s = float(self._curriculum_totals['total_successes'])
                                tot_c = float(self._curriculum_totals['total_crashes'])
                                tot_t = float(self._curriculum_totals['total_timeouts'])
                                total = max(1.0, tot_s + tot_c + tot_t)
                                extra.setdefault('curriculum/success_rate', tot_s / total)
                                extra.setdefault('curriculum/crash_rate', tot_c / total)
                                extra.setdefault('curriculum/timeout_rate', tot_t / total)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    infos['episode_extra_stats'] = extra
                    try:
                        # Existing compact log (kept for continuity)
                        # print(f"[W&B_DEBUG][worker] episode_extra_stats added: curriculum_level={extra['curriculum_level']}, curriculum_level_minus_1={extra['curriculum_level_minus_1']}, traj_keys={[k for k in extra.keys() if k in ['path_efficiency','time_to_gate_steps','min_gate_distance','center_offset_success','height_offset_success']]}")
                        # New temporary debug: show newly added groups
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
                                # print(f"[W&B_DEBUG][worker] {group_name}: {preview}")
                    except Exception:
                        pass
        except Exception:
            pass
        
        # DYNAMIC OBSERVATION PROCESSING: Handle both standard DCE (81D) and gate navigation (145D)
        # Task provides "observations" with correct dimensionality (81D or 145D) based on task type
        # We pass this through as "obs" for Sample Factory
        transformed_obs = {"obs": obs["observations"]}
        # Apply return-drop ablation if requested
        transformed_obs["obs"] = self._apply_obs_ablation(transformed_obs["obs"]) 
        
        self.step_count += 1
        # Removed step-based GIF debugging - now using episode-based saving every 50 episodes
        
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
                        gate_config.disable_gate_size_randomization = bool(getattr(cfg, 'disable_gate_size_randomization', False))
                    except Exception:
                        gate_config.disable_gate_size_randomization = False
                    try:
                        gate_config.fixed_gate_scale_percent = int(getattr(cfg, 'fixed_gate_scale_percent', 100))
                    except Exception:
                        gate_config.fixed_gate_scale_percent = 100
                    # Obstacle ablation flags
                    try:
                        gate_config.disable_obstacle_randomization = bool(getattr(cfg, 'disable_obstacle_randomization', False))
                    except Exception:
                        gate_config.disable_obstacle_randomization = False
                    try:
                        gate_config.fixed_obstacles_behind_gate = int(getattr(cfg, 'fixed_obstacles_behind_gate', 0))
                    except Exception:
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
                    print(f"[SUBPROCESS] Config batch_size: {getattr(cfg, 'batch_size', 'not set')}")
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
                    print(f"[SUBPROCESS] env_agents={getattr(cfg, 'env_agents', 'not set')}, using default num_envs")
                
                task_registry.register_task(register_name, TaskClass, config)
                # Also register backup name for backward compatibility
                task_registry.register_task(backup_name, TaskClass, config)
                print(f"Registered {register_name} and {backup_name} in subprocess")
            except Exception as e:
                print(f"Failed to register quad_with_obstacles in subprocess: {e}")

    # Get save_gifs parameter from config
    save_gifs = getattr(cfg, 'save_gifs', False)

    # Create the environment and force correct action space for inference compatibility
    env = AerialGymVecEnv(task_registry.make_task(task_name=full_task_name), "obs", save_gifs=save_gifs)
    
    # Debug: list available gate variants if present
    try:
        gd = getattr(env.env, 'global_tensor_dict', {})
        names0 = gd.get('gate_variant_names_per_env', [])
        if names0 and len(names0) > 0:
            print(f"[GateVariant] Available gate variants for env0: {names0[0]}")
        active = gd.get('active_gate_variant_index', None)
        if active is not None:
            print(f"[GateVariant] Active gate variant index tensor: {active}")
    except Exception as e:
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


def add_extra_params_func(parser):
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
    # State noise randomization ablation flag (drone & static pose noise)
    parser.add_argument("--disable_state_noise_randomization", type=lambda x: x.lower() == 'true', default=False, help="Disable pose state noise randomization for drone and static camera")
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
    
    # Complete observation influence tracking arguments
    parser.add_argument("--enable_gradient_monitoring", type=lambda x: x.lower() == 'true', default=False, help="Enable complete observation influence tracking")
    parser.add_argument("--gradient_log_interval", default=100, type=int, help="Log influence metrics every N steps")
    parser.add_argument("--gradient_print_interval", default=100, type=int, help="Print analysis summary every N steps")
    parser.add_argument("--enable_grad_attribution", type=lambda x: x.lower() == 'true', default=True, help="Enable gradient-based attribution alongside correlation analysis")
    
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
        rollout=32,  # REVERTED: Issue was tensor reference bug, not rollout frequency
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
        env_agents=16,  # Default to 16 environments for standard training
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
        actor_worker_gpus=[0],  # Match original config
        obs_key="obs",  # Match original config
        subtask=None,  # Match original config
        ige_api_version="preview4",  # Match original config
        eval_stats=False,  # Match original config
    ),
    quad_with_obstacles_gate=dict(
        # Gate Navigation Task Configuration
        # PURE VISION NAVIGATION EXPERIMENT: 141D observation space (13D basic + 64D drone VAE + 64D static camera VAE)
        # Removed explicit target guidance to test vision-based navigation through gate
        # X500 robot with D455 camera flying through static gate with static camera
        adaptive_stddev=True,  # Can use adaptive_stddev with 4D actions
        action_space_dim=4,  # 4D action space for VELOCITY CONTROLLER (x_vel, y_vel, z_vel, yaw_rate)
        
        # Gate Navigation Training Configuration
        train_for_env_steps=200000000,  # 200M steps for comprehensive gate navigation learning
        encoder_mlp_layers=[512, 256, 128],  # Larger network for 145D observation space
        encoder_conv_mlp_layers=[],  # No ConvNet - VAE latents handled by gate task
        use_rnn=True,
        rnn_num_layers=1,
        rnn_size=128,  # Larger RNN for gate navigation complexity
        rnn_type="gru",
        recurrence=32,
        gamma=0.98,
        reward_scale=0.1,
        rollout=32,  # REVERTED: Issue was tensor reference bug, not rollout frequency
        learning_rate=0.0003,
        lr_schedule="kl_adaptive_epoch",
        lr_schedule_kl_threshold=0.016,
        
        # Optimized for 128 environments with dual camera processing
        batch_size=8192,  # Reduced batch size for 145D observations and dual cameras
        num_batches_to_accumulate=2,  # Accumulate for effective batch size 16384
        num_batches_per_epoch=4,  # Keep same as base DCE
        num_epochs=4,
        max_grad_norm=1.0,
        exploration_loss_coeff=0.001,
        value_loss_coeff=2.0,
        kl_loss_coeff=0.1,
        normalize_input=True,
        normalize_returns=True,
        async_rl=True,
        serial_mode=True,
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        
        # Gate Navigation Environment Configuration
        env_agents=16,  # Default to 16 environments; can still be overridden via CLI/env
        worker_num_splits=1,
        policy_workers_per_policy=1,
        nonlinearity="elu",
        shuffle_minibatches=False,
        gae_lambda=0.95,
        ppo_clip_ratio=0.2,
        ppo_clip_value=1.0,
        with_vtrace=False,
        value_bootstrap=True,
        reward_clip=1000.0,
        obs_subtract_mean=0.0,
        obs_scale=1.0,
        decorrelate_experience_max_seconds=0,
        decorrelate_envs_on_one_worker=True,
        max_policy_lag=1000,
        vtrace_rho=1.0,
        vtrace_c=1.0,
        lr_adaptive_min=1e-06,
        lr_adaptive_max=0.01,
        
        # Checkpoint and logging
        save_every_sec=120,  # Save every 2 minutes
        keep_checkpoints=5,
        save_milestones_sec=-1,
        save_best_every_sec=5,
        save_best_metric="reward",
        save_best_after=100000,  # Save best models after 100K steps
        
        # Model configuration
        policy_initialization="torch_default",
        policy_init_gain=1.0,
        actor_critic_share_weights=True,
        continuous_tanh_scale=0.0,
        initial_stddev=1.0,
        restart_behavior="resume",
        optimizer="adam",
        adam_eps=1e-06,
        adam_beta1=0.9,
        adam_beta2=0.999,
        exploration_loss="entropy",
        decoder_mlp_layers=[],
        
        # Environment settings
        env_frameskip=1,
        env_framestack=1,
        pixel_format="CHW",
        use_record_episode_statistics=False,
        normalize_input_keys=None,
        set_workers_cpu_affinity=True,
        force_envs_single_thread=False,
        default_niceness=0,
        
        # Logging and monitoring
        log_to_file=True,
        experiment_summaries_interval=10,
        flush_summaries_interval=30,
        stats_avg=100,
        summaries_use_frameskip=True,
        heartbeat_interval=20,
        heartbeat_reporting_interval=180,
        train_for_seconds=10000000000,
        load_checkpoint_kind="latest",
        benchmark=False,
        
        # Weights & Biases logging
        with_wandb=True,
        wandb_project="gate_navigation_dual_camera",  # New project for gate navigation
        wandb_user="ziya-ruso-ucl",
        wandb_group="gate_navigation_training",
        wandb_tags=["aerial_gym", "gate_navigation", "dual_camera", "x500", "sample_factory"],
        wandb_job_type="SF",
        
        # Population Based Training (disabled for now)
        with_pbt=False,
        pbt_mix_policies_in_one_env=True,
        pbt_period_env_steps=5000000,
        pbt_start_mutation=20000000,
        pbt_replace_fraction=0.3,
        pbt_mutation_rate=0.15,
        pbt_replace_reward_gap=0.1,
        pbt_replace_reward_gap_absolute=1e-06,
        pbt_optimize_gamma=False,
        pbt_target_objective="true_objective",
        pbt_perturb_min=1.1,
        pbt_perturb_max=1.5,
        
        # Sample Factory specific
        help=False,
        algo="APPO",
        device="gpu",
        seed=None,
        num_policies=1,
        actor_worker_gpus=[0],
        obs_key="obs",
        subtask=None,
        ige_api_version="preview4",
        eval_stats=False,
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
    except Exception as e:
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
    except Exception as e:
        print(f"Warning: Could not register quad_with_obstacles_gate: {e}")
    
    for env_name in env_configs:
        register_env(env_name, make_aerialgym_env)

    # Don't register custom encoder since DCE task handles VAE encoding internally


def parse_aerialgym_cfg(evaluation=False):
    parser, partial_cfg = parse_sf_args(evaluation=evaluation)
    add_extra_params_func(parser)
    override_default_params_func(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser)
    # Bridge CLI flag to environment variable so worker processes can read it reliably
    try:
        if hasattr(final_cfg, 'disable_static_camera_orientation_randomization'):
            os.environ['SF_DISABLE_STATIC_CAMERA_ORIENT_RANDOMIZATION'] = 'true' if final_cfg.disable_static_camera_orientation_randomization else 'false'
            print(f"[CFG] static camera orientation randomization disabled: {final_cfg.disable_static_camera_orientation_randomization}")
        if hasattr(final_cfg, 'disable_camera_noise_randomization'):
            os.environ['SF_DISABLE_CAMERA_NOISE_RANDOMIZATION'] = 'true' if final_cfg.disable_camera_noise_randomization else 'false'
            print(f"[CFG] camera noise randomization disabled: {final_cfg.disable_camera_noise_randomization}")
        if hasattr(final_cfg, 'disable_camera_frame_dropout_randomization'):
            os.environ['SF_DISABLE_CAMERA_FRAME_DROPOUT_RANDOMIZATION'] = 'true' if final_cfg.disable_camera_frame_dropout_randomization else 'false'
            print(f"[CFG] camera frame dropout randomization disabled: {final_cfg.disable_camera_frame_dropout_randomization}")
        if hasattr(final_cfg, 'disable_state_noise_randomization'):
            os.environ['SF_DISABLE_STATE_NOISE_RANDOMIZATION'] = 'true' if final_cfg.disable_state_noise_randomization else 'false'
            print(f"[CFG] state noise randomization disabled: {final_cfg.disable_state_noise_randomization}")
        if hasattr(final_cfg, 'disable_spawn_position_randomization'):
            os.environ['SF_DISABLE_SPAWN_POSITION_RANDOMIZATION'] = 'true' if final_cfg.disable_spawn_position_randomization else 'false'
            print(f"[CFG] spawn position randomization disabled: {final_cfg.disable_spawn_position_randomization}")
        if hasattr(final_cfg, 'disable_spawn_orientation_randomization'):
            os.environ['SF_DISABLE_SPAWN_ORIENTATION_RANDOMIZATION'] = 'true' if final_cfg.disable_spawn_orientation_randomization else 'false'
            print(f"[CFG] spawn orientation randomization disabled: {final_cfg.disable_spawn_orientation_randomization}")
        if hasattr(final_cfg, 'disable_curriculum_multiplier'):
            os.environ['SF_DISABLE_CURRICULUM_MULTIPLIER'] = 'true' if final_cfg.disable_curriculum_multiplier else 'false'
            print(f"[CFG] curriculum multiplier disabled: {final_cfg.disable_curriculum_multiplier}")
        if hasattr(final_cfg, 'force_curriculum_level') and (final_cfg.force_curriculum_level is not None):
            try:
                lvl_str = str(final_cfg.force_curriculum_level).strip().lower()
                if lvl_str and lvl_str != 'none':
                    os.environ['SF_FORCE_CURRICULUM_LEVEL'] = str(int(lvl_str))
                    print(f"[CFG] forcing curriculum level: {lvl_str}")
                else:
                    # ensure any previous env var is cleared
                    os.environ.pop('SF_FORCE_CURRICULUM_LEVEL', None)
                    print("[CFG] force curriculum level: none (disabled)")
            except Exception:
                pass
    except Exception:
        pass
    return final_cfg


def main():
    """Script entry point."""
    register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg()
    
    # Check if complete observation influence tracking is enabled
    # Always attempt enhanced run; it falls back internally if trackers are unavailable
    return run_with_influence_tracking(cfg)


def run_with_influence_tracking(cfg: Config):
    """Enhanced training with complete observation influence tracking."""
    
    # Import the complete observation influence tracker and gradient attribution
    try:
        from aerial_gym.utils.gradient_monitor import create_influence_tracker, INFLUENCE_MONITOR_AVAILABLE
        from aerial_gym.utils.gradient_attribution import create_gradient_tracker, GRAD_ATTR_AVAILABLE
    except ImportError:
        print("❌ Influence/gradient trackers not available")
        INFLUENCE_MONITOR_AVAILABLE = False
        GRAD_ATTR_AVAILABLE = False
    
    if not INFLUENCE_MONITOR_AVAILABLE:
        print("❌ Complete observation influence tracker not available - falling back to standard training")
        return run_rl(cfg)

    print("🔬 Complete observation influence tracking ENABLED - analyzing ALL 150D observation components")
    print("   📊 Log interval: {} steps".format(getattr(cfg, 'gradient_log_interval', 100)))
    print("   📋 Print interval: {} steps".format(getattr(cfg, 'gradient_print_interval', 100)))
    print("✅ Complete observation influence tracker ready")
    print("🔍 Will analyze ALL 150D observation components for neural network influence")
    
    # Store original wandb.log if wandb is used
    original_wandb_log = None
    if cfg.with_wandb:
        try:
            import wandb
            original_wandb_log = wandb.log
        except ImportError:
            pass

    # Create tracker instances (attached to model later)
    influence_tracker = None
    grad_tracker = None

    # Cache last non-empty obs_grad metrics to mirror every step
    _last_obsgrad_from_influence = {}
    _last_obsgrad_from_grad = {}
    # Continuous curriculum logging: remember last known values and emit every step
    CURRICULUM_KEYS = [
        'curriculum/total_timeouts',
        'curriculum/total_successes',
        'curriculum/total_crashes',
        'curriculum/total_resets',
        'curriculum/timeout_rate',
        'curriculum/success_rate',
        'curriculum/state_noise_static_pos_std_m',
        'curriculum/state_noise_static_orient_std_deg',
        'curriculum/state_noise_drone_pos_std_m',
        'curriculum/state_noise_drone_orient_std_deg',
        'curriculum/progress',
        'curriculum/obstacles_behind_gate',
        'curriculum/max_level_reached',
        'curriculum/level',
        'curriculum/crash_rate',
        'curriculum/camera_max_angle',
        'curriculum/camera_gaussian_std',
        'curriculum/camera_frame_freeze_static',
        'curriculum/camera_frame_freeze_drone',
        'curriculum/camera_frame_dropout_static_total',
        'curriculum/camera_frame_dropout_drone_total',
        'curriculum/camera_frame_blank_static',
        'curriculum/camera_frame_blank_drone',
        'curriculum/camera_dropout_rate',
        'curriculum/camera_current_angle',
    ]
    _last_curriculum = {k: 0.0 for k in CURRICULUM_KEYS}

    def enhanced_wandb_log(metrics, **kwargs):
        """Enhanced wandb logging that includes influence monitoring metrics"""
        nonlocal influence_tracker
        nonlocal _last_obsgrad_from_influence
        nonlocal _last_obsgrad_from_grad
        # Ensure a dict
        if metrics is None:
            metrics = {}
        else:
            metrics = dict(metrics)
        
        # Inject step from cfg if not provided
        step_key = None
        if 'step' in kwargs:
            step_key = 'step'
        elif 'commit' in kwargs and isinstance(kwargs.get('commit'), bool):
            # no explicit step in kwargs
            pass
        
        # Prefer Sample Factory train_step as global step
        global_step = None
        try:
            from sample_factory.algo.learning.learner import Learner as _L
            # Try to read from a live learner if available via a global reference (best-effort)
        except Exception:
            pass
        
        # Fallback: attach frames/env_steps from cfg if available
        frames = None
        if hasattr(cfg, 'train_step') and isinstance(getattr(cfg, 'train_step'), (int, float)):
            frames = int(getattr(cfg, 'train_step'))
        elif hasattr(cfg, 'env_steps') and isinstance(getattr(cfg, 'env_steps'), (int, float)):
            frames = int(getattr(cfg, 'env_steps'))
        if frames is not None:
            metrics.setdefault('frames', frames)
            kwargs.setdefault('step', frames)
        
        # Merge influence and grad metrics
        if influence_tracker:
            if not influence_tracker.should_log():
                try:
                    print("[W&B_DEBUG][obs_grad] influence_tracker present but not scheduled to log at this step (forcing minimal log header)")
                except Exception:
                    pass
            influence_metrics = influence_tracker.get_logging_metrics()
            # Cast to plain floats
            for k, v in list(influence_metrics.items()):
                try:
                    influence_metrics[k] = float(v)
                except Exception:
                    del influence_metrics[k]
            metrics.update(influence_metrics)
            # Update cache if we received any obs/influence keys
            try:
                had = any(isinstance(k, str) and k.startswith(('obs_grad/', 'influence/', 'grad_attr/')) for k in influence_metrics.keys())
                if had:
                    _last_obsgrad_from_influence = dict(influence_metrics)
            except Exception:
                pass
            # Also mirror per-slice obs_grad/influence summaries under episode_extra_stats for dashboard grouping
            try:
                episode_extra = {}
                # Expected keys provided by tracker (examples):
                #  - obs_grad/slice/..., influence/slice/...
                #  - obs_grad/total_recent, influence/total_recent, ...
                source_metrics = influence_metrics if len(influence_metrics) > 0 else _last_obsgrad_from_influence
                for name, val in list(source_metrics.items()):
                    if not isinstance(name, str):
                        continue
                    if name.startswith(('obs_grad/', 'influence/', 'grad_attr/')):
                        try:
                            prefix_removed = name.split('/', 1)[1] if '/' in name else name
                            new_key = 'episode_extra_stats/obs_grad/' + prefix_removed
                            episode_extra[new_key] = float(val)
                        except Exception:
                            pass
                if len(episode_extra) > 0:
                    metrics.update(episode_extra)
                    # try:
                    #     print(f"[W&B_DEBUG][obs_grad] mirrored {len([k for k in episode_extra.keys() if k.startswith('episode_extra_stats/obs_grad/')])} obs_grad keys from influence tracker")
                    # except Exception:
                    #     pass
                # Derived metrics: camera/state shares and per-slice magnitudes + shares
                # Collect slice values by suffix (ignore totals/backward_passes)
                SLICE_NAMES_STATE = {
                    'drone_position', 'static_camera_pos', 'static_camera_orient',
                    'drone_orientation',
                    'linear_vel', 'angular_vel', 'actions',
                    'drone_linear_vel', 'drone_angular_vel', 'drone_actions'
                }
                SLICE_NAMES_CAMERA = {'drone_camera_vae', 'static_camera_vae'}
                total_val = 0.0
                camera_val = 0.0
                state_val = 0.0
                slice_values = {}
                for name, val in list(source_metrics.items()):
                    if not isinstance(name, str):
                        continue
                    if not (name.startswith(('obs_grad/', 'influence/', 'grad_attr/'))):
                        continue
                    parts = name.split('/')
                    # Extract slice label after 'slice_pct'/'slice_mag' if present
                    label = parts[-1]
                    if 'slice_pct' in parts:
                        try:
                            idx = parts.index('slice_pct')
                            if idx + 1 < len(parts):
                                label = parts[idx + 1]
                        except Exception:
                            pass
                    elif 'slice_mag' in parts:
                        try:
                            idx = parts.index('slice_mag')
                            if idx + 1 < len(parts):
                                label = parts[idx + 1]
                        except Exception:
                            pass
                    if suffix.startswith('total_') or suffix == 'backward_passes':
                        continue
                    try:
                        scalar = float(val)
                    except Exception:
                        continue
                    total_val += scalar
                    # Normalize label by removing common postfixes
                    base = label
                    for post in ['_mean_norm_recent', '_mean_norm_overall', '_recent', '_overall', '_mean_norm', '_mean', '_norm']:
                        if base.endswith(post):
                            base = base[: -len(post)]
                            break
                    slice_values[base] = scalar
                    # Camera bucket
                    if base in SLICE_NAMES_CAMERA or 'camera_vae' in base:
                        camera_val += scalar
                    # State bucket
                    elif (base in SLICE_NAMES_STATE or
                          base.startswith('drone_position') or base.startswith('drone_orientation') or
                          base.startswith('drone_linear_vel') or base.startswith('drone_angular_vel') or
                          base.startswith('drone_actions') or
                          base.startswith('static_camera_pos') or base.startswith('static_camera_orient')):
                        state_val += scalar
                    else:
                        # Unknown slices count toward total but not camera/state buckets
                        pass
                if total_val > 0.0:
                    camera_share = camera_val / total_val
                    # Fallback if state set did not match tracker naming: infer as residual
                    if state_val <= 0.0 and camera_val > 0.0:
                        state_val = max(total_val - camera_val, 0.0)
                    state_share = state_val / total_val
                    # Ratios [0,1]
                    metrics['episode_extra_stats/obs_grad/camera_share'] = float(camera_share)
                    metrics['episode_extra_stats/obs_grad/state_share'] = float(state_share)
                    # Percentages [0,100]
                    metrics['episode_extra_stats/obs_grad/camera_share_pct'] = float(camera_share * 100.0)
                    metrics['episode_extra_stats/obs_grad/state_share_pct'] = float(state_share * 100.0)
                    # Per-slice magnitudes and percentages
                    for base, sval in slice_values.items():
                        metrics['episode_extra_stats/obs_grad/slice_mag/' + base] = float(sval)
                        metrics['episode_extra_stats/obs_grad/slice_pct/' + base] = float((sval / total_val) * 100.0)
                    # try:
                    #     captured = ','.join(sorted(list(slice_values.keys()))[:8])
                    #     print(f"[W&B_DEBUG][obs_grad] shares computed: camera={camera_share*100.0:.1f}% state={state_share*100.0:.1f}% total_slices={len(slice_values)} captured=[{captured}…]")
                    # except Exception:
                    #     pass
            except Exception:
                pass
            if influence_tracker.should_log():
                influence_tracker.step()
                if hasattr(influence_tracker, 'step_count'):
                    if influence_tracker.step_count % getattr(cfg, 'gradient_print_interval', 100) == 0:
                        influence_tracker.print_analysis_summary()
        else:
            pass
        if grad_tracker and grad_tracker.should_log():
            grad_metrics = grad_tracker.get_logging_metrics()
            for k, v in list(grad_metrics.items()):
                try:
                    grad_metrics[k] = float(v)
                except Exception:
                    del grad_metrics[k]
            metrics.update(grad_metrics)
            # Mirror obs_grad from gradient attribution tracker as well
            try:
                mirrored = {}
                for name, val in list(grad_metrics.items()):
                    if isinstance(name, str) and name.startswith(('obs_grad/', 'influence/', 'grad_attr/')):
                        prefix_removed = name.split('/', 1)[1] if '/' in name else name
                        mirrored['episode_extra_stats/obs_grad/' + prefix_removed] = float(val)
                if len(mirrored) > 0:
                    metrics.update(mirrored)
                    _last_obsgrad_from_grad = dict(grad_metrics)
                    # try:
                    #     print(f"[W&B_DEBUG][obs_grad] mirrored {len([k for k in mirrored.keys() if k.startswith('episode_extra_stats/obs_grad/')])} obs_grad keys from grad tracker")
                    # except Exception:
                    #     pass
            except Exception:
                pass
        else:
            pass
        
        # Cast any remaining tensor/np values to Python scalars
        for k, v in list(metrics.items()):
            try:
                metrics[k] = float(v)
            except Exception:
                try:
                    metrics[k] = int(v)
                except Exception:
                    # Drop non-loggable values
                    del metrics[k]
        
        # Mirror episode_extra_stats/curriculum/* -> curriculum/* and derive totals/rates every log
        try:
            # Direct mirror from episode_extra_stats to curriculum namespace
            for name, val in list(metrics.items()):
                if isinstance(name, str) and name.startswith('episode_extra_stats/curriculum/'):
                    suffix = name.split('episode_extra_stats/', 1)[1]  # 'curriculum/...'
                    try:
                        metrics[suffix] = float(val)
                    except Exception:
                        pass
            # Alias current_* to canonical names
            alias_map = {
                'curriculum/current_level': 'curriculum/level',
                'curriculum/current_progress': 'curriculum/progress',
            }
            for src, dst in alias_map.items():
                if src in metrics and dst not in metrics:
                    try:
                        metrics[dst] = float(metrics[src])
                    except Exception:
                        pass
            # Derive totals/rates when components are present
            def _get_float(key: str):
                v = metrics.get(key, None)
                try:
                    return float(v)
                except Exception:
                    return None
            s = _get_float('curriculum/total_successes') or _get_float('episode_extra_stats/curriculum/total_successes')
            c = _get_float('curriculum/total_crashes') or _get_float('episode_extra_stats/curriculum/total_crashes')
            tmo = _get_float('curriculum/total_timeouts') or _get_float('episode_extra_stats/curriculum/total_timeouts')
            if s is not None and c is not None and tmo is not None:
                total_resets = s + c + tmo
                metrics['curriculum/total_resets'] = float(total_resets)
                if total_resets > 0.0:
                    metrics.setdefault('curriculum/success_rate', float(s / total_resets))
                    metrics.setdefault('curriculum/crash_rate', float(c / total_resets))
                    metrics.setdefault('curriculum/timeout_rate', float(tmo / total_resets))
        except Exception:
            pass
        
        # Define metrics with step mapping once
        try:
            import wandb
            if hasattr(wandb, 'define_metric'):
                wandb.define_metric('frames')
                # Namespace common custom groups
                for name in list(metrics.keys()):
                    if name.startswith(('obs_grad/', 'influence/', 'curriculum/', 'gpu/', 'reward_breakdown/', 'episode_extra_stats/obs_grad/', 'episode_extra_stats/curriculum/')):
                        wandb.define_metric(name, step_metric='frames')
                # Ensure episode_extra_stats trajectory keys are tracked against frames
                for key in (
                    'episode_extra_stats/path_efficiency',
                    'episode_extra_stats/time_to_gate_steps',
                    'episode_extra_stats/min_gate_distance',
                    'episode_extra_stats/center_offset_success',
                    'episode_extra_stats/height_offset_success',
                ):
                    wandb.define_metric(key, step_metric='frames')
        except Exception:
            pass
        
        if original_wandb_log:
            original_wandb_log(metrics, **kwargs)

    # Store original functions before monkey patching
    from sample_factory.algo.learning.learner import Learner
    original_learner_init = Learner.init
    original_learner_train = Learner.train

    # Tracker configuration
    tracker_config = {
        'log_interval': getattr(cfg, 'gradient_log_interval', 100),
        'print_interval': getattr(cfg, 'gradient_print_interval', 100),
    }
    grad_config = {
        'log_interval': getattr(cfg, 'gradient_log_interval', 100),
        'print_interval': getattr(cfg, 'gradient_print_interval', 100),
    }

    def enhanced_learner_init(self):
        """Enhanced learner init that attaches influence tracker to the model"""
        nonlocal influence_tracker
        nonlocal grad_tracker
        result = original_learner_init(self)
        
        print(f"🔧 Enhanced learner init called")
        print(f"🔧 actor_critic type: {type(self.actor_critic)}")
        print(f"🔧 actor_critic is None: {self.actor_critic is None}")
        
        if hasattr(self, 'actor_critic') and self.actor_critic is not None:
            print("🔧 Creating influence tracker with actual model...")
            print(f"🔧 Model structure: {self.actor_critic}")
            
            try:
                influence_tracker = create_influence_tracker(self.actor_critic, tracker_config)
                print(f"🔧 Created influence tracker: {type(influence_tracker)}")
                print(f"🔧 Influence tracker enabled: {influence_tracker.enabled if influence_tracker else 'None'}")
                
                if influence_tracker and influence_tracker.enabled:
                    print("✅ Influence tracker successfully attached to model")
                    print(f"   🔍 Monitoring ALL observation components based on 150D observation structure:")
                    print(f"      • [0:3] Drone position | [3:9] Static camera pose | [9:12] Drone orientation")
                    print(f"      • [12:18] Velocities | [18:22] Actions | [22:86] Drone camera | [86:150] Static camera")
                    print(f"   📊 Complete 150D observation breakdown:")
                    print(f"      🎯 Spatial: 15D (position + pose + orientation)")
                    print(f"      ⚡ Kinematic: 10D (velocities + actions)")
                    print(f"      📹 Visual: 128D (dual camera VAE latents)")
                    print(f"   📊 Logging every {cfg.gradient_log_interval} steps")
                    print(f"   📺 Analysis summary every {cfg.gradient_print_interval} steps")
                else:
                    print("❌ Failed to attach influence tracker")
            except Exception as e:
                print(f"❌ Error creating influence tracker: {e}")
                influence_tracker = None
            
            # Optionally create gradient attribution tracker
            try:
                if getattr(cfg, 'enable_grad_attribution', True):
                    grad_tracker = create_gradient_tracker(self.actor_critic, grad_config)
                    if grad_tracker and grad_tracker.enabled:
                        print("✅ Gradient attribution tracker successfully attached")
                    else:
                        print("❌ Failed to attach gradient attribution tracker")
            except Exception as e:
                print(f"❌ Error creating gradient attribution tracker: {e}")
                grad_tracker = None

            # Attach a small forward hook to the actor_critic input path to mirror the obs tensor and capture its grad
            try:
                tracker_ref = grad_tracker
                if tracker_ref and tracker_ref.enabled:
                    def _ac_forward_hook(mod, inp):
                        # Try to locate the 150D obs inside Sample Factory's normalized_obs_dict
                        try:
                            arg = inp[0] if isinstance(inp, tuple) and len(inp) > 0 else inp
                            # Case 1: dict input with key 'obs'
                            if isinstance(arg, dict):
                                t = arg.get('obs', None)
                                if torch.is_tensor(t) and t.dim() == 2 and t.shape[1] == 150:
                                    x = t.detach().requires_grad_(True)
                                    # Stash proxy for backward hook
                                    mod._obs_proxy = x
                                    # Replace in a shallow-copied dict to avoid in-place side-effects
                                    new_arg = dict(arg)
                                    new_arg['obs'] = x
                                    if isinstance(inp, tuple):
                                        return (new_arg,) + tuple(inp[1:])
                                    else:
                                        return new_arg
                                return None
                            # Case 2: raw tensor input
                            if torch.is_tensor(arg) and arg.dim() == 2 and arg.shape[1] == 150:
                                x = arg.detach().requires_grad_(True)
                                mod._obs_proxy = x
                                if isinstance(inp, tuple):
                                    lst = list(inp)
                                    lst[0] = x
                                    return tuple(lst)
                                else:
                                    return x
                            return None
                        except Exception:
                            return None
                    # Register on the encoder if present; otherwise on actor_critic itself
                    target = getattr(self.actor_critic, 'encoder', self.actor_critic)
                    self._grad_attr_forward_handle = target.register_forward_pre_hook(_ac_forward_hook)

                    def _ac_backward_hook(mod, grad_in, grad_out):
                        try:
                            x = getattr(mod, '_obs_proxy', None)
                            if x is not None and x.grad is not None and hasattr(self, '_grad_tracker') and self._grad_tracker:
                                self._grad_tracker.consume_grad(x.grad)
                        except Exception:
                            pass
                    self._grad_attr_backward_handle = target.register_full_backward_hook(_ac_backward_hook)
            except Exception as e:
                print(f"❌ Failed to attach actor_critic grad mirror hooks: {e}")
        else:
            print("🔧 Cannot create influence/gradient trackers - model not available")
            if hasattr(self, 'actor_critic'):
                print(f"   • actor_critic exists but is None: {self.actor_critic is None}")
            else:
                print("   • actor_critic attribute doesn't exist")
        
        # Store tracker references on learner for access in train method
        self._influence_tracker = influence_tracker
        self._grad_tracker = grad_tracker

        # One-time: emit initial curriculum keys so W&B materializes all series early
        try:
            import wandb
            frames0 = int(getattr(self, 'train_step', 0))
            curriculum_keys = [
                'curriculum/level','curriculum/progress','curriculum/success_rate','curriculum/crash_rate','curriculum/timeout_rate',
                'curriculum/obstacles_behind_gate','curriculum/total_assets','curriculum/max_level_reached',
                'curriculum/camera_gaussian_std','curriculum/camera_dropout_rate',
                'curriculum/camera_frame_dropout_drone_total','curriculum/camera_frame_dropout_static_total',
                'curriculum/camera_frame_freeze_drone','curriculum/camera_frame_blank_drone',
                'curriculum/camera_frame_freeze_static','curriculum/camera_frame_blank_static',
                'curriculum/camera_max_angle','curriculum/camera_current_angle',
                'curriculum/state_noise_drone_pos_std_m','curriculum/state_noise_drone_orient_std_deg',
                'curriculum/state_noise_static_pos_std_m','curriculum/state_noise_static_orient_std_deg',
                'curriculum/total_successes','curriculum/total_crashes','curriculum/total_timeouts',
            ]
            boot = {}
            for k in curriculum_keys:
                boot[f'episode_extra_stats/{k}'] = 0.0
                boot[k] = 0.0
            boot['frames'] = frames0
            wandb.log(boot, step=frames0)
        except Exception:
            pass
        return result

    def enhanced_train(self, *args, **kwargs):
        """Enhanced train method that updates influence tracker"""
        # Log when training method is called
        current_step_before = getattr(self, 'train_step', 0)
        print(f"🔧 enhanced_train() called - current step BEFORE: {current_step_before}")
        
        result = original_learner_train(self, *args, **kwargs)
        
        current_step_after = getattr(self, 'train_step', 0)
        print(f"🔧 enhanced_train() finished - current step AFTER: {current_step_after}")
        
        # Learner-side W&B logging of curriculum level if present in episode stats
        try:
            import wandb
            frames = int(current_step_after)
            if hasattr(self, 'all_episodic_stats'):
                # Sample Factory aggregates episode stats; we can pull our injected keys if present
                # latest_stats is a dict of lists; take last value for curriculum/level if available
                latest = getattr(self, 'last_episodic_stats', None)
                curr_level = None
                curr_level_minus_1 = None
                path_efficiency = None
                time_to_gate_steps = None
                min_gate_distance = None
                center_offset_success = None
                height_offset_success = None
                if isinstance(latest, dict):
                    # common aggregation stores arrays; try several namespaces
                    for key in ('curriculum/level', 'curriculum_level', 'episode_extra_stats/curriculum_level'):
                        if key in latest:
                            try:
                                v = latest[key]
                                if isinstance(v, (list, tuple)) and len(v) > 0:
                                    curr_level = float(v[-1])
                                elif isinstance(v, (int, float)):
                                    curr_level = float(v)
                                break
                            except Exception:
                                pass
                    # Also try to read the minus_1 variant
                    for key in ('curriculum/level_minus_1', 'curriculum_level_minus_1', 'episode_extra_stats/curriculum_level_minus_1'):
                        if key in latest:
                            try:
                                v = latest[key]
                                if isinstance(v, (list, tuple)) and len(v) > 0:
                                    curr_level_minus_1 = float(v[-1])
                                elif isinstance(v, (int, float)):
                                    curr_level_minus_1 = float(v)
                                break
                            except Exception:
                                pass
                    # Fetch episode_extra_stats trajectory metrics if available
                    def _get_last(key_name):
                        try:
                            if key_name in latest:
                                v = latest[key_name]
                                if isinstance(v, (list, tuple)) and len(v) > 0:
                                    return float(v[-1])
                                elif isinstance(v, (int, float)):
                                    return float(v)
                        except Exception:
                            return None
                        return None
                    path_efficiency = _get_last('episode_extra_stats/path_efficiency') or _get_last('path_efficiency')
                    time_to_gate_steps = _get_last('episode_extra_stats/time_to_gate_steps') or _get_last('time_to_gate_steps')
                    min_gate_distance = _get_last('episode_extra_stats/min_gate_distance') or _get_last('min_gate_distance')
                    center_offset_success = _get_last('episode_extra_stats/center_offset_success') or _get_last('center_offset_success')
                    height_offset_success = _get_last('episode_extra_stats/height_offset_success') or _get_last('height_offset_success')
                if curr_level is not None:
                    # Force integer logging for curriculum/level to match discrete levels
                    wandb.log({'frames': frames, 'curriculum/level': int(curr_level)}, step=frames)
                    try:
                        print(f"[W&B_DEBUG][learner] logged curriculum/level={int(curr_level)} at frames={frames}")
                    except Exception:
                        pass
                if curr_level_minus_1 is not None:
                    wandb.log({'frames': frames, 'curriculum/level_minus_1': int(curr_level_minus_1)}, step=frames)
                    try:
                        print(f"[W&B_DEBUG][learner] logged curriculum/level_minus_1={int(curr_level_minus_1)} at frames={frames}")
                    except Exception:
                        pass
                # --- Explicit curriculum mirror block: ensure ~25+ curriculum keys are present each step ---
                try:
                    # Helper that tries multiple namespaces to find the latest value
                    def _get_last_with_prefixes(key_name: str):
                        v0 = _get_last(key_name)
                        if v0 is not None:
                            return v0
                        v1 = _get_last('episode_extra_stats/' + key_name)
                        if v1 is not None:
                            return v1
                        if key_name.startswith('curriculum/'):
                            bare = key_name.split('/', 1)[1]
                            v2 = _get_last(bare)
                            if v2 is not None:
                                return v2
                        return None
                    curriculum_keys = CURRICULUM_KEYS
                    cur_payload = {}
                    if isinstance(latest, dict):
                        for k in curriculum_keys:
                            v = _get_last_with_prefixes(k)
                            if v is not None:
                                # Log under both namespaces for dashboard compatibility
                                cur_payload[f'episode_extra_stats/{k}'] = v
                                cur_payload[k] = v
                        # Derive total_resets if components available
                        def _pl_get(name: str):
                            if name in cur_payload:
                                return cur_payload[name]
                            if f'episode_extra_stats/{name}' in cur_payload:
                                return cur_payload[f'episode_extra_stats/{name}']
                            return None
                        ts = _pl_get('curriculum/total_successes')
                        tc = _pl_get('curriculum/total_crashes')
                        tt = _pl_get('curriculum/total_timeouts')
                        if ts is not None and tc is not None and tt is not None:
                            cur_payload['curriculum/total_resets'] = float(ts + tc + tt)
                            cur_payload['episode_extra_stats/curriculum/total_resets'] = float(ts + tc + tt)
                    if len(cur_payload) > 0:
                        cur_payload['frames'] = frames
                        wandb.log(cur_payload, step=frames)
                        try:
                            first_keys = list(cur_payload.keys())[:6]
                            print(f"[W&B_DEBUG][learner] logged curriculum keys: {first_keys} ... (total {len(cur_payload)})")
                        except Exception:
                            pass
                    # Update last-known curriculum values
                    try:
                        for k in curriculum_keys:
                            if k in cur_payload:
                                _last_curriculum[k] = float(cur_payload[k])
                    except Exception:
                        pass
                    # Emit continuous curriculum series each step, carrying forward last-known values
                    try:
                        forward_payload = {'frames': frames}
                        for k in curriculum_keys:
                            forward_payload[k] = float(_last_curriculum.get(k, 0.0))
                            forward_payload[f'episode_extra_stats/{k}'] = float(_last_curriculum.get(k, 0.0))
                        wandb.log(forward_payload, step=frames)
                    except Exception:
                        pass
                except Exception:
                    pass
                # Log trajectory metrics if present (NaN will be ignored by W&B)
                traj_payload = {}
                # Prefer running means if available; fall back to base keys
                def _pref(keys):
                    for k in keys:
                        val = _get_last(k)
                        if val is not None:
                            return val
                    return None
                pe_out = _pref(['episode_extra_stats/path_efficiency_running_mean','path_efficiency_running_mean','episode_extra_stats/path_efficiency'])
                ttg_out = _pref(['episode_extra_stats/time_to_gate_running_mean','time_to_gate_running_mean','episode_extra_stats/time_to_gate_steps'])
                mgd_out = _pref(['episode_extra_stats/min_gate_distance_running_mean','min_gate_distance_running_mean','episode_extra_stats/min_gate_distance'])
                co_out  = _pref(['episode_extra_stats/center_offset_running_mean','center_offset_running_mean','episode_extra_stats/center_offset_success'])
                ho_out  = _pref(['episode_extra_stats/height_offset_running_mean','height_offset_running_mean','episode_extra_stats/height_offset_success'])
                pass_rate = _get_last('episode_extra_stats/gate_pass_rate')
                ep_total  = _get_last('episode_extra_stats/episodes_total')
                ep_cross  = _get_last('episode_extra_stats/episodes_crossed')
                if pe_out is not None:
                    traj_payload['episode_extra_stats/path_efficiency'] = pe_out
                if ttg_out is not None:
                    traj_payload['episode_extra_stats/time_to_gate_steps'] = ttg_out
                if mgd_out is not None:
                    traj_payload['episode_extra_stats/min_gate_distance'] = mgd_out
                if co_out is not None:
                    traj_payload['episode_extra_stats/center_offset_success'] = co_out
                if ho_out is not None:
                    traj_payload['episode_extra_stats/height_offset_success'] = ho_out
                if pass_rate is not None:
                    traj_payload['episode_extra_stats/gate_pass_rate'] = pass_rate
                if ep_total is not None:
                    traj_payload['episode_extra_stats/episodes_total'] = ep_total
                if ep_cross is not None:
                    traj_payload['episode_extra_stats/episodes_crossed'] = ep_cross
                if len(traj_payload) > 0:
                    traj_payload['frames'] = frames
                    wandb.log(traj_payload, step=frames)
                    try:
                        print(f"[W&B_DEBUG][learner] logged traj metrics keys: {list(traj_payload.keys())}")
                    except Exception:
                        pass
        except Exception:
            pass
        
        if hasattr(self, '_influence_tracker') and self._influence_tracker and self._influence_tracker.enabled:
            tracker = self._influence_tracker
            print(f"🔧 Calling tracker.step() - Sample Factory training step {current_step_after}")
            
            # Update tracker to use Sample Factory's actual step count
            tracker.step_count = current_step_after
            
            # Show data collection progress
            if hasattr(tracker, 'activation_history'):
                influence_samples = len(tracker.activation_history.get('drone_camera_vae', {}).get('correlations', []))
                print(f"🔧 Tracker synced to step {current_step_after} - collected {influence_samples} influence samples")
            
            # Print analysis at specified intervals
            if current_step_after > 0 and current_step_after % cfg.gradient_print_interval == 0:
                print(f"🔧 Influence analysis at training step {current_step_after}")
                tracker.print_analysis_summary()
        else:
            print(f"🔧 No influence tracker available for step {current_step_after}")
        
        # Optionally print gradient summary at intervals
        if hasattr(self, '_grad_tracker') and self._grad_tracker and self._grad_tracker.enabled:
            if current_step_after > 0 and current_step_after % cfg.gradient_print_interval == 0:
                print(f"🔧 Gradient attribution analysis at training step {current_step_after}")
                self._grad_tracker.print_gradient_summary()
        
        # Explicit obs-grad logging to W&B under episode_extra_stats (same path as other episodic metrics)
        try:
            import wandb
            frames = int(current_step_after)
            # Prefer influence tracker; fall back to grad tracker
            metric_sources = []
            if hasattr(self, '_influence_tracker') and self._influence_tracker and self._influence_tracker.enabled:
                try:
                    metric_sources.append(self._influence_tracker.get_logging_metrics())
                except Exception:
                    pass
            if hasattr(self, '_grad_tracker') and self._grad_tracker and self._grad_tracker.enabled:
                try:
                    metric_sources.append(self._grad_tracker.get_logging_metrics())
                except Exception:
                    pass
            merged = {}
            for src in metric_sources:
                for k, v in list(src.items()):
                    try:
                        merged[k] = float(v)
                    except Exception:
                        pass
            # Build payload like curriculum block
            obs_payload = {}
            # Mirror raw slice metrics
            for name, val in merged.items():
                if isinstance(name, str) and name.startswith(('obs_grad/', 'influence/', 'grad_attr/')):
                    key_tail = name.split('/', 1)[1] if '/' in name else name
                    obs_payload['episode_extra_stats/obs_grad/' + key_tail] = float(val)
            # Derived shares
            SLICE_NAMES_STATE = {'drone_position','static_camera_pos','static_camera_orient','drone_orientation','linear_vel','angular_vel','actions','drone_linear_vel','drone_angular_vel','drone_actions'}
            SLICE_NAMES_CAMERA = {'drone_camera_vae','static_camera_vae'}
            total_val = 0.0
            camera_val = 0.0
            state_val = 0.0
            slice_vals = {}
            for name, val in merged.items():
                if not isinstance(name, str) or not name.startswith(('obs_grad/','influence/','grad_attr/')):
                    continue
                parts = name.split('/')
                suffix = parts[-1]
                if suffix.startswith('total_') or suffix == 'backward_passes':
                    continue
                try:
                    scalar = float(val)
                except Exception:
                    continue
                total_val += scalar
                slice_vals[suffix] = scalar
                base = suffix
                for post in ['_mean_norm_recent','_mean_norm_overall','_recent','_overall','_mean_norm','_mean','_norm']:
                    if base.endswith(post):
                        base = base[: -len(post)]
                        break
                if (base in SLICE_NAMES_CAMERA) or ('camera_vae' in base):
                    camera_val += scalar
                elif (base in SLICE_NAMES_STATE or
                      base.startswith('drone_position') or base.startswith('drone_orientation') or
                      base.startswith('drone_linear_vel') or base.startswith('drone_angular_vel') or
                      base.startswith('drone_actions') or
                      base.startswith('static_camera_pos') or base.startswith('static_camera_orient')):
                    state_val += scalar
            if total_val > 0.0:
                cam_share = camera_val / total_val
                if state_val <= 0.0 and camera_val > 0.0:
                    state_val = max(total_val - camera_val, 0.0)
                st_share = state_val / total_val
                obs_payload['episode_extra_stats/obs_grad/camera_share'] = float(cam_share)
                obs_payload['episode_extra_stats/obs_grad/state_share'] = float(st_share)
                obs_payload['episode_extra_stats/obs_grad/camera_share_pct'] = float(cam_share * 100.0)
                obs_payload['episode_extra_stats/obs_grad/state_share_pct'] = float(st_share * 100.0)
                for sfx, sval in slice_vals.items():
                    obs_payload[f'episode_extra_stats/obs_grad/slice_mag/{sfx}'] = float(sval)
                    obs_payload[f'episode_extra_stats/obs_grad/slice_pct/{sfx}'] = float((sval / total_val) * 100.0)
            if len(obs_payload) > 0:
                obs_payload['frames'] = frames
                # Print the two target metrics for debugging
                try:
                    ss = obs_payload.get('episode_extra_stats/obs_grad/state_share', None)
                    ssp = obs_payload.get('episode_extra_stats/obs_grad/state_share_pct', None)
                    if (ss is not None) or (ssp is not None):
                        print(f"[OBS_GRAD_DEBUG] frames={frames} state_share={ss if ss is not None else 'None'} state_share_pct={ssp if ssp is not None else 'None'}", flush=True)
                except Exception:
                    pass
                wandb.log(obs_payload, step=frames)
                try:
                    print(f"[W&B_DEBUG][obs_grad] explicit_log keys={len(obs_payload)}", flush=True)
                except Exception:
                    pass
        except Exception:
            pass
        
        return result

    # Apply monkey patches
    Learner.init = enhanced_learner_init
    Learner.train = enhanced_train

    # Apply wandb monkey patch if available
    if original_wandb_log:
        import wandb
        wandb.log = enhanced_wandb_log

    try:
        print("🚀 Starting enhanced training with complete observation influence tracking...")
        result = run_rl(cfg)
        
        # Final analysis and cleanup
        print("=" * 80)
        print("🎯 COMPLETE OBSERVATION INFLUENCE ANALYSIS - FINAL SUMMARY")
        print("=" * 80)
        
        if influence_tracker:
            print(f"📊 Training completed with {influence_tracker.step_count} analysis steps")
            influence_tracker.print_analysis_summary()
            influence_tracker.cleanup()
        else:
            print("❌ No influence tracker was created - analysis unavailable")
        
        # Gradient attribution final summary
        print("=" * 80)
        print("🧮 GRADIENT-BASED ATTRIBUTION - FINAL SUMMARY")
        print("=" * 80)
        if grad_tracker:
            grad_tracker.print_gradient_summary()
            grad_tracker.cleanup()
        else:
            print("❌ No gradient attribution tracker was created - analysis unavailable")
            
        print("=" * 80)
        
        return result
        
    finally:
        # Restore original functions
        Learner.init = original_learner_init
        Learner.train = original_learner_train
        # Remove grad mirror hooks if present
        try:
            if hasattr(self, '_grad_attr_forward_handle'):
                self._grad_attr_forward_handle.remove()
            if hasattr(self, '_grad_attr_backward_handle'):
                self._grad_attr_backward_handle.remove()
        except Exception:
            pass
        if original_wandb_log:
            import wandb
            wandb.log = original_wandb_log


if __name__ == "__main__":
    sys.exit(main())
