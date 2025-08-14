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

    def _process_camera_image(self, image_data, camera_type="depth"):
        """Process camera image for GIF saving."""
        if camera_type == "depth":
            # Process depth image (similar to dce_nn_navigation.py)
            image = (255.0 * image_data.cpu().numpy()).astype(np.uint8)
            return Image.fromarray(image)
        elif camera_type == "segmentation":
            # Process segmentation image with colormap
            seg_image = image_data.cpu().numpy()
            seg_image[seg_image <= 0] = seg_image[seg_image > 0].min() if seg_image[seg_image > 0].size > 0 else 1
            seg_image_normalized = (seg_image - seg_image.min()) / (seg_image.max() - seg_image.min() + 1e-8)
            seg_image_plasma = matplotlib.cm.plasma(seg_image_normalized)
            return Image.fromarray((seg_image_plasma * 255.0).astype(np.uint8))
        return None

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
            
            # Save drone camera GIFs
            if len(self.drone_depth_frames[env_id]) > 0:
                gif_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_drone_depth.gif")
                self.drone_depth_frames[env_id][0].save(
                    gif_path,
                    save_all=True,
                    append_images=self.drone_depth_frames[env_id][1:],
                    duration=100,
                    loop=0
                )
                print(f"[GIF] Saved drone depth: {gif_path}")
            
            if len(self.drone_seg_frames[env_id]) > 0:
                gif_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_drone_seg.gif")
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
                gif_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_static_depth.gif")
                self.static_depth_frames[env_id][0].save(
                    gif_path,
                    save_all=True,
                    append_images=self.static_depth_frames[env_id][1:],
                    duration=100,
                    loop=0
                )
                print(f"[GIF] Saved static depth: {gif_path}")
            
            if len(self.static_seg_frames[env_id]) > 0:
                gif_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_static_seg.gif")
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
                gif_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_merged_dual_camera_CLEAN.gif")
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
                gif_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_drone_depth_D455_NOISED.gif")
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
                gif_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_static_depth_D455_NOISED.gif")
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
                gif_path = os.path.join(self.gif_output_dir, f"episode_{episode_num:04d}_merged_dual_camera_D455_NOISED.gif")
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
        
        # DYNAMIC OBSERVATION PROCESSING: Handle both standard DCE (81D) and gate navigation (145D)
        # Task provides "observations" with correct dimensionality (81D or 145D) based on task type
        # We pass this through as "obs" for Sample Factory
        transformed_obs = {"obs": obs["observations"]}
        
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
    
    # Complete observation influence tracking arguments
    parser.add_argument("--enable_gradient_monitoring", type=lambda x: x.lower() == 'true', default=False, help="Enable complete observation influence tracking")
    parser.add_argument("--gradient_log_interval", default=100, type=int, help="Log influence metrics every N steps")
    parser.add_argument("--gradient_print_interval", default=100, type=int, help="Print analysis summary every N steps")
    
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
        train_for_env_steps=100000000,  # 100M steps for comprehensive gate navigation learning
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
        env_agents=16,  # 16 environments for stable testing (can be overridden via --env_agents)
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
    
    # Use environment variable from shell script if set, otherwise default to 128 (maximum parallelization config)
    # This will be updated based on the actual env_agents parameter when env is created
    current_env_agents = os.environ.get('SF_ENV_AGENTS', '128')  # Default to maximum parallelization DCE configuration
    os.environ['SF_ENV_AGENTS'] = current_env_agents
    
    # Set train_dir for curriculum logging
    import os
    if 'SF_TRAIN_DIR' not in os.environ:
        os.environ['SF_TRAIN_DIR'] = './train_dir'  # Default train directory
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
    return final_cfg


def main():
    """Script entry point."""
    register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg()
    
    # Check if complete observation influence tracking is enabled
    if getattr(cfg, 'enable_gradient_monitoring', False):
        return run_with_influence_tracking(cfg)
    else:
        return run_rl(cfg)


def run_with_influence_tracking(cfg: Config):
    """Enhanced training with complete observation influence tracking."""
    
    # Import the complete observation influence tracker
    try:
        from aerial_gym.utils.gradient_monitor import create_influence_tracker, INFLUENCE_MONITOR_AVAILABLE
        # For compatibility during transition
        GRADIENT_MONITOR_AVAILABLE = INFLUENCE_MONITOR_AVAILABLE
    except ImportError:
        print("❌ Complete observation influence tracker not available")
        INFLUENCE_MONITOR_AVAILABLE = False
        GRADIENT_MONITOR_AVAILABLE = False
    
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

    # Create influence tracker instance (will be attached to model later)
    influence_tracker = None

    def enhanced_wandb_log(metrics, **kwargs):
        """Enhanced wandb logging that includes influence monitoring metrics"""
        nonlocal influence_tracker
        if influence_tracker and influence_tracker.should_log():
            influence_metrics = influence_tracker.get_logging_metrics()
            metrics.update(influence_metrics)
            influence_tracker.step()
            if hasattr(influence_tracker, 'step_count'):
                if influence_tracker.step_count % getattr(cfg, 'gradient_print_interval', 100) == 0:
                    influence_tracker.print_analysis_summary()
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

    def enhanced_learner_init(self):
        """Enhanced learner init that attaches influence tracker to the model"""
        nonlocal influence_tracker
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
        else:
            print("🔧 Cannot create influence tracker - model not available")
            if hasattr(self, 'actor_critic'):
                print(f"   • actor_critic exists but is None: {self.actor_critic is None}")
            else:
                print("   • actor_critic attribute doesn't exist")
        
        # Store tracker reference on learner for access in train method
        self._influence_tracker = influence_tracker
        return result

    def enhanced_train(self, *args, **kwargs):
        """Enhanced train method that updates influence tracker"""
        # Log when training method is called
        current_step_before = getattr(self, 'train_step', 0)
        print(f"🔧 enhanced_train() called - current step BEFORE: {current_step_before}")
        
        result = original_learner_train(self, *args, **kwargs)
        
        current_step_after = getattr(self, 'train_step', 0)
        print(f"🔧 enhanced_train() finished - current step AFTER: {current_step_after}")
        
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
            
        print("=" * 80)
        
        return result
        
    finally:
        # Restore original functions
        Learner.init = original_learner_init
        Learner.train = original_learner_train
        if original_wandb_log:
            import wandb
            wandb.log = original_wandb_log


if __name__ == "__main__":
    sys.exit(main())
