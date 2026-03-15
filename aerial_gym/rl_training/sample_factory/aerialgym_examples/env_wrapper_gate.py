from __future__ import annotations

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import gymnasium as gym
from torch import Tensor
from sample_factory.algo.utils.gymnasium_utils import convert_space

from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_common import (
    AerialGymVecEnv as AerialGymVecEnvBase,
)

VERBOSE = os.environ.get('TRAIN_VERBOSE', 'false').lower() == 'true'


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

