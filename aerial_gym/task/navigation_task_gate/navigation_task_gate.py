from __future__ import annotations

from aerial_gym.task.base_task import BaseTask, StepReturn
from aerial_gym.task.task_config_protocol import TaskConfig
from aerial_gym.task.schemas import (
    GATE_OBS_LAYOUT,
    EpisodeRewardAccumulators,
    EpisodeTrajectoryState,
)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.tensor_utils import invalid_mask_per_env, has_invalid, sanitize_tensor
import numpy as np
import os

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

from aerial_gym.utils.vae.vae_image_encoder import VAEImageEncoder

from gym.spaces import Dict, Box

# Isaac Gym imports for static camera management

from aerial_gym.sensors.static_camera_manager import StaticCameraManager
from aerial_gym.utils.env_flag_utils import (
    read_env_bool,
    read_env_int,
    parse_ablation_flags,
    apply_ablation_flags_to_tensor_dict,
)
from aerial_gym.task.navigation_task_gate.reward_functions import compute_gate_reward
from aerial_gym.task.navigation_task_gate.init_helpers import InitHelpers
from aerial_gym.task.navigation_task_gate.step_helpers import StepHelpers
from aerial_gym.task.navigation_task_gate.reward_helpers import RewardHelpers
from aerial_gym.task.navigation_task_gate.gate_geometry import GateGeometry
from aerial_gym.task.navigation_task_gate.curriculum_management import CurriculumManager
from aerial_gym.task.navigation_task_gate.camera_observations import CameraObservations
from aerial_gym.task.navigation_task_gate.reward_tracking import RewardTracking
from aerial_gym.task.navigation_task_gate.curriculum_logging import CurriculumLogging


logger = CustomLogger("navigation_task_gate")


def dict_to_class(dict) -> None:
    return type("ClassFromDict", (object,), dict)


class NavigationTaskGate(BaseTask):
    def __init__(
        self,
        task_config: TaskConfig,
        seed: int | None = None,
        num_envs: int | None = None,
        headless: bool | None = None,
        device: str | None = None,
        use_warp: bool | None = None,
    ) -> None:
        # overwrite the params if user has provided them
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp
        super().__init__(task_config)
        self.device = self.task_config.device

        # Compose helper objects (composition over inheritance)
        self._init = InitHelpers(self)
        self._step = StepHelpers(self)
        self._rewards = RewardHelpers(self)
        self._geometry = GateGeometry(self)
        self._curriculum = CurriculumManager(self)
        self._camera = CameraObservations(self)
        self._reward_tracking = RewardTracking(self)
        self._curriculum_log = CurriculumLogging(self)

        
        # If static latents (86:150) are fully ablated, disable static FOV visibility reward
        try:
            spec_str = os.environ.get('ABLATE_OBS_RANGES', '').strip()
            static_ablated = False
            if spec_str:
                for spec in [s.strip() for s in spec_str.split(',') if s.strip() and '=' in s]:
                    lhs, rhs = spec.split('=', 1)
                    lhs = lhs.strip(); rhs = rhs.strip()
                    if ':' in lhs:
                        try:
                            a, b = lhs.split(':', 1)
                            a = int(a); b = int(b)
                        except (ValueError, TypeError):
                            continue
                        if rhs in ('zero', 'zerograd') and a <= 86 and b >= 150:
                            static_ablated = True
                            break
            if static_ablated:
                self.task_config.reward_parameters["static_fov_visibility_reward_magnitude"] = 0.0
        except (ValueError, TypeError):
            pass

        # set the each of the elements of reward parameter to a torch tensor
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )
        
        logger.info("Building environment for gate navigation task.")
        logger.info(
            "Sim Name: {}, Env Name: {}, Robot Name: {}, Controller Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )

        # This ensures the asset manager gets the correct count from the start
        self.curriculum_level = self.task_config.curriculum.min_level
        # Obstacle ablation: if disabled, force to fixed count (default 0)
        obstacles_disable = read_env_bool(
            "SF_DISABLE_OBSTACLE_RANDOMIZATION", self.task_config.disable_obstacle_randomization
        )
        obstacles_fixed = read_env_int(
            "SF_FIXED_OBSTACLES_BEHIND_GATE", self.task_config.fixed_obstacles_behind_gate
        )
        if obstacles_disable:
            obstacles_behind_gate = max(0, obstacles_fixed)
        else:
            obstacles_behind_gate = self.task_config.curriculum.get_obstacle_count_behind_gate(self.curriculum_level)
        
        # NOTE: Even though 11 gate variants are loaded, only 1 will be visible at any time
        # The other 10 gates are hidden by moving them to (-1000, -1000, -1000)
        visible_gates = 0  # Only 1 gate visible at a time (others hidden by gate selection system)
        walls = 6  # 6 boundary walls
        robot = 0  # Robot is NOT part of env_asset_state_tensor (handled separately)
        fixed_assets_visible = visible_gates + walls  # = 7 visible fixed assets
        total_obstacles_in_env = fixed_assets_visible + obstacles_behind_gate
        
        logger.info(f"PRE-INIT: Setting curriculum level {self.curriculum_level} with {obstacles_behind_gate} curriculum obstacles")
        logger.info(f"PRE-INIT: Visible assets (env assets only): {visible_gates} gate + {walls} walls + {obstacles_behind_gate} curriculum = {total_obstacles_in_env} total")
        logger.info(f"PRE-INIT: Total obstacle count for asset manager: {total_obstacles_in_env}")

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )
        
        # Ensure num_envs is available before any observation processing that may rely on it
        self.num_envs = self.sim_env.num_envs
        
        # Propagate ablation flags to global tensor dict for EnvManager to consume
        if hasattr(self.sim_env, "global_tensor_dict"):
            flags = parse_ablation_flags(self.task_config)
            apply_ablation_flags_to_tensor_dict(
                self.sim_env.global_tensor_dict, flags, self.task_config, logger
            )
            # Mirror commonly-accessed flags onto self for quick access
            self.disable_static_camera_orientation_randomization = flags[
                "static_camera_orient_disabled"
            ]
            self.disable_camera_frame_dropout_randomization = flags[
                "camera_frame_dropout_disabled"
            ]
            self.disable_camera_noise_randomization = flags["camera_noise_disabled"]
            self.disable_state_noise_randomization = flags["state_noise_disabled"]
            self.disable_dynamic_camera_following = flags[
                "dynamic_camera_following_disabled"
            ]

        # Immediately select a random gate variant once after creation (safety)
        if hasattr(self.sim_env, 'apply_gate_variant_selection'):
            logger.warning("[GateVariant] Initial selection after build (one-time)")
            self.sim_env.apply_gate_variant_selection(env_ids=torch.arange(self.sim_env.num_envs, device=self.device))
        
        if hasattr(self.sim_env, 'global_tensor_dict'):
            # Override count if obstacle randomization disabled
            try:
                obs_dis = bool(self.sim_env.global_tensor_dict.get('obstacles_randomization/disabled', False))
            except (KeyError, TypeError):
                obs_dis = False
            if obs_dis:
                try:
                    fixed_count = int(self.sim_env.global_tensor_dict.get('obstacles_randomization/fixed_count', 0))
                except (ValueError, TypeError):
                    fixed_count = 0
                total_obstacles_in_env = fixed_assets_visible + max(0, fixed_count)
            self.sim_env.global_tensor_dict["num_obstacles_in_env"] = total_obstacles_in_env
            logger.info(f"POST-INIT: Updated global_tensor_dict with obstacle count: {total_obstacles_in_env}")

        # Target position: will be set to adaptive gate center on each reset
        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        # Target position is now always set to adaptive gate center instead of random sampling

        self.success_aggregate = 0
        self.crashes_aggregate = 0
        self.timeouts_aggregate = 0

        self._init._init_gate_tracking_tensors()
        self._init._init_vae_model()

        # Static camera management using Isaac Gym native API
        self.static_camera_manager = StaticCameraManager(self.sim_env, self.task_config)

        # Get the dictionary once from the environment and use it to get the observations later.
        self.obs_dict = self.sim_env.get_obs()
        
        # DEBUG: Print all available observations at startup for debugging
        logger.warning("="*80)
        logger.warning("🔍 AVAILABLE OBSERVATIONS AT INITIALIZATION:")
        for key, value in self.obs_dict.items():
            if hasattr(value, 'shape'):
                logger.warning(f"  📊 {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                logger.warning(f"  📊 {key}: {type(value)} = {value}")
        logger.warning("="*80)
        
        # IMMEDIATE DEBUG: Test observation processing right after initialization  
        logger.warning("🧪 TESTING OBSERVATION PROCESSING AT INITIALIZATION:")
        try:
            # Initialize task_obs first to avoid AttributeError
            if not True:
                self.task_obs = {
                    "observations": torch.zeros(
                        (self.sim_env.num_envs, self.task_config.observation_space_dim),
                        device=self.device,
                        requires_grad=False,
                    ),
                }
            
            self.process_obs_for_task()
            logger.warning("✅ process_obs_for_task() executed successfully")
            
            # Check if task_obs was populated
            if'observations' in self.task_obs:
                obs_shape = self.task_obs['observations'].shape
                logger.warning(f"📊 task_obs shape: {obs_shape}")
                
                if obs_shape[1] == 150:  # 150D observation space
                    obs_sample = self.task_obs["observations"][0]  # First environment
                    
                    # Static camera data verification
                    static_cam_pos = obs_sample[3:6]
                    static_cam_orient = obs_sample[6:9] 
                    static_vae = obs_sample[86:150]
                    
                    logger.warning("🔍 IMMEDIATE STATIC CAMERA VERIFICATION:")
                    logger.warning(f"  📍 Static cam pos [3:6]: {static_cam_pos.cpu().numpy()}")
                    logger.warning(f"  📍 Static cam pos sum: {torch.sum(static_cam_pos).item():.6f}")
                    logger.warning(f"  🧭 Static cam orient [6:9]: {static_cam_orient.cpu().numpy()}")
                    logger.warning(f"  🧭 Static cam orient sum: {torch.sum(static_cam_orient).item():.6f}")
                    logger.warning(f"  📷 Static VAE sum: {torch.sum(static_vae).item():.6f}")
                    logger.warning(f"  📷 Static VAE norm: {torch.norm(static_vae).item():.6f}")
                    
                    # Critical checks
                    pos_is_zero = torch.allclose(static_cam_pos, torch.zeros_like(static_cam_pos), atol=1e-6)
                    orient_is_zero = torch.allclose(static_cam_orient, torch.zeros_like(static_cam_orient), atol=1e-6)
                    vae_is_zero = torch.allclose(static_vae, torch.zeros_like(static_vae), atol=1e-6)
                    
                    logger.warning(f"  ✅ Position populated: {'❌ ALL ZEROS' if pos_is_zero else '✅ NON-ZERO'}")
                    logger.warning(f"  ✅ Orientation populated: {'❌ ALL ZEROS' if orient_is_zero else '✅ NON-ZERO'}")
                    logger.warning(f"  ✅ VAE latents populated: {'❌ ALL ZEROS' if vae_is_zero else '✅ NON-ZERO'}")
                else:
                    logger.warning(f"❌ Wrong observation space dimension: {obs_shape[1]} (expected 150)")
            else:
                logger.warning("❌ task_obs not found or observations key missing")
                
        except RuntimeError as e:
            logger.warning(f"❌ Error in process_obs_for_task(): {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
        
        logger.warning("="*80)
        
        self._init._init_curriculum()

        # Use dedicated terminations tensor if provided by env_manager; fallback to crashes
        try:
            self.terminations = self.obs_dict["terminations"]
        except (KeyError, TypeError):
            self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        self._init._init_observation_action_spaces()
        self._init._init_task_observations()

        self.num_task_steps = 0
        
        # Curriculum logging already initialized earlier in __init__

        self.pos_error_vehicle_frame = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.pos_error_vehicle_frame_prev = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.gate_passed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Episode-level tracking flags
        self._ep_target_success_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.camera_alignment_debug = torch.zeros(self.num_envs, device=self.device)
        self.num_task_steps = 0
        self.curriculum_progress_fraction = 0.0
        
        self._init._init_episode_reward_tracking()
        self._init._init_episode_trajectory_state()
        self._init._init_debug_flags()

        # Initialize gate dimensions for all environments after full initialization
        logger.warning("[GATE_ADAPTIVE] Initializing gate dimensions for all environments")
        self._geometry.update_gate_dimensions_for_environments(torch.arange(self.sim_env.num_envs, device=self.device))

        # Ensure infos survive resets for logging back to the learner
        self._infos_to_return = None

    # Private init helpers (extracted from __init__ for readability)










    def logging_sanity_check(self, infos: dict[str, torch.Tensor]) -> None:
        """Sanity check for logging to detect issues with success/crash/timeout logic."""
        successes = infos["successes"]
        crashes = infos["crashes"]
        timeouts = infos["timeouts"]
        time_at_crash = torch.where(
            crashes > 0,
            self.sim_env.sim_steps,
            self.task_config.episode_len_steps * torch.ones_like(self.sim_env.sim_steps),
        )
        env_list_for_toc = (time_at_crash < 5).nonzero(as_tuple=False).squeeze(-1)
        crash_envs = crashes.nonzero(as_tuple=False).squeeze(-1)
        success_envs = successes.nonzero(as_tuple=False).squeeze(-1)
        timeout_envs = timeouts.nonzero(as_tuple=False).squeeze(-1)

        if len(env_list_for_toc) > 0:
            logger.critical("Crash is happening too soon.")
            logger.critical(f"Envs crashing too soon: {env_list_for_toc}")
            logger.critical(f"Time at crash: {time_at_crash[env_list_for_toc]}")

        if torch.sum(torch.logical_and(successes, crashes)) > 0:
            logger.critical("Success and crash are occuring at the same time")
            logger.critical(
                f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
            )
            logger.critical(
                f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, successes))}"
            )

    def close(self) -> None:
        try:
            if hasattr(self.sim_env, 'delete_env'):
                self.sim_env.delete_env()
            elif hasattr(self.sim_env, 'close'):
                self.sim_env.close()
            else:
                print("[DEBUG] No cleanup method found for sim_env")
        except (ValueError, TypeError) as e:
            print(f"[DEBUG] Error during close: {e}")
        
        # Close curriculum log file
        if self.curriculum_log_file:
            try:
                import datetime
                self.curriculum_log_file.write(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training session ended.\n")
                self.curriculum_log_file.write("="*80 + "\n")
                self.curriculum_log_file.close()
            except OSError as e:
                print(f"[DEBUG] Error closing curriculum log: {e}")



    def reset(self) -> StepReturn:
        self.reset_idx(torch.arange(self.sim_env.num_envs))
        return self.get_return_tuple()

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """
        SIMPLIFIED RESET WITH FIXED SPAWNING PARAMETERS
        
        Uses LMF2 robot config for spawning with fixed parameters:
        - ±0.5m lateral variation from gate center
        - ±45° orientation randomization  
        - Minimal initial velocity for randomization
        """
        # This ensures "Getting Closer" and "position reward" are consistently aimed at the gate center
        # Gate position comes from obs_dict["env_bounds_min/max"] or is set by environment
        # We need to get gate position from observations after environment reset
        if "gate_position" in self.obs_dict:
            # If gate position is directly available in observations
            self.gate_position[env_ids] = self.obs_dict["gate_position"][env_ids]
        else:
            # Fallback: assume gate is at origin (0, 0, 0) for each environment
            # This matches the typical gate environment setup
            self.gate_position[env_ids, 0] = 0.0  # X: center
            self.gate_position[env_ids, 1] = 0.0  # Y: gate plane
            self.gate_position[env_ids, 2] = 0.0  # Z: ground level
        
        # Robot spawning is now handled by the normal Isaac Gym reset mechanism
        # which uses the min_init_state and max_init_state from LMF2 config
        # This provides ±0.5m lateral variation and ±45° orientation automatically
        
        # Reset gate-specific tracking
        self.gate_passed[env_ids] = False
        # Reset episode flags for these environments
        self._ep_target_success_flag[env_ids] = False
        self.gate_approach_distance[env_ids] = 0.0
        
        # RESET EPISODE REWARD TRACKING: Store completed episode data and reset trackers
        self._reward_tracking.reset_episode_reward_tracking(env_ids)
        
        # Update static camera position based on curriculum level (ONLY for resetting environments)
        if len(env_ids) > 0:
            self.static_camera_manager.update_camera_positions(self.curriculum_level, env_ids)
            logger.debug(f"Updated static camera angles for {len(env_ids)} resetting environments: {env_ids.tolist()}")
        
        # Add debugging for gate randomization on reset
        
        # Persist curriculum level for env manager (for gated gate randomization)
        self.sim_env.global_tensor_dict["curriculum_level"] = int(self.curriculum_level)
        # Propagate eval-stretch flags from task_config if present (for gate size unlocking etc.)
        stretch_enabled = bool(self.task_config.curriculum.eval_stretch_enabled)
        stretch_end = int(self.task_config.curriculum.eval_stretch_end_level)
        self.sim_env.global_tensor_dict['eval_stretch_enabled'] = stretch_enabled
        self.sim_env.global_tensor_dict['eval_stretch_end_level'] = stretch_end
        # Update gate dimensions for adaptive rewards after gate selection
        self._geometry.update_gate_dimensions_for_environments(env_ids)
        
        # CRITICAL: Set target position to adaptive gate center AFTER gate dimensions are updated
        # This ensures "Getting Closer" and "position reward" terms aim at the correct gate center
        gate_center_x = self.gate_position[env_ids, 0]  # Same X as gate base
        gate_center_y = self.gate_position[env_ids, 1]  # Same Y as gate base  
        gate_center_z = self.gate_position[env_ids, 2] + self.gate_center_height[env_ids]  # Base Z + adaptive center height
        
        self.target_position[env_ids, 0] = gate_center_x
        self.target_position[env_ids, 1] = gate_center_y
        self.target_position[env_ids, 2] = gate_center_z
        
        # Verification logging: show sample target position and gate dimensions for debugging
        if len(env_ids) > 0:
            sample_env = env_ids[0].item()
            sample_target = self.target_position[sample_env]
            sample_gate_pos = self.gate_position[sample_env] 
            sample_center_height = self.gate_center_height[sample_env].item()
            sample_width = self.gate_width[sample_env].item()
            sample_height = self.gate_height[sample_env].item()
            
            logger.debug(f"[TARGET_VERIFICATION] Env {sample_env}: target=({sample_target[0]:.3f}, {sample_target[1]:.3f}, {sample_target[2]:.3f}), gate_base=({sample_gate_pos[0]:.3f}, {sample_gate_pos[1]:.3f}, {sample_gate_pos[2]:.3f}), center_height={sample_center_height:.3f}, dims=({sample_width:.3f}x{sample_height:.3f})")
        
        logger.debug(f"[TARGET_UPDATE] Set target positions for {len(env_ids)} envs to adaptive gate centers")
        
        # Reset per-env episode trajectory state for these environments
        self.trajectory.reset_envs(env_ids)
        
        self.infos = {}
        return
    
    
    
    
    # These methods have been removed as we now use fixed parameters from LMF2 config
    # The normal Isaac Gym reset mechanism handles spawning using min_init_state/max_init_state

    def render(self) -> None:
        return self.sim_env.render()










    def step(self, actions: torch.Tensor) -> StepReturn:
        # VELOCITY CONTROLLER: Transform 4D actions to direct velocity commands for LMF2 robot
        # Input: [x_vel_cmd, y_vel_cmd, z_vel_cmd, yaw_rate_cmd] in [-1, 1]^4
        # Output: [x_vel, y_vel, z_vel, yaw_rate] applied directly as velocity commands

        transformed_action, nan_trunc_mask = self._step._validate_and_step(actions)

        # This step must be done since the reset is done after the reward is calculated.
        # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
        # This is important for the RL agent to get the correct state after the reset.
        self.rewards[:], self.terminations[:], camera_gate_alignment = self.compute_rewards_and_crashes(self.obs_dict)
        # Reward NaN/Inf guard: sanitize invalid rewards and truncate offending envs
        invalid_reward_mask = torch.isnan(self.rewards) | torch.isinf(self.rewards)
        if torch.any(invalid_reward_mask):
            if self.task_config.guard_debug_enabled:
                _ids = torch.nonzero(invalid_reward_mask, as_tuple=False).squeeze(-1).tolist()
                logger.warning(f"[NaNGuard] Invalid REWARD in envs {_ids}; zeroed and truncating.")
            self.rewards[invalid_reward_mask] = 0.0
            # Ensure truncation to reset these envs safely
            self.truncations[invalid_reward_mask] = 1

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )
        # Apply NaN/Inf-triggered truncations (takes precedence)
        if torch.any(nan_trunc_mask):
            self.truncations[nan_trunc_mask] = 1
            # Guard debug: final truncation set due to NaN/Inf
            if self.task_config.guard_debug_enabled:
                _ids = torch.nonzero(nan_trunc_mask, as_tuple=False).squeeze(-1).tolist()
                logger.warning(f"[NaNGuard] Truncating envs due to NaN/Inf: {_ids}")

        robot_position = self.obs_dict["robot_position"]
        robot_position_before_reset = robot_position.clone()
        successes, target_successes, gate_passage_success = self._step._detect_gate_passage(robot_position)

        self._step._apply_timeout_and_populate_infos(successes)

        robot_position = self.obs_dict["robot_position"]
        gate_center_position, gate_passed_current = self._step._compute_gate_navigation_metrics(
            robot_position, camera_gate_alignment,
        )

        # Update per-env episode trajectory state
        self._step._update_trajectory_state(robot_position, gate_center_position, gate_passed_current)

        self._curriculum.check_and_update_curriculum_level(
            self.infos["successes"], self.infos["crashes"], self.infos["timeouts"]
        )

        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self._step._handle_post_reward_reset(
                robot_position, robot_position_before_reset, gate_center_position,
                successes, target_successes, reset_envs,
            )
        self.num_task_steps += 1

        self._step._process_images_and_finalize()

        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()
        return return_tuple




    def get_return_tuple(self) -> StepReturn:
        self.process_obs_for_task()
        # If we have stashed infos from the previous step (pre-reset), use them once
        if self._infos_to_return is not None:
            infos_to_return = self._infos_to_return
            self._infos_to_return = None
        else:
            infos_to_return = self.infos
        
        # ADDITIONAL DEBUG: Verify observations in get_return_tuple (called every step)
        if not self._return_tuple_debug_printed:
            self._return_tuple_debug_printed = True
            logger.warning("🎯 OBSERVATION VERIFICATION IN get_return_tuple():")
            
            if'observations' in self.task_obs:
                obs_shape = self.task_obs['observations'].shape
                logger.warning(f"📊 Final task_obs shape: {obs_shape}")
                
                if obs_shape[1] >= 150:
                    obs_sample = self.task_obs["observations"][0]
                    
                    # Core verification of static camera data
                    static_pos = obs_sample[3:6]
                    static_orient = obs_sample[6:9]
                    static_vae = obs_sample[86:150] if obs_shape[1] >= 150 else obs_sample[86:]
                    
                    logger.warning(f"🔍 FINAL VERIFICATION:")
                    logger.warning(f"  Static pos: {static_pos.cpu().numpy()}")
                    logger.warning(f"  Static orient: {static_orient.cpu().numpy()}")
                    logger.warning(f"  Static VAE range: [{static_vae.min().item():.3f}, {static_vae.max().item():.3f}]")
                    
                    # Check if properly populated
                    pos_nonzero = not torch.allclose(static_pos, torch.zeros_like(static_pos), atol=1e-6)
                    orient_nonzero = not torch.allclose(static_orient, torch.zeros_like(static_orient), atol=1e-6) 
                    vae_nonzero = not torch.allclose(static_vae, torch.zeros_like(static_vae), atol=1e-6)
                    
                    logger.warning(f"  ✅ Static camera working: pos={pos_nonzero}, orient={orient_nonzero}, vae={vae_nonzero}")
                    
                    if not (pos_nonzero and orient_nonzero and vae_nonzero):
                        logger.error("❌ CRITICAL: Static camera data is missing!")
                    else:
                        logger.warning("✅ SUCCESS: All static camera data populated correctly!")
                else:
                    logger.error(f"❌ WRONG OBSERVATION DIMENSION: {obs_shape[1]} (expected 150)")
            else:
                logger.error("❌ task_obs not available in get_return_tuple")
        # Periodic static camera obs debug (env 0): print every 120 steps
        if(int(self.num_task_steps) % 120 == 0):
            if'observations' in self.task_obs:
                obs_all = self.task_obs['observations']
                if obs_all.shape[0] > 0 and obs_all.shape[1] >= 9:
                    sp = obs_all[0, 3:6]
                    so = obs_all[0, 6:9]
                    # Drone and camera world positions
                    rp = self.obs_dict.get('robot_position', None)
                    rp0 = rp[0] if isinstance(rp, torch.Tensor) and rp.shape[0] > 0 else torch.zeros(3, device=self.device)
                    cw = self._debug_cam_world if self._debug_cam_world is not None else None
                    cw0 = cw[0] if isinstance(cw, torch.Tensor) and cw.shape[0] > 0 else torch.zeros(3, device=self.device)
                    # Drone & camera world orientations (Euler)
                    de = self._debug_drone_eul if self._debug_drone_eul is not None else None
                    ce = self._debug_cam_eul if self._debug_cam_eul is not None else None
                    de0 = de[0] if isinstance(de, torch.Tensor) and de.shape[0] > 0 else torch.zeros(3, device=self.device)
                    ce0 = ce[0] if isinstance(ce, torch.Tensor) and ce.shape[0] > 0 else torch.zeros(3, device=self.device)
        
        # Update camera modes in priority order: arc-follow (new mode) > dynamic-follow > yaw-sweep/locked-follow
        dynamic_enabled = self.task_config.curriculum.enable_dynamic_camera_following
        dynamic_disabled = bool(self.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False))
        arc_follow_enabled = bool(self.sim_env.global_tensor_dict.get('static_camera/arc_follow_enabled', False))
        
        if arc_follow_enabled:
            self.static_camera_manager.update_arc_follow(
                self.obs_dict["robot_position"],
                self.gate_position,
                self.gate_center_height,
                float(self.sim_env.global_tensor_dict.get('static_camera/arc_follow_radius_m', 2.0))
            )
        elif dynamic_enabled and not dynamic_disabled:
            self.static_camera_manager.update_dynamic_camera_following(
                self.obs_dict["robot_position"], 
                self.gate_position, 
                self.gate_center_height
            )
        
        # If static camera yaw sweep is enabled, update static camera orientation every frame
        try:
            gtd = self.sim_env.global_tensor_dict
            sweep_enabled_flag = str(gtd.get('static_camera/yaw_sweep_enabled', 'false')).lower() == 'true'
            # Do not run sweep updates when dynamic following is active
            locked_follow = bool(gtd.get('static_camera/locked_follow', False))
            if locked_follow and not (dynamic_enabled and not dynamic_disabled) and not arc_follow_enabled:
                # Update orientation-only to keep the drone centered
                self.static_camera_manager.update_locked_follow(self.obs_dict["robot_position"]) 
            elif sweep_enabled_flag and not (dynamic_enabled and not dynamic_disabled) and not arc_follow_enabled:
                env_ids_all = torch.arange(self.sim_env.num_envs, device=self.device)
                self.static_camera_manager.update_camera_positions(self.curriculum_level, env_ids_all)
                # Temporary debug for yaw sweep status (print ~once every 5 seconds at 60Hz)
                try:
                    sim_steps = int(gtd.get('sim_steps', torch.tensor([0], device=self.device))[0].item())
                except (ValueError, TypeError):
                    sim_steps = 0
                # Reduce spam: every 300 steps instead of every 60
                if (sim_steps % 300) == 0:
                    logger.warning("[YawSweep] Camera orientation update running (sweeping/locked-follow active)")
        except (ValueError, TypeError) as e:
            logger.debug(f"[YawSweep] Per-step update skipped due to: {e}")

        self._camera._compute_visibility_metrics(infos_to_return)

        
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            infos_to_return,
        )


    def process_obs_for_task(self) -> None:
        """
        Process observations for the gate navigation task.
        
        UPDATED: Now matches DCE navigation task format with static camera pose observations.
        
        Observation space (150D):
        - 0-3: Drone absolute position in world coordinates
        - 3-6: Static camera position relative to drone  
        - 6-9: Static camera orientation relative to drone
        - 9-12: Full drone orientation including yaw
        - 12-15: Robot body linear velocity
        - 15-18: Robot body angular velocity
        - 18-22: Robot actions (4D for gate navigation)
        - 22-86: Drone camera VAE latents (64D)
        - 86-150: Static camera VAE latents (64D)
        """
        # This provides the agent with complete spatial awareness of its state and static camera relative position
        
        drone_pos_clean = self.obs_dict["robot_position"]
        # Apply curriculum-driven state noise (drone position)
        if self.task_config.curriculum.enable_state_noise and not bool(self.sim_env.global_tensor_dict.get('state_randomization/noise_disabled', False)):
            noise_cfg = self.task_config.curriculum.get_state_noise(self.curriculum_level)
            dp_std = float(noise_cfg.get("drone_pos_std_m", 0.0))
            if dp_std > 0.0:
                drone_pos_noised = drone_pos_clean + torch.randn_like(drone_pos_clean) * dp_std
            else:
                drone_pos_noised = drone_pos_clean
        else:
            drone_pos_noised = drone_pos_clean
        obs = self.task_obs["observations"]
        layout = GATE_OBS_LAYOUT

        obs[:, layout.drone_position] = drone_pos_noised

        # Static camera pose (relative to drone)
        static_camera_pos, static_camera_orientation = self._camera._get_static_camera_pose_relative_to_drone()
        if self.task_config.curriculum.enable_state_noise and not bool(self.sim_env.global_tensor_dict.get('state_randomization/noise_disabled', False)):
            noise_cfg = self.task_config.curriculum.get_state_noise(self.curriculum_level)
            sp_std = float(noise_cfg.get("static_pos_std_m", 0.0))
            so_std = float(noise_cfg.get("static_orient_std_rad", 0.0))
            if sp_std > 0.0:
                static_camera_pos = static_camera_pos + torch.randn_like(static_camera_pos) * sp_std
            if so_std > 0.0:
                static_camera_orientation = static_camera_orientation + torch.randn_like(static_camera_orientation) * so_std
                static_camera_orientation = torch.atan2(torch.sin(static_camera_orientation), torch.cos(static_camera_orientation))

        obs[:, layout.static_camera_position] = static_camera_pos
        obs[:, layout.static_camera_orientation] = static_camera_orientation

        # Drone orientation (roll, pitch, yaw)
        euler_angles = ssa(get_euler_xyz_tensor(self.obs_dict["robot_vehicle_orientation"]))
        if self.task_config.curriculum.enable_state_noise and not bool(self.sim_env.global_tensor_dict.get('state_randomization/noise_disabled', False)):
            noise_cfg = self.task_config.curriculum.get_state_noise(self.curriculum_level)
            do_std = float(noise_cfg.get("drone_orient_std_rad", 0.0))
            if do_std > 0.0:
                euler_angles = euler_angles + torch.randn_like(euler_angles) * do_std
                euler_angles = torch.atan2(torch.sin(euler_angles), torch.cos(euler_angles))
        obs[:, layout.drone_orientation] = euler_angles

        # Velocities and actions
        obs[:, layout.body_linear_velocity] = self.obs_dict["robot_body_linvel"]
        obs[:, layout.body_angular_velocity] = self.obs_dict["robot_body_angvel"]
        obs[:, layout.actions] = self.obs_dict["robot_actions"]

        # VAE latents (drone and static camera)
        if isinstance(self.image_latents, torch.Tensor) and self.image_latents.shape[1] >= 64:
            obs[:, layout.drone_vae_latents] = self.image_latents[:, :64]
        if isinstance(self.static_image_latents, torch.Tensor) and self.static_image_latents.shape[1] >= 64:
            obs[:, layout.static_vae_latents] = self.static_image_latents[:, :64]

        # (Removed W&B latent stats logging per request)

        # Final observation NaN/Inf guard: sanitize outgoing observations tensor
        obs_tensor = self.task_obs.get("observations", None)
        if isinstance(obs_tensor, torch.Tensor):
            bad = ~torch.isfinite(obs_tensor)
            if torch.any(bad):
                if self.task_config.guard_debug_enabled:
                    logger.warning(f"[NaNGuard] Sanitizing {int(torch.sum(bad).item())} invalid obs entries before return.")
                obs_tensor[bad] = 0.0



    def compute_rewards_and_crashes(self, obs_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute rewards with gate-specific components."""
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        
        self.pos_error_vehicle_frame_prev[:] = self.pos_error_vehicle_frame
        self.pos_error_vehicle_frame[:] = quat_rotate_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        
        # obs_dict contains direct references to global tensors that get updated simultaneously
        current_actions = obs_dict["robot_actions"].clone()
        previous_actions = obs_dict["robot_prev_actions"].clone()
        
        # First-step stabilization: align previous error/actions with current for fresh episodes
        prev_actions_for_reward = previous_actions
        fresh_mask = self._episode_fresh
        if isinstance(fresh_mask, torch.Tensor) and fresh_mask.shape[0] == self.num_envs:
            if torch.any(fresh_mask):
                # Set previous error equal to current on the first step after reset
                self.pos_error_vehicle_frame_prev[fresh_mask] = self.pos_error_vehicle_frame[fresh_mask]
                # Also zero action diff on the first step after reset
                prev_actions_for_reward = previous_actions.clone()
                prev_actions_for_reward[fresh_mask] = current_actions[fresh_mask]
        
        # Curriculum multiplier ablation: pass effective fraction to scripted reward
        cm_disabled = read_env_bool("SF_DISABLE_CURRICULUM_MULTIPLIER", self.task_config.disable_curriculum_multiplier)
        if not cm_disabled:
            cm_disabled = bool(self.task_config.disable_curriculum_multiplier)
        try:
            frac_current = (
                self.curriculum_level - self.task_config.curriculum.min_level
            ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)
        except (ZeroDivisionError, AttributeError, TypeError):
            frac_current = 0.0
        frac_eff = 0.0 if cm_disabled else float(frac_current)
        self._curriculum_multiplier_factor = 1.0 + 0.5 * frac_eff

        boundary_violation_one_shot_mask = self._rewards._detect_boundary_violation(robot_position)

        # Disable nvFuser optimizations for this scripted block to avoid CUDA fuser fallback warnings
        # (no change in semantics; prevents noisy warnings from TorchScript fuser)
        with torch.jit.optimized_execution(False):
            rewards, crashes, camera_gate_alignment = compute_gate_reward(
                self.pos_error_vehicle_frame,
                self.pos_error_vehicle_frame_prev,
                obs_dict["crashes"],
                current_actions,
                prev_actions_for_reward,
                robot_position,
                robot_vehicle_orientation,
                self.gate_position,
                self.gate_passed,
                frac_eff,
                self.task_config.reward_parameters,
                self.gate_width,
                self.gate_height,
                self.gate_center_height,
                boundary_violation_one_shot_mask,
            )

        rewards = self._rewards._apply_time_penalty(rewards, robot_position)
        rewards = self._rewards._apply_static_fov_reward(rewards, robot_position)
        # UPDATE EPISODE REWARD TRACKING: Track cumulative reward components
        self._reward_tracking.update_episode_reward_tracking(obs_dict, rewards, crashes)
        self._reward_tracking._log_comprehensive_reward_debug(obs_dict, rewards, crashes, boundary_violation_one_shot_mask, camera_gate_alignment)
        
        # Store camera alignment for debugging
        self.camera_alignment_debug = camera_gate_alignment
        
        return rewards, crashes, camera_gate_alignment


        



