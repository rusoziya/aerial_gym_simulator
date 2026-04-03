from __future__ import annotations

import os

import torch

from aerial_gym.sensors.static_camera_manager import StaticCameraManager
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.task.base_task import BaseTask, StepReturn
from aerial_gym.task.navigation_task_gate.camera_observations import CameraObservations
from aerial_gym.task.navigation_task_gate.curriculum_logging import CurriculumLogging
from aerial_gym.task.navigation_task_gate.curriculum_management import CurriculumManager
from aerial_gym.task.navigation_task_gate.gate_geometry import GateGeometry
from aerial_gym.task.navigation_task_gate.init_helpers import InitHelpers
from aerial_gym.task.navigation_task_gate.obs_reward_helpers import (
    compute_rewards_and_crashes as _compute_rewards_and_crashes,
)
from aerial_gym.task.navigation_task_gate.obs_reward_helpers import (
    logging_sanity_check as _logging_sanity_check,
)
from aerial_gym.task.navigation_task_gate.obs_reward_helpers import (
    process_obs_for_task as _process_obs_for_task,
)
from aerial_gym.task.navigation_task_gate.obs_reward_helpers import (
    update_camera_modes as _update_camera_modes,
)
from aerial_gym.task.navigation_task_gate.reward_helpers import RewardHelpers
from aerial_gym.task.navigation_task_gate.reward_tracking import RewardTracking
from aerial_gym.task.navigation_task_gate.step_helpers import StepHelpers
from aerial_gym.task.task_config_protocol import TaskConfig
from aerial_gym.utils.env_flag_utils import (
    apply_ablation_flags_to_tensor_dict,
    parse_ablation_flags,
    read_env_bool,
    read_env_int,
)
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.tensor_utils import invalid_mask_per_env

logger = CustomLogger("navigation_task_gate")


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
            spec_str = os.environ.get("ABLATE_OBS_RANGES", "").strip()
            static_ablated = False
            if spec_str:
                for spec in [s.strip() for s in spec_str.split(",") if s.strip() and "=" in s]:
                    lhs, rhs = spec.split("=", 1)
                    lhs = lhs.strip()
                    rhs = rhs.strip()
                    if ":" in lhs:
                        try:
                            a, b = lhs.split(":", 1)
                            a = int(a)
                            b = int(b)
                        except (ValueError, TypeError):
                            continue
                        if rhs in ("zero", "zerograd") and a <= 86 and b >= 150:
                            static_ablated = True
                            break
            if static_ablated:
                self.task_config.reward_parameters["static_fov_visibility_reward_magnitude"] = 0.0
        except (ValueError, TypeError):
            pass

        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )

        logger.info("Building environment for gate navigation task.")
        logger.info(
            f"Sim Name: {self.task_config.sim_name}, Env Name: {self.task_config.env_name}, Robot Name: {self.task_config.robot_name}, Controller Name: {self.task_config.controller_name}"
        )

        self.curriculum_level = self.task_config.curriculum.min_level
        obstacles_disable = read_env_bool(
            "SF_DISABLE_OBSTACLE_RANDOMIZATION", self.task_config.disable_obstacle_randomization
        )
        obstacles_fixed = read_env_int(
            "SF_FIXED_OBSTACLES_BEHIND_GATE", self.task_config.fixed_obstacles_behind_gate
        )
        if obstacles_disable:
            obstacles_behind_gate = max(0, obstacles_fixed)
        else:
            obstacles_behind_gate = self.task_config.curriculum.get_obstacle_count_behind_gate(
                self.curriculum_level
            )

        visible_gates = 0
        walls = 6
        fixed_assets_visible = visible_gates + walls
        total_obstacles_in_env = fixed_assets_visible + obstacles_behind_gate

        logger.info(
            f"PRE-INIT: Setting curriculum level {self.curriculum_level} with {obstacles_behind_gate} curriculum obstacles"
        )
        logger.info(
            f"PRE-INIT: Visible assets (env assets only): {visible_gates} gate + {walls} walls + {obstacles_behind_gate} curriculum = {total_obstacles_in_env} total"
        )
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
        self.num_envs = self.sim_env.num_envs

        flags = parse_ablation_flags(self.task_config)
        apply_ablation_flags_to_tensor_dict(
            self.sim_env.global_tensor_dict, flags, self.task_config, logger
        )
        self.disable_static_camera_orientation_randomization = flags[
            "static_camera_orient_disabled"
        ]
        self.disable_camera_frame_dropout_randomization = flags["camera_frame_dropout_disabled"]
        self.disable_camera_noise_randomization = flags["camera_noise_disabled"]
        self.disable_state_noise_randomization = flags["state_noise_disabled"]
        self.disable_dynamic_camera_following = flags["dynamic_camera_following_disabled"]

        logger.info("[GateVariant] Initial selection after build")
        self.sim_env.apply_gate_variant_selection(
            env_ids=torch.arange(self.sim_env.num_envs, device=self.device)
        )

        # Override count if obstacle randomization disabled
        obs_dis = bool(
            self.sim_env.global_tensor_dict.get("obstacles_randomization/disabled", False)
        )
        if obs_dis:
            fixed_count = int(
                self.sim_env.global_tensor_dict.get("obstacles_randomization/fixed_count", 0)
            )
            total_obstacles_in_env = fixed_assets_visible + max(0, fixed_count)
        self.sim_env.global_tensor_dict.num_obstacles_in_env = total_obstacles_in_env
        logger.info(
            f"POST-INIT: Updated global_tensor_dict with obstacle count: {total_obstacles_in_env}"
        )

        # Target position: will be set to adaptive gate center on each reset
        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        self.success_aggregate = 0
        self.crashes_aggregate = 0
        self.timeouts_aggregate = 0

        self._init._init_gate_tracking_tensors()
        self._init._init_vae_model()

        self.static_camera_manager = StaticCameraManager(self.sim_env, self.task_config)

        self.obs_dict = self.sim_env.get_obs()

        self._init._init_curriculum()

        try:
            self.terminations = self.obs_dict.terminations
        except (KeyError, TypeError):
            self.terminations = self.obs_dict.crashes
        self.truncations = self.obs_dict.truncations
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        self._init._init_observation_action_spaces()
        self._init._init_task_observations()

        self.pos_error_vehicle_frame = torch.zeros(self.num_envs, 3, device=self.device)
        self.pos_error_vehicle_frame_prev = torch.zeros(self.num_envs, 3, device=self.device)
        self.gate_passed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._ep_target_success_flag = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.camera_alignment_debug = torch.zeros(self.num_envs, device=self.device)
        self.num_task_steps = 0
        self.curriculum_progress_fraction = 0.0

        self._init._init_episode_reward_tracking()
        self._init._init_episode_trajectory_state()
        self._init._init_debug_flags()

        # Initialize gate dimensions for all environments after full initialization
        self._geometry.update_gate_dimensions_for_environments(
            torch.arange(self.sim_env.num_envs, device=self.device)
        )

        # Ensure infos survive resets for logging back to the learner
        self._infos_to_return = None

    def logging_sanity_check(self, infos: dict[str, torch.Tensor]) -> None:
        """Sanity check for logging to detect issues with success/crash/timeout logic."""
        _logging_sanity_check(self, infos)

    # Delegation methods so composed helpers can call through self.task
    def setup_curriculum_logging(self):
        self._curriculum.setup_curriculum_logging()

    def log_curriculum_update(self, msg):
        self._curriculum.log_curriculum_update(msg)

    def extract_gate_dimensions_from_urdf(self, p):
        return self._geometry.extract_gate_dimensions_from_urdf(p)

    def calculate_gate_dimensions_from_name(self, n):
        return self._geometry.calculate_gate_dimensions_from_name(n)

    def update_gate_dimensions_for_environments(self, ids):
        self._geometry.update_gate_dimensions_for_environments(ids)

    def process_image_observation(self):
        self._camera.process_image_observation()

    def process_static_camera_observation(self):
        self._camera.process_static_camera_observation()

    def post_image_reward_addition(self):
        self._rewards.post_image_reward_addition()

    def close(self) -> None:
        self.sim_env.delete_env()

        if self.curriculum_log_file:
            try:
                import datetime

                self.curriculum_log_file.write(
                    f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    "Training session ended.\n"
                )
                self.curriculum_log_file.close()
            except OSError as e:
                logger.warning(f"Error closing curriculum log: {e}")

    def reset(self) -> StepReturn:
        self.reset_idx(torch.arange(self.sim_env.num_envs))
        return self.get_return_tuple()

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset specified environments: update gate tracking, camera, and target positions."""
        if "gate_position" in self.obs_dict:
            self.gate_position[env_ids] = self.obs_dict.gate_position[env_ids]
        else:
            self.gate_position[env_ids] = 0.0

        self.gate_passed[env_ids] = False
        self._ep_target_success_flag[env_ids] = False
        self.gate_approach_distance[env_ids] = 0.0

        self._reward_tracking.reset_episode_reward_tracking(env_ids)

        if len(env_ids) > 0:
            self.static_camera_manager.update_camera_positions(self.curriculum_level, env_ids)
            logger.debug(
                f"Updated static camera angles for {len(env_ids)} resetting environments: {env_ids.tolist()}"
            )

        self.sim_env.global_tensor_dict.curriculum_level = int(self.curriculum_level)
        self.sim_env.global_tensor_dict.eval_stretch_enabled = bool(
            self.task_config.curriculum.eval_stretch_enabled
        )
        self.sim_env.global_tensor_dict.eval_stretch_end_level = int(
            self.task_config.curriculum.eval_stretch_end_level
        )
        self._geometry.update_gate_dimensions_for_environments(env_ids)

        # Set target position to adaptive gate center after gate dimensions are updated
        gate_center_x = self.gate_position[env_ids, 0]
        gate_center_y = self.gate_position[env_ids, 1]
        gate_center_z = self.gate_position[env_ids, 2] + self.gate_center_height[env_ids]

        self.target_position[env_ids, 0] = gate_center_x
        self.target_position[env_ids, 1] = gate_center_y
        self.target_position[env_ids, 2] = gate_center_z

        self.trajectory.reset_envs(env_ids)
        self.infos = {}

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
        self.rewards[:], self.terminations[:], camera_gate_alignment = (
            self.compute_rewards_and_crashes(self.obs_dict)
        )
        # Reward NaN/Inf guard: sanitize invalid rewards and truncate offending envs
        invalid_reward_mask = invalid_mask_per_env(self.rewards)
        if torch.any(invalid_reward_mask):
            if self.task_config.guard_debug_enabled:
                _ids = torch.nonzero(invalid_reward_mask, as_tuple=False).squeeze(-1).tolist()
                logger.warning(f"[NaNGuard] Invalid REWARD in envs {_ids}; zeroed and truncating.")
            self.rewards[invalid_reward_mask] = 0.0
            # Ensure truncation to reset these envs safely
            self.truncations[invalid_reward_mask] = 1

        if self.task_config.return_state_before_reset:
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

        robot_position = self.obs_dict.robot_position
        robot_position_before_reset = robot_position.clone()
        successes, target_successes, gate_passage_success = self._step._detect_gate_passage(
            robot_position
        )

        self._step._apply_timeout_and_populate_infos(successes)

        robot_position = self.obs_dict.robot_position
        gate_center_position, gate_passed_current = self._step._compute_gate_navigation_metrics(
            robot_position,
            camera_gate_alignment,
        )

        # Update per-env episode trajectory state
        self._step._update_trajectory_state(
            robot_position, gate_center_position, gate_passed_current
        )

        self._curriculum.check_and_update_curriculum_level(
            self.infos["successes"], self.infos["crashes"], self.infos["timeouts"]
        )

        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self._step._handle_post_reward_reset(
                robot_position,
                robot_position_before_reset,
                gate_center_position,
                successes,
                target_successes,
                reset_envs,
            )
        self.num_task_steps += 1

        self._step._process_images_and_finalize()

        if not self.task_config.return_state_before_reset:
            return_tuple = self.get_return_tuple()
        return return_tuple

    def get_return_tuple(self) -> StepReturn:
        self.process_obs_for_task()
        if self._infos_to_return is not None:
            infos_to_return = self._infos_to_return
            self._infos_to_return = None
        else:
            infos_to_return = self.infos

        _update_camera_modes(self)
        self._camera._compute_visibility_metrics(infos_to_return)

        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            infos_to_return,
        )

    def process_obs_for_task(self) -> None:
        """Assemble 150D observation vector from raw sensor/state data (see GATE_OBS_LAYOUT)."""
        _process_obs_for_task(self)

    def compute_rewards_and_crashes(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute rewards with gate-specific components."""
        return _compute_rewards_and_crashes(self, obs_dict)
