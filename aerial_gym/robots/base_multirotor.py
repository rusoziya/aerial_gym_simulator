from __future__ import annotations

from aerial_gym.robots.base_robot import BaseRobot

from aerial_gym.control.control_allocation import ControlAllocator
from aerial_gym.registry.controller_registry import controller_registry

import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("base_multirotor")


class BaseMultirotor(BaseRobot):
    """
    Base class for the quadrotor robot. Does not contain sensors or actuators.
    This class should be inherited by the specific quadrotor class with sensors or actuators.

    The controller config for the robot is used to initialize the controller for the robot.
    """

    def __init__(self, robot_config: object, controller_name: str, env_config: object, device: str) -> None:
        logger.debug("Initializing BaseQuadrotor")
        super().__init__(
            robot_config=robot_config,
            controller_name=controller_name,
            env_config=env_config,
            device=device,
        )
        logger.warning(f"Creating {self.num_envs} multirotors.")
        self.force_application_level = self.cfg.control_allocator_config.force_application_level
        if controller_name == "no_control":
            self.output_mode = "forces"
        else:
            self.output_mode = "wrench"

        if self.force_application_level == "root_link" and controller_name == "no_control":
            raise ValueError(
                "Force application level 'root_link' cannot be used with 'no_control'."
            )

        # Initialize the tensors
        self.robot_state = None
        self.robot_force_tensors = None
        self.robot_torque_tensors = None
        self.action_tensor = None
        self.max_init_state = None
        self.min_init_state = None
        self.max_force_and_torque_disturbance = None
        self.max_torque_disturbance = None
        self.controller_input = None
        self.control_allocator = None
        self.output_forces = None
        self.output_torques = None

        logger.debug("[DONE] Initializing BaseQuadrotor")

    def init_tensors(self, global_tensor_dict: dict[str, object]) -> None:
        """
        Initialize the tensors for the robot state, force, torque, and action.
        The tensors used in this function call are sent as slices from the main tensors in the environment.
        These slices are only detemine the robot state, force, torque, and action.
        The full tensors are not passed to this function to avoid access to data that is not needed by the robot.
        """
        super().init_tensors(global_tensor_dict)
        # Adding more tensors to the global tensor dictionary
        self.robot_vehicle_orientation = torch.zeros_like(
            self.robot_orientation, requires_grad=False, device=self.device
        )
        self.robot_vehicle_linvel = torch.zeros_like(
            self.robot_linvel, requires_grad=False, device=self.device
        )
        self.robot_body_angvel = torch.zeros_like(
            self.robot_linvel, requires_grad=False, device=self.device
        )
        self.robot_body_linvel = torch.zeros_like(
            self.robot_linvel, requires_grad=False, device=self.device
        )
        self.robot_euler_angles = torch.zeros_like(
            self.robot_linvel, requires_grad=False, device=self.device
        )
        # Add to tensor dictionary
        global_tensor_dict["robot_vehicle_orientation"] = self.robot_vehicle_orientation
        global_tensor_dict["robot_vehicle_linvel"] = self.robot_vehicle_linvel
        global_tensor_dict["robot_body_angvel"] = self.robot_body_angvel
        global_tensor_dict["robot_body_linvel"] = self.robot_body_linvel
        global_tensor_dict["robot_euler_angles"] = self.robot_euler_angles

        global_tensor_dict["num_robot_actions"] = self.controller_config.num_actions

        self.controller.init_tensors(global_tensor_dict)
        self.action_tensor = torch.zeros(
            (self.num_envs, self.controller_config.num_actions),
            device=self.device,
            requires_grad=False,
        )

        # Initialize the robot state
        # [x, y, z, roll, pitch, yaw, 1.0 (for maintaining shape), vx, vy, vz, wx, wy, wz]
        self.min_init_state = torch.tensor(
            self.cfg.init_config.min_init_state, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.max_init_state = torch.tensor(
            self.cfg.init_config.max_init_state, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)

        # Disturbance params
        # [fx, fy, fz, tx, ty, tz]
        self.max_force_and_torque_disturbance = torch.tensor(
            self.cfg.disturbance.max_force_and_torque_disturbance,
            device=self.device,
            requires_grad=False,
        ).expand(self.num_envs, -1)

        # Controller params
        self.controller_input = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, requires_grad=False
        )
        self.control_allocator = ControlAllocator(
            num_envs=self.num_envs,
            dt=self.dt,
            config=self.cfg.control_allocator_config,
            device=self.device,
        )

        self.body_vel_linear_damping_coefficient = torch.tensor(
            self.cfg.damping.linvel_linear_damping_coefficient,
            device=self.device,
            requires_grad=False,
        )

        self.body_vel_quadratic_damping_coefficient = torch.tensor(
            self.cfg.damping.linvel_quadratic_damping_coefficient,
            device=self.device,
            requires_grad=False,
        )

        self.angvel_linear_damping_coefficient = torch.tensor(
            self.cfg.damping.angular_linear_damping_coefficient,
            device=self.device,
            requires_grad=False,
        )

        self.angvel_quadratic_damping_coefficient = torch.tensor(
            self.cfg.damping.angular_quadratic_damping_coefficient,
            device=self.device,
            requires_grad=False,
        )

        if self.force_application_level == "motor_link":
            self.application_mask = torch.tensor(
                self.cfg.control_allocator_config.application_mask,
                device=self.device,
                requires_grad=False,
            )
        else:
            self.application_mask = torch.tensor([0], device=self.device, requires_grad=False)

        self.motor_directions = torch.tensor(
            self.cfg.control_allocator_config.motor_directions,
            device=self.device,
            requires_grad=False,
        )

        self.output_forces = torch.zeros_like(
            global_tensor_dict["robot_force_tensor"], device=self.device
        )
        self.output_torques = torch.zeros_like(
            global_tensor_dict["robot_torque_tensor"], device=self.device
        )

    def reset(self) -> None:
        self.reset_idx(torch.arange(self.num_envs))

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        # robot_state is defined as a tensor of shape (num_envs, 13)
        # init_state tensor if of the format [ratio_x, ratio_y, ratio_z, roll, pitch, yaw, 1.0 (for maintaining shape), vx, vy, vz, wx, wy, wz]
        # Curriculum-controlled spawn if task provides curriculum_level and helper; else fallback to fixed
        use_curriculum_spawn = False
        try:
            # Try to read curriculum from global tensors
            if 'curriculum_level' in self._global_tensor_dict:
                from aerial_gym.config.task_config.navigation_task_config_gate import task_config
                level = int(self._global_tensor_dict['curriculum_level'])
                # Read spawn ablation flags
                pos_dis = bool(self._global_tensor_dict.get('spawn_randomization/position_disabled', False))
                yaw_dis = bool(self._global_tensor_dict.get('spawn_randomization/orientation_disabled', False))
                # Active and baseline (min_level) spawn ranges
                sr_active = task_config.curriculum.get_spawn_ranges(level)
                baseline_level = int(task_config.curriculum.min_level)
                sr_base = task_config.curriculum.get_spawn_ranges(baseline_level)
                # Choose ranges according to ablation flags
                sr = {
                    'x_half_span_m': sr_base['x_half_span_m'] if pos_dis else sr_active['x_half_span_m'],
                    'y_center_m':    sr_base['y_center_m']    if pos_dis else sr_active['y_center_m'],
                    'y_half_span_m': sr_base['y_half_span_m'] if pos_dis else sr_active['y_half_span_m'],
                    'z_center_m':    sr_base['z_center_m']    if pos_dis else sr_active['z_center_m'],
                    'z_half_span_m': sr_base['z_half_span_m'] if pos_dis else sr_active['z_half_span_m'],
                    'yaw_abs_rad':   sr_base['yaw_abs_rad']   if yaw_dis else sr_active['yaw_abs_rad'],
                }
                # Convert meters to ratios using env bounds
                # Use environment bounds to set center and scale
                env_x_min = float(self.env_bounds_min[0, 0].item())
                env_x_max = float(self.env_bounds_max[0, 0].item())
                env_y_min = float(self.env_bounds_min[0, 1].item())
                env_y_max = float(self.env_bounds_max[0, 1].item())
                env_z_min = float(self.env_bounds_min[0, 2].item())
                env_z_max = float(self.env_bounds_max[0, 2].item())
                x_center = 0.5 * (env_x_min + env_x_max)  # center of X
                y_center = float(sr['y_center_m'])
                x_half = float(sr['x_half_span_m'])
                y_half = float(sr['y_half_span_m'])
                z_center = float(sr['z_center_m'])
                z_half = float(sr['z_half_span_m'])
                # Build min/max ratios (positions first three entries)
                # Clamp spans to env bounds
                x_min_m = max(env_x_min, x_center - x_half)
                x_max_m = min(env_x_max, x_center + x_half)
                y_min_m = max(env_y_min, y_center - y_half)
                y_max_m = min(env_y_max, y_center + y_half)
                z_min_m = max(env_z_min, z_center - z_half)
                z_max_m = min(env_z_max, z_center + z_half)
                # Convert meters to ratios based on env bounds
                min_ratio_x = (x_min_m - env_x_min) / (env_x_max - env_x_min)
                max_ratio_x = (x_max_m - env_x_min) / (env_x_max - env_x_min)
                min_ratio_y = (y_min_m - env_y_min) / (env_y_max - env_y_min)
                max_ratio_y = (y_max_m - env_y_min) / (env_y_max - env_y_min)
                min_ratio_z = (z_min_m - env_z_min) / (env_z_max - env_z_min)
                max_ratio_z = (z_max_m - env_z_min) / (env_z_max - env_z_min)
                # Yaw in radians
                yaw_abs = float(sr['yaw_abs_rad'])
                min_yaw = -yaw_abs
                max_yaw = +yaw_abs
                # Clone base ranges and override pos/orient
                min_init = self.min_init_state.clone()
                max_init = self.max_init_state.clone()
                min_init[:, 0] = min_ratio_x
                max_init[:, 0] = max_ratio_x
                min_init[:, 1] = min_ratio_y
                max_init[:, 1] = max_ratio_y
                min_init[:, 2] = min_ratio_z
                max_init[:, 2] = max_ratio_z
                # roll, pitch kept from config (0). yaw override
                min_init[:, 5] = min_yaw
                max_init[:, 5] = max_yaw
                random_state = torch_rand_float_tensor(min_init, max_init)
                use_curriculum_spawn = True
                logger.debug(f"[SPAWN_CURRICULUM] Level {level}: X∈[{x_min_m:.2f},{x_max_m:.2f}] m, Y∈[{y_min_m:.2f},{y_max_m:.2f}] m, Z∈[{z_min_m:.2f},{z_max_m:.2f}] m; yaw∈[{min_yaw:.2f},{max_yaw:.2f}]rad")
        except (AttributeError, KeyError, ValueError, TypeError, RuntimeError) as e:
            logger.warning(f"[SPAWN_CURRICULUM] Disabled due to exception: {e}")
        if not use_curriculum_spawn:
            random_state = torch_rand_float_tensor(self.min_init_state, self.max_init_state)

        self.robot_state[env_ids, 0:3] = torch_interpolate_ratio(
            self.env_bounds_min, self.env_bounds_max, random_state[:, 0:3]
        )[env_ids]

        # Align spawn yaw so the camera (+X body axis) faces the gate center, with curriculum jitter
        try:
            from aerial_gym.config.task_config.navigation_task_config_gate import task_config as _tc
            if 'curriculum_level' in self._global_tensor_dict:
                level = int(self._global_tensor_dict['curriculum_level'])
                # Respect orientation ablation flag by selecting baseline yaw if disabled
                yaw_dis = bool(self._global_tensor_dict.get('spawn_randomization/orientation_disabled', False))
                if yaw_dis:
                    baseline_level = int(_tc.curriculum.min_level)
                    sr_local = _tc.curriculum.get_spawn_ranges(baseline_level)
                else:
                    sr_local = _tc.curriculum.get_spawn_ranges(level)
                yaw_abs = float(sr_local.get('yaw_abs_rad', 0.0))
            else:
                yaw_abs = 0.0
        except ImportError:
            yaw_abs = 0.0

        # Compute yaw to face the gate center at (0, 0) in world X-Y (gate opening faces +Y)
        pos_xy = self.robot_state[env_ids, 0:2]
        to_gate_xy = -pos_xy  # gate assumed at (0,0)
        yaw_face = torch.atan2(to_gate_xy[:, 1], to_gate_xy[:, 0])
        if yaw_abs > 0.0:
            jitter = (torch.rand_like(yaw_face) * 2.0 - 1.0) * yaw_abs
        else:
            jitter = torch.zeros_like(yaw_face)
        # Overwrite yaw in random_state before quaternion conversion
        random_state[env_ids, 5] = yaw_face + jitter

        # Optional debug: print a few spawned positions and yaw to verify ranges
        try:
            from aerial_gym.config.task_config.navigation_task_config_gate import task_config
            do_debug = bool(task_config.curriculum.enable_detailed_logging)
        except ImportError:
            do_debug = False
        if do_debug:
            sample_n = min(3, env_ids.shape[0])
            sample_envs = env_ids[:sample_n]
            pos_samples = self.robot_state[sample_envs, 0:3].detach().cpu()
            yaw_samples = random_state[sample_envs, 5].detach().cpu()

        # quat conversion is handled separately (uses our yaw override above)
        self.robot_state[env_ids, 3:7] = quat_from_euler_xyz_tensor(random_state[env_ids, 3:6])

        self.robot_state[env_ids, 7:10] = random_state[env_ids, 7:10]
        self.robot_state[env_ids, 10:13] = random_state[env_ids, 10:13]

        self.controller.randomize_params(env_ids=env_ids)
        self.control_allocator.reset_idx(env_ids)

        # update the states after resetting because the RL agent gets the first state after reset
        self.update_states()

    def clip_actions(self) -> None:
        """
        Clip the action tensor to the range of the controller inputs.
        """
        self.action_tensor[:] = torch.clamp(self.action_tensor, -10.0, 10.0)

    def apply_disturbance(self) -> None:
        if not self.cfg.disturbance.enable_disturbance:
            return
        disturbance_occurence = torch.bernoulli(
            self.cfg.disturbance.prob_apply_disturbance
            * torch.ones((self.num_envs), device=self.device)
        )
        self.robot_force_tensors[:, 0, 0:3] += torch_rand_float_tensor(
            -self.max_force_and_torque_disturbance[:, 0:3],
            self.max_force_and_torque_disturbance[:, 0:3],
        ) * disturbance_occurence.unsqueeze(1)
        self.robot_torque_tensors[:, 0, 0:3] += torch_rand_float_tensor(
            -self.max_force_and_torque_disturbance[:, 3:6],
            self.max_force_and_torque_disturbance[:, 3:6],
        ) * disturbance_occurence.unsqueeze(1)

    def control_allocation(self, command_wrench: torch.Tensor, output_mode: str) -> None:
        """
        Allocate the thrust and torque commands to the motors. The motor model is also used to update the motor thrusts.
        """

        forces, torques = self.control_allocator.allocate_output(command_wrench, output_mode)

        self.output_forces[:, self.application_mask, :] = forces
        self.output_torques[:, self.application_mask, :] = torques

    def call_controller(self) -> None:
        """
        Convert the action tensor to the controller inputs. The action tensor is the input and can be parametrized as desired by the user.
        This function serves the purpose of converting the action tensor to the controller inputs.
        """

        self.clip_actions()

        controller_output = self.controller(self.action_tensor)
        self.control_allocation(controller_output, self.output_mode)

        self.robot_force_tensors[:] = self.output_forces
        self.robot_torque_tensors[:] = self.output_torques

    def simulate_drag(self) -> None:
        self.robot_body_vel_drag_linear = (
            -self.body_vel_linear_damping_coefficient * self.robot_body_linvel
        )
        self.robot_body_vel_drag_quadratic = (
            -self.body_vel_quadratic_damping_coefficient
            * torch.norm(self.robot_body_linvel, dim=-1).unsqueeze(-1)
            * self.robot_body_linvel
        )
        self.robot_body_vel_drag = (
            self.robot_body_vel_drag_linear + self.robot_body_vel_drag_quadratic
        )
        self.robot_force_tensors[:, 0, 0:3] += self.robot_body_vel_drag

        self.robot_body_angvel_drag_linear = (
            -self.angvel_linear_damping_coefficient * self.robot_body_angvel
        )
        self.robot_body_angvel_drag_quadratic = (
            -self.angvel_quadratic_damping_coefficient
            * self.robot_body_angvel.abs()
            * self.robot_body_angvel
        )
        self.robot_body_angvel_drag = (
            self.robot_body_angvel_drag_linear + self.robot_body_angvel_drag_quadratic
        )
        self.robot_torque_tensors[:, 0, 0:3] += self.robot_body_angvel_drag

    def update_states(self) -> None:
        self.robot_euler_angles[:] = ssa(get_euler_xyz_tensor(self.robot_orientation))
        self.robot_vehicle_orientation[:] = vehicle_frame_quat_from_quat(self.robot_orientation)
        self.robot_vehicle_linvel[:] = quat_rotate_inverse(
            self.robot_vehicle_orientation, self.robot_linvel
        )
        self.robot_body_linvel[:] = quat_rotate_inverse(self.robot_orientation, self.robot_linvel)
        self.robot_body_angvel[:] = quat_rotate_inverse(self.robot_orientation, self.robot_angvel)

    def step(self, action_tensor: torch.Tensor) -> None:
        """
        Update the state of the quadrotor. This function is called every simulation step.
        """
        self.update_states()
        if action_tensor.shape[0] != self.num_envs:
            raise ValueError("Action tensor does not have the correct number of environments")
        self.action_tensor[:] = action_tensor
        # calling controller leads to control allocation happening, and
        self.call_controller()
        self.simulate_drag()
        self.apply_disturbance()
