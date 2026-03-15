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

    def __init__(
        self, robot_config: object, controller_name: str, env_config: object, device: str
    ) -> None:
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
        random_state = self._compute_spawn_state()

        self.robot_state[env_ids, 0:3] = torch_interpolate_ratio(
            self.env_bounds_min, self.env_bounds_max, random_state[:, 0:3]
        )[env_ids]

        yaw_abs = self._get_spawn_yaw_jitter()
        self._apply_gate_facing_yaw(env_ids, random_state, yaw_abs)

        self.robot_state[env_ids, 3:7] = quat_from_euler_xyz_tensor(random_state[env_ids, 3:6])
        self.robot_state[env_ids, 7:10] = random_state[env_ids, 7:10]
        self.robot_state[env_ids, 10:13] = random_state[env_ids, 10:13]

        self.controller.randomize_params(env_ids=env_ids)
        self.control_allocator.reset_idx(env_ids)
        self.update_states()

    def _compute_spawn_state(self) -> torch.Tensor:
        """Compute initial random state using curriculum spawn ranges if available."""
        try:
            if "curriculum_level" in self._global_tensor_dict:
                return self._compute_curriculum_spawn_state()
        except (AttributeError, KeyError, ValueError, TypeError, RuntimeError) as e:
            logger.warning(f"[SPAWN_CURRICULUM] Disabled due to exception: {e}")
        return torch_rand_float_tensor(self.min_init_state, self.max_init_state)

    def _compute_curriculum_spawn_state(self) -> torch.Tensor:
        """Compute spawn state based on curriculum-controlled spawn ranges."""
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config

        level = int(self._global_tensor_dict["curriculum_level"])
        sr = self._get_effective_spawn_ranges(task_config, level)

        env_bounds = self._extract_env_bounds()
        min_init, max_init = self._build_curriculum_init_ranges(sr, env_bounds)

        random_state = torch_rand_float_tensor(min_init, max_init)
        logger.debug(
            f"[SPAWN_CURRICULUM] Level {level}: "
            f"X∈[{env_bounds['x_min_m']:.2f},{env_bounds['x_max_m']:.2f}] m"
        )
        return random_state

    def _get_effective_spawn_ranges(self, task_config: object, level: int) -> dict[str, float]:
        """Get spawn ranges respecting ablation flags."""
        pos_dis = bool(self._global_tensor_dict.get("spawn_randomization/position_disabled", False))
        yaw_dis = bool(
            self._global_tensor_dict.get("spawn_randomization/orientation_disabled", False)
        )
        sr_active = task_config.curriculum.get_spawn_ranges(level)
        baseline_level = int(task_config.curriculum.min_level)
        sr_base = task_config.curriculum.get_spawn_ranges(baseline_level)
        return {
            "x_half_span_m": sr_base["x_half_span_m"] if pos_dis else sr_active["x_half_span_m"],
            "y_center_m": sr_base["y_center_m"] if pos_dis else sr_active["y_center_m"],
            "y_half_span_m": sr_base["y_half_span_m"] if pos_dis else sr_active["y_half_span_m"],
            "z_center_m": sr_base["z_center_m"] if pos_dis else sr_active["z_center_m"],
            "z_half_span_m": sr_base["z_half_span_m"] if pos_dis else sr_active["z_half_span_m"],
            "yaw_abs_rad": sr_base["yaw_abs_rad"] if yaw_dis else sr_active["yaw_abs_rad"],
        }

    def _extract_env_bounds(self) -> dict[str, float]:
        return {
            "x_min": float(self.env_bounds_min[0, 0].item()),
            "x_max": float(self.env_bounds_max[0, 0].item()),
            "y_min": float(self.env_bounds_min[0, 1].item()),
            "y_max": float(self.env_bounds_max[0, 1].item()),
            "z_min": float(self.env_bounds_min[0, 2].item()),
            "z_max": float(self.env_bounds_max[0, 2].item()),
        }

    def _build_curriculum_init_ranges(
        self, sr: dict[str, float], eb: dict[str, float]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build min/max init state tensors from curriculum spawn ranges and env bounds."""
        x_center = 0.5 * (eb["x_min"] + eb["x_max"])
        x_min_m = max(eb["x_min"], x_center - sr["x_half_span_m"])
        x_max_m = min(eb["x_max"], x_center + sr["x_half_span_m"])
        y_min_m = max(eb["y_min"], sr["y_center_m"] - sr["y_half_span_m"])
        y_max_m = min(eb["y_max"], sr["y_center_m"] + sr["y_half_span_m"])
        z_min_m = max(eb["z_min"], sr["z_center_m"] - sr["z_half_span_m"])
        z_max_m = min(eb["z_max"], sr["z_center_m"] + sr["z_half_span_m"])

        min_init = self.min_init_state.clone()
        max_init = self.max_init_state.clone()
        min_init[:, 0] = (x_min_m - eb["x_min"]) / (eb["x_max"] - eb["x_min"])
        max_init[:, 0] = (x_max_m - eb["x_min"]) / (eb["x_max"] - eb["x_min"])
        min_init[:, 1] = (y_min_m - eb["y_min"]) / (eb["y_max"] - eb["y_min"])
        max_init[:, 1] = (y_max_m - eb["y_min"]) / (eb["y_max"] - eb["y_min"])
        min_init[:, 2] = (z_min_m - eb["z_min"]) / (eb["z_max"] - eb["z_min"])
        max_init[:, 2] = (z_max_m - eb["z_min"]) / (eb["z_max"] - eb["z_min"])
        min_init[:, 5] = -sr["yaw_abs_rad"]
        max_init[:, 5] = sr["yaw_abs_rad"]
        return min_init, max_init

    def _get_spawn_yaw_jitter(self) -> float:
        """Get yaw jitter range from curriculum config."""
        try:
            from aerial_gym.config.task_config.navigation_task_config_gate import task_config as _tc

            if "curriculum_level" not in self._global_tensor_dict:
                return 0.0
            level = int(self._global_tensor_dict["curriculum_level"])
            yaw_dis = bool(
                self._global_tensor_dict.get("spawn_randomization/orientation_disabled", False)
            )
            if yaw_dis:
                sr_local = _tc.curriculum.get_spawn_ranges(int(_tc.curriculum.min_level))
            else:
                sr_local = _tc.curriculum.get_spawn_ranges(level)
            return float(sr_local.get("yaw_abs_rad", 0.0))
        except ImportError:
            return 0.0

    def _apply_gate_facing_yaw(
        self, env_ids: torch.Tensor, random_state: torch.Tensor, yaw_abs: float
    ) -> None:
        """Align spawn yaw so camera faces the gate center, with optional jitter."""
        pos_xy = self.robot_state[env_ids, 0:2]
        to_gate_xy = -pos_xy
        yaw_face = torch.atan2(to_gate_xy[:, 1], to_gate_xy[:, 0])
        if yaw_abs > 0.0:
            jitter = (torch.rand_like(yaw_face) * 2.0 - 1.0) * yaw_abs
        else:
            jitter = torch.zeros_like(yaw_face)
        random_state[env_ids, 5] = yaw_face + jitter

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
