from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaacgym import gymapi, gymtorch

from aerial_gym.env_manager.base_env_manager import BaseManager

if TYPE_CHECKING:
    from aerial_gym.env_manager.global_tensor_dict_schema import GlobalTensorDict
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.robots.inertia_computation import compute_composite_inertia, compute_robot_com
from aerial_gym.sensors.imu_sensor import IMUSensor
from aerial_gym.sensors.isaacgym_camera_sensor import IsaacGymCameraSensor
from aerial_gym.sensors.warp.warp_sensor import WarpSensor
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("robot_manager")


class RobotManagerIGE(BaseManager):
    def __init__(
        self,
        global_sim_dict: dict[str, object],
        robot_name: str,
        controller_name: str,
        device: str,
        robot_id: int = 0,
    ) -> None:
        logger.debug("Initializing RobotManagerIGE")
        self.gym = global_sim_dict["gym"]
        self.sim = global_sim_dict["sim"]
        self.env_config = global_sim_dict["env_cfg"]
        self.use_warp = global_sim_dict["use_warp"]
        self.num_envs = global_sim_dict["num_envs"]
        # create the robot from the name registry and use the configs from this created robot.
        self.robot, robot_config = robot_registry.make_robot(
            robot_name, controller_name, self.env_config, device
        )

        # Super-class is initialized when the robot registry tells us what the robot config is
        super().__init__(robot_config, device)

        self.robot_handles = []  # list of robot handles

        self.camera_sensor = None
        self.warp_sensor = None
        self.lidar_sensor = None
        self.imu_sensor = None
        self.has_IGE_sensors = False

        self.robot_inertia = None
        self.robot_mass = None
        self.robot_masses = torch.zeros(self.num_envs, device=self.device)
        self.robot_inertias = torch.zeros((self.num_envs, 3, 3), device=self.device)

        self.dof_control_mode: str = "none"

        self.robot_id: int = robot_id
        self.robot_name_prefix: str = f"robot_{robot_id}_"
        self.env_robot_mapping: dict[int, int] = {}

        if self.use_warp == False:
            if self.cfg.sensor_config.enable_camera:
                logger.debug("Initializing Isaac Gym camera sensor")
                self.camera_sensor = IsaacGymCameraSensor(
                    self.cfg.sensor_config.camera_config,
                    self.num_envs,
                    self.gym,
                    self.sim,
                    self.device,
                )
                logger.debug("[DONE] Initializing Isaac Gym camera sensor")
            if self.cfg.sensor_config.enable_lidar:
                raise ValueError(
                    "Lidar sensors are not supported using Isaac Gym Rendering. Please enable warp."
                )
        elif self.use_warp == True and (
            self.cfg.sensor_config.enable_camera and self.cfg.sensor_config.enable_lidar
        ):
            logger.warning(
                "Warp is enabled. Appropriate camera sensors will be spawned using warp."
            )
            logger.error(
                "This error is here because you have enabled both camera and lidar sensors with warp."
            )
            logger.error(
                "There is no reason for the simulation to kill itself really, but both have not been extensively tested together."
            )
            logger.error(
                "if you really need to use both, just comment out the exception here and these lines and things should mostly work okay :) "
            )
            logger.error(
                "You might need to declare another tensor for sensor data for the other sensor though because they currently use the same tensor."
            )
            raise ValueError(
                "Both camera and lidar are enabled. But there is no reason for this error other than preventing undesired behaviors. Just comment out this error line and things should be okay."
            )

            # Warp sensors are not instantiated here. They are prepared in the prepare_for_sim function as they require a ready environment to work with.

        logger.debug("[DONE] Initializing RobotManagerIGE")

        return

    def create_robot(self, asset_loader_class: object) -> None:
        # create the robot from the name registry and use the configs from this created robot.
        logger.debug("Creating robot asset for Isaac Gym")
        robot_asset_class = self.cfg.robot_asset
        self.robot_asset_dict = asset_loader_class.load_selected_file_from_config(
            "robot", robot_asset_class, robot_asset_class.file, is_robot=True
        )
        logger.debug("[DONE] Creating robot asset for Isaac Gym")

        return

    def _init_sensors(self) -> None:
        """Initialize camera/lidar sensors (warp or Isaac Gym native)."""
        if not self.use_warp:
            logger.error("Not using warp. Initializing sensors")
            if self.cfg.sensor_config.enable_lidar:
                raise ValueError(
                    "Lidar sensors are not supported using Isaac Gym Rendering. Please enable warp."
                )

            if self.cfg.sensor_config.enable_camera:
                self.image_tensor = torch.zeros(
                    (
                        self.num_envs,
                        self.cfg.sensor_config.camera_config.num_sensors,
                        self.cfg.sensor_config.camera_config.height,
                        self.cfg.sensor_config.camera_config.width,
                    ),
                    device=self.device,
                    requires_grad=False,
                )
                self.global_tensor_dict["depth_range_pixels"] = self.image_tensor
                self.rgb_image_tensor = torch.zeros(
                    (
                        self.num_envs,
                        self.cfg.sensor_config.camera_config.num_sensors,
                        self.cfg.sensor_config.camera_config.height,
                        self.cfg.sensor_config.camera_config.width,
                        4,
                    ),
                    device=self.device,
                    requires_grad=False,
                )
                self.global_tensor_dict["rgb_pixels"] = self.rgb_image_tensor

                if self.cfg.sensor_config.camera_config.segmentation_camera:
                    self.segmentation_tensor = torch.zeros(
                        (
                            self.num_envs,
                            self.cfg.sensor_config.camera_config.num_sensors,
                            self.cfg.sensor_config.camera_config.height,
                            self.cfg.sensor_config.camera_config.width,
                        ),
                        dtype=torch.int32,
                        device=self.device,
                        requires_grad=False,
                    )
                    self.global_tensor_dict["segmentation_pixels"] = self.segmentation_tensor
                    logger.critical(
                        f"Segmentation pixels shape: {self.global_tensor_dict['segmentation_pixels'].shape}"
                    )
                logger.critical(
                    f"Depth range pixels shape: {self.global_tensor_dict['depth_range_pixels'].shape}"
                )

                self.camera_sensor.init_tensors(global_tensor_dict=self.global_tensor_dict)
        else:
            # assert that only one of camera or lidar is used at once
            assert not (
                self.cfg.sensor_config.enable_camera and self.cfg.sensor_config.enable_lidar
            ), "Do not use both camera and lidar sensors together for now."

            self.warp_sensor_config = None
            if self.cfg.sensor_config.enable_camera:
                self.warp_sensor_config = self.cfg.sensor_config.camera_config
                self.warp_sensor_class = WarpSensor
            elif self.cfg.sensor_config.enable_lidar:
                self.warp_sensor_config = self.cfg.sensor_config.lidar_config
                self.warp_sensor_class = WarpSensor

            if self.warp_sensor_config is not None:
                logger.debug("Initializing warp sensor")
                # prepare the tensors for simulation before preparing the tensors for the sensors
                image_tensor_dims = 3 * (self.warp_sensor_config.return_pointcloud == True)
                if self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"] is None:
                    logger.critical(
                        "Warp camera is enabled but there is nothing in the environment. No rendering will take place and the camera tensor will not be populated."
                    )
                else:
                    if image_tensor_dims == 0:
                        self.image_tensor = torch.zeros(
                            (
                                self.num_envs,
                                self.warp_sensor_config.num_sensors,
                                self.warp_sensor_config.height,
                                self.warp_sensor_config.width,
                            ),
                            device=self.device,
                            requires_grad=False,
                        )
                    else:
                        self.image_tensor = torch.zeros(
                            (
                                self.num_envs,
                                self.warp_sensor_config.num_sensors,
                                self.warp_sensor_config.height,
                                self.warp_sensor_config.width,
                                image_tensor_dims,
                            ),
                            device=self.device,
                            requires_grad=False,
                        )
                    self.global_tensor_dict["depth_range_pixels"] = self.image_tensor

                    if self.warp_sensor_config.segmentation_camera:
                        self.segmentation_tensor = torch.zeros(
                            (
                                self.num_envs,
                                self.warp_sensor_config.num_sensors,
                                self.warp_sensor_config.height,
                                self.warp_sensor_config.width,
                            ),
                            dtype=torch.int32,
                            device=self.device,
                            requires_grad=False,
                        )
                        self.global_tensor_dict["segmentation_pixels"] = self.segmentation_tensor
                    self.warp_sensor = self.warp_sensor_class(
                        self.warp_sensor_config,
                        self.num_envs,
                        self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"],
                        self.device,
                    )
                    self.warp_sensor.init_tensors(global_tensor_dict=self.global_tensor_dict)
                    logger.debug("[DONE] Initializing warp sensor")
                    logger.debug("Capturing warp sensor")
                    self.warp_sensor.update()
                    logger.debug("[DONE] Capturing warp sensor")

        if self.cfg.sensor_config.enable_imu:
            logger.debug("Initializing IMU sensor")
            # acquire force tensors for each of the assets
            self.force_sensor_tensor = gymtorch.wrap_tensor(
                self.gym.acquire_force_sensor_tensor(self.sim)
            )
            self.global_tensor_dict["force_sensor_tensor"] = self.force_sensor_tensor

            self.imu_sensor = IMUSensor(
                self.cfg.sensor_config.imu_config, self.num_envs, self.device
            )
            self.imu_sensor.init_tensors(global_tensor_dict=self.global_tensor_dict)
            logger.debug("[DONE] Initializing IMU sensor")

        elif self.use_warp == False and self.camera_sensor is not None:
            self.has_IGE_sensors = True
        return

    def prepare_for_sim(self, global_tensor_dict: GlobalTensorDict) -> None:
        self.global_tensor_dict: GlobalTensorDict = global_tensor_dict

        self.global_tensor_dict["robot_mass"] = self.robot_masses
        self.global_tensor_dict["robot_inertia"] = self.robot_inertias

        self.global_tensor_dict["robot_actions"] = torch.zeros(
            (self.num_envs, self.robot.num_actions), device=self.device
        )

        self.global_tensor_dict["robot_prev_actions"] = torch.zeros_like(
            self.global_tensor_dict["robot_actions"]
        )

        self.actions = self.global_tensor_dict["robot_actions"]
        self.prev_actions = self.global_tensor_dict["robot_prev_actions"]

        self.global_tensor_dict["dof_control_mode"] = self.dof_control_mode

        self.robot.init_tensors(self.global_tensor_dict)

        self._init_sensors()

    def _compute_robot_inertia(self, env_handle: object, env_id: int) -> None:
        """Compute robot mass and inertia from rigid body properties."""
        if self.robot_inertia is None or self.robot_mass is None:
            rbp = self.gym.get_actor_rigid_body_properties(env_handle, self.actor_handle)
            state_list = self.gym.get_actor_rigid_body_states(
                env_handle, self.actor_handle, gymapi.STATE_ALL
            )
            robot_com, _ = compute_robot_com(state_list, rbp, self.device)
            self.robot_mass, self.robot_inertia = compute_composite_inertia(
                state_list, rbp, robot_com, self.device
            )
            logger.warning(
                f"\nRobot mass: {self.robot_mass},\nInertia: {self.robot_inertia},\nRobot COM: {robot_com}"
            )
            logger.warning(
                "Calculated robot mass and inertia for this robot. This code assumes that your robot is the same across environments."
            )
            logger.critical(
                "If your robot differs across environments you need to perform this computation for each different robot here."
            )
        else:
            logger.debug(
                "It's the same robot as before. Not calculating the inertia and mass again. Change this if your robot differs across envs."
            )

        self._configure_dof_drive_mode(env_handle)
        self.robot_masses[env_id] = self.robot_mass
        self.robot_inertias[env_id] = self.robot_inertia

    def _configure_dof_drive_mode(self, env_handle: object) -> None:
        """Set drive mode for the robot's degrees of freedom."""
        props = self.gym.get_actor_dof_properties(env_handle, self.actor_handle)
        try:
            if len(props["driveMode"]) > 0:
                if self.cfg.reconfiguration_config.dof_mode == "position":
                    props["driveMode"].fill(gymapi.DOF_MODE_POS)
                    for j_index in range(len(props["stiffness"])):
                        props["stiffness"][j_index] = self.cfg.reconfiguration_config.stiffness[
                            j_index
                        ]
                        props["damping"][j_index] = self.cfg.reconfiguration_config.damping[j_index]
                elif self.cfg.reconfiguration_config.dof_mode == "velocity":
                    props["driveMode"].fill(gymapi.DOF_MODE_VEL)
                    for j_index in range(len(props["damping"])):
                        props["damping"][j_index] = self.cfg.reconfiguration_config.damping[j_index]
                elif self.cfg.reconfiguration_config.dof_mode == "effort":
                    props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
                else:
                    props["driveMode"].fill(gymapi.DOF_MODE_NONE)
                self.dof_control_mode = self.cfg.reconfiguration_config.dof_mode
                self.gym.set_actor_dof_properties(env_handle, self.actor_handle, props)
        except (AttributeError, IndexError, KeyError, RuntimeError) as e:
            logger.error(
                "Something unexpected happened while setting parameters for the DOF modes of the robot. "
                "Please check if the correct reconfiguration_config params are set in the robot config file."
            )
            raise e

    def add_robot_to_env(
        self,
        simulation_env_class: object,
        env_handle: object,
        global_asset_counter: int,
        env_id: int,
        segmentation_counter: int,
        robot_idx_in_env: int | None = None,
    ) -> int:
        if robot_idx_in_env is None:
            robot_idx_in_env = self.robot_id

        # Create unique robot name
        robot_name = f"{self.robot_name_prefix}env_{env_id}"

        self.actor_handle, _ = simulation_env_class.add_asset_to_env(
            self.robot_asset_dict,
            env_handle,
            env_id,
            global_asset_counter,
            segmentation_counter,
        )
        self.robot_handles.append(self.actor_handle)
        # currently the robot is not having segmentation IDs. Can change if needed.
        if self.camera_sensor is not None:
            for i in range(self.camera_sensor.cfg.num_sensors):
                self.camera_sensor.add_sensor_to_env(env_id, env_handle, self.actor_handle)

        self._compute_robot_inertia(env_handle, env_id)
        # Store mapping of environment to robot index within that environment
        self.env_robot_mapping[env_id] = robot_idx_in_env

        return segmentation_counter + 1

    def reset(self) -> None:
        self.reset_idx(torch.arange(self.cfg.num_envs, device=self.device))

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        self.robot.reset_idx(env_ids)
        if self.warp_sensor is not None:
            self.warp_sensor.reset_idx(env_ids)
        if self.imu_sensor is not None:
            self.imu_sensor.reset_idx(env_ids)
        if self.camera_sensor is not None:
            self.camera_sensor.reset_idx(env_ids)

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        # FIXED: Action tracking now works correctly with tensor cloning in reward computation
        self.prev_actions[:] = self.actions[:]  # Save current actions as previous
        self.actions[:] = actions  # Set new actions
        self.robot.step(self.actions)

    def post_physics_step(self) -> None:
        # have this sensor here rather than at capture_sensors
        # as this will still update the sensor without the user forgetting to call render()
        if self.imu_sensor is not None:
            self.imu_sensor.update()

    def capture_sensors(self) -> None:
        if self.warp_sensor is not None:
            self.warp_sensor.update()
        if self.camera_sensor is not None:
            self.camera_sensor.update()

    def get_observations(self) -> dict[str, object]:
        """
        Get observations with robot ID information
        """
        observations = super().get_observations()  # Get base observations

        # Add robot ID to observations for identification
        observations["robot_id"] = self.robot_id
        observations["robot_position"] = self.get_robot_positions()
        observations["robot_velocity"] = self.get_robot_velocities()

        return observations

    def get_robot_positions(self) -> torch.Tensor | None:
        """Get current robot positions across all environments"""
        # Implementation depends on existing robot state access
        # This should return tensor of shape (num_envs, 3)
        pass

    def get_robot_velocities(self) -> torch.Tensor | None:
        """Get current robot velocities across all environments"""
        # Implementation depends on existing robot state access
        # This should return tensor of shape (num_envs, 3)
        pass

    def set_robot_positions(self, positions: torch.Tensor) -> None:
        """Set robot positions for reset"""
        # Implementation depends on existing robot control interface
        # positions: tensor of shape (num_envs, 3)
        pass
