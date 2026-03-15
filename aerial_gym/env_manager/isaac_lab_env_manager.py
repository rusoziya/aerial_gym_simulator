"""Isaac Lab physics backend — stub implementation.

This module provides the Isaac Lab equivalent of IGE_env_manager.py.
It implements the PhysicsBackend Protocol using NVIDIA Isaac Lab APIs
instead of Isaac Gym Preview 4.

Requirements:
- Python >= 3.10
- Isaac Lab (omni.isaac.lab)
- Isaac Sim (omni.isaac.core)

Status: STUB — method signatures and docstrings are complete,
implementations are TODO placeholders. Fill in each method to
complete the migration.

Migration guide: each method has a comment showing the Isaac Gym
equivalent and what the Isaac Lab replacement should do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from aerial_gym.env_manager.base_env_manager import BaseManager
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import torch_rand_float_tensor

if TYPE_CHECKING:
    from aerial_gym.env_manager.tensor_state import TensorState

logger = CustomLogger("IsaacLabEnvManager")


class IsaacLabEnv(BaseManager):
    """Isaac Lab physics backend — implements PhysicsBackend Protocol.

    Drop-in replacement for IsaacGymEnv. EnvManager can use either backend
    via the PhysicsBackend Protocol without code changes.

    Key differences from Isaac Gym:
    - Single scene with instanced clones (not per-env create_env loop)
    - Tensors are native PyTorch (no wrap_tensor/unwrap_tensor)
    - Automatic tensor sync on step (no manual refresh_* calls)
    - Camera sensors via CameraCfg (not gym.create_camera_sensor)
    """

    def __init__(
        self,
        config: type,
        sim_config: type,
        has_cameras: bool,
        device: str,
    ) -> None:
        super().__init__(config, device)
        self.sim_config = sim_config
        self.has_IGE_cameras = has_cameras
        self.env_handles: list = []
        self.asset_handles: list = []
        self.num_rigid_bodies_robot: Optional[int] = None
        self.num_assets_per_env: int = 0
        self.sim_has_dof: bool = False

        logger.info("Creating Isaac Lab Environment")

        # TODO: Replace Isaac Gym calls:
        #   gym = gymapi.acquire_gym()
        #   sim = gym.create_sim(device_id, graphics_id, physics_engine, sim_params)
        #
        # Isaac Lab equivalent:
        #   from omni.isaac.lab.sim import SimulationCfg, SimulationContext
        #   sim_cfg = SimulationCfg(dt=config.sim.dt, ...)
        #   self.sim_context = SimulationContext(sim_cfg)
        #   self.sim_context.set_camera_view(eye, target)
        self.sim_context = None  # TODO: SimulationContext(...)

        # Environment bounds (same logic as Isaac Gym)
        self.env_lower_bound_min = torch.tensor(
            self.cfg.env.lower_bound_min, device=self.device, requires_grad=False
        ).expand(self.cfg.env.num_envs, -1)
        self.env_lower_bound_max = torch.tensor(
            self.cfg.env.lower_bound_max, device=self.device, requires_grad=False
        ).expand(self.cfg.env.num_envs, -1)
        self.env_upper_bound_min = torch.tensor(
            self.cfg.env.upper_bound_min, device=self.device, requires_grad=False
        ).expand(self.cfg.env.num_envs, -1)
        self.env_upper_bound_max = torch.tensor(
            self.cfg.env.upper_bound_max, device=self.device, requires_grad=False
        ).expand(self.cfg.env.num_envs, -1)

        self.env_lower_bound = torch_rand_float_tensor(
            self.env_lower_bound_min, self.env_lower_bound_max
        )
        self.env_upper_bound = torch_rand_float_tensor(
            self.env_upper_bound_min, self.env_upper_bound_max
        )

        self.viewer = None
        self.graphics_are_stepped = True

        logger.info("Isaac Lab Environment initialized (stub)")

    def create_ground_plane(self) -> None:
        """Add ground plane to the scene.

        Isaac Gym equivalent: gym.add_ground(sim, plane_params)
        Isaac Lab: Add a GroundPlaneCfg to the scene.

        TODO:
            from omni.isaac.lab.terrains import TerrainImporterCfg
            ground_cfg = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
            self.scene.add(ground_cfg)
        """
        logger.info("TODO: Add ground plane via Isaac Lab terrain API")

    def create_env(self, env_id: int) -> object:
        """Create environment instance.

        Isaac Gym equivalent: gym.create_env(sim, lower_bound, upper_bound, num_per_row)

        Isaac Lab: Environments are created via scene cloning, not explicit loops.
        The first call creates the prototype, subsequent calls are no-ops since
        Isaac Lab handles cloning internally via num_envs.

        TODO:
            if env_id == 0:
                from omni.isaac.lab.scene import InteractiveSceneCfg
                self.scene = InteractiveScene(InteractiveSceneCfg(num_envs=self.cfg.env.num_envs))
            return env_id  # Isaac Lab uses integer indices, not handles
        """
        if len(self.env_handles) <= env_id:
            self.env_handles.append(env_id)
            self.asset_handles.append([])
        return env_id

    def add_asset_to_env(
        self,
        asset_info_dict: dict,
        env_handle: object,
        env_id: int,
        global_asset_counter: int,
        segmentation_counter: int,
    ) -> Tuple[object, int]:
        """Add rigid body or articulation to environment.

        Isaac Gym equivalent: gym.create_actor(env, asset, transform, name, env_id, collision, seg)

        Isaac Lab: Register articulation or rigid body in scene.

        TODO:
            from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
            if asset_info_dict["asset_type"] == "robot":
                cfg = ArticulationCfg(prim_path=f"/World/envs/env_{env_id}/robot", ...)
            else:
                cfg = RigidObjectCfg(prim_path=f"/World/envs/env_{env_id}/asset_{global_asset_counter}", ...)
            handle = self.scene.add(cfg)
        """
        asset_handle = global_asset_counter
        self.asset_handles[env_id].append(asset_handle)

        if asset_info_dict["asset_type"] == "robot":
            self.num_rigid_bodies_robot = 1  # TODO: query from scene

        seg_increment = 1
        return (asset_handle, seg_increment)

    def prepare_for_simulation(
        self,
        env_manager: object,
        global_tensor_dict: TensorState,
    ) -> bool:
        """Finalize simulation: acquire state tensors, set up viewer.

        Isaac Gym equivalent:
            gym.prepare_sim(sim)
            root_tensor = gym.acquire_actor_root_state_tensor(sim)
            contact_tensor = gym.acquire_net_contact_force_tensor(sim)
            dof_tensor = gym.acquire_dof_state_tensor(sim)

        Isaac Lab: Tensors are available after scene.reset().
            self.sim_context.reset()
            root_state = self.scene["robot"].data.root_state_w  # (num_envs, 13)
            contact = self.scene.sensors["contact"].data.net_forces_w  # (num_envs, num_bodies, 3)

        TODO: Populate global_tensor_dict with Isaac Lab tensors:
            global_tensor_dict.robot_state_tensor = root_state
            global_tensor_dict.robot_position = root_state[:, :3]
            global_tensor_dict.robot_orientation = root_state[:, 3:7]
            # etc.
        """
        logger.info("TODO: Acquire Isaac Lab state tensors and populate TensorState")

        self.num_envs = len(self.env_handles)
        self.num_assets_per_env = len(self.asset_handles[0]) if self.asset_handles else 0

        self.global_tensor_dict = global_tensor_dict

        # TODO: Create viewer if not headless
        self.create_viewer(env_manager)

        return True

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply forces/torques before stepping.

        Isaac Gym equivalent:
            gym.apply_rigid_body_force_tensors(sim, force_tensor, torque_tensor, LOCAL_SPACE)
            gym.set_dof_position_target_tensor(sim, target_tensor)

        Isaac Lab:
            self.scene["robot"].set_joint_position_target(targets)
            # OR for direct force application:
            self.scene["robot"].set_external_force_and_torque(forces, torques)

        TODO: Map DOF control modes and force application.
        """

    def physics_step(self) -> None:
        """Run one simulation step.

        Isaac Gym: gym.simulate(sim)
        Isaac Lab: self.sim_context.step()

        Note: Isaac Lab automatically handles fetch_results internally.
        """
        if self.sim_context is not None:
            self.sim_context.step()
        self.graphics_are_stepped = False

    def post_physics_step(self) -> None:
        """Synchronize tensors after physics step.

        Isaac Gym: gym.fetch_results(sim, True) + gym.refresh_*_tensor() calls
        Isaac Lab: Automatic — tensors are updated after sim_context.step().
                   May need sim_context.render() for camera data.

        TODO: If explicit refresh needed:
            self.sim_context.render()  # for camera sensors
        """
        self.refresh_tensors()

    def refresh_tensors(self) -> None:
        """Refresh all state tensors from simulation.

        Isaac Gym: 5 separate gym.refresh_*() calls
        Isaac Lab: Automatic after step(). This is a no-op.

        Note: Camera data may require explicit sim_context.render().
        """

    def step_graphics(self) -> None:
        """Update graphics for camera rendering.

        Isaac Gym: gym.step_graphics(sim)
        Isaac Lab: self.sim_context.render()
        """
        if not self.graphics_are_stepped:
            if self.sim_context is not None:
                self.sim_context.render()
            self.graphics_are_stepped = True

    def render_viewer(self) -> None:
        """Render viewer window.

        Isaac Gym: gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)
        Isaac Lab: Handled by Omniverse viewport automatically.
        """
        if self.viewer is not None:
            if not self.graphics_are_stepped:
                self.step_graphics()

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset specified environments.

        Isaac Gym: Randomize env bounds
        Isaac Lab: Same logic — randomize bounds, then write_to_sim.

        TODO: Also reset articulation/rigid body states via:
            self.scene["robot"].write_root_state_to_sim(root_state, env_ids)
        """
        self.env_lower_bound[env_ids, :] = torch_rand_float_tensor(
            self.env_lower_bound_min, self.env_lower_bound_max
        )[env_ids]
        self.env_upper_bound[env_ids, :] = torch_rand_float_tensor(
            self.env_upper_bound_min, self.env_upper_bound_max
        )[env_ids]

    def write_to_sim(self) -> None:
        """Write state tensors back to simulation.

        Isaac Gym:
            gym.set_actor_root_state_tensor(sim, unwrap(root_tensor))
            gym.set_dof_state_tensor(sim, unwrap(dof_tensor))

        Isaac Lab:
            self.scene["robot"].write_root_state_to_sim(root_state)
            self.scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)

        TODO: Implement state write-back.
        """

    def create_viewer(self, env_manager: object) -> None:
        """Create visualization window.

        Isaac Gym: gym.create_viewer(sim, CameraProperties())
        Isaac Lab: Omniverse viewport is created automatically if not headless.

        TODO: Set up Isaac Lab viewport camera if needed.
        """
        if not self.sim_config.viewer.headless:
            logger.info("TODO: Create Isaac Lab viewer/viewport")
        else:
            logger.info("Headless mode — no viewer created")
