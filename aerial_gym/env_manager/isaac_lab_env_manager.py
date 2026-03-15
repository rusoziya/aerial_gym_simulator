"""Isaac Lab physics backend for Aerial Gym.

Implements the same interface as IsaacGymEnv using Isaac Lab APIs.
SimulationApp must be created before any omni/isaaclab imports, so this
module lazily imports Isaac Lab symbols inside methods that need them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from aerial_gym.env_manager.base_env_manager import BaseManager
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import torch_rand_float_tensor

if TYPE_CHECKING:
    from aerial_gym.env_manager.env_manager import EnvManager
    from aerial_gym.env_manager.tensor_state import TensorState

logger = CustomLogger("IsaacLabEnvManager")


@dataclass
class _PendingAsset:
    """Metadata for an asset registered during add_asset_to_env, awaiting scene build."""

    prim_path: str
    asset_info_dict: dict[str, object]
    is_robot: bool
    global_asset_counter: int


_ISAAC_LAB_SRC_PATH = (
    "/home/ziyar/miniforge3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab"
)


def _ensure_isaac_lab_on_path() -> None:
    """Add Isaac Lab source to sys.path if not already present."""
    if _ISAAC_LAB_SRC_PATH not in sys.path:
        sys.path.insert(0, _ISAAC_LAB_SRC_PATH)


class IsaacLabEnv(BaseManager):
    """Isaac Lab physics backend — drop-in replacement for IsaacGymEnv.

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
        self.has_IGE_cameras: bool = has_cameras
        self.env_handles: list[int] = []
        self.asset_handles: list[list[int]] = []
        self.num_rigid_bodies_robot: int = 1
        self.num_assets_per_env: int = 0
        self.num_envs: int = self.cfg.env.num_envs
        self.num_rigid_bodies_per_env: int = 0
        self.sim_has_dof: bool = False
        self.dof_control_mode: str = "none"
        self.global_tensor_dict: Optional[TensorState] = None

        # Isaac Lab articulations and rigid objects registered during add_asset_to_env
        self._robot_prim_paths: list[str] = []
        self._asset_prim_paths: list[str] = []
        self._robot_articulation: Optional[object] = None
        self._asset_rigid_objects: list[object] = []

        # Pending assets collected during add_asset_to_env for deferred scene build
        self._pending_assets: list[_PendingAsset] = []
        from aerial_gym.utils.urdf_to_usd import UrdfToUsdConverter

        self._urdf_converter = UrdfToUsdConverter()

        logger.info("Creating Isaac Lab Environment")

        _ensure_isaac_lab_on_path()
        self._create_simulation_app_and_context()

        # Environment bounds (same logic as Isaac Gym backend)
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

        self.viewer: Optional[object] = None
        self.graphics_are_stepped: bool = True

        logger.info("Isaac Lab Environment initialized")

    def _create_simulation_app_and_context(self) -> None:
        """Create SimulationApp and SimulationContext — must happen before other omni imports."""
        from isaacsim import SimulationApp

        headless = self.sim_config.viewer.headless
        self._app = SimulationApp({"headless": headless})

        from isaaclab.sim import SimulationCfg, SimulationContext

        dt = self.sim_config.sim.dt
        gravity = tuple(self.sim_config.sim.gravity)
        sim_cfg = SimulationCfg(
            dt=dt,
            device=self.device,
            gravity=gravity,
        )
        self.sim_context = SimulationContext(sim_cfg)
        logger.info(f"SimulationContext created (dt={dt}, device={self.device})")

    def create_ground_plane(self) -> None:
        """Add a ground plane to the USD stage via Isaac Lab."""
        from isaaclab.sim.spawners.shapes import GroundPlaneCfg

        ground_cfg = GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)
        logger.info("Ground plane added")

    def create_env(self, env_id: int) -> int:
        """Register an environment index.

        Isaac Lab uses scene cloning via num_envs, so individual env creation
        is a no-op. We just track handles for compatibility with EnvManager's
        per-env asset population loop.
        """
        if len(self.env_handles) <= env_id:
            self.env_handles.append(env_id)
            self.asset_handles.append([])
        return env_id

    def add_asset_to_env(
        self,
        asset_info_dict: dict[str, int | str | bool | list[float] | None],
        env_handle: int,
        env_id: int,
        global_asset_counter: int,
        segmentation_counter: int,
    ) -> Tuple[int, int]:
        """Register an asset for later scene construction.

        Isaac Lab spawns all prims at scene-build time via cloning, so we only
        record metadata here. The actual Articulation / RigidObject creation
        happens in prepare_for_simulation for env_id == 0 (the prototype env).
        """
        asset_handle = global_asset_counter
        self.asset_handles[env_id].append(asset_handle)

        is_robot = asset_info_dict["asset_type"] == "robot"

        # Only record prim paths and pending assets for the prototype environment (env 0)
        if env_id == 0:
            if is_robot:
                prim_path = f"/World/envs/env_.*/robot_{global_asset_counter}"
                self._robot_prim_paths.append(prim_path)
            else:
                prim_path = f"/World/envs/env_.*/asset_{global_asset_counter}"
                self._asset_prim_paths.append(prim_path)

            self._pending_assets.append(
                _PendingAsset(
                    prim_path=prim_path,
                    asset_info_dict=asset_info_dict,
                    is_robot=is_robot,
                    global_asset_counter=global_asset_counter,
                )
            )

        if is_robot:
            self.num_rigid_bodies_robot = 1

        seg_increment = 1
        return (asset_handle, seg_increment)

    def prepare_for_simulation(
        self,
        env_manager: EnvManager,
        global_tensor_dict: TensorState,
    ) -> bool:
        """Finalize the scene: spawn assets via cloning, reset physics, populate tensors."""
        self.num_envs = len(self.env_handles)
        self.num_assets_per_env = len(self.asset_handles[0]) if self.asset_handles else 0

        self.global_tensor_dict = global_tensor_dict

        self._build_scene()
        self.sim_context.reset()

        self._populate_state_tensors()
        self._populate_force_tensors()
        self._populate_dof_and_contact_tensors()
        self._populate_robot_tensor_slices()
        if self.num_assets_per_env > 1:
            self._populate_obstacle_tensor_slices()
        self._populate_env_metadata()

        self.create_viewer(env_manager)
        return True

    def _build_scene(self) -> None:
        """Spawn the robot articulation and obstacle rigid objects into the USD stage.

        Isaac Lab clones the prototype environment across num_envs automatically
        when using regex prim paths like ``/World/envs/env_.*/...``.
        """
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.sim import schemas as sim_utils

        for pending in self._pending_assets:
            usd_path = self._urdf_converter.resolve_usd_path(pending.asset_info_dict)
            if usd_path is None:
                logger.warning(
                    f"Could not resolve USD for asset at '{pending.prim_path}', skipping"
                )
                continue

            if pending.is_robot:
                self._spawn_robot(pending, usd_path, ArticulationCfg, Articulation, sim_utils)
            else:
                self._spawn_obstacle(pending, usd_path, RigidObjectCfg, RigidObject, sim_utils)

    def _spawn_robot(
        self,
        pending: _PendingAsset,
        usd_path: str,
        articulation_cfg_cls: type,
        articulation_cls: type,
        sim_utils: object,
    ) -> None:
        """Create and register an ArticulationCfg for the robot."""
        robot_cfg = articulation_cfg_cls(
            prim_path=pending.prim_path,
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
            init_state=articulation_cfg_cls.InitialStateCfg(
                pos=(0.0, 0.0, 1.0),
            ),
        )
        self._robot_articulation = articulation_cls(robot_cfg)
        logger.info(f"Robot articulation spawned at '{pending.prim_path}' from {usd_path}")

    def _spawn_obstacle(
        self,
        pending: _PendingAsset,
        usd_path: str,
        rigid_object_cfg_cls: type,
        rigid_object_cls: type,
        sim_utils: object,
    ) -> None:
        """Create and register a RigidObjectCfg for a static/dynamic obstacle."""
        obj_cfg = rigid_object_cfg_cls(
            prim_path=pending.prim_path,
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
            init_state=rigid_object_cfg_cls.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
            ),
        )
        rigid_obj = rigid_object_cls(obj_cfg)
        self._asset_rigid_objects.append(rigid_obj)
        logger.info(f"Rigid object spawned at '{pending.prim_path}' from {usd_path}")

    def _populate_state_tensors(self) -> None:
        """Fill global_tensor_dict with root state tensors from the Isaac Lab scene."""
        n = self.num_envs
        total_actors = self.num_assets_per_env  # robot + obstacles

        if self._robot_articulation is not None:
            root_state = self._robot_articulation.data.root_state_w  # (num_envs, 13)
        else:
            # Placeholder tensors until the scene is fully wired
            root_state = torch.zeros((n, 13), device=self.device)
            root_state[:, 6] = 1.0  # unit quaternion w component

        # Build the full vec_root_tensor: (num_envs, num_assets_per_env, 13)
        if total_actors > 0:
            all_states = [root_state.unsqueeze(1)]
            for obj in self._asset_rigid_objects:
                all_states.append(obj.data.root_state_w.unsqueeze(1))
            # Pad with placeholders for unregistered assets
            while len(all_states) < total_actors:
                placeholder = torch.zeros((n, 1, 13), device=self.device)
                placeholder[:, :, 6] = 1.0
                all_states.append(placeholder)
            vec_root = torch.cat(all_states, dim=1)
        else:
            vec_root = root_state.unsqueeze(1)

        # Flat view used by write_to_sim
        self._unfolded_root_tensor = vec_root.reshape(-1, 13).contiguous()

        self.global_tensor_dict.vec_root_tensor = vec_root
        self.global_tensor_dict.robot_state_tensor = vec_root[:, 0, :]
        self.global_tensor_dict.env_asset_state_tensor = vec_root[:, 1:, :]
        self.global_tensor_dict.unfolded_env_asset_state_tensor = self._unfolded_root_tensor
        self.global_tensor_dict.unfolded_env_asset_state_tensor_const = (
            self._unfolded_root_tensor.clone()
        )

    def _populate_force_tensors(self) -> None:
        """Create zero-initialized force/torque tensors matching Isaac Gym layout."""
        n = self.num_envs
        # Each env has num_rigid_bodies_robot + obstacle rigid bodies
        # For now approximate 1 rigid body per asset
        self.num_rigid_bodies_per_env = (
            self.num_assets_per_env if self.num_assets_per_env > 0 else 1
        )
        total_bodies = n * self.num_rigid_bodies_per_env

        # Rigid body state: Isaac Lab provides via articulation.data.body_pos_w etc.
        # We create a compatible (total_bodies, 13) tensor
        self.global_tensor_dict.rigid_body_state_tensor = torch.zeros(
            (total_bodies, 13), device=self.device
        )

        self.global_tensor_dict.global_force_tensor = torch.zeros(
            (total_bodies, 3), device=self.device
        )
        self.global_tensor_dict.global_torque_tensor = torch.zeros(
            (total_bodies, 3), device=self.device
        )

        idx = self.num_rigid_bodies_robot
        self.global_tensor_dict.robot_force_tensor = (
            self.global_tensor_dict.global_force_tensor.view(n, self.num_rigid_bodies_per_env, 3)[
                :, :idx, :
            ]
        )
        self.global_tensor_dict.robot_torque_tensor = (
            self.global_tensor_dict.global_torque_tensor.view(n, self.num_rigid_bodies_per_env, 3)[
                :, :idx, :
            ]
        )

    def _populate_dof_and_contact_tensors(self) -> None:
        """Set up DOF state and contact force tensors."""
        n = self.num_envs

        # DOF state — only relevant when the robot has joints
        if self._robot_articulation is not None:
            # TODO: query actual DOF count from articulation
            self.sim_has_dof = True
        # Placeholder empty DOF tensor
        self.global_tensor_dict.unfolded_dof_state_tensor = torch.zeros(
            (n, 0, 2), device=self.device
        )
        self.global_tensor_dict.dof_state_tensor = torch.zeros((n, 0, 2), device=self.device)

        # Contact forces: (num_envs, num_rigid_bodies_per_env, 3)
        self.global_tensor_dict.global_contact_force_tensor = torch.zeros(
            (n, self.num_rigid_bodies_per_env, 3), device=self.device
        )
        self.global_tensor_dict.robot_contact_force_tensor = (
            self.global_tensor_dict.global_contact_force_tensor[:, 0, :]
        )

    def _populate_robot_tensor_slices(self) -> None:
        """Slice the root state tensor into robot-specific views."""
        rs = self.global_tensor_dict.robot_state_tensor
        self.global_tensor_dict.robot_position = rs[:, :3]
        self.global_tensor_dict.robot_orientation = rs[:, 3:7]
        self.global_tensor_dict.robot_linvel = rs[:, 7:10]
        self.global_tensor_dict.robot_angvel = rs[:, 10:]
        self.global_tensor_dict.robot_body_angvel = torch.zeros_like(rs[:, 10:13])
        self.global_tensor_dict.robot_body_linvel = torch.zeros_like(rs[:, 7:10])
        self.global_tensor_dict.robot_euler_angles = torch.zeros_like(rs[:, 7:10])

    def _populate_obstacle_tensor_slices(self) -> None:
        """Slice the env asset state tensor into obstacle-specific views."""
        n = self.num_envs
        eas = self.global_tensor_dict.env_asset_state_tensor
        self.global_tensor_dict.obstacle_position = eas[:, :, 0:3]
        self.global_tensor_dict.obstacle_orientation = eas[:, :, 3:7]
        self.global_tensor_dict.obstacle_linvel = eas[:, :, 7:10]
        self.global_tensor_dict.obstacle_angvel = eas[:, :, 10:]
        self.global_tensor_dict.obstacle_body_angvel = torch.zeros_like(eas[:, :, 10:13])
        self.global_tensor_dict.obstacle_body_linvel = torch.zeros_like(eas[:, :, 7:10])
        self.global_tensor_dict.obstacle_euler_angles = torch.zeros_like(eas[:, :, 7:10])

        idx = self.num_rigid_bodies_robot
        self.global_tensor_dict.obstacle_force_tensor = (
            self.global_tensor_dict.global_force_tensor.view(n, self.num_rigid_bodies_per_env, 3)[
                :, idx:, :
            ]
        )
        self.global_tensor_dict.obstacle_torque_tensor = (
            self.global_tensor_dict.global_torque_tensor.view(n, self.num_rigid_bodies_per_env, 3)[
                :, idx:, :
            ]
        )

    def _populate_env_metadata(self) -> None:
        """Store environment bounds, gravity, and time step."""
        self.global_tensor_dict.env_bounds_max = self.env_upper_bound
        self.global_tensor_dict.env_bounds_min = self.env_lower_bound
        self.global_tensor_dict.gravity = torch.tensor(
            self.sim_config.sim.gravity, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.global_tensor_dict.dt = self.sim_config.sim.dt

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply forces/torques and DOF targets before stepping physics.

        Isaac Lab equivalent of gym.apply_rigid_body_force_tensors and
        gym.set_dof_*_target_tensor.
        """
        if self.cfg.env.write_to_sim_at_every_timestep:
            self.write_to_sim()

        if self._robot_articulation is not None:
            # Apply external forces and torques to the robot
            forces = self.global_tensor_dict.robot_force_tensor
            torques = self.global_tensor_dict.robot_torque_tensor
            self._robot_articulation.set_external_force_and_torque(forces, torques)

            # Apply DOF targets based on control mode
            if self.sim_has_dof:
                self.dof_control_mode = self.global_tensor_dict.dof_control_mode
                if self.dof_control_mode == "position":
                    self._robot_articulation.set_joint_position_target(
                        self.global_tensor_dict.dof_position_setpoint_tensor
                    )
                elif self.dof_control_mode == "velocity":
                    self._robot_articulation.set_joint_velocity_target(
                        self.global_tensor_dict.dof_velocity_setpoint_tensor
                    )
                elif self.dof_control_mode == "effort":
                    self._robot_articulation.set_joint_effort_target(
                        self.global_tensor_dict.dof_effort_tensor
                    )

    def physics_step(self) -> None:
        """Run one simulation step via Isaac Lab."""
        self.sim_context.step()
        self.graphics_are_stepped = False

    def post_physics_step(self) -> None:
        """Synchronize tensors after physics step.

        Isaac Lab updates articulation/rigid-object data automatically after
        step(). We refresh our tensor dict views from the live data.
        """
        self._sync_tensors_from_sim()

    def _sync_tensors_from_sim(self) -> None:
        """Pull latest state from Isaac Lab articulations into the tensor dict."""
        if self._robot_articulation is None:
            return

        robot = self._robot_articulation
        root_state = robot.data.root_state_w  # (num_envs, 13)

        # Update the robot slice of vec_root_tensor (it's a view, so downstream sees changes)
        self.global_tensor_dict.vec_root_tensor[:, 0, :] = root_state

        # Update rigid object states
        for i, obj in enumerate(self._asset_rigid_objects):
            self.global_tensor_dict.vec_root_tensor[:, i + 1, :] = obj.data.root_state_w

        # Sync the flat unfolded tensor
        self._unfolded_root_tensor[:] = self.global_tensor_dict.vec_root_tensor.reshape(-1, 13)

        # Contact forces — Isaac Lab provides net contact forces on articulation bodies
        # TODO: Wire up isaaclab.sensors.ContactSensorCfg for accurate contact data

    def refresh_tensors(self) -> None:
        """No-op — Isaac Lab auto-refreshes tensors after step()."""

    def step_graphics(self) -> None:
        """Render the scene (needed for camera sensor data)."""
        if not self.graphics_are_stepped:
            self.sim_context.render()
            self.graphics_are_stepped = True

    def render_viewer(self) -> None:
        """Update the Omniverse viewport. Isaac Lab handles this via render()."""
        if self.viewer is not None:
            if not self.graphics_are_stepped:
                self.step_graphics()

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset specified environments: randomize bounds and write robot state."""
        self.env_lower_bound[env_ids, :] = torch_rand_float_tensor(
            self.env_lower_bound_min, self.env_lower_bound_max
        )[env_ids]
        self.env_upper_bound[env_ids, :] = torch_rand_float_tensor(
            self.env_upper_bound_min, self.env_upper_bound_max
        )[env_ids]

        # Write reset state for the affected environments
        if self._robot_articulation is not None:
            root_state = self.global_tensor_dict.vec_root_tensor[:, 0, :]
            self._robot_articulation.write_root_state_to_sim(root_state[env_ids], env_ids)

            for i, obj in enumerate(self._asset_rigid_objects):
                asset_state = self.global_tensor_dict.vec_root_tensor[:, i + 1, :]
                obj.write_root_state_to_sim(asset_state[env_ids], env_ids)

    def write_to_sim(self) -> None:
        """Write all state tensors back to the Isaac Lab simulation."""
        if self._robot_articulation is not None:
            root_state = self.global_tensor_dict.vec_root_tensor[:, 0, :]
            self._robot_articulation.write_root_state_to_sim(root_state)

            if self.sim_has_dof and self.global_tensor_dict.dof_state_tensor is not None:
                dof = self.global_tensor_dict.dof_state_tensor
                if dof.shape[1] > 0:
                    self._robot_articulation.write_joint_state_to_sim(dof[:, :, 0], dof[:, :, 1])

        for i, obj in enumerate(self._asset_rigid_objects):
            asset_state = self.global_tensor_dict.vec_root_tensor[:, i + 1, :]
            obj.write_root_state_to_sim(asset_state)

    def create_viewer(self, env_manager: EnvManager) -> None:
        """Set up the Omniverse viewport camera if not headless."""
        if not self.sim_config.viewer.headless:
            logger.info("Isaac Lab viewport active — viewer camera managed by Omniverse Kit")
            # The Omniverse viewport is created automatically by SimulationApp.
            # Store a sentinel so render_viewer knows we're not headless.
            self.viewer = True
        else:
            logger.info("Headless mode — no viewer created")
            self.viewer = None
