from __future__ import annotations

from aerial_gym.assets.asset_backend import AssetBackend
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("isaac_lab_asset")


class IsaacLabBackend:
    """Isaac Lab implementation of ``AssetBackend``.

    Isaac Lab replaces Isaac Gym's ``gymapi`` with Omniverse / USD-based APIs.
    This stub maps each ``AssetBackend`` method to its Isaac Lab equivalent.

    Isaac Lab import path (when available):
        ``import omni.isaac.lab.sim as sim_utils``
        ``from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg``
        ``from omni.isaac.lab.sim.spawners import UsdFileCfg, UrdfFileCfg``
    """

    def load_asset(
        self,
        sim: object,
        asset_folder: str,
        asset_file: str,
        asset_options: object,
    ) -> object:
        """Load an asset via Isaac Lab's USD/URDF spawner pipeline.

        Isaac Gym equivalent:
            ``gym.load_asset(sim, asset_folder, asset_file, asset_options)``

        Isaac Lab equivalent:
            For URDF files::

                from omni.isaac.lab.sim.spawners import UrdfFileCfg
                spawner = UrdfFileCfg(
                    asset_path=f"{asset_folder}/{asset_file}",
                    fix_base=asset_options.fix_base_link,
                    merge_fixed_joints=asset_options.collapse_fixed_joints,
                )
                prim = spawner.func(prim_path, spawner)

            For USD files::

                from omni.isaac.lab.sim.spawners import UsdFileCfg
                spawner = UsdFileCfg(usd_path=f"{asset_folder}/{asset_file}")
                prim = spawner.func(prim_path, spawner)

            Alternatively, using the high-level ArticulationCfg::

                from omni.isaac.lab.assets import ArticulationCfg
                cfg = ArticulationCfg(
                    spawn=UsdFileCfg(usd_path=...),
                    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 1)),
                )
                articulation = Articulation(cfg)

        TODO: Implement URDF-to-USD conversion if assets are only available as URDF.
            Isaac Lab provides ``omni.isaac.lab.sim.converters.UrdfConverter`` for this.
        TODO: Map ``asset_options`` fields (density, damping, gravity) to
            ``sim_utils.RigidBodyPropertiesCfg`` and ``sim_utils.MassPropertiesCfg``.
        TODO: Handle ``replace_cylinder_with_capsule`` via USD schema overrides
            or pre-process the URDF before conversion.
        """
        raise NotImplementedError(
            "IsaacLabBackend.load_asset is not yet implemented. "
            "Requires omni.isaac.lab.sim.spawners (UrdfFileCfg / UsdFileCfg)."
        )

    def find_body_index(self, asset: object, body_name: str) -> int:
        """Find the rigid body index for a named link.

        Isaac Gym equivalent:
            ``gym.find_asset_rigid_body_index(asset, body_name)``

        Isaac Lab equivalent:
            After spawning an ``Articulation``::

                articulation = Articulation(cfg)
                body_index = articulation.find_bodies(body_name)[0]

            Or via the USD prim introspection::

                from pxr import UsdPhysics
                # traverse prim children to locate the rigid body by name

        TODO: Implement using ``Articulation.find_bodies()`` once the asset
            is instantiated as an Isaac Lab Articulation.
        TODO: For non-articulated rigid objects, use
            ``RigidObject.find_bodies(body_name)`` instead.
        """
        raise NotImplementedError(
            "IsaacLabBackend.find_body_index is not yet implemented. "
            "Requires Articulation.find_bodies() from omni.isaac.lab.assets."
        )

    def create_force_sensor(
        self,
        asset: object,
        body_index: int,
        transform_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        transform_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        enable_forward_dynamics_forces: bool = True,
        enable_constraint_solver_forces: bool = True,
        use_world_frame: bool = False,
    ) -> object:
        """Attach a force/torque sensor to a rigid body.

        Isaac Gym equivalent::

            sensor_props = gymapi.ForceSensorProperties()
            sensor_props.enable_forward_dynamics_forces = True
            sensor_props.enable_constraint_solver_forces = True
            sensor_props.use_world_frame = False
            transform = gymapi.Transform(gymapi.Vec3(*pos), gymapi.Quat(*quat))
            gym.create_asset_force_sensor(asset, body_index, transform, sensor_props)

        Isaac Lab equivalent:
            Isaac Lab uses ``ContactSensorCfg`` for contact/force sensing::

                from omni.isaac.lab.sensors import ContactSensorCfg
                sensor_cfg = ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/body_name",
                    update_period=0.0,
                    history_length=1,
                    track_air_time=False,
                    filter_prim_paths_expr=[],
                )
                contact_sensor = ContactSensor(sensor_cfg)

            For full 6-axis force/torque, use the articulation's built-in
            joint force reporting::

                articulation.root_physx_view.get_force_sensor_forces()

            Or attach a ``FrameTransformerCfg`` for transform tracking.

        TODO: Implement using ``ContactSensorCfg`` from
            ``omni.isaac.lab.sensors``.
        TODO: Map ``enable_forward_dynamics_forces`` and
            ``enable_constraint_solver_forces`` to the appropriate PhysX
            contact report flags.
        TODO: Handle the sensor transform offset (``transform_pos``,
            ``transform_quat``) via the ``offset`` parameter of the sensor cfg.
        """
        raise NotImplementedError(
            "IsaacLabBackend.create_force_sensor is not yet implemented. "
            "Requires ContactSensorCfg from omni.isaac.lab.sensors."
        )

    def create_asset_options(
        self,
        collapse_fixed_joints: bool = True,
        replace_cylinder_with_capsule: bool = False,
        flip_visual_attachments: bool = False,
        fix_base_link: bool = True,
        density: float = 1.0,
        angular_damping: float = 0.0,
        linear_damping: float = 0.0,
        max_angular_velocity: float = 100.0,
        max_linear_velocity: float = 100.0,
        disable_gravity: bool = False,
    ) -> object:
        """Create asset import options for Isaac Lab.

        Isaac Gym equivalent:
            ``gymapi.AssetOptions()`` with fields set directly.

        Isaac Lab equivalent:
            Options are split across multiple config objects::

                from omni.isaac.lab import sim as sim_utils

                rigid_props = sim_utils.RigidBodyPropertiesCfg(
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                    linear_damping=linear_damping,
                    angular_damping=angular_damping,
                    disable_gravity=disable_gravity,
                )

                mass_props = sim_utils.MassPropertiesCfg(
                    density=density,
                )

                # For URDF import, merge_fixed_joints replaces collapse_fixed_joints:
                from omni.isaac.lab.sim.spawners import UrdfFileCfg
                spawner = UrdfFileCfg(
                    asset_path=...,
                    fix_base=fix_base_link,
                    merge_fixed_joints=collapse_fixed_joints,
                    rigid_props=rigid_props,
                    mass_props=mass_props,
                )

        TODO: Implement and return a structured config dataclass that
            downstream ``load_asset`` can consume.
        TODO: Map ``flip_visual_attachments`` — Isaac Lab handles visual
            meshes differently via USD; this flag may not be needed.
        TODO: Map ``replace_cylinder_with_capsule`` — may require a
            pre-processing step on the URDF or a collision schema override.
        """
        raise NotImplementedError(
            "IsaacLabBackend.create_asset_options is not yet implemented. "
            "Requires sim_utils.RigidBodyPropertiesCfg and MassPropertiesCfg."
        )


# Verify that IsaacLabBackend satisfies the AssetBackend protocol at import time
_: AssetBackend = IsaacLabBackend()  # type: ignore[assignment]
del _
