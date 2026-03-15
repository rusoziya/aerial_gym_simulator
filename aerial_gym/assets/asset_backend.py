from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AssetBackend(Protocol):
    """Protocol defining the asset loading interface for physics backends.

    Each physics backend (Isaac Gym, Isaac Lab) must implement this protocol
    to provide asset loading, body introspection, and force sensor creation.
    """

    def load_asset(
        self,
        sim: object,
        asset_folder: str,
        asset_file: str,
        asset_options: object,
    ) -> object:
        """Load a URDF/USD asset from disk into the simulation.

        Args:
            sim: The simulation instance handle.
            asset_folder: Root directory containing the asset file.
            asset_file: Filename of the asset (e.g. "quadrotor.urdf").
            asset_options: Backend-specific options controlling how the asset
                is imported (fixed joints, density, damping, etc.).

        Returns:
            An opaque asset handle used by subsequent backend calls.
        """
        ...

    def find_body_index(self, asset: object, body_name: str) -> int:
        """Return the rigid body index for a named link within an asset.

        Args:
            asset: Asset handle returned by ``load_asset``.
            body_name: Name of the rigid body / link to look up.

        Returns:
            Zero-based index of the body inside the asset.
        """
        ...

    def create_force_sensor(
        self,
        asset: object,
        body_index: int,
        transform_pos: tuple[float, float, float],
        transform_quat: tuple[float, float, float, float],
        enable_forward_dynamics_forces: bool = True,
        enable_constraint_solver_forces: bool = True,
        use_world_frame: bool = False,
    ) -> object:
        """Attach a force sensor to a rigid body of the asset.

        Args:
            asset: Asset handle returned by ``load_asset``.
            body_index: Index of the body to attach the sensor to
                (from ``find_body_index``).
            transform_pos: (x, y, z) position offset of the sensor relative
                to the body frame.
            transform_quat: (x, y, z, w) quaternion orientation of the sensor
                relative to the body frame.
            enable_forward_dynamics_forces: Whether to report forward-dynamics forces.
            enable_constraint_solver_forces: Whether to report constraint-solver forces.
            use_world_frame: Whether forces are reported in the world frame.

        Returns:
            An opaque sensor handle.
        """
        ...

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
        """Create a backend-specific asset options object.

        Returns:
            An opaque options object to pass to ``load_asset``.
        """
        ...
