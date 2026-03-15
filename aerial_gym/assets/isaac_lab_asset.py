from __future__ import annotations

import os
from dataclasses import dataclass

from aerial_gym.assets.asset_backend import AssetBackend
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("isaac_lab_asset")


@dataclass
class IsaacLabAssetOptions:
    """Typed mirror of Isaac Gym's ``AssetOptions`` for the Isaac Lab backend."""

    collapse_fixed_joints: bool = True
    replace_cylinder_with_capsule: bool = False
    flip_visual_attachments: bool = False
    fix_base_link: bool = True
    density: float = 1.0
    angular_damping: float = 0.0
    linear_damping: float = 0.0
    max_angular_velocity: float = 100.0
    max_linear_velocity: float = 100.0
    disable_gravity: bool = False


class IsaacLabBackend:
    """Isaac Lab implementation of ``AssetBackend``.

    Uses ``UrdfToUsdConverter`` to convert URDF assets to USD, and stores body
    name mappings for index lookups. Force sensors are handled via Isaac Lab's
    ``ContactSensorCfg``.
    """

    def __init__(self) -> None:
        from aerial_gym.utils.urdf_to_usd import UrdfToUsdConverter

        self._converter = UrdfToUsdConverter()
        self._asset_body_names: dict[str, list[str]] = {}

    def load_asset(
        self,
        sim: object,
        asset_folder: str,
        asset_file: str,
        asset_options: object,
    ) -> str:
        """Convert URDF to USD and return the USD file path.

        Args:
            sim: The simulation instance handle (unused for conversion).
            asset_folder: Root directory containing the asset file.
            asset_file: Filename of the asset (e.g. "quadrotor.urdf").
            asset_options: An ``IsaacLabAssetOptions`` instance.

        Returns:
            The filesystem path to the converted USD file.
        """
        urdf_path = os.path.join(asset_folder, asset_file)

        if asset_file.endswith(".usd") or asset_file.endswith(".usda"):
            return urdf_path

        fix_base = True
        merge_fixed = True
        if isinstance(asset_options, IsaacLabAssetOptions):
            fix_base = asset_options.fix_base_link
            merge_fixed = asset_options.collapse_fixed_joints

        usd_path = self._converter.convert(
            urdf_path, fix_base=fix_base, merge_fixed_joints=merge_fixed
        )

        body_names = self._extract_body_names(urdf_path)
        self._asset_body_names[usd_path] = body_names

        logger.info(f"Loaded asset: {urdf_path} -> {usd_path}")
        return usd_path

    def find_body_index(self, asset: object, body_name: str) -> int:
        """Return the rigid body index for a named link within an asset.

        Args:
            asset: USD path string returned by ``load_asset``.
            body_name: Name of the rigid body / link to look up.

        Returns:
            Zero-based index of the body inside the asset.
        """
        usd_path = str(asset)
        body_names = self._asset_body_names.get(usd_path, [])
        if body_name in body_names:
            return body_names.index(body_name)

        logger.warning(
            f"Body '{body_name}' not found in asset '{usd_path}'. "
            f"Known bodies: {body_names}. Returning 0."
        )
        return 0

    def create_force_sensor(
        self,
        asset: object,
        body_index: int,
        transform_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        transform_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        enable_forward_dynamics_forces: bool = True,
        enable_constraint_solver_forces: bool = True,
        use_world_frame: bool = False,
    ) -> dict[str, object]:
        """Store force sensor config for later instantiation in the Isaac Lab scene.

        Isaac Lab uses ``ContactSensorCfg`` which must be added to the scene
        during setup. This method returns a config dict that the scene builder
        can consume.

        Args:
            asset: USD path string returned by ``load_asset``.
            body_index: Index of the body to attach the sensor to.
            transform_pos: (x, y, z) position offset relative to the body.
            transform_quat: (x, y, z, w) quaternion orientation offset.
            enable_forward_dynamics_forces: Report forward-dynamics forces.
            enable_constraint_solver_forces: Report constraint-solver forces.
            use_world_frame: Report forces in world frame.

        Returns:
            A config dict describing the sensor for scene registration.
        """
        usd_path = str(asset)
        body_names = self._asset_body_names.get(usd_path, [])
        body_name = body_names[body_index] if body_index < len(body_names) else "base_link"

        sensor_config: dict[str, object] = {
            "body_name": body_name,
            "body_index": body_index,
            "transform_pos": transform_pos,
            "transform_quat": transform_quat,
            "enable_forward_dynamics_forces": enable_forward_dynamics_forces,
            "enable_constraint_solver_forces": enable_constraint_solver_forces,
            "use_world_frame": use_world_frame,
        }
        logger.info(f"Created force sensor config for body '{body_name}' (index {body_index})")
        return sensor_config

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
    ) -> IsaacLabAssetOptions:
        """Create an ``IsaacLabAssetOptions`` dataclass with the given parameters.

        Returns:
            A typed options object to pass to ``load_asset``.
        """
        return IsaacLabAssetOptions(
            collapse_fixed_joints=collapse_fixed_joints,
            replace_cylinder_with_capsule=replace_cylinder_with_capsule,
            flip_visual_attachments=flip_visual_attachments,
            fix_base_link=fix_base_link,
            density=density,
            angular_damping=angular_damping,
            linear_damping=linear_damping,
            max_angular_velocity=max_angular_velocity,
            max_linear_velocity=max_linear_velocity,
            disable_gravity=disable_gravity,
        )

    def _extract_body_names(self, urdf_path: str) -> list[str]:
        """Parse link names from a URDF file for body index lookups."""
        body_names: list[str] = []
        if not os.path.exists(urdf_path):
            return body_names

        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(urdf_path)
            root = tree.getroot()
            for link in root.findall("link"):
                name = link.get("name")
                if name is not None:
                    body_names.append(name)
        except Exception as exc:
            logger.warning(f"Failed to parse URDF body names from {urdf_path}: {exc}")

        return body_names


_: AssetBackend = IsaacLabBackend()  # type: ignore[assignment]
del _
