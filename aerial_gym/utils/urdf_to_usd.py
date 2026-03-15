"""URDF-to-USD conversion utilities for Isaac Lab backend.

Provides cached conversion of URDF assets to USD format using Isaac Lab's
UrdfConverter, with filesystem-level caching keyed on URDF path and
conversion parameters.
"""

from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger("urdf_to_usd")

_USD_CACHE_DIR = os.path.join("/tmp", "aerial_gym_usd_cache")


class UrdfToUsdConverter:
    """Converts URDF files to USD with in-memory and filesystem caching."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def resolve_usd_path(self, asset_info_dict: dict[str, object]) -> str | None:
        """Get the USD path for an asset, converting from URDF if needed."""
        urdf_path = self._find_urdf_path(asset_info_dict)
        if urdf_path is None:
            return None

        if urdf_path.endswith(".usd") or urdf_path.endswith(".usda"):
            return urdf_path

        fix_base = _should_fix_base(asset_info_dict)
        merge_fixed = _should_merge_fixed_joints(asset_info_dict)
        return self.convert(urdf_path, fix_base, merge_fixed)

    def convert(self, urdf_path: str, fix_base: bool, merge_fixed_joints: bool) -> str:
        """Convert a URDF to USD, returning the cached or newly generated USD path."""
        cache_key = f"{urdf_path}|fix={fix_base}|merge={merge_fixed_joints}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        usd_path = self._try_filesystem_cache(cache_key, urdf_path)
        if usd_path is not None:
            self._cache[cache_key] = usd_path
            return usd_path

        usd_path = self._run_conversion(cache_key, urdf_path, fix_base, merge_fixed_joints)
        self._cache[cache_key] = usd_path
        return usd_path

    def _find_urdf_path(self, asset_info_dict: dict[str, object]) -> str | None:
        """Locate the URDF file from asset_info_dict fields."""
        asset_folder: str = asset_info_dict.get("asset_folder", "")
        filename: str = asset_info_dict.get("filename", "")

        if not filename:
            return None

        # filename from load_selected_file_from_config is already the full path
        if os.path.isabs(filename) and os.path.exists(filename):
            return filename
        if asset_folder and os.path.exists(os.path.join(asset_folder, filename)):
            return os.path.join(asset_folder, filename)

        if os.path.exists(filename):
            return filename

        logger.warning(f"URDF not found: {filename}")
        return None

    def _try_filesystem_cache(self, cache_key: str, urdf_path: str) -> str | None:
        """Check if a previously converted USD exists on disk and is still fresh."""
        usd_dir, usd_file_name = _cache_dir_and_name(cache_key, urdf_path)
        expected_path = os.path.join(usd_dir, usd_file_name)

        if not os.path.exists(expected_path):
            return None

        urdf_mtime = os.path.getmtime(urdf_path)
        usd_mtime = os.path.getmtime(expected_path)
        if usd_mtime >= urdf_mtime:
            logger.debug(f"Reusing cached USD: {expected_path}")
            return expected_path
        return None

    def _run_conversion(
        self, cache_key: str, urdf_path: str, fix_base: bool, merge_fixed_joints: bool
    ) -> str:
        """Execute the Isaac Lab URDF-to-USD conversion."""
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        usd_dir, usd_file_name = _cache_dir_and_name(cache_key, urdf_path)
        logger.info(f"Converting URDF to USD: {urdf_path} -> {usd_dir}")
        os.makedirs(usd_dir, exist_ok=True)

        urdf_cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=usd_dir,
            usd_file_name=usd_file_name,
            fix_base=fix_base,
            merge_fixed_joints=merge_fixed_joints,
        )
        converter = UrdfConverter(urdf_cfg)
        usd_path = converter.usd_path
        logger.info(f"USD generated at: {usd_path}")
        return usd_path


def _cache_dir_and_name(cache_key: str, urdf_path: str) -> tuple[str, str]:
    """Compute a stable output directory and filename for a URDF conversion."""
    hash_digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    urdf_stem = os.path.splitext(os.path.basename(urdf_path))[0]
    usd_dir = os.path.join(_USD_CACHE_DIR, f"{urdf_stem}_{hash_digest}")
    usd_file_name = f"{urdf_stem}.usd"
    return usd_dir, usd_file_name


def _should_fix_base(asset_info_dict: dict[str, object]) -> bool:
    """Determine whether the asset's base link should be fixed."""
    asset_options = asset_info_dict.get("asset_options")
    if asset_options is not None and hasattr(asset_options, "fix_base_link"):
        return bool(asset_options.fix_base_link)
    return asset_info_dict.get("asset_type") != "robot"


def _should_merge_fixed_joints(asset_info_dict: dict[str, object]) -> bool:
    """Determine whether to merge fixed joints during URDF conversion."""
    asset_options = asset_info_dict.get("asset_options")
    if asset_options is not None and hasattr(asset_options, "collapse_fixed_joints"):
        return bool(asset_options.collapse_fixed_joints)
    return True
