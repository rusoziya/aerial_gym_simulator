from __future__ import annotations

import os
import torch

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("navigation_task_gate_geometry")


class NavigationTaskGateGeometryMixin:
    """Gate dimension calculation and URDF parsing methods."""

    def extract_gate_dimensions_from_urdf(self, urdf_path: str) -> tuple[float, float]:
        """
        Extract gate dimensions from URDF file.
        Returns (width, height, center_height, scale_factor)
        """
        import xml.etree.ElementTree as ET
        
        try:
            if not os.path.exists(urdf_path):
                logger.warning(f"[GATE_ADAPTIVE] URDF file not found: {urdf_path}, using default dimensions")
                return 2.5, 2.4, 1.2, 1.0  # Default 100% scale gate
            
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            
            # Extract scale factor from filename
            filename = os.path.basename(urdf_path)
            scale_factor = 1.0
            if "gate_scale_" in filename:
                try:
                    scale_str = filename.replace("gate_scale_", "").replace(".urdf", "")
                    scale_factor = int(scale_str) / 100.0
                except:
                    scale_factor = 1.0
            
            # Find left and right post positions to calculate width
            width = 2.5 * scale_factor  # Default scaled width
            height = 2.4 * scale_factor  # Default scaled height
            center_height = 1.2 * scale_factor  # Default scaled center height
            
            # Parse joint positions for more accurate dimensions
            for joint in root.iter('joint'):
                if joint.get('name') == 'base_to_left_post':
                    origin = joint.find('origin')
                    if origin is not None:
                        xyz = origin.get('xyz', '0 0 0').split()
                        left_y = abs(float(xyz[1]))
                        width = left_y * 2  # Total width = 2 * distance from center
                
                elif joint.get('name') == 'base_to_top_bar':
                    origin = joint.find('origin')
                    if origin is not None:
                        xyz = origin.get('xyz', '0 0 0').split()
                        top_z = float(xyz[2])
                        height = top_z  # Height to top bar
                        center_height = top_z / 2  # Center height
            
            return width, height, center_height, scale_factor
            
        except (ValueError, TypeError) as e:
            logger.warning(f"[GATE_ADAPTIVE] Error parsing URDF {urdf_path}: {e}, using default dimensions")
            return 2.5, 2.4, 1.2, 1.0

    def calculate_gate_dimensions_from_name(self, gate_name: str) -> tuple[float, float, float]:
        """
        Calculate gate dimensions from the gate name (e.g., gate_scale_060 -> 60% scale).
        Returns (width, height, center_height, scale_factor)
        """
        try:
            # Extract scale factor from gate name
            if "gate_scale_" in gate_name:
                scale_str = gate_name.replace("gate_scale_", "")
                scale_factor = int(scale_str) / 100.0
            else:
                scale_factor = 1.0
            
            # Base dimensions for 100% gate
            base_width = 2.5
            base_height = 2.4
            base_center_height = 1.2
            
            # Calculate scaled dimensions
            width = base_width * scale_factor
            height = base_height * scale_factor
            center_height = base_center_height * scale_factor
            
            logger.warning(f"[GATE_ADAPTIVE] Calculated dimensions from name '{gate_name}': width={width:.3f}m, height={height:.3f}m, center_height={center_height:.3f}m, scale={scale_factor:.2f}")
            return width, height, center_height, scale_factor
            
        except (ValueError, TypeError) as e:
            logger.warning(f"[GATE_ADAPTIVE] Error calculating dimensions from name '{gate_name}': {e}, using default")
            return 2.5, 2.4, 1.2, 1.0

    def update_gate_dimensions_for_environments(self, env_ids: torch.Tensor) -> None:
        """
        Update gate dimensions for specified environments based on their selected gate variants.
        """
        if not hasattr(self.sim_env, 'global_tensor_dict'):
            return
            
        # Safety check: ensure gate dimension attributes exist
        if not True or not True:
            logger.warning("[GATE_ADAPTIVE] Gate dimension attributes not initialized yet, skipping update")
            return
            
        gate_variant_names = self.sim_env.global_tensor_dict.get("gate_variant_names_per_env", [])
        active_gate_array_indices = self.sim_env.global_tensor_dict.get("active_gate_variant_array_index", torch.zeros(self.sim_env.num_envs))
        
        for env_id in (env_ids.tolist() if hasattr(env_ids, 'tolist') else [env_ids]):
            if env_id >= len(gate_variant_names):
                continue
                
            env_gate_names = gate_variant_names[env_id]
            active_idx = active_gate_array_indices[env_id].item()
            
            if active_idx >= 0 and active_idx < len(env_gate_names):
                # Get the active gate variant name
                active_gate_name = env_gate_names[active_idx] if env_gate_names else "gate_scale_100"
                
                # Construct URDF path - find the correct base directory
                urdf_filename = f"{active_gate_name}.urdf"
                
                # Try multiple possible base directories to find the URDF files
                possible_base_dirs = [
                    os.getcwd(),  # Current working directory
                    os.path.dirname(os.path.abspath(__file__)),  # Directory of this file
                    "/home/ziyar/aerialgym/aerialgym_ws/src/aerial_gym_simulator",  # Known project root
                ]
                
                # Add parent directories up to 5 levels
                current_dir = os.getcwd()
                for _ in range(5):
                    current_dir = os.path.dirname(current_dir)
                    possible_base_dirs.append(current_dir)
                
                urdf_path = None
                for base_dir in possible_base_dirs:
                    for sub_dir in (
                        "resources/models/environment_assets/gates",
                        "resources/models/environment_assets/smaller gates",
                    ):
                        test_path = os.path.join(base_dir, sub_dir, urdf_filename)
                        if os.path.exists(test_path):
                            urdf_path = test_path
                            break
                    if urdf_path is not None:
                        break
                
                if urdf_path is None:
                    # Fallback: construct path anyway for the error message
                    urdf_path = os.path.join(possible_base_dirs[0], "resources/models/environment_assets/gates", urdf_filename)
                
                # Extract dimensions from URDF or calculate from filename
                if urdf_path and os.path.exists(urdf_path):
                    width, height, center_height, scale_factor = self.extract_gate_dimensions_from_urdf(urdf_path)
                else:
                    # Fallback: calculate dimensions from scale factor in filename
                    width, height, center_height, scale_factor = self.calculate_gate_dimensions_from_name(active_gate_name)
                
                # Update environment-specific dimensions
                self.gate_width[env_id] = width
                self.gate_height[env_id] = height
                self.gate_center_height[env_id] = center_height
                self.gate_scale_factors[env_id] = scale_factor
                
            else:
                # Default dimensions if no active gate found
                self.gate_width[env_id] = 2.5
                self.gate_height[env_id] = 2.4
                self.gate_center_height[env_id] = 1.2
                self.gate_scale_factors[env_id] = 1.0
                logger.warning(f"[GATE_ADAPTIVE] Env {env_id}: No active gate found (active_idx={active_idx}, num_gates={len(env_gate_names)}), using default gate dimensions")
        # Expose adaptive gate center heights per env to global tensor dict for camera spawning
        if hasattr(self.sim_env, 'global_tensor_dict'):
            self.sim_env.global_tensor_dict['gate/center_height_per_env'] = self.gate_center_height.detach().clone()

