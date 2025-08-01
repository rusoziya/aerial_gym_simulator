from aerial_gym.config.asset_config.gate_asset_config import GateAssetConfig
from aerial_gym.config.asset_config.base_asset import BaseAssetParams
from aerial_gym import AERIAL_GYM_DIRECTORY
import numpy as np

# Semantic ID for gate objects (same as original)
GATE_SEMANTIC_ID = 10


class GateScalingConfig:
    """
    Gate scaling configuration for curriculum learning.
    
    Creates multiple gate instances with different scale factors:
    - Full size (scale 1.0): 2.5m x 2.3m opening
    - Medium size (scale 0.7): 1.75m x 1.61m opening  
    - Small size (scale 0.5): 1.25m x 1.15m opening
    - Minimum size (scale 0.4): 1.0m x 0.92m opening
    """
    
    # Gate scaling levels for curriculum progression
    GATE_SCALES = [1.0, 0.7, 0.5, 0.4]  # From largest to smallest
    NUM_GATE_INSTANCES = len(GATE_SCALES)
    
    # Minimum gate dimensions (1 meter opening)
    MIN_GATE_SCALE = 0.4
    MAX_GATE_SCALE = 1.0
    
    class FullSizeGateParams(GateAssetConfig.gate_asset_params):
        """Full size gate (scale 1.0) - Original gate dimensions"""
        file = "gate.urdf"  # Use original gate file
    
    class MediumGateParams(GateAssetConfig.gate_asset_params):
        """Medium size gate (scale 0.7) - 70% of original size"""
        file = "gate_medium.urdf"  # Use scaled medium gate file
        # Scale is built into the URDF, so we use normal scaling (1.0)
        # Start hidden off-screen initially (positioning handled by task)
        min_position_ratio = [-10.0, -10.0, 0.0]  # Hidden position
        max_position_ratio = [-10.0, -10.0, 0.0]
    
    class SmallGateParams(GateAssetConfig.gate_asset_params):
        """Small size gate (scale 0.5) - 50% of original size"""
        file = "gate_small.urdf"  # Use scaled small gate file
        # Scale is built into the URDF, so we use normal scaling (1.0)
        # Start hidden off-screen initially (positioning handled by task)
        min_position_ratio = [-10.0, -10.0, 0.0]  # Hidden position
        max_position_ratio = [-10.0, -10.0, 0.0]
    
    class MinimumGateParams(GateAssetConfig.gate_asset_params):
        """Minimum size gate (scale 0.4) - 40% of original (1m opening)"""
        file = "gate_minimum.urdf"  # Use scaled minimum gate file
        # Scale is built into the URDF, so we use normal scaling (1.0)
        # Start hidden off-screen initially (positioning handled by task)
        min_position_ratio = [-10.0, -10.0, 0.0]  # Hidden position
        max_position_ratio = [-10.0, -10.0, 0.0]
    
    @classmethod
    def get_gate_params_by_scale(cls, scale_factor):
        """Get gate parameters for a specific scale factor"""
        if scale_factor >= 1.0:
            return cls.FullSizeGateParams
        elif scale_factor >= 0.7:
            return cls.MediumGateParams
        elif scale_factor >= 0.5:
            return cls.SmallGateParams
        else:
            return cls.MinimumGateParams
    
    @classmethod
    def get_gate_tolerance_for_scale(cls, scale_factor):
        """
        Get adaptive success tolerance for gate scale.
        
        Original tolerance: ±1.3m width, 0.2-2.2m height
        Scales proportionally with gate size.
        
        Args:
            scale_factor: Gate scale factor (0.4 to 1.0)
            
        Returns:
            tuple: (width_tolerance, height_min, height_max)
        """
        # Base tolerance for full-size gate
        base_width_tolerance = 1.3  # ±1.3m for original 2.5m gate
        base_height_min = 0.2
        base_height_max = 2.2
        base_height_range = base_height_max - base_height_min  # 2.0m
        
        # Scale tolerances proportionally
        scaled_width_tolerance = base_width_tolerance * scale_factor
        scaled_height_range = base_height_range * scale_factor
        
        # Keep minimum height at ground level, scale the range
        scaled_height_min = base_height_min
        scaled_height_max = scaled_height_min + scaled_height_range
        
        return scaled_width_tolerance, scaled_height_min, scaled_height_max
    
    @classmethod
    def get_available_scales_for_level(cls, curriculum_level):
        """
        Get available gate scales for a given curriculum level.
        
        Early levels: Only largest gates
        Later levels: Mix of all scales
        
        Args:
            curriculum_level: Current curriculum level (3-23)
            
        Returns:
            list: Available scale factors for this level
        """
        # Define curriculum progression for gate scaling
        if curriculum_level <= 8:
            # Levels 3-8: Only full size gates (easiest)
            return [1.0]
        elif curriculum_level <= 13:
            # Levels 9-13: Full and medium size gates
            return [1.0, 0.7]
        elif curriculum_level <= 18:
            # Levels 14-18: Full, medium, and small gates
            return [1.0, 0.7, 0.5]
        else:
            # Levels 19-23: All gate sizes including minimum
            return [1.0, 0.7, 0.5, 0.4]


# Create instances for different gate scales
full_size_gate_params = GateScalingConfig.FullSizeGateParams
medium_gate_params = GateScalingConfig.MediumGateParams
small_gate_params = GateScalingConfig.SmallGateParams
minimum_gate_params = GateScalingConfig.MinimumGateParams 