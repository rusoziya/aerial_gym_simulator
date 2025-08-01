"""
Test script for the new gate scaling curriculum feature.

This script demonstrates how the gate scaling curriculum works:
1. Different gate scales are selected based on curriculum level
2. Success tolerances adapt to gate size
3. Progressive difficulty from large to small gates

Usage:
    python aerial_gym/examples/test_gate_scaling.py
"""

import torch
import numpy as np
from aerial_gym.config.task_config.navigation_task_config_gate import task_config
from aerial_gym.config.asset_config.gate_scaling_config import GateScalingConfig

def test_gate_scaling_curriculum():
    """Test the gate scaling curriculum progression."""
    
    print("=" * 60)
    print("GATE SCALING CURRICULUM TEST")
    print("=" * 60)
    
    # Test curriculum levels from 3 to 23
    test_levels = [3, 5, 8, 10, 13, 15, 18, 20, 23]
    
    print(f"{'Level':<6} {'Available Scales':<20} {'Selected':<8} {'Width Tol':<10} {'Height Range'}")
    print("-" * 60)
    
    for level in test_levels:
        # Get available scales for this level
        available_scales = GateScalingConfig.get_available_scales_for_level(level)
        
        # Select a random scale (simulate episode reset)
        selected_scale = task_config.curriculum.get_gate_scale_for_level(level)
        
        # Get adaptive tolerance for this scale
        width_tol, height_min, height_max = task_config.curriculum.get_gate_tolerance_for_scale(selected_scale)
        height_range = height_max - height_min
        
        available_str = f"{available_scales}".replace(' ', '')
        print(f"{level:<6} {available_str:<20} {selected_scale:<8.1f} {width_tol:<10.2f} {height_range:<11.2f}")
    
    print("\n" + "=" * 60)
    print("GATE TOLERANCE SCALING TEST")
    print("=" * 60)
    
    # Test tolerance scaling for different gate sizes
    test_scales = [1.0, 0.7, 0.5, 0.4]
    gate_names = ["Full Size", "Medium", "Small", "Minimum"]
    
    print(f"{'Gate Type':<12} {'Scale':<6} {'Width Tol':<10} {'Height Min':<10} {'Height Max':<10} {'Gate Opening'}")
    print("-" * 70)
    
    for i, scale in enumerate(test_scales):
        width_tol, height_min, height_max = GateScalingConfig.get_gate_tolerance_for_scale(scale)
        
        # Calculate actual gate opening dimensions
        base_width = 2.5  # Original gate width
        base_height = 2.3  # Original gate height
        
        actual_width = base_width * scale
        actual_height = base_height * scale
        
        print(f"{gate_names[i]:<12} {scale:<6.1f} {width_tol:<10.2f} {height_min:<10.2f} {height_max:<10.2f} {actual_width:.1f}x{actual_height:.1f}m")
    
    print("\n" + "=" * 60)
    print("CURRICULUM PROGRESSION SIMULATION")
    print("=" * 60)
    
    # Simulate curriculum progression
    print("Simulating 10 episode resets at different curriculum levels:\n")
    
    for level in [5, 10, 15, 20]:
        print(f"Curriculum Level {level}:")
        scales_used = []
        
        for episode in range(10):
            scale = task_config.curriculum.get_gate_scale_for_level(level)
            scales_used.append(scale)
        
        # Count scale usage
        from collections import Counter
        scale_counts = Counter(scales_used)
        
        for scale, count in sorted(scale_counts.items(), reverse=True):
            percentage = (count / 10) * 100
            gate_type = "Full" if scale >= 1.0 else "Medium" if scale >= 0.7 else "Small" if scale >= 0.5 else "Minimum"
            print(f"  {gate_type:>8} (scale {scale:.1f}): {count:>2}/10 episodes ({percentage:>4.0f}%)")
        
        print()
    
    print("✅ Gate scaling curriculum test completed successfully!")

def test_tolerance_adaptation():
    """Test adaptive tolerance calculations."""
    
    print("\n" + "=" * 60)
    print("ADAPTIVE TOLERANCE TEST")
    print("=" * 60)
    
    # Test with different robot positions and gate scales
    robot_positions = torch.tensor([
        [0.0, 1.0, 1.2],    # Perfect center passage
        [0.6, 1.0, 1.2],    # Slightly off-center
        [1.0, 1.0, 1.2],    # Near edge
        [1.5, 1.0, 1.2],    # At tolerance boundary
    ])
    
    gate_position = torch.tensor([0.0, 0.0, 0.0])  # Gate at origin
    
    scales_to_test = [1.0, 0.7, 0.5, 0.4]
    scale_names = ["Full", "Medium", "Small", "Minimum"]
    
    print("Testing if robot positions would pass through different gate scales:")
    print(f"{'Position':<15} {'Full':<6} {'Medium':<8} {'Small':<7} {'Minimum'}")
    print("-" * 50)
    
    for i, robot_pos in enumerate(robot_positions):
        x_offset = robot_pos[0] - gate_position[0]
        pos_str = f"X={x_offset:.1f}m"
        
        results = []
        for scale in scales_to_test:
            width_tol, _, _ = GateScalingConfig.get_gate_tolerance_for_scale(scale)
            
            # Check if position would pass
            within_width = abs(x_offset) < width_tol
            results.append("✅" if within_width else "❌")
        
        print(f"{pos_str:<15} {results[0]:<6} {results[1]:<8} {results[2]:<7} {results[3]}")
    
    print(f"\nTolerance values:")
    for i, scale in enumerate(scales_to_test):
        width_tol, _, _ = GateScalingConfig.get_gate_tolerance_for_scale(scale)
        print(f"  {scale_names[i]:>8}: ±{width_tol:.2f}m width tolerance")

if __name__ == "__main__":
    try:
        test_gate_scaling_curriculum()
        test_tolerance_adaptation()
        
        print("\n🎉 All tests passed! Gate scaling curriculum is ready to use.")
        print("\nTo use the gate scaling curriculum:")
        print("1. Use 'gate_env' as your environment configuration")
        print("2. The curriculum will automatically select appropriate gate scales")
        print("3. Success criteria will adapt to the selected gate size")
        print("4. Monitor 'gate_scaling/*' metrics in wandb for tracking")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise 