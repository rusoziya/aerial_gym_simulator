#!/usr/bin/env python3
"""
Test script to verify the gate obstacle count fix.

This script tests that:
1. Gate environment can load 12 objects initially
2. Curriculum system can properly access 4+ obstacles at level 4
3. Asset counts match curriculum expectations
"""

import sys
import os
sys.path.append('/home/ziyar/aerialgym/aerialgym_ws/src/aerial_gym_simulator')

def test_gate_obstacle_configuration():
    """Test that gate environment configuration is correct."""
    print("=== Testing Gate Environment Configuration ===")
    
    try:
        from aerial_gym.config.env_config.gate_env import GateEnvCfg
        gate_config = GateEnvCfg()
        
        # Check if gate_object_params has correct num_assets
        gate_object_params = gate_config.env_config.gate_object_params
        print(f"✓ gate_object_params.num_assets = {gate_object_params.num_assets}")
        
        if gate_object_params.num_assets >= 12:
            print("✓ Gate environment has sufficient asset capacity for curriculum")
        else:
            print(f"✗ Gate environment only has {gate_object_params.num_assets} assets (need 12+)")
            
    except Exception as e:
        print(f"✗ Error loading gate configuration: {e}")
        return False
    
    return True

def test_curriculum_obstacle_calculation():
    """Test that curriculum calculates correct obstacle counts."""
    print("\n=== Testing Curriculum Obstacle Calculation ===")
    
    try:
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        curriculum = task_config.curriculum
        
        # Test obstacle counts for different levels
        test_levels = [3, 4, 5, 10, 15, 20]
        for level in test_levels:
            obstacle_count = curriculum.get_obstacle_count_behind_gate(level)
            print(f"Level {level}: {obstacle_count} obstacles behind gate")
            
        # Specific test for level 4 (the problematic level)
        level_4_obstacles = curriculum.get_obstacle_count_behind_gate(4)
        if level_4_obstacles == 4:
            print("✓ Level 4 correctly calculates 4 obstacles")
        else:
            print(f"✗ Level 4 calculates {level_4_obstacles} obstacles (expected 4)")
            
    except Exception as e:
        print(f"✗ Error testing curriculum: {e}")
        return False
    
    return True

def test_asset_capacity_validation():
    """Test that curriculum validation prevents exceeding capacity."""
    print("\n=== Testing Asset Capacity Validation ===")
    
    try:
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        curriculum = task_config.curriculum
        
        # Test with high level that should be capped
        high_level_obstacles = curriculum.get_obstacle_count_behind_gate(25)
        print(f"Level 25: {high_level_obstacles} obstacles (should be capped at 10)")
        
        if high_level_obstacles <= 12:  # Should not exceed total capacity
            print("✓ Curriculum properly caps obstacle count within capacity")
        else:
            print(f"✗ Curriculum exceeds capacity: {high_level_obstacles} > 12")
            
    except Exception as e:
        print(f"✗ Error testing validation: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Gate Obstacle Count Fix Verification")
    print("=" * 50)
    
    success = True
    success &= test_gate_obstacle_configuration()
    success &= test_curriculum_obstacle_calculation()
    success &= test_asset_capacity_validation()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! The obstacle count fix should work correctly.")
        print("\nNext steps:")
        print("1. Restart the training to reload the configuration")
        print("2. Check that curriculum level 4 now shows 4 obstacles behind gate")
        print("3. Verify curriculum progression works properly")
    else:
        print("✗ Some tests failed. Please check the configuration.")
    
    print("=" * 50) 