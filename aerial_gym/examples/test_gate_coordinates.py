#!/usr/bin/env python3

import sys
import os
import numpy as np

# Import Isaac Gym DIRECTLY first to establish order
import isaacgym
from isaacgym import gymapi

# Now import torch and other modules
import torch

# Now import aerial_gym modules
from aerial_gym.task.navigation_task_gate.navigation_task_gate import NavigationTaskGate
from aerial_gym.config.task_config.navigation_task_config_gate import task_config

def test_gate_coordinates():
    """Test the actual gate coordinates and spawn positions"""
    print("TESTING GATE COORDINATES AND SPAWN POSITIONS")
    print("="*60)
    
    # Create task instance
    config = task_config()
    task = NavigationTaskGate(
        task_config=config,
        num_envs=1,
        headless=True,
        device="cuda:0"
    )
    
    # Get environment info
    print("ENVIRONMENT BOUNDS:")
    env_bounds_min = task.obs_dict["env_bounds_min"][0].cpu().numpy()
    env_bounds_max = task.obs_dict["env_bounds_max"][0].cpu().numpy()
    print(f"  Min bounds: {env_bounds_min}")
    print(f"  Max bounds: {env_bounds_max}")
    
    # Check gate position
    print(f"GATE POSITION:")
    gate_pos = task.gate_position.cpu().numpy()
    print(f"  Gate center: {gate_pos}")
    
    # Test current spawn configuration
    print(f"CURRENT DRONE SPAWN CONFIGURATION:")
    spawn_min = config.drone_spawn_min_ratio
    spawn_max = config.drone_spawn_max_ratio
    print(f"  Spawn ratios: {spawn_min} to {spawn_max}")
    
    # Convert spawn ratios to actual positions
    spawn_pos_min = env_bounds_min + np.array(spawn_min) * (env_bounds_max - env_bounds_min)
    spawn_pos_max = env_bounds_min + np.array(spawn_max) * (env_bounds_max - env_bounds_min)
    print(f"  X range: {spawn_pos_min[0]:.2f} to {spawn_pos_max[0]:.2f}")
    print(f"  Y range: {spawn_pos_min[1]:.2f} to {spawn_pos_max[1]:.2f}")
    
    print("\nTESTING ACTUAL SPAWN POSITIONS (10 samples):")
    for i in range(10):
        # Reset environment 0 to trigger spawn position generation
        task.reset_idx(torch.tensor([0], device="cuda:0"))
        
        # Get robot position after reset
        robot_pos = task.obs_dict["robot_position"][0].cpu().numpy()
        print(f"Sample {i+1}: Position = [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}]")
    
    print("\nChecking if positions are in expected quadrants...")
    behind_gate_count = 0
    left_count = 0
    right_count = 0
    
    for i in range(50):  # Test more samples
        task.reset_idx(torch.tensor([0], device="cuda:0"))
        robot_pos = task.obs_dict["robot_position"][0].cpu().numpy()
        
        if robot_pos[1] < 0:  # Behind gate (negative Y)
            behind_gate_count += 1
        if robot_pos[0] < 0:  # Left side (negative X)
            left_count += 1
        if robot_pos[0] > 0:  # Right side (positive X)
            right_count += 1
    
    print(f"Out of 50 samples:")
    print(f"  Behind gate (Y < 0): {behind_gate_count}/50 ({behind_gate_count/50*100:.1f}%)")
    print(f"  Left side (X < 0): {left_count}/50 ({left_count/50*100:.1f}%)")
    print(f"  Right side (X > 0): {right_count}/50 ({right_count/50*100:.1f}%)")
    
    # Clean up
    task.close()

if __name__ == "__main__":
    test_gate_coordinates() 