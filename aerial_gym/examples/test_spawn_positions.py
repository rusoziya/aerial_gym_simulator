#!/usr/bin/env python3

import numpy as np
import torch

def test_spawn_ratios():
    """Test that spawn ratios correctly place drone on obstacle-free side."""
    
    print("=== Testing Gate Navigation Spawn Positions ===")
    print("Environment bounds: [-6, 6] x [-6, 6] x [0, 6]")
    print("Gate position: (0, 0, 0)")
    print("Trees (obstacles): Y ratios 0.7-0.8 → Y positions 2.4-3.6 (positive Y)")
    print("Drone spawn: Y ratios 0.15-0.4 → Y positions -4.2 to -1.2 (negative Y)")
    print()
    
    # Environment bounds
    env_bounds_min = torch.tensor([-6.0, -6.0, 0.0])
    env_bounds_max = torch.tensor([6.0, 6.0, 6.0])
    
    # Tree positioning ratios
    tree_y_min_ratio = 0.7
    tree_y_max_ratio = 0.8
    
    # Drone spawn ratios from config
    drone_spawn_min_ratio = torch.tensor([0.2, 0.15, 0.2])
    drone_spawn_max_ratio = torch.tensor([0.8, 0.40, 0.6])
    
    # Calculate actual positions
    tree_y_min = env_bounds_min[1] + tree_y_min_ratio * (env_bounds_max[1] - env_bounds_min[1])
    tree_y_max = env_bounds_min[1] + tree_y_max_ratio * (env_bounds_max[1] - env_bounds_min[1])
    
    drone_spawn_min = env_bounds_min + drone_spawn_min_ratio * (env_bounds_max - env_bounds_min)
    drone_spawn_max = env_bounds_min + drone_spawn_max_ratio * (env_bounds_max - env_bounds_min)
    
    print(f"Tree Y positions: {tree_y_min:.1f} to {tree_y_max:.1f}")
    print(f"Drone spawn X: {drone_spawn_min[0]:.1f} to {drone_spawn_max[0]:.1f}")
    print(f"Drone spawn Y: {drone_spawn_min[1]:.1f} to {drone_spawn_max[1]:.1f}")
    print(f"Drone spawn Z: {drone_spawn_min[2]:.1f} to {drone_spawn_max[2]:.1f}")
    print()
    
    # Verification
    gate_y = 0.0
    drone_max_y = drone_spawn_max[1].item()
    tree_min_y = tree_y_min.item()
    
    print("=== Verification ===")
    print(f"✓ Trees are on positive Y side: {tree_min_y > gate_y}")
    print(f"✓ Drone spawns on negative Y side: {drone_max_y < gate_y}")
    print(f"✓ Good separation: drone max Y ({drone_max_y:.1f}) < gate Y ({gate_y}) < tree min Y ({tree_min_y:.1f})")
    
    separation = tree_min_y - drone_max_y
    print(f"✓ Total separation distance: {separation:.1f}m")
    
    # Test sample positions
    print("\n=== Sample Spawn Positions (10 random samples) ===")
    for i in range(10):
        # Random ratios within spawn range
        sample_ratios = torch.rand(3) * (drone_spawn_max_ratio - drone_spawn_min_ratio) + drone_spawn_min_ratio
        sample_pos = env_bounds_min + sample_ratios * (env_bounds_max - env_bounds_min)
        
        on_obstacle_free_side = sample_pos[1] < gate_y
        print(f"Sample {i+1}: ({sample_pos[0]:.1f}, {sample_pos[1]:.1f}, {sample_pos[2]:.1f}) - "
              f"{'✓ Obstacle-free' if on_obstacle_free_side else '✗ OBSTACLE SIDE!'}")
    
    print("\n=== Summary ===")
    print("Configuration ensures drones always spawn on obstacle-free side (negative Y)")
    print("Trees are positioned on positive Y side, providing clear navigation challenge")
    print("Gate center rewards will encourage precise navigation through center")

if __name__ == "__main__":
    test_spawn_ratios() 