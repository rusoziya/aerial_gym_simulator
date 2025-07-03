#!/usr/bin/env python3

import torch
import numpy as np
from aerial_gym.task.navigation_task_gate.navigation_task_gate import NavigationTaskGate
from aerial_gym.config.task_config.navigation_task_config_gate import NavigationTaskGateConfig
from aerial_gym.config.env_config.gate_env import GateEnvConfig
from aerial_gym.config.robot_config.lmf2_config import LMF2Config
from aerial_gym.config.controller_config.lmf2_controller_config import LMF2ControllerConfig
from aerial_gym.config.sim_config.base_sim_config import BaseSimConfig
from aerial_gym.sim.sim_builder import SimBuilder

def test_gate_center_rewards():
    """Test gate center alignment rewards and easier spawn positions."""
    
    # Create configs
    task_config = NavigationTaskGateConfig()
    env_config = GateEnvConfig()
    robot_config = LMF2Config()
    controller_config = LMF2ControllerConfig()
    sim_config = BaseSimConfig()
    
    # Build simulator
    sim_builder = SimBuilder(
        sim_config=sim_config,
        env_config=env_config,
        robot_config=robot_config,
        controller_config=controller_config,
        task_config=task_config,
        num_envs=4,
        headless=True,
        device="cuda:0"
    )
    
    sim_env = sim_builder.build()
    
    # Create task
    task = NavigationTaskGate(
        task_config=task_config,
        sim_env=sim_env,
        num_envs=4,
    )
    
    print("=== Testing Gate Center Rewards and Easier Spawn Positions ===")
    
    # Test 1: Check spawn positions
    print("\nTest 1: Spawn Position Analysis")
    env_ids = torch.tensor([0, 1, 2, 3], device="cuda:0")
    
    spawn_positions = []
    for i in range(10):
        task.reset_idx(env_ids)
        obs = task.get_observations()
        robot_positions = obs["robot_position"]
        spawn_positions.append(robot_positions.cpu().numpy())
        
        print(f"Reset {i}: Robot positions (X, Y, Z):")
        for j, pos in enumerate(robot_positions):
            print(f"  Env {j}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    # Analyze spawn distribution
    all_spawns = np.concatenate(spawn_positions, axis=0)
    print(f"\nSpawn Analysis (10 resets, 4 envs):")
    print(f"X range: [{all_spawns[:, 0].min():.2f}, {all_spawns[:, 0].max():.2f}]")
    print(f"Y range: [{all_spawns[:, 1].min():.2f}, {all_spawns[:, 1].max():.2f}]")
    print(f"Z range: [{all_spawns[:, 2].min():.2f}, {all_spawns[:, 2].max():.2f}]")
    
    # Check if spawns are behind gate (Y > 0 for gate at center)
    behind_gate = all_spawns[:, 1] > 0
    print(f"Spawns behind gate: {behind_gate.sum()}/{len(behind_gate)} ({100*behind_gate.sum()/len(behind_gate):.1f}%)")
    
    # Test 2: Check gate center rewards
    print("\nTest 2: Gate Center Reward Analysis")
    
    # Position drone at different locations relative to gate
    test_positions = [
        [0.0, -1.0, 1.2],   # Perfectly centered behind gate
        [0.3, -1.0, 1.2],   # Slightly off-center horizontally 
        [0.0, -1.0, 1.0],   # Slightly off-center vertically
        [0.6, -1.0, 1.2],   # Outside center threshold
        [0.0, 0.5, 1.2],    # In front of gate, centered
    ]
    
    for i, pos in enumerate(test_positions):
        # Set robot position manually
        robot_pos = torch.tensor([pos], device="cuda:0").expand(4, -1)
        
        # Create fake obs dict with this position
        obs = {
            "robot_position": robot_pos,
            "robot_orientation": torch.zeros(4, 4, device="cuda:0"),
            "robot_linvel": torch.zeros(4, 3, device="cuda:0"),
            "robot_angvel": torch.zeros(4, 3, device="cuda:0"),
            "gate_position": torch.tensor([[0.0, 0.0, 0.0]], device="cuda:0").expand(4, -1),
            "gate_passed": torch.zeros(4, dtype=torch.bool, device="cuda:0"),
            "prev_distance_to_goal": torch.ones(4, device="cuda:0") * 5.0,
            "distance_to_goal": torch.norm(robot_pos - torch.tensor([[2.0, 2.0, 1.0]], device="cuda:0"), dim=1),
            "distance_to_gate": torch.norm(robot_pos - torch.tensor([[0.0, 0.0, 0.0]], device="cuda:0"), dim=1),
        }
        
        task.obs_dict = obs
        
        # Mock actions
        actions = torch.zeros(4, 3, device="cuda:0")
        
        # Compute rewards
        rewards = task.compute_reward(actions)
        
        print(f"\nPosition {i+1}: {pos}")
        print(f"  Reward: {rewards[0]:.3f}")
        
        # Check specific reward components by looking at reward breakdown
        if i == 0:
            print("  -> Should have gate center bonus (perfectly centered)")
        elif i == 4:
            print("  -> Should have gate center bonus + potential passage bonus")
    
    print("\n=== Test Complete ===")
    print("Check that:")
    print("1. Spawn positions are behind gate (Y > 0) and in reasonable ranges")
    print("2. Center rewards are higher for positions closer to gate center")
    print("3. All spawns are in obstacle-free areas")

if __name__ == "__main__":
    test_gate_center_rewards() 