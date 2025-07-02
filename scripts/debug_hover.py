#!/usr/bin/env python3

"""
Debug script to understand why hover episodes terminate immediately
"""

import os
import numpy as np

def main():
    print("=" * 60)
    print("DEBUG: HOVER EPISODE TERMINATION")
    print("=" * 60)
    
    # Import Aerial Gym components FIRST
    print("Importing Aerial Gym components...")
    from aerial_gym.envs.navigation_task.multi_agent_navigation_task_config import MultiAgentHoverTaskConfig
    from aerial_gym.algorithms.multi_agent_wrapper import create_multi_agent_env
    
    # Import PyTorch AFTER Aerial Gym
    print("Importing PyTorch...")
    import torch
    
    # Create config
    print("Setting up configuration...")
    config = MultiAgentHoverTaskConfig()
    config.num_envs = 1
    config.headless = False
    config.episode_len_steps = 500
    
    print("Creating environment...")
    env = create_multi_agent_env(
        task_config_class=lambda: config,
        algorithm="mappo",
        device="cuda" if torch.cuda.is_available() else "cpu",
        headless=False,
        num_envs=1
    )
    
    print("✅ Environment created!")
    print(f"   - Agent IDs: {env.agent_ids}")
    print(f"   - Episode length: {env.task.episode_len_steps}")
    
    # Reset and examine
    print("\n🔍 Detailed Environment Analysis:")
    
    obs_dict = env.reset()
    print(f"\n1. Reset completed - got observations")
    
    # Get raw simulation data
    raw_obs = env.task.sim_env.get_obs()
    print(f"\n2. Raw observations available: {list(raw_obs.keys())}")
    
    robot_states = None
    if "robot_state" in raw_obs:
        robot_states = raw_obs["robot_state"].view(1, 3, 13)
        print(f"   Raw robot states shape: {robot_states.shape}")
        
        for i in range(3):
            pos = robot_states[0, i, :3]
            vel = robot_states[0, i, 7:10]
            print(f"   Robot {i}: pos={pos}, vel={vel}")
    elif "robot_state_tensor" in raw_obs:
        # Try using robot_state_tensor instead
        robot_states = raw_obs["robot_state_tensor"].view(env.task.num_envs, env.task.num_robots_per_env, 13)
        print(f"   Using robot_state_tensor shape: {robot_states.shape}")
        
        for i in range(env.task.num_robots_per_env):
            pos = robot_states[0, i, :3]
            vel = robot_states[0, i, 7:10]
            print(f"   Robot {i}: pos={pos}, vel={vel}")
    else:
        print("   ⚠️  Neither 'robot_state' nor 'robot_state_tensor' found in raw observations!")
        # Try to get robot positions another way
        if hasattr(env.task, 'robot_positions'):
            print(f"   Found robot_positions: {env.task.robot_positions}")
        if hasattr(env.task.sim_env, 'robot_positions'):
            print(f"   Found sim_env.robot_positions: {env.task.sim_env.robot_positions}")
    
    # Check termination logic step by step
    print(f"\n3. Termination Analysis:")
    print(f"   Current episode steps: {env.task.current_episode_steps}")
    print(f"   Episode length limit: {env.task.episode_len_steps}")
    
    # Manual termination check
    dones = env.task._check_multi_agent_termination()
    print(f"   Termination result: {dones}")
    
    # Check bounds manually if we have robot states
    if robot_states is not None:
        print(f"\n4. Detailed Bounds Analysis:")
        for i in range(env.task.num_robots_per_env):
            pos = robot_states[0, i, :3]
            x, y, z = pos[0], pos[1], pos[2]
            
            out_of_bounds_x = (x < -50) or (x > 50)
            out_of_bounds_y = (y < -50) or (y > 50)
            out_of_bounds_z = (z < 0.5) or (z > 20)
            
            print(f"   Robot {i} bounds check:")
            print(f"     Position: x={x:.3f}, y={y:.3f}, z={z:.3f}")
            print(f"     Out of bounds X: {out_of_bounds_x} (bounds: -50 to 50)")
            print(f"     Out of bounds Y: {out_of_bounds_y} (bounds: -50 to 50)")
            print(f"     Out of bounds Z: {out_of_bounds_z} (bounds: 0.5 to 20)")
            
            # Check collisions
            for j in range(env.task.num_robots_per_env):
                if i != j:
                    other_pos = robot_states[0, j, :3]
                    distance = torch.norm(pos - other_pos)
                    collision = distance < env.task.hard_collision_radius
                    print(f"     Distance to robot {j}: {distance:.3f} (collision if < {env.task.hard_collision_radius}): {collision}")
    else:
        print(f"\n4. Cannot analyze bounds - robot_states not available")
        
        # Try to access termination method directly
        print("   Attempting to inspect termination logic...")
        if hasattr(env.task, '_check_out_of_bounds'):
            try:
                bounds_result = env.task._check_out_of_bounds()
                print(f"   Out of bounds check: {bounds_result}")
            except Exception as e:
                print(f"   Error checking bounds: {e}")
        
        if hasattr(env.task, '_check_collisions'):
            try:
                collision_result = env.task._check_collisions()
                print(f"   Collision check: {collision_result}")
            except Exception as e:
                print(f"   Error checking collisions: {e}")
    
    # Try one step
    print(f"\n5. Single Step Test:")
    
    # Create dummy actions
    actions_dict = {}
    for agent_key in env.agent_ids:
        actions_dict[agent_key] = np.array([0.0, 0.0, 0.0, 0.0])  # Zero actions
    
    print(f"   Taking step with zero actions...")
    new_obs_dict, reward_dict, done_dict, _, info_dict = env.step(actions_dict)
    
    print(f"   Step results:")
    print(f"     Rewards: {reward_dict}")
    print(f"     Done: {done_dict}")
    print(f"     Episode steps after: {env.task.current_episode_steps}")
    
    # Check positions after step
    raw_obs_after = env.task.sim_env.get_obs()
    if "robot_state" in raw_obs_after:
        robot_states_after = raw_obs_after["robot_state"].view(1, 3, 13)
        print(f"   Positions after step:")
        for i in range(3):
            pos_after = robot_states_after[0, i, :3]
            print(f"     Robot {i}: {pos_after}")
    elif "robot_state_tensor" in raw_obs_after:
        robot_states_after = raw_obs_after["robot_state_tensor"].view(env.task.num_envs, env.task.num_robots_per_env, 13)
        print(f"   Positions after step:")
        for i in range(env.task.num_robots_per_env):
            pos_after = robot_states_after[0, i, :3]
            print(f"     Robot {i}: {pos_after}")
    
    # Let's also check the environment's internal state
    print(f"\n6. Environment Internal State:")
    print(f"   env.task type: {type(env.task)}")
    print(f"   env.task.sim_env type: {type(env.task.sim_env)}")
    
    # Check if there are any specific termination flags
    if hasattr(env.task, 'is_collision'):
        print(f"   is_collision: {env.task.is_collision}")
    if hasattr(env.task, 'out_of_bounds'):
        print(f"   out_of_bounds: {env.task.out_of_bounds}")
    if hasattr(env.task, 'termination_reason'):
        print(f"   termination_reason: {env.task.termination_reason}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("Check the detailed output above for termination causes.")
    print("=" * 60)

if __name__ == "__main__":
    main() 