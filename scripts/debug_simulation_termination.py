#!/usr/bin/env python3

"""
Debug script to check what the underlying simulation is reporting for termination
"""

import numpy as np

def main():
    print("=" * 60)
    print("DEBUG: SIMULATION TERMINATION")
    print("=" * 60)
    
    # Import Aerial Gym components FIRST
    print("Importing Aerial Gym components...")
    from aerial_gym.envs.navigation_task.multi_agent_navigation_task_config import MultiAgentHoverTaskConfig
    from aerial_gym.algorithms.multi_agent_wrapper import create_multi_agent_env
    
    # Also import the working single-agent navigation task for comparison
    from aerial_gym.envs.navigation_task.multi_agent_navigation_task_config import MultiAgentFormationTaskConfigLight
    
    # Import PyTorch AFTER Aerial Gym
    print("Importing PyTorch...")
    import torch
    
    # Create config
    print("Setting up configuration...")
    # Use the lighter config which might be more stable
    config = MultiAgentFormationTaskConfigLight()
    
    # Override environment to use better bounds
    config.task_config["env_name"] = "env_with_obstacles"
    config.sim_config["env_name"] = "env_with_obstacles"
    config.num_envs = 1  # Use only 1 environment for debugging
    config.headless = True
    config.episode_len_steps = 100  # Longer episodes
    
    print("Creating environment...")
    env = create_multi_agent_env(
        task_config_class=lambda: config,
        algorithm="mappo",
        device="cuda" if torch.cuda.is_available() else "cpu",
        headless=True,
        num_envs=1
    )
    
    print("✅ Environment created!")
    
    # Reset
    print("\n🔍 Debugging Simulation Termination:")
    obs_dict = env.reset()
    print(f"After reset - Episode steps: {env.task.current_episode_steps.item()}")
    
    # Check what the underlying sim_env reports
    print("\n📊 Checking underlying simulation state:")
    sim_obs = env.task.sim_env.get_obs()
    
    if 'crashes' in sim_obs:
        print(f"   Crashes: {sim_obs['crashes']}")
    if 'truncations' in sim_obs:
        print(f"   Truncations: {sim_obs['truncations']}")
    
    # Check robot state tensor directly
    robot_states = sim_obs.get("robot_state_tensor")
    if robot_states is not None:
        robot_states = robot_states.view(env.task.num_envs, env.task.num_robots_per_env, 13)
        print(f"   Robot states shape: {robot_states.shape}")
        
        for i in range(env.task.num_robots_per_env):
            pos = robot_states[0, i, :3]
            vel = robot_states[0, i, 7:10]
            print(f"   Robot {i}: pos={pos}, vel={vel}")
    
    # Now manually check each termination condition step by step
    print(f"\n🔍 Manual Termination Check:")
    
    # Call the task's termination check directly
    dones = env.task._check_multi_agent_termination()
    print(f"   Task termination result: {dones}")
    
    # Check bounds manually
    print(f"   Manual bounds check:")
    for env_idx in range(env.task.num_envs):
        for robot_idx in range(env.task.num_robots_per_env):
            robot_pos = robot_states[env_idx, robot_idx, :3]
            
            out_of_bounds_x = (robot_pos[0] < -50) or (robot_pos[0] > 50)
            out_of_bounds_y = (robot_pos[1] < -50) or (robot_pos[1] > 50) 
            out_of_bounds_z = (robot_pos[2] < 0.5) or (robot_pos[2] > 20)
            
            print(f"     Robot {robot_idx}: pos={robot_pos}")
            print(f"       X bounds ok: {not out_of_bounds_x}")
            print(f"       Y bounds ok: {not out_of_bounds_y}")
            print(f"       Z bounds ok: {not out_of_bounds_z}")
            
            # Check collisions
            for other_idx in range(env.task.num_robots_per_env):
                if other_idx != robot_idx:
                    other_pos = robot_states[env_idx, other_idx, :3]
                    distance = torch.norm(robot_pos - other_pos, dim=0)
                    collision = distance < env.task.hard_collision_radius
                    print(f"       Distance to robot {other_idx}: {distance:.3f} (collision: {collision})")
    
    # Check episode steps
    print(f"   Episode steps: {env.task.current_episode_steps} / {env.task.episode_len_steps}")
    
    # Take one step and see what happens
    print(f"\n🚀 Taking one step...")
    
    # Use position commands for lee_position_control controller
    # Actions are [x, y, z, yaw] scaled between -1 and 1
    # Since robots start at z=2 and targets are at z=3, we want slight upward motion
    # Small positive z command to hover and maintain altitude
    actions_dict = {}
    for agent_key in env.agent_ids:
        actions_dict[agent_key] = np.array([0.0, 0.0, 0.1, 0.0])  # Small upward command
    
    print(f"   Stepping with position commands: [x=0, y=0, z=0.1, yaw=0]...")
    print(f"   Agent IDs: {env.agent_ids}")
    print(f"   Actions dict: {actions_dict}")
    
    # Just run the actual environment step (no manual stepping)
    new_obs_dict, reward_dict, done_dict, _, info_dict = env.step(actions_dict)
    
    print(f"   After step:")
    print(f"     Episode steps: {env.task.current_episode_steps.item()}")
    print(f"     Rewards: {[f'{k}: {v:.1f}' for k, v in reward_dict.items()]}")
    print(f"     Done: {list(done_dict.values())}")
    
    # Check what info_dict contains
    print(f"     Info dict keys: {info_dict.keys()}")
    if "sim_crashes" in info_dict:
        print(f"     Sim crashes: {info_dict['sim_crashes']}")
    if "custom_terminations" in info_dict:
        print(f"     Custom terminations: {info_dict['custom_terminations']}")
    if "truncations" in info_dict:
        print(f"     Truncations: {info_dict['truncations']}")
    
    # Check final robot positions
    print(f"   Final robot positions after step:")
    final_obs = env.task.sim_env.get_obs()
    final_robot_states = final_obs.get("robot_state_tensor", torch.zeros(3, 13, device=env.task.device))
    final_robot_states = final_robot_states.view(env.task.num_envs, env.task.num_robots_per_env, 13)
    for i in range(env.task.num_robots_per_env):
        final_pos = final_robot_states[0, i, :3]
        final_vel = final_robot_states[0, i, 7:10]
        print(f"     Robot {i}: pos={final_pos}, vel={final_vel}")
    
    # Check simulation state after step
    sim_obs_after = env.task.sim_env.get_obs()
    if 'crashes' in sim_obs_after:
        print(f"     Crashes after: {sim_obs_after['crashes']}")
    if 'truncations' in sim_obs_after:
        print(f"     Truncations after: {sim_obs_after['truncations']}")
    
    print("\n" + "=" * 60)
    print("SIMULATION TERMINATION DEBUG COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main() 