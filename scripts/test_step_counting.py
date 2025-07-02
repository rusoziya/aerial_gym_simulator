#!/usr/bin/env python3

"""
Test script to verify that step counting works properly in the multi-agent hover task
"""

import numpy as np

def main():
    print("=" * 60)
    print("TESTING STEP COUNTING")
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
    config.headless = True  # Headless for faster testing
    config.episode_len_steps = 10  # Short episodes for testing
    
    print("Creating environment...")
    env = create_multi_agent_env(
        task_config_class=lambda: config,
        algorithm="mappo",
        device="cuda" if torch.cuda.is_available() else "cpu",
        headless=True,
        num_envs=1
    )
    
    print("✅ Environment created!")
    print(f"   - Episode length: {env.task.episode_len_steps}")
    
    # Reset and test multiple steps
    print("\n🔍 Testing Step Counting:")
    
    obs_dict = env.reset()
    print(f"After reset - Episode steps: {env.task.current_episode_steps.item()}")
    
    # Create dummy actions
    actions_dict = {}
    for agent_key in env.agent_ids:
        actions_dict[agent_key] = np.array([0.0, 0.0, 0.0, 0.0])  # Zero actions
    
    # Take several steps
    for step_num in range(5):
        print(f"\nStep {step_num + 1}:")
        print(f"  Before step - Episode steps: {env.task.current_episode_steps.item()}")
        
        new_obs_dict, reward_dict, done_dict, _, info_dict = env.step(actions_dict)
        
        print(f"  After step - Episode steps: {env.task.current_episode_steps.item()}")
        print(f"  Rewards: {[f'{k}: {v:.1f}' for k, v in reward_dict.items()]}")
        print(f"  Done: {list(done_dict.values())}")
        
        # Check if episode terminated
        if any(done_dict.values()):
            print(f"  🏁 Episode terminated at step {step_num + 1}")
            break
    
    print("\n" + "=" * 60)
    print("STEP COUNTING TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main() 