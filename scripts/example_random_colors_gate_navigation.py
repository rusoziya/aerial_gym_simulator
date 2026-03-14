#!/usr/bin/env python3
"""
Example script demonstrating random color functionality for gate navigation.

This script shows how to:
1. Create a gate navigation task with random colors
2. Run the simulation and observe the colored assets
3. Reset environments to get new random colors
"""

import sys
import os
sys.path.append('/home/ziyar/aerialgym/aerialgym_ws/src/aerial_gym_simulator')

import torch
import time

def main():
    """Main function demonstrating random color gate navigation."""
    print("=== Random Color Gate Navigation Example ===")
    
    try:
        # Import the modified navigation task
        from aerial_gym.task.navigation_task_gate.navigation_task_gate import NavigationTaskGate
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        
        print("✓ Successfully imported NavigationTaskGate")
        
        # Create task configuration
        config = task_config()
        config.num_envs = 4  # Use 4 environments to see different colors
        config.headless = False  # Enable visualization
        config.episode_len_steps = 50  # Shorter episodes for demonstration
        
        print(f"✓ Configuration: {config.num_envs} environments, headless={config.headless}")
        
        # Create the navigation task
        print("\nCreating NavigationTaskGate with random colors...")
        task = NavigationTaskGate(config)
        
        print("✓ Task created successfully!")
        print("✓ Random colors have been applied to gate and obstacle assets")
        print("✓ You should see the Isaac Gym viewer with colored assets")
        
        # Run simulation for a few episodes
        print("\nRunning simulation for 3 episodes...")
        
        for episode in range(3):
            print(f"\n--- Episode {episode + 1} ---")
            
            # Reset for new episode (this will apply new random colors)
            obs = task.reset()
            print(f"✓ Episode {episode + 1} reset - new random colors applied")
            
            # Run episode
            episode_reward = 0.0
            for step in range(config.episode_len_steps):
                # Simple hover action (no movement)
                actions = torch.zeros((config.num_envs, 4), device=task.device)
                
                # Step simulation
                obs, rewards, terminations, truncations, infos = task.step(actions)
                
                episode_reward += rewards.mean().item()
                
                # Check if episode ended
                if terminations.any() or truncations.any():
                    break
                
                # Small delay to see the simulation
                time.sleep(0.01)
            
            print(f"✓ Episode {episode + 1} completed - Average reward: {episode_reward:.3f}")
            print(f"✓ Each episode should have different random colors for the assets")
        
        # Clean up
        print("\nCleaning up...")
        task.close()
        print("✓ Task closed successfully")
        
        print("\n🎉 Random color gate navigation example completed!")
        print("Key features demonstrated:")
        print("- Random colors applied to gate assets during initialization")
        print("- Random colors applied to obstacle assets (objects_gate)")
        print("- New random colors applied each time environments are reset")
        print("- Colors are applied to all rigid bodies of each asset")
        print("- Each environment has different random colors")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in example: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Example completed successfully!")
    else:
        print("\n❌ Example failed!")
        sys.exit(1) 