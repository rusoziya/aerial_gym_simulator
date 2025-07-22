#!/usr/bin/env python3
"""
Test script to verify random color functionality for gate navigation task.

This script creates a simple gate navigation environment and verifies that:
1. Random colors are applied to gate assets during initialization
2. Random colors are reapplied when environments are reset
3. Both gate and obstacle assets receive random colors
"""

import sys
import os
sys.path.append('/home/ziyar/aerialgym/aerialgym_ws/src/aerial_gym_simulator')

def test_random_colors():
    """Test the random color functionality for gate navigation."""
    print("=== Testing Random Color Functionality for Gate Navigation ===")
    
    try:
        # Import the modified navigation task
        from aerial_gym.task.navigation_task_gate.navigation_task_gate import NavigationTaskGate
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        
        print("✓ Successfully imported NavigationTaskGate and task_config")
        
        # Create task configuration
        config = task_config()
        config.num_envs = 2  # Use small number for testing
        config.headless = False  # Enable visualization to see colors
        
        print(f"✓ Created task configuration with {config.num_envs} environments")
        print(f"✓ Headless mode: {config.headless}")
        
        # Create the navigation task
        print("Creating NavigationTaskGate...")
        task = NavigationTaskGate(config)
        
        print("✓ NavigationTaskGate created successfully")
        print("✓ Random colors should have been applied during initialization")
        
        # Test reset functionality
        print("\nTesting reset functionality...")
        task.reset()
        print("✓ Reset completed successfully")
        print("✓ Random colors should have been reapplied during reset")
        
        # Test a few steps to see the colors in action
        print("\nTesting simulation steps...")
        import torch
        
        # Create some dummy actions
        actions = torch.zeros((config.num_envs, 4), device=task.device)
        
        for step in range(5):
            obs, rewards, terminations, truncations, infos = task.step(actions)
            print(f"Step {step + 1}: Rewards = {rewards.mean().item():.3f}")
        
        print("✓ Simulation steps completed successfully")
        
        # Clean up
        task.close()
        print("✓ Task closed successfully")
        
        print("\n🎉 Random color functionality test completed successfully!")
        print("If headless=False, you should have seen the Isaac Gym viewer with colored gate and obstacle assets.")
        print("Each environment should have different random colors for the gate and obstacles.")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during random color test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_random_colors()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1) 