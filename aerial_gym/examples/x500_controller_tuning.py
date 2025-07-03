#!/usr/bin/env python3

"""
X500 Controller Tuning Script

This script helps you test and tune the X500 controller gains interactively.
It creates a simple hovering task where you can observe the drone's behavior
and adjust gains in real-time.

Usage:
    python x500_controller_tuning.py [--headless]

Controls:
    - The drone will try to hover at a target position
    - Watch for oscillations, overshooting, or instability
    - Press Ctrl+C to stop and get tuning recommendations

Tuning Guidelines:
    Position Gains (K_pos): Controls position tracking
        - Too high: Overshooting, oscillations
        - Too low: Slow response, poor tracking
    
    Velocity Gains (K_vel): Controls damping
        - Too high: Sluggish response
        - Too low: Oscillations, instability
    
    Rotation Gains (K_rot): Controls attitude response
        - Too high: High-frequency oscillations
        - Too low: Poor attitude control
    
    Angular Velocity Gains (K_angvel): Controls angular damping
        - Too high: Sluggish rotation
        - Too low: Angular oscillations
"""

import time
import numpy as np

# CRITICAL: Import isaacgym before torch (Isaac Gym requirement)
import isaacgym  # Must be imported first!
import torch

from aerial_gym.registry.task_registry import task_registry

def main():
    """Main tuning loop."""
    print("🚁 X500 Controller Tuning Script")
    print("=" * 50)
    
    # Create the gate navigation task for testing
    try:
        from aerial_gym.examples.dce_rl_navigation.dce_navigation_task_gate import DCE_RL_Navigation_Task_Gate
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        
        # Register the gate navigation task
        task_registry.register_task(
            "x500_tuning_task",
            DCE_RL_Navigation_Task_Gate,
            task_config()
        )
        
        # Create task instance
        task = task_registry.make_task("x500_tuning_task", num_envs=1, headless=False)
        print("✓ Created X500 tuning environment")
        
    except Exception as e:
        print(f"❌ Error creating tuning environment: {e}")
        print("Make sure the gate navigation task is properly configured.")
        return
    
    # Simple hover test
    print("\n🎯 Starting Hover Test")
    print("Watch the drone's behavior and note any issues:")
    print("  - Oscillations: Reduce position gains or increase velocity gains")
    print("  - Overshooting: Reduce position gains")
    print("  - Slow response: Increase position gains (carefully)")
    print("  - Instability: Increase velocity gains for damping")
    print("\nPress Ctrl+C to stop the test\n")
    
    # Reset environment
    obs = task.reset()
    
    # Hover target (slightly above starting position)
    hover_commands = torch.tensor([[0.0, 0.0, 0.5]], device=task.device)  # Small upward movement
    
    step_count = 0
    try:
        while True:
            # Step the simulation with hover command
            obs, rewards, terminated, truncated, infos = task.step(hover_commands)
            
            step_count += 1
            
            # Print status every 60 steps (~1 second at 60Hz)
            if step_count % 60 == 0:
                # Get robot position from observations
                robot_pos = task.obs_dict["robot_position"][0].cpu().numpy()
                robot_vel = task.obs_dict["robot_body_linvel"][0].cpu().numpy()
                reward = rewards[0].item()
                
                print(f"Step {step_count:4d} | Pos: [{robot_pos[0]:+6.2f}, {robot_pos[1]:+6.2f}, {robot_pos[2]:+6.2f}] | "
                      f"Vel: [{robot_vel[0]:+5.2f}, {robot_vel[1]:+5.2f}, {robot_vel[2]:+5.2f}] | "
                      f"Reward: {reward:+7.2f}")
            
            # Reset if episode ends
            if terminated.any() or truncated.any():
                print("\n⚠️  Episode ended - resetting...")
                obs = task.reset()
                step_count = 0
                time.sleep(1)  # Brief pause
                
            time.sleep(0.016)  # ~60 Hz
            
    except KeyboardInterrupt:
        print("\n\n🔧 Tuning Recommendations")
        print("=" * 40)
        
        # Get final state for analysis
        robot_pos = task.obs_dict["robot_position"][0].cpu().numpy()
        robot_vel = task.obs_dict["robot_body_linvel"][0].cpu().numpy()
        
        print(f"Final Position: [{robot_pos[0]:+6.2f}, {robot_pos[1]:+6.2f}, {robot_pos[2]:+6.2f}]")
        print(f"Final Velocity: [{robot_vel[0]:+5.2f}, {robot_vel[1]:+5.2f}, {robot_vel[2]:+5.2f}]")
        
        # Basic analysis and recommendations
        pos_error = np.linalg.norm(robot_pos - np.array([0, 0, 1]))  # Distance from hover target
        vel_magnitude = np.linalg.norm(robot_vel)
        
        print(f"\nPosition Error: {pos_error:.3f}m")
        print(f"Velocity Magnitude: {vel_magnitude:.3f}m/s")
        
        print("\n📝 Tuning Suggestions:")
        
        if pos_error > 0.5:
            print("  - Position error is large, consider increasing position gains")
            print("    Edit aerial_gym/config/controller_config/x500_controller_config.py")
            print("    Increase K_pos_tensor_max and K_pos_tensor_min by 10-20%")
        
        if vel_magnitude > 0.2:
            print("  - High residual velocity, consider increasing velocity gains")
            print("    Edit aerial_gym/config/controller_config/x500_controller_config.py")
            print("    Increase K_vel_tensor_max and K_vel_tensor_min by 10-20%")
        
        if pos_error < 0.1 and vel_magnitude < 0.1:
            print("  - Controller performance looks good!")
            print("  - You can now train with confidence")
        
        print("\n💡 Advanced Tuning Tips:")
        print("  1. If you see oscillations, the gains might be too high")
        print("  2. If response is too slow, gains might be too low")
        print("  3. Start with small adjustments (±10%) and test again")
        print("  4. The current gains are based on X500 characteristics but may need fine-tuning")
        
        print(f"\n📍 Current Controller Config Location:")
        print(f"    aerial_gym/config/controller_config/x500_controller_config.py")
        
    finally:
        print("\n🏁 Tuning session complete")
        task.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="X500 Controller Tuning")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    args = parser.parse_args()
    
    main() 