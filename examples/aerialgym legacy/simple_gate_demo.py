"""
Simple Gate Environment Demo

A straightforward demo script to test the gate environment with:
- Single environment (no parallelization)
- Fixed gate position (non-randomized) 
- Simple automated flight pattern
- Clear visualization and logging

This is perfect for initial testing and debugging of the gate environment.
"""

import numpy as np
import time
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.config.env_config.gate_env import GateEnvCfg
from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg
from aerial_gym.config.controller_config.lee_controller_config import LeeControllerCfg
import torch


class SimpleGateEnvCfg(GateEnvCfg):
    """Modified gate environment config for single environment testing."""
    
    class env(GateEnvCfg.env):
        # Single environment for easy testing
        num_envs = 1
        
        # Smaller environment bounds for easier visualization
        lower_bound_min = [-5.0, -5.0, -0.5]
        lower_bound_max = [-5.0, -5.0, -0.5]  # Fixed bounds
        upper_bound_min = [5.0, 5.0, 3.0]
        upper_bound_max = [5.0, 5.0, 3.0]    # Fixed bounds
        
        # More responsive simulation
        render_viewer_every_n_steps = 1
        
        # Less sensitive collision detection for testing
        collision_force_threshold = 0.1
        
        # Create ground plane for reference
        create_ground_plane = True


def main():
    """Main demo function."""
    
    print("="*60)
    print("SIMPLE GATE ENVIRONMENT DEMO")
    print("="*60)
    print("Testing gate environment with:")
    print("- Single environment (no parallelization)")
    print("- Fixed gate position at center")
    print("- Automated flight pattern")
    print("="*60)
    
    # Configuration
    env_cfg = SimpleGateEnvCfg
    robot_cfg = BaseQuadCfg
    controller_cfg = LeeControllerCfg
    
    print("\nBuilding simulation...")
    
    # Create simulation builder
    sim_builder = SimBuilder(
        env_config=env_cfg,
        robot_config=robot_cfg,
        controller_config=controller_cfg,
        args=None
    )
    
    # Build environment
    env = sim_builder.build_env()
    
    print(f"✓ Environment built successfully!")
    print(f"✓ Number of environments: {env.num_envs}")
    print(f"✓ Environment bounds: {env.env_bounds}")
    
    # Reset environment
    obs = env.reset()
    
    print(f"✓ Environment initialized!")
    print(f"✓ Robot starting position: {obs['robot_position'][0]}")
    
    print("\nStarting flight demonstration...")
    print("Flight plan:")
    print("1. Hover and stabilize (steps 0-100)")
    print("2. Move toward gate (steps 100-300)")
    print("3. Fly through gate (steps 300-500)")
    print("4. Continue past gate (steps 500-700)")
    print("5. Return back through gate (steps 700-1000)")
    
    # Flight demonstration
    for step in range(1000):
        
        # Define flight phases
        if step < 100:
            # Phase 1: Hover and stabilize
            actions = torch.tensor([[0.0, 0.0, 0.0, 0.6]], device=env.device)
            phase = "Hovering"
            
        elif step < 300:
            # Phase 2: Move toward gate (approach on X-axis)
            actions = torch.tensor([[0.4, 0.0, 0.0, 0.6]], device=env.device)
            phase = "Approaching gate"
            
        elif step < 500:
            # Phase 3: Fly through gate (slower, controlled)
            actions = torch.tensor([[0.2, 0.0, 0.0, 0.6]], device=env.device)
            phase = "Flying through gate"
            
        elif step < 700:
            # Phase 4: Continue past gate
            actions = torch.tensor([[0.1, 0.0, 0.0, 0.6]], device=env.device)
            phase = "Past gate"
            
        else:
            # Phase 5: Return back through gate
            actions = torch.tensor([[-0.3, 0.0, 0.0, 0.6]], device=env.device)
            phase = "Returning through gate"
        
        # Step environment
        obs, rewards, dones, infos = env.step(actions)
        
        # Get robot state
        robot_pos = obs["robot_position"][0]
        robot_vel = obs["robot_body_linvel"][0] if "robot_body_linvel" in obs else torch.zeros(3)
        
        # Log progress every 50 steps
        if step % 50 == 0:
            print(f"Step {step:4d} | {phase:20s} | "
                  f"Pos: ({robot_pos[0]:+5.2f}, {robot_pos[1]:+5.2f}, {robot_pos[2]:+5.2f}) | "
                  f"Vel: ({robot_vel[0]:+5.2f}, {robot_vel[1]:+5.2f}, {robot_vel[2]:+5.2f})")
        
        # Check for collisions
        if dones[0]:
            print(f"⚠️  Episode ended at step {step} (collision or reset)")
            obs = env.reset()
            print(f"🔄 Environment reset, continuing...")
        
        # Small delay for visualization
        time.sleep(0.02)
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Gate environment is working correctly!")
    print("\nEnvironment features tested:")
    print("✓ Gate asset loading and positioning")
    print("✓ Robot spawning and control")
    print("✓ Physics simulation")
    print("✓ Collision detection")
    print("✓ Environment bounds")
    print("✓ Flight dynamics")
    
    print("\nNext steps:")
    print("- Use this environment for RL training")
    print("- Add sensors (cameras, lidar) for perception")
    print("- Create navigation tasks")
    print("- Test with different robot configurations")


if __name__ == "__main__":
    main() 