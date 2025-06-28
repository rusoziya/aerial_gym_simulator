#!/usr/bin/env python3

"""
Example: Static Environment Camera with Multi-Agent Navigation

This example demonstrates how to integrate a static environment camera 
using the Warp rendering pipeline to observe multi-agent drone training.
"""

import torch
import numpy as np
import cv2
import os
from datetime import datetime

def main():
    print("=" * 60)
    print("STATIC ENVIRONMENT CAMERA EXAMPLE")
    print("=" * 60)
    
    # Import Aerial Gym components FIRST (before torch) to avoid IsaacGym conflicts
    print("Importing Aerial Gym components...")
    from aerial_gym.envs.navigation_task.multi_agent_navigation_task_config import MultiAgentFormationTaskConfigLight
    from aerial_gym.algorithms.multi_agent_wrapper import create_multi_agent_env
    from aerial_gym.sensors.warp.static_environment_camera import (
        StaticEnvironmentCamera,
        OverheadCameraConfig,
        SideViewCameraConfig,
        MultiAngleCameraConfig
    )
    
    print("All imports successful!")
    
    # Create multi-agent environment
    print("Creating multi-agent environment...")
    
    def create_config():
        config = MultiAgentFormationTaskConfigLight()
        config.task_config["num_envs"] = 1  # Single environment for demonstration
        config.env_config["num_envs"] = 1
        config.task_config["num_robots_per_env"] = 3
        config.env_config["num_robots_per_env"] = 3
        
        # Enable Warp rendering (required for static cameras)
        config.task_config["use_warp"] = True
        config.sim_config["use_warp"] = True
        
        # Override environment to use better bounds
        config.task_config["env_name"] = "env_with_obstacles"
        config.sim_config["env_name"] = "env_with_obstacles"
        
        # Disable robot cameras to focus on static environment cameras
        # You could also enable them for comparison
        config.task_config["enable_onboard_cameras"] = False
        
        return config
    
    env = create_multi_agent_env(
        task_config_class=create_config,
        algorithm="mappo",
        device="cuda" if torch.cuda.is_available() else "cpu",
        headless=True,  # Use headless mode since we have our own camera
        num_envs=1
    )
    
    print("Environment created successfully!")
    
    # Get mesh IDs for Warp rendering (required for static cameras)
    mesh_ids = env.task.sim_env.global_tensor_dict.get("CONST_WARP_MESH_ID_LIST")
    if mesh_ids is None:
        raise ValueError("Warp mesh IDs not available. Make sure use_warp=True in config.")
    
    device = env.task.device
    
    # Create different types of static cameras
    print("Creating static environment cameras...")
    
    # 1. Overhead camera for bird's eye view
    overhead_config = OverheadCameraConfig()
    overhead_camera = StaticEnvironmentCamera(
        camera_config=overhead_config,
        num_envs=1,
        mesh_ids_array=mesh_ids,
        device=device
    )
    
    # Set up overhead view 15m above the environment center
    overhead_camera.set_overhead_view(height=15.0)
    
    # Create image tensors for overhead camera
    overhead_pixels = torch.zeros(
        (1, 1, overhead_config.height, overhead_config.width),
        device=device, requires_grad=False
    )
    overhead_segmentation = torch.zeros(
        (1, 1, overhead_config.height, overhead_config.width),
        dtype=torch.int32, device=device, requires_grad=False
    )
    overhead_camera.set_image_tensors(overhead_pixels, overhead_segmentation)
    
    # 2. Side view camera
    side_config = SideViewCameraConfig()
    side_camera = StaticEnvironmentCamera(
        camera_config=side_config,
        num_envs=1,
        mesh_ids_array=mesh_ids,
        device=device
    )
    
    # Set up side view 20m away, 8m high, looking from the side
    side_camera.set_side_view(distance=20.0, height=8.0, angle_degrees=45.0)
    
    # Create image tensors for side camera
    side_pixels = torch.zeros(
        (1, 1, side_config.height, side_config.width),
        device=device, requires_grad=False
    )
    side_segmentation = torch.zeros(
        (1, 1, side_config.height, side_config.width),
        dtype=torch.int32, device=device, requires_grad=False
    )
    side_camera.set_image_tensors(side_pixels, side_segmentation)
    
    print("Static cameras created and configured!")
    
    # Create output directory for captured images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"static_camera_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Images will be saved to: {output_dir}")
    
    # Simple policy for demonstration
    class SimplePolicy(torch.nn.Module):
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, action_dim),
                torch.nn.Tanh()
            )
            
        def forward(self, x):
            return self.network(x)
    
    # Get dimensions and create policies
    obs_dim = list(env.observation_spaces.values())[0].shape[0]
    action_dim = list(env.action_spaces.values())[0].shape[0]
    
    policies = {}
    for agent_id in env.agent_ids:
        policies[agent_id] = SimplePolicy(obs_dim, action_dim).to(device)
    
    print(f"Running demonstration with static camera recording...")
    print(f"Agent IDs: {env.agent_ids}")
    
    # Run demonstration with camera recording
    num_steps = 200  # Number of steps to record
    
    for step in range(num_steps):
        # Reset environment if needed
        if step == 0:
            obs_dict = env.reset()
            
        # Get actions from policies
        actions_dict = {}
        for agent_env_key, obs in obs_dict.items():
            agent_id = agent_env_key.split('_env_')[0]
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action = policies[agent_id](obs_tensor).squeeze(0).cpu().numpy()
                actions_dict[agent_env_key] = action
        
        # Step environment
        obs_dict, reward_dict, done_dict, _, info_dict = env.step(actions_dict)
        
        # Capture images from static cameras
        overhead_camera.capture()
        side_camera.capture()
        
        # Save images every 10 steps
        if step % 10 == 0:
            # Convert overhead camera image to numpy and save
            overhead_img = overhead_pixels[0, 0].cpu().numpy()
            overhead_img_normalized = (overhead_img * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(output_dir, f"overhead_step_{step:04d}.png"),
                overhead_img_normalized
            )
            
            # Convert side camera image to numpy and save
            side_img = side_pixels[0, 0].cpu().numpy()
            side_img_normalized = (side_img * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(output_dir, f"side_view_step_{step:04d}.png"),
                side_img_normalized
            )
            
            # Save segmentation images
            overhead_seg = overhead_segmentation[0, 0].cpu().numpy().astype(np.uint8)
            side_seg = side_segmentation[0, 0].cpu().numpy().astype(np.uint8)
            
            cv2.imwrite(
                os.path.join(output_dir, f"overhead_seg_step_{step:04d}.png"),
                overhead_seg * 50  # Scale for visibility
            )
            cv2.imwrite(
                os.path.join(output_dir, f"side_seg_step_{step:04d}.png"),
                side_seg * 50  # Scale for visibility
            )
            
            print(f"Step {step}: Captured and saved images")
            
            # Print some statistics
            if step % 50 == 0:
                avg_rewards = {aid.split('_env_')[0]: reward_dict[aid] for aid in reward_dict.keys()}
                print(f"Step {step}: Rewards = {avg_rewards}")
        
        # Check if episode ended
        if any(done_dict.values()):
            print(f"Episode ended at step {step}")
            obs_dict = env.reset()
    
    print("\n" + "=" * 60)
    print("STATIC CAMERA DEMONSTRATION COMPLETE")
    print("=" * 60)
    print(f"Images saved to: {output_dir}")
    print("Files created:")
    print("  - overhead_step_XXXX.png: Overhead depth images")
    print("  - side_view_step_XXXX.png: Side view depth images") 
    print("  - overhead_seg_step_XXXX.png: Overhead segmentation images")
    print("  - side_seg_step_XXXX.png: Side view segmentation images")
    print("\nYou can create videos from these images using:")
    print(f"  ffmpeg -r 10 -i {output_dir}/overhead_step_%04d.png -vcodec libx264 overhead_video.mp4")
    print(f"  ffmpeg -r 10 -i {output_dir}/side_view_step_%04d.png -vcodec libx264 side_video.mp4")


if __name__ == "__main__":
    main() 