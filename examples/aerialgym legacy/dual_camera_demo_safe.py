#!/usr/bin/env python3

"""
Dual Camera Demo: Static Environment Camera + Drone Camera

This example demonstrates:
1. A static environment camera providing overhead/side views
2. A drone with an onboard camera
3. Real-time visualization of depth and segmentation outputs from both cameras
4. No RL training - just visualization of camera outputs

The demo shows depth and segmentation information extraction similar to the DCE navigation task.
"""

import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm
from PIL import Image
from datetime import datetime
from typing import List

def main():
    print("=" * 70)
    print("DUAL CAMERA DEMO: Static Environment Camera + Drone Camera")
    print("=" * 70)
    
    # Import Aerial Gym components FIRST (before torch) to avoid IsaacGym conflicts
    print("Importing Aerial Gym components...")
    from aerial_gym.registry.task_registry import task_registry
    from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import DCE_RL_Navigation_Task
    from aerial_gym.sensors.warp.static_environment_camera import (
        StaticEnvironmentCamera,
        OverheadCameraConfig,
        SideViewCameraConfig
    )
    
    # Import torch AFTER aerial gym components
    import torch
    
    print("All imports successful!")
    
    # Register the DCE navigation task (which has camera enabled)
    task_registry.register_task(
        task_name="dual_camera_demo_task",
        task_class=DCE_RL_Navigation_Task,
        task_config=task_registry.get_task_config("navigation_task"),
    )
    
    # Create the navigation task with camera enabled
    print("Creating navigation task with drone camera...")
    use_warp = True
    headless = True  # Use headless mode since we have our own visualization
    
    rl_task = task_registry.make_task(
        "dual_camera_demo_task",
        seed=42,
        use_warp=use_warp,
        headless=headless
    )
    
    print(f"Created task with {rl_task.num_envs} environments")
    device = rl_task.device
    
    # Initialize the task to ensure all tensors are properly set up
    print("Initializing task and resetting environments...")
    rl_task.reset()
    
    # Get mesh IDs for Warp rendering (required for static cameras)
    mesh_ids_list = rl_task.sim_env.global_tensor_dict.get("CONST_WARP_MESH_ID_LIST")
    if mesh_ids_list is None:
        raise ValueError("Warp mesh IDs not available. Make sure use_warp=True in config.")
    
    print(f"Mesh IDs type: {type(mesh_ids_list)}, length: {len(mesh_ids_list) if hasattr(mesh_ids_list, '__len__') else 'N/A'}")
    
    # Convert mesh_ids list to warp array
    import warp as wp
    mesh_ids = wp.array(mesh_ids_list, dtype=wp.uint64, device=device)
    print(f"Converted mesh IDs to warp array: {mesh_ids.shape}")
    
    # Create static environment cameras
    print("Creating static environment cameras...")
    
    # 1. Overhead camera for bird's eye view
    overhead_config = OverheadCameraConfig()
    overhead_config.width = 480
    overhead_config.height = 480
    overhead_camera = StaticEnvironmentCamera(
        camera_config=overhead_config,
        num_envs=rl_task.num_envs,
        mesh_ids_array=mesh_ids,
        device=device
    )
    
    # Set up overhead view 20m above the environment center
    overhead_camera.set_overhead_view(height=20.0)
    
    # Create image tensors for overhead camera
    overhead_pixels = torch.zeros(
        (rl_task.num_envs, 1, overhead_config.height, overhead_config.width),
        device=device, requires_grad=False
    )
    overhead_segmentation = torch.zeros(
        (rl_task.num_envs, 1, overhead_config.height, overhead_config.width),
        dtype=torch.int32, device=device, requires_grad=False
    )
    overhead_camera.set_image_tensors(overhead_pixels, overhead_segmentation)
    
    # 2. Side view camera
    side_config = SideViewCameraConfig()
    side_config.width = 480
    side_config.height = 480
    side_camera = StaticEnvironmentCamera(
        camera_config=side_config,
        num_envs=rl_task.num_envs,
        mesh_ids_array=mesh_ids,
        device=device
    )
    
    # Set up side view 25m away, 10m high, looking from 45 degrees
    side_camera.set_side_view(distance=25.0, height=10.0, angle_degrees=45.0)
    
    # Create image tensors for side camera
    side_pixels = torch.zeros(
        (rl_task.num_envs, 1, side_config.height, side_config.width),
        device=device, requires_grad=False
    )
    side_segmentation = torch.zeros(
        (rl_task.num_envs, 1, side_config.height, side_config.width),
        dtype=torch.int32, device=device, requires_grad=False
    )
    side_camera.set_image_tensors(side_pixels, side_segmentation)
    
    print("Static cameras created and configured!")
    
    # Create output directory for captured images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"dual_camera_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Images will be saved to: {output_dir}")
    
    # Set up real-time visualization windows
    print("Setting up real-time visualization...")
    cv2.namedWindow("Drone Camera - Depth & Segmentation", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Overhead Camera - Depth & Segmentation", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Side Camera - Depth & Segmentation", cv2.WINDOW_NORMAL)
    
    # Resize windows
    cv2.resizeWindow("Drone Camera - Depth & Segmentation", 800, 400)
    cv2.resizeWindow("Overhead Camera - Depth & Segmentation", 800, 400)
    cv2.resizeWindow("Side Camera - Depth & Segmentation", 800, 400)
    
    print("Real-time camera views enabled. Press 'q' to exit, 's' to save images.")
    
    # Initialize simple movement commands  
    command_actions = torch.zeros((rl_task.num_envs, rl_task.task_config.action_space_dim), device=device)
    
    # Set initial hovering command to prevent early crashes
    command_actions[:, 0] = 0.0  # No forward movement initially
    command_actions[:, 1] = 0.0  # No lateral movement initially  
    command_actions[:, 2] = 0.0  # No vertical movement initially
    
    # Main visualization loop
    frame_count = 0
    save_images = False
    
    def normalize_segmentation_image(seg_image):
        """Normalize segmentation image for display"""
        if np.any(seg_image > 0):
            min_positive = seg_image[seg_image > 0].min()
            seg_image[seg_image <= 0] = min_positive
        else:
            seg_image[:] = 0.1
            
        seg_normalized = (seg_image - seg_image.min()) / (seg_image.max() - seg_image.min() + 1e-8)
        return seg_normalized
    
    def create_combined_image(depth_img, seg_img, title="Camera"):
        """Create a combined depth + segmentation image"""
        # Normalize depth image
        depth_normalized = (depth_img * 255).astype(np.uint8)
        
        # Create depth image as 3-channel
        depth_rgb = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2RGB)
        
        # Normalize and colorize segmentation
        seg_normalized = normalize_segmentation_image(seg_img.copy())
        seg_colored = matplotlib.cm.plasma(seg_normalized)
        seg_rgb = (seg_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Combine images side by side
        combined = np.hstack([depth_rgb, seg_rgb])
        
        # Add title
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, f"{title} - Depth | Segmentation", (10, 30), font, 0.7, (255, 255, 255), 2)
        
        return combined
    
    try:
        while True:
            start_time = time.time()
            
            # Simple circular movement pattern (start gentle after initial frames)
            if frame_count > 50:  # Allow some time for initialization
                t = (frame_count - 50) * 0.01  # Slower movement
                command_actions[:, 0] = 0.3 * torch.cos(torch.tensor(t))  # Forward/backward
                command_actions[:, 1] = 0.2 * torch.sin(torch.tensor(t))  # Left/right
                command_actions[:, 2] = 0.1 * torch.sin(torch.tensor(t * 0.5))  # Up/down
            else:
                # Hover in place for first 50 frames
                command_actions[:, 0] = 0.0
                command_actions[:, 1] = 0.0
                command_actions[:, 2] = 0.0
            
            # Step the simulation
            obs, rewards, termination, truncation, infos = rl_task.step(command_actions)
            
            # Capture static camera images
            overhead_camera.capture()
            side_camera.capture()
            
            # Get drone camera images (from obs_dict)
            drone_depth = rl_task.obs_dict["depth_range_pixels"][0, 0].cpu().numpy()
            drone_seg = rl_task.obs_dict["segmentation_pixels"][0, 0].cpu().numpy()
            
            # Get static camera images
            overhead_depth = overhead_pixels[0, 0].cpu().numpy()
            overhead_seg = overhead_segmentation[0, 0].cpu().numpy()
            
            side_depth = side_pixels[0, 0].cpu().numpy()
            side_seg = side_segmentation[0, 0].cpu().numpy()
            
            # Create combined images for display
            drone_combined = create_combined_image(drone_depth, drone_seg, "Drone")
            overhead_combined = create_combined_image(overhead_depth, overhead_seg, "Overhead")
            side_combined = create_combined_image(side_depth, side_seg, "Side View")
            
            # Display images
            cv2.imshow("Drone Camera - Depth & Segmentation", drone_combined)
            cv2.imshow("Overhead Camera - Depth & Segmentation", overhead_combined)
            cv2.imshow("Side Camera - Depth & Segmentation", side_combined)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User requested exit")
                break
            elif key == ord('s'):
                save_images = True
            
            # Save images every 30 frames or when 's' is pressed
            if frame_count % 30 == 0 or save_images:
                print(f"Saving frame {frame_count}...")
                
                # Save individual images
                cv2.imwrite(
                    os.path.join(output_dir, f"drone_combined_frame_{frame_count:04d}.png"),
                    drone_combined
                )
                cv2.imwrite(
                    os.path.join(output_dir, f"overhead_combined_frame_{frame_count:04d}.png"),
                    overhead_combined
                )
                cv2.imwrite(
                    os.path.join(output_dir, f"side_combined_frame_{frame_count:04d}.png"),
                    side_combined
                )
                
                # Save raw depth and segmentation data
                np.save(
                    os.path.join(output_dir, f"drone_depth_frame_{frame_count:04d}.npy"),
                    drone_depth
                )
                np.save(
                    os.path.join(output_dir, f"drone_seg_frame_{frame_count:04d}.npy"),
                    drone_seg
                )
                
                save_images = False
            
            # Reset environments if needed
            reset_ids = (termination + truncation).nonzero(as_tuple=True)
            if torch.any(termination):
                terminated_envs = termination.nonzero(as_tuple=True)
                print(f"Resetting environments {terminated_envs} due to termination")
                rl_task.reset()
            if torch.any(truncation):
                truncated_envs = truncation.nonzero(as_tuple=True)
                print(f"Resetting environments {truncated_envs} due to timeout")
                rl_task.reset()
            
            frame_count += 1
            
            # Print FPS info every 100 frames
            if frame_count % 100 == 0:
                end_time = time.time()
                fps = 100 / (end_time - start_time) if end_time > start_time else 0
                print(f"Frame {frame_count}, FPS: {fps:.1f}")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        # Clean up
        cv2.destroyAllWindows()
        print(f"Demo completed. Output saved to: {output_dir}")
        print("Summary:")
        print(f"- Total frames processed: {frame_count}")
        print(f"- Drone camera: Depth + Segmentation from onboard sensor")
        print(f"- Static cameras: Overhead and side view perspectives")
        print(f"- All images saved to: {output_dir}")


if __name__ == "__main__":
    main() 