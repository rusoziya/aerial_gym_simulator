#!/usr/bin/env python3

"""
Gate Camera Demo: Static Camera Behind Gate + Drone Navigation

This example demonstrates:
1. A static camera positioned behind the gate with full view of gate panels
2. A drone with an onboard camera navigating through the gate
3. Real-time visualization of depth and segmentation outputs from both cameras
4. Uses proper gate environment configuration
5. No RL training - just visualization of gate navigation

The demo focuses specifically on gate observation and drone-gate interaction.
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
import math

def main():
    print("=" * 80)
    print("GATE CAMERA DEMO: Static Gate View Camera + Drone Navigation")
    print("=" * 80)
    
    # Import Aerial Gym components FIRST (before torch) to avoid IsaacGym conflicts
    print("Importing Aerial Gym components...")
    from aerial_gym.registry.task_registry import task_registry
    from aerial_gym.task.navigation_task.navigation_task import NavigationTask
    from aerial_gym.config.env_config.gate_env import GateEnvCfg
    from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg
    from aerial_gym.config.controller_config.lee_controller_config import LeeControllerCfg
    from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import BaseDepthCameraCfg
    from aerial_gym.config.sim_config.base_sim_config import BaseSimConfig
    from aerial_gym.sensors.warp.static_environment_camera import (
        StaticEnvironmentCamera,
        OverheadCameraConfig,
        SideViewCameraConfig
    )
    
    # Import torch AFTER aerial gym components
    import torch
    
    print("All imports successful!")
    
    # Create a custom gate task configuration
    class GateNavigationTaskConfig:
        def __init__(self):
            self.task = NavigationTask
            self.env_config = GateEnvCfg
            self.robot_config = BaseQuadCfg
            self.controller_config = LeeControllerCfg
            self.sensor_config = BaseDepthCameraCfg
            self.sim_config = BaseSimConfig
            
            # Override some configurations for gate navigation
            self.env_config.env.use_warp = True  # Enable Warp for static cameras
            self.env_config.env.num_envs = 1     # Single environment for demo
            self.sim_config.use_warp = True      # Enable Warp in sim config too
            self.sensor_config.enable_camera = True
            
    # Create and register the gate navigation task
    gate_task_config = GateNavigationTaskConfig()
    
    task_registry.register_task(
        task_name="gate_camera_demo_task",
        task_class=NavigationTask,
        task_config=gate_task_config,
    )
    
    # Create the gate navigation task with camera enabled
    print("Creating gate navigation task with drone camera...")
    use_warp = True
    headless = True  # Use headless mode since we have our own visualization
    
    rl_task = task_registry.make_task(
        "gate_camera_demo_task",
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
    
    # 1. Gate View Camera - positioned behind the gate with full view of gate panels
    print("Creating gate view camera positioned behind the gate...")
    gate_config = SideViewCameraConfig()
    gate_config.width = 640  # Higher resolution for detailed gate view
    gate_config.height = 480
    gate_config.horizontal_fov_deg = 60.0  # Narrower FOV for focused gate view
    
    gate_camera = StaticEnvironmentCamera(
        camera_config=gate_config,
        num_envs=rl_task.num_envs,
        mesh_ids_array=mesh_ids,
        device=device
    )
    
    # Position camera behind the gate (gate is at center 0,0,0)
    # Gate has scale 1.0 by default, so it's roughly 2m wide x 2m tall
    # Position camera 6m behind the gate, at height 1.5m (middle of gate height)
    # This should give a full frontal view of the gate panels
    gate_positions = torch.zeros((rl_task.num_envs, 1, 3), device=device)
    gate_positions[:, 0, 0] = 0.0   # Centered on X-axis with gate
    gate_positions[:, 0, 1] = -6.0  # 6m behind the gate (negative Y direction)
    gate_positions[:, 0, 2] = 1.5   # At height 1.5m (middle of gate height)
    
    # Create orientation quaternion to look forward toward the gate (positive Y direction)
    # The gate should be directly in front of the camera
    gate_orientations = torch.zeros((rl_task.num_envs, 1, 4), device=device)
    gate_orientations[:, 0, 0] = 0.0   # x component
    gate_orientations[:, 0, 1] = 0.0   # y component  
    gate_orientations[:, 0, 2] = 0.0   # z component (no rotation around Z)
    gate_orientations[:, 0, 3] = 1.0   # w component (identity quaternion)
    
    gate_camera.set_camera_poses(gate_positions, gate_orientations)
    
    # Create image tensors for gate camera
    gate_pixels = torch.zeros(
        (rl_task.num_envs, 1, gate_config.height, gate_config.width),
        device=device, requires_grad=False
    )
    gate_segmentation = torch.zeros(
        (rl_task.num_envs, 1, gate_config.height, gate_config.width),
        dtype=torch.int32, device=device, requires_grad=False
    )
    gate_camera.set_image_tensors(gate_pixels, gate_segmentation)
    
    print("Gate view camera positioned for full frontal view of gate panels!")
    print(f"Gate camera position: {gate_positions[0, 0].cpu().numpy()}")
    print(f"Gate camera orientation: {gate_orientations[0, 0].cpu().numpy()}")
    
    # 2. Overhead camera for context
    overhead_config = OverheadCameraConfig()
    overhead_config.width = 480
    overhead_config.height = 480
    overhead_camera = StaticEnvironmentCamera(
        camera_config=overhead_config,
        num_envs=rl_task.num_envs,
        mesh_ids_array=mesh_ids,
        device=device
    )
    
    # Set up overhead view 15m above the environment center
    overhead_camera.set_overhead_view(height=15.0)
    
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
    
    print("Static cameras created and configured!")
    
    # Create output directory for captured images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"gate_camera_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Images will be saved to: {output_dir}")
    
    # Set up real-time visualization windows
    print("Setting up real-time visualization...")
    cv2.namedWindow("Drone Camera - Depth & Segmentation", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Gate View Camera - Depth & Segmentation", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Overhead Camera - Depth & Segmentation", cv2.WINDOW_NORMAL)
    
    # Resize windows
    cv2.resizeWindow("Drone Camera - Depth & Segmentation", 800, 400)
    cv2.resizeWindow("Gate View Camera - Depth & Segmentation", 1000, 400)
    cv2.resizeWindow("Overhead Camera - Depth & Segmentation", 600, 600)
    
    print("Real-time camera views enabled. Press 'q' to exit, 's' to save images.")
    print("Flight pattern: Approach gate -> Fly through -> Return")
    
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
            
            # Enhanced gate navigation pattern
            if frame_count > 50:  # Allow some time for initialization
                t = (frame_count - 50) * 0.015  # Moderate speed movement
                
                # Create a flight pattern that specifically focuses on gate navigation
                phase = t % (4 * math.pi)  # Longer cycle for complete gate traversal
                
                if phase < math.pi:  # First phase: approach gate from behind camera
                    # Start from behind and approach the gate
                    command_actions[:, 0] = 0.4 * torch.sin(torch.tensor(phase * 0.5))  # Gradual approach
                    command_actions[:, 1] = 0.6 * torch.cos(torch.tensor(phase * 0.3))  # Side movement for better camera view
                    command_actions[:, 2] = 0.2 * torch.sin(torch.tensor(phase * 0.7))  # Height variation
                elif phase < 2 * math.pi:  # Second phase: fly through gate
                    # Steady movement through the gate
                    command_actions[:, 0] = 0.5  # Steady forward movement through gate
                    command_actions[:, 1] = 0.0  # Straight through center
                    command_actions[:, 2] = 0.0  # Maintain height
                elif phase < 3 * math.pi:  # Third phase: continue past gate  
                    # Continue past gate and maneuver
                    command_actions[:, 0] = 0.3 * torch.cos(torch.tensor(phase))  # Slow down
                    command_actions[:, 1] = 0.4 * torch.sin(torch.tensor(phase))  # Side movement
                    command_actions[:, 2] = -0.2 * torch.sin(torch.tensor(phase * 0.5))  # Descent
                else:  # Fourth phase: return through gate
                    # Return back through gate
                    command_actions[:, 0] = -0.4  # Move back toward gate
                    command_actions[:, 1] = 0.2 * torch.sin(torch.tensor(phase * 2))  # Slight weaving
                    command_actions[:, 2] = 0.1 * torch.cos(torch.tensor(phase))  # Height adjustment
            else:
                # Hover in place for first 50 frames
                command_actions[:, 0] = 0.0
                command_actions[:, 1] = 0.0
                command_actions[:, 2] = 0.0
            
            # Step the simulation
            obs, rewards, termination, truncation, infos = rl_task.step(command_actions)
            
            # Capture static camera images
            gate_camera.capture()  # Gate view is primary focus
            overhead_camera.capture()
            
            # Get drone camera images (from obs_dict)
            drone_depth = rl_task.obs_dict["depth_range_pixels"][0, 0].cpu().numpy()
            drone_seg = rl_task.obs_dict["segmentation_pixels"][0, 0].cpu().numpy()
            
            # Get static camera images
            gate_depth = gate_pixels[0, 0].cpu().numpy()
            gate_seg = gate_segmentation[0, 0].cpu().numpy()
            
            overhead_depth = overhead_pixels[0, 0].cpu().numpy()
            overhead_seg = overhead_segmentation[0, 0].cpu().numpy()
            
            # Create combined images for display
            drone_combined = create_combined_image(drone_depth, drone_seg, "Drone View")
            gate_combined = create_combined_image(gate_depth, gate_seg, "Gate View (Behind)")
            overhead_combined = create_combined_image(overhead_depth, overhead_seg, "Overhead")
            
            # Display images with gate view as primary focus
            cv2.imshow("Gate View Camera - Depth & Segmentation", gate_combined)
            cv2.imshow("Drone Camera - Depth & Segmentation", drone_combined)
            cv2.imshow("Overhead Camera - Depth & Segmentation", overhead_combined)
            
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
                    os.path.join(output_dir, f"gate_view_combined_frame_{frame_count:04d}.png"),
                    gate_combined
                )
                cv2.imwrite(
                    os.path.join(output_dir, f"drone_combined_frame_{frame_count:04d}.png"),
                    drone_combined
                )
                cv2.imwrite(
                    os.path.join(output_dir, f"overhead_combined_frame_{frame_count:04d}.png"),
                    overhead_combined
                )
                
                # Save raw depth and segmentation data
                np.save(
                    os.path.join(output_dir, f"gate_depth_frame_{frame_count:04d}.npy"),
                    gate_depth
                )
                np.save(
                    os.path.join(output_dir, f"gate_seg_frame_{frame_count:04d}.npy"),
                    gate_seg
                )
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
            
            # Print FPS and drone position info every 100 frames
            if frame_count % 100 == 0:
                end_time = time.time()
                fps = 100 / (end_time - start_time) if end_time > start_time else 0
                
                # Get drone position for debugging
                drone_pos = obs["robot_position"][0].cpu().numpy() if "robot_position" in obs else [0, 0, 0]
                
                print(f"Frame {frame_count}, FPS: {fps:.1f}, Drone position: [{drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}]")
                
                # Print gate camera info and analysis
                if frame_count == 100:
                    print(f"Gate camera view active - showing full frontal view of gate panels")
                    print(f"Gate semantic ID in segmentation should be: 10")
                    print(f"Looking for gate at center (0, 0) in environment")
                    
                    # Analyze gate segmentation
                    gate_pixels_count = np.sum(gate_seg == 10)  # Gate semantic ID is 10
                    total_pixels = gate_seg.shape[0] * gate_seg.shape[1]
                    gate_coverage = gate_pixels_count / total_pixels * 100
                    print(f"Gate coverage in view: {gate_coverage:.1f}% of image")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        # Clean up
        cv2.destroyAllWindows()
        print(f"Demo completed. Output saved to: {output_dir}")
        print("Summary:")
        print(f"- Total frames processed: {frame_count}")
        print(f"- Primary focus: Gate view camera positioned behind gate")
        print(f"- Secondary views: Drone camera and overhead perspective")
        print(f"- Gate navigation: Approach, traverse, and return flight pattern")
        print(f"- Gate semantic ID: 10 (visible in segmentation)")
        print(f"- All images and data saved to: {output_dir}")


if __name__ == "__main__":
    main() 