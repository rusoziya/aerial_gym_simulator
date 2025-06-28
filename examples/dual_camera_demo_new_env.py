#!/usr/bin/env python3

"""
Dual Drone Camera Demo: Dense Forest Environment with Two Stationary Drones

This example demonstrates:
1. A dense forest environment with multiple trees, objects, and ground plane
2. Four static cameras positioned at environment boundaries, all facing center
3. TWO stationary drones that don't move (hover in place at different positions)
4. Real-time visualization of depth and segmentation outputs from all 4 boundary cameras
5. Onboard camera feeds from both drones showing their individual perspectives
6. No RL training - just visualization of camera outputs

Features:
- Dense forest environment with 6 trees
- Ground plane with scattered objects and multiple trees
- No walls, just trees, objects and ground
- 4 boundary cameras at congruent marks (N, S, E, W) all facing center
- 2 stationary drones with different IDs positioned at different locations:
  * Drone 1: (-2, 0, 1.5) - left side
  * Drone 2: (2, 0, 1.5) - right side
- Each drone has its own onboard camera feed
- Dual-drone surveillance system with 6 total camera views
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
    print("=" * 80)
    print("DUAL DRONE CAMERA DEMO: Dense Forest Environment with Two Stationary Drones")
    print("=" * 80)
    
    # Import Aerial Gym components FIRST (before torch) to avoid IsaacGym conflicts
    print("Importing Aerial Gym components...")
    from aerial_gym.registry.task_registry import task_registry
    from aerial_gym.task.position_setpoint_task.position_setpoint_task import PositionSetpointTask
    from aerial_gym.sensors.warp.static_environment_camera import (
        StaticEnvironmentCamera,
        OverheadCameraConfig,
        SideViewCameraConfig
    )
    
    # Import torch AFTER aerial gym components
    import torch
    
    print("All imports successful!")
    
    # Create custom environment configuration for dual drones
    print("Creating dual drone forest environment configuration...")
    
    # Get the base position setpoint task configuration and modify it
    task_config = task_registry.get_task_config("position_setpoint_task")
    
    # Configure for dual drone setup in forest environment
    task_config.env_name = "forest_env"  # Dense forest environment
    task_config.num_envs = 2  # 2 environments = 2 drones
    task_config.sim_name = "base_sim"
    task_config.robot_name = "lmf2"
    task_config.controller_name = "lmf2_velocity_control"
    task_config.headless = False  # Enable Isaac Gym viewer
    task_config.use_warp = True
    task_config.episode_len_steps = 10000  # Long episode for demo
    
    print(f"Task configuration set up for dual drone forest environment")
    print(f"  - Number of environments (drones): {task_config.num_envs}")
    print(f"  - Environment type: Dense forest with 6 trees")
    print(f"  - Robot type: {task_config.robot_name}")
    
    # Create the position setpoint task with dual drones
    print("Creating position setpoint task with dual drone setup...")
    use_warp = True
    headless = False  # Enable Isaac Gym viewer to see the simulation
    
    rl_task = task_registry.make_task(
        "position_setpoint_task",
        seed=42,
        use_warp=use_warp,
        headless=headless
    )
    
    print(f"Created task with {rl_task.num_envs} environments (drones)")
    device = rl_task.device
    
    # Initialize the task to ensure all tensors are properly set up
    print("Initializing task and resetting environments...")
    rl_task.reset()
    
    # Manually set drone positions to ensure they are at different locations
    print("Setting specific drone positions...")
    
    # Define desired drone positions: Drone 1 left, Drone 2 right
    drone_1_pos = np.array([-2.0, 0.0, 1.5])  # Left side
    drone_2_pos = np.array([2.0, 0.0, 1.5])   # Right side
    
    # Access the root states and manually set positions
    if hasattr(rl_task, 'root_states') and rl_task.root_states is not None:
        # Environment 0: Drone 1 at left position
        rl_task.root_states[0, 0] = drone_1_pos[0]  # X position
        rl_task.root_states[0, 1] = drone_1_pos[1]  # Y position  
        rl_task.root_states[0, 2] = drone_1_pos[2]  # Z position
        
        # Environment 1: Drone 2 at right position  
        rl_task.root_states[1, 0] = drone_2_pos[0]  # X position
        rl_task.root_states[1, 1] = drone_2_pos[1]  # Y position
        rl_task.root_states[1, 2] = drone_2_pos[2]  # Z position
        
        # Reset velocities to zero for both drones
        rl_task.root_states[0, 7:10] = 0.0  # Linear velocity for drone 1
        rl_task.root_states[0, 10:13] = 0.0  # Angular velocity for drone 1
        rl_task.root_states[1, 7:10] = 0.0  # Linear velocity for drone 2
        rl_task.root_states[1, 10:13] = 0.0  # Angular velocity for drone 2
        
        print(f"✅ Set Drone 1 position to: {drone_1_pos}")
        print(f"✅ Set Drone 2 position to: {drone_2_pos}")
        
        # Also try to update robot manager state if available
        if hasattr(rl_task.sim_env, 'robot_manager') and hasattr(rl_task.sim_env.robot_manager, 'robot'):
            robot = rl_task.sim_env.robot_manager.robot
            if hasattr(robot, 'robot_state') and robot.robot_state is not None:
                robot.robot_state[0, :3] = torch.from_numpy(drone_1_pos).to(device)
                robot.robot_state[1, :3] = torch.from_numpy(drone_2_pos).to(device)
                robot.robot_state[0, 7:10] = 0.0  # Zero velocity
                robot.robot_state[0, 10:13] = 0.0
                robot.robot_state[1, 7:10] = 0.0  # Zero velocity  
                robot.robot_state[1, 10:13] = 0.0
                print("✅ Updated robot manager states as well")
    
    # Get drone positions for camera targeting
    print("Getting drone positions for camera targeting...")
    drone_positions = []
    
    if hasattr(rl_task, 'root_states') and rl_task.root_states is not None:
        # root_states contains [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
        for env_idx in range(rl_task.num_envs):
            drone_pos = rl_task.root_states[env_idx, :3].cpu().numpy()  # position (x,y,z)
            drone_positions.append(drone_pos)
            print(f"Drone {env_idx + 1} position: {drone_pos}")
    else:
        # Use the manually set positions as defaults
        drone_positions = [drone_1_pos, drone_2_pos]
        print(f"Using manually set drone positions:")
        for i, pos in enumerate(drone_positions):
            print(f"  Drone {i + 1}: {pos}")
    
    # Check if we have access to warp mesh IDs
    print("Checking for Warp mesh IDs...")
    try:
        mesh_ids_list = rl_task.sim_env.global_tensor_dict.get("CONST_WARP_MESH_ID_LIST")
        if mesh_ids_list is None or len(mesh_ids_list) == 0:
            print("WARNING: No Warp mesh IDs found. Creating dummy mesh ID list...")
            mesh_ids_list = [0]  # Single dummy mesh ID
        
        print(f"Mesh IDs type: {type(mesh_ids_list)}, length: {len(mesh_ids_list) if hasattr(mesh_ids_list, '__len__') else 'N/A'}")
        
        # Convert mesh_ids list to warp array
        import warp as wp
        mesh_ids = wp.array(mesh_ids_list, dtype=wp.uint64, device=device)
        print(f"Converted mesh IDs to warp array: {mesh_ids.shape}")
    except Exception as e:
        print(f"Error accessing mesh IDs: {e}")
        print("Creating dummy mesh IDs for camera setup...")
        import warp as wp
        mesh_ids = wp.array([0], dtype=wp.uint64, device=device)
    
    # Create static environment cameras and onboard drone cameras
    print("Creating static environment cameras and onboard drone cameras...")
    
    # Camera configurations and storage
    static_cameras = []
    camera_pixels = []
    camera_segmentations = []
    camera_names = []
    
    # Standard camera config
    def create_camera_config():
        config = OverheadCameraConfig()
        config.width = 480
        config.height = 480
        config.horizontal_fov_deg = 80.0  # Good field of view for drone observation
        return config
    
    try:
        # === 1. BOUNDARY CAMERAS (4 cameras) ===
        # Define 4 cameras inside environment bounds at congruent positions
        # Environment bounds: [-5, -5] to [5, 5], position cameras inside bounds
        # All cameras at same height, facing perpendicular towards center (0, 0, 1.5)
        
        camera_height = 2.0  # Same height for all cameras
        boundary_offset = 1.0  # Position cameras 1m inside the boundaries for visibility
        
        # 1. North Camera - north side inside bounds, facing south
        north_config = create_camera_config()
        north_camera = StaticEnvironmentCamera(
            camera_config=north_config,
            num_envs=1,  # Single environment perspective for boundary cameras
            mesh_ids_array=mesh_ids,
            device=device
        )
        
        north_positions = torch.zeros((1, 1, 3), device=device)
        north_positions[:, 0, 0] = 0.0      # Middle of X axis
        north_positions[:, 0, 1] = 5.0 - boundary_offset  # North side, inside bounds
        north_positions[:, 0, 2] = camera_height
        
        # Quaternion for looking south (negative Y direction)
        north_orientations = torch.zeros((1, 1, 4), device=device)
        north_orientations[:, 0, 0] = 0.0   # x
        north_orientations[:, 0, 1] = 0.0   # y
        north_orientations[:, 0, 2] = 1.0   # z (180 degree rotation)
        north_orientations[:, 0, 3] = 0.0   # w
        
        north_camera.set_camera_poses(north_positions, north_orientations)
        north_pixels = torch.zeros((1, 1, 480, 480), device=device, requires_grad=False)
        north_segmentation = torch.zeros((1, 1, 480, 480), dtype=torch.int32, device=device, requires_grad=False)
        north_camera.set_image_tensors(north_pixels, north_segmentation)
        
        static_cameras.append(north_camera)
        camera_pixels.append(north_pixels)
        camera_segmentations.append(north_segmentation)
        camera_names.append("North Boundary")
        
        # 2. South Camera - south side inside bounds, facing north
        south_config = create_camera_config()
        south_camera = StaticEnvironmentCamera(
            camera_config=south_config,
            num_envs=1,
            mesh_ids_array=mesh_ids,
            device=device
        )
        
        south_positions = torch.zeros((1, 1, 3), device=device)
        south_positions[:, 0, 0] = 0.0      # Middle of X axis
        south_positions[:, 0, 1] = -5.0 + boundary_offset  # South side, inside bounds
        south_positions[:, 0, 2] = camera_height
        
        # Quaternion for looking north (positive Y direction) - no rotation needed
        south_orientations = torch.zeros((1, 1, 4), device=device)
        south_orientations[:, 0, 0] = 0.0   # x
        south_orientations[:, 0, 1] = 0.0   # y
        south_orientations[:, 0, 2] = 0.0   # z (no rotation)
        south_orientations[:, 0, 3] = 1.0   # w
        
        south_camera.set_camera_poses(south_positions, south_orientations)
        south_pixels = torch.zeros((1, 1, 480, 480), device=device, requires_grad=False)
        south_segmentation = torch.zeros((1, 1, 480, 480), dtype=torch.int32, device=device, requires_grad=False)
        south_camera.set_image_tensors(south_pixels, south_segmentation)
        
        static_cameras.append(south_camera)
        camera_pixels.append(south_pixels)
        camera_segmentations.append(south_segmentation)
        camera_names.append("South Boundary")
        
        # 3. East Camera - east side inside bounds, facing west
        east_config = create_camera_config()
        east_camera = StaticEnvironmentCamera(
            camera_config=east_config,
            num_envs=1,
            mesh_ids_array=mesh_ids,
            device=device
        )
        
        east_positions = torch.zeros((1, 1, 3), device=device)
        east_positions[:, 0, 0] = 5.0 - boundary_offset   # East side, inside bounds
        east_positions[:, 0, 1] = 0.0      # Middle of Y axis
        east_positions[:, 0, 2] = camera_height
        
        # Quaternion for looking west (negative X direction) - 90 degree rotation around Z
        east_orientations = torch.zeros((1, 1, 4), device=device)
        east_orientations[:, 0, 0] = 0.0    # x
        east_orientations[:, 0, 1] = 0.0    # y
        east_orientations[:, 0, 2] = 0.707  # z (90 degree rotation)
        east_orientations[:, 0, 3] = 0.707  # w
        
        east_camera.set_camera_poses(east_positions, east_orientations)
        east_pixels = torch.zeros((1, 1, 480, 480), device=device, requires_grad=False)
        east_segmentation = torch.zeros((1, 1, 480, 480), dtype=torch.int32, device=device, requires_grad=False)
        east_camera.set_image_tensors(east_pixels, east_segmentation)
        
        static_cameras.append(east_camera)
        camera_pixels.append(east_pixels)
        camera_segmentations.append(east_segmentation)
        camera_names.append("East Boundary")
        
        # 4. West Camera - west side inside bounds, facing east  
        west_config = create_camera_config()
        west_camera = StaticEnvironmentCamera(
            camera_config=west_config,
            num_envs=1,
            mesh_ids_array=mesh_ids,
            device=device
        )
        
        west_positions = torch.zeros((1, 1, 3), device=device)
        west_positions[:, 0, 0] = -5.0 + boundary_offset  # West side, inside bounds
        west_positions[:, 0, 1] = 0.0      # Middle of Y axis
        west_positions[:, 0, 2] = camera_height
        
        # Quaternion for looking east (positive X direction) - -90 degree rotation around Z
        west_orientations = torch.zeros((1, 1, 4), device=device)
        west_orientations[:, 0, 0] = 0.0    # x
        west_orientations[:, 0, 1] = 0.0    # y
        west_orientations[:, 0, 2] = -0.707 # z (-90 degree rotation)
        west_orientations[:, 0, 3] = 0.707  # w
        
        west_camera.set_camera_poses(west_positions, west_orientations)
        west_pixels = torch.zeros((1, 1, 480, 480), device=device, requires_grad=False)
        west_segmentation = torch.zeros((1, 1, 480, 480), dtype=torch.int32, device=device, requires_grad=False)
        west_camera.set_image_tensors(west_pixels, west_segmentation)
        
        static_cameras.append(west_camera)
        camera_pixels.append(west_pixels)
        camera_segmentations.append(west_segmentation)
        camera_names.append("West Boundary")
        
        # === 2. ONBOARD DRONE CAMERAS (2 cameras) ===
        # Create onboard cameras for each drone, looking forward
        
        for drone_idx in range(rl_task.num_envs):
            drone_pos = drone_positions[drone_idx]
            
            # Create onboard camera for this drone
            onboard_config = create_camera_config()
            onboard_config.width = 320  # Smaller resolution for onboard cameras
            onboard_config.height = 240
            onboard_config.horizontal_fov_deg = 60.0  # Slightly narrower FOV for onboard view
            
            onboard_camera = StaticEnvironmentCamera(
                camera_config=onboard_config,
                num_envs=1,  # Each drone camera is independent
                mesh_ids_array=mesh_ids,
                device=device
            )
            
            # Position camera slightly in front of and above the drone
            onboard_positions = torch.zeros((1, 1, 3), device=device)
            onboard_positions[:, 0, 0] = drone_pos[0] + 0.3  # 30cm forward from drone
            onboard_positions[:, 0, 1] = drone_pos[1]        # Same Y position
            onboard_positions[:, 0, 2] = drone_pos[2] + 0.1  # 10cm above drone
            
            # Quaternion for looking forward (positive X direction for drone's forward view)
            onboard_orientations = torch.zeros((1, 1, 4), device=device)
            onboard_orientations[:, 0, 0] = 0.0   # x
            onboard_orientations[:, 0, 1] = 0.0   # y
            onboard_orientations[:, 0, 2] = 0.0   # z (no rotation - looking forward)
            onboard_orientations[:, 0, 3] = 1.0   # w
            
            onboard_camera.set_camera_poses(onboard_positions, onboard_orientations)
            onboard_pixels = torch.zeros((1, 1, 240, 320), device=device, requires_grad=False)
            onboard_segmentation = torch.zeros((1, 1, 240, 320), dtype=torch.int32, device=device, requires_grad=False)
            onboard_camera.set_image_tensors(onboard_pixels, onboard_segmentation)
        
            static_cameras.append(onboard_camera)
            camera_pixels.append(onboard_pixels)
            camera_segmentations.append(onboard_segmentation)
            camera_names.append(f"Drone {drone_idx + 1} Onboard")
        
        print("Static environment cameras and onboard drone cameras created successfully!")
        
        # Camera positions information
        print("📍 CAMERA SYSTEM OVERVIEW:")
        print("   BOUNDARY CAMERAS (4x 480x480):")
        print("   • North Camera: (0, 4.0, 2.0) - looking SOUTH toward center")
        print("     🟢 GREEN CUBE marker at (0, 3.0, 2.5)")
        print("   • South Camera: (0, -4.0, 2.0) - looking NORTH toward center")
        print("     🔴 RED CUBE marker at (0, -3.0, 2.5)") 
        print("   • East Camera: (4.0, 0, 2.0) - looking WEST toward center")
        print("     🔵 BLUE CUBE marker at (3.0, 0, 2.5)")
        print("   • West Camera: (-4.0, 0, 2.0) - looking EAST toward center")
        print("     🟡 YELLOW CUBE marker at (-3.0, 0, 2.5)")
        print("")
        print("   ONBOARD DRONE CAMERAS (2x 320x240):")
        for i, pos in enumerate(drone_positions):
            print(f"   • Drone {i+1} Onboard: ({pos[0]+0.3:.1f}, {pos[1]:.1f}, {pos[2]+0.1:.1f}) - looking FORWARD")
        print("")
        print("   DRONE POSITIONS:")
        for i, pos in enumerate(drone_positions):
            print(f"   • Drone {i+1}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        print("")
        print("   Look for the COLORED CUBE MARKERS in the 3D viewer!")
        
    except Exception as e:
        print(f"Error creating cameras: {e}")
        print("Continuing with empty camera list - you'll only see the Isaac Gym viewer.")
        static_cameras = []
        camera_pixels = []
        camera_segmentations = []
        camera_names = []
    
    # Create output directory for captured images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"dual_drone_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Images will be saved to: {output_dir}")
    
    # Set up real-time visualization windows only if we have cameras
    if static_cameras:
        print("Setting up real-time visualization with 6 camera windows...")
        cv2.namedWindow("North Boundary - Depth & Segmentation", cv2.WINDOW_NORMAL)
        cv2.namedWindow("South Boundary - Depth & Segmentation", cv2.WINDOW_NORMAL)
        cv2.namedWindow("East Boundary - Depth & Segmentation", cv2.WINDOW_NORMAL)
        cv2.namedWindow("West Boundary - Depth & Segmentation", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Drone 1 Onboard - Depth & Segmentation", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Drone 2 Onboard - Depth & Segmentation", cv2.WINDOW_NORMAL)
        
        # Resize windows for better layout
        cv2.resizeWindow("North Boundary - Depth & Segmentation", 800, 400)
        cv2.resizeWindow("South Boundary - Depth & Segmentation", 800, 400)
        cv2.resizeWindow("East Boundary - Depth & Segmentation", 800, 400)
        cv2.resizeWindow("West Boundary - Depth & Segmentation", 800, 400)
        cv2.resizeWindow("Drone 1 Onboard - Depth & Segmentation", 640, 240)
        cv2.resizeWindow("Drone 2 Onboard - Depth & Segmentation", 640, 240)
        
        print("Real-time camera views enabled. Press 'q' to exit, 's' to save images.")
    else:
        print("No cameras available - only Isaac Gym viewer will be shown.")
        print("You can still see both stationary drones in the Isaac Gym viewer window.")
        print("Press 'q' in the console to exit.")
    
    print("Both drones will remain stationary (hover in place)")
    
    # Initialize stationary hover commands - both drones stay in place
    command_actions = torch.zeros((rl_task.num_envs, rl_task.task_config.action_space_dim), device=device)
    
    # Set to hover commands (all zeros for stationary hover)
    for drone_idx in range(rl_task.num_envs):
        command_actions[drone_idx, 0] = 0.0  # No forward movement
        command_actions[drone_idx, 1] = 0.0  # No lateral movement  
        command_actions[drone_idx, 2] = 0.0  # No vertical movement
        command_actions[drone_idx, 3] = 0.0  # No yaw rotation
    
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
    
    print("\nStarting main loop...")
    print("Watch the Isaac Gym viewer window to see both stationary drones!")
    if static_cameras:
        print("Additional camera windows should show:")
        print("  - 4 boundary cameras showing overview of both drones")
        print("  - 2 onboard cameras showing each drone's forward view")
    
    try:
        while True:
            start_time = time.time()
            
            # Keep both drones stationary - no movement commands
            # Actions remain at zero for hover
            
            # Step the simulation with stationary commands
            obs, rewards, termination, truncation, infos = rl_task.step(command_actions)
            
            # Only process cameras if they exist and are working
            if static_cameras:
                try:
                    # Capture all static camera images
                    for camera in static_cameras:
                        camera.capture()
                    
                    # Get static camera images
                    camera_images = []
                    for i, (pixels, segmentation, name) in enumerate(zip(camera_pixels, camera_segmentations, camera_names)):
                        depth = pixels[0, 0].cpu().numpy()
                        seg = segmentation[0, 0].cpu().numpy()
                        combined = create_combined_image(depth, seg, name)
                        camera_images.append(combined)
                    
                    # Display all images
                    cv2.imshow("North Boundary - Depth & Segmentation", camera_images[0])
                    cv2.imshow("South Boundary - Depth & Segmentation", camera_images[1])
                    cv2.imshow("East Boundary - Depth & Segmentation", camera_images[2])
                    cv2.imshow("West Boundary - Depth & Segmentation", camera_images[3])
                    cv2.imshow("Drone 1 Onboard - Depth & Segmentation", camera_images[4])
                    cv2.imshow("Drone 2 Onboard - Depth & Segmentation", camera_images[5])
                    
                    # Save images if requested
                    if save_images:
                        # Save all cameras
                        for i, (combined, name) in enumerate(zip(camera_images, camera_names)):
                            safe_name = name.lower().replace(" ", "_")
                            cv2.imwrite(f"{output_dir}/{safe_name}_combined_frame_{frame_count:04d}.png", combined)
                        
                        print(f"Saved frame {frame_count}")
                    
                    # Handle keyboard input for camera windows
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quit requested by user")
                        break
                    elif key == ord('s'):
                        save_images = not save_images
                        print(f"Image saving {'enabled' if save_images else 'disabled'}")
                        
                except Exception as e:
                    print(f"Error processing cameras: {e}")
                    # Continue the simulation loop even if cameras fail
            else:
                # No cameras - just run simulation and check for quit in console
                # Check for keyboard interrupt to quit
                pass
            
            frame_count += 1
            
            # Print frame rate every 100 frames
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
                print(f"Frame {frame_count}, FPS: {fps:.1f} - Both drones hovering in place")
                if not static_cameras:
                    print("  (No camera windows - check Isaac Gym viewer for dual drone visualization)")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        print("Cleaning up...")
        if static_cameras:
            cv2.destroyAllWindows()
        print(f"Demo completed. {frame_count} frames processed.")
        if save_images and static_cameras:
            print(f"Images saved to: {output_dir}")

if __name__ == "__main__":
    main()
