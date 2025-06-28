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
    
    # Get the base navigation task configuration and modify it
    task_config = task_registry.get_task_config("navigation_task")
    
    # Disable VAE processing - we just want raw camera outputs for the demo
    task_config.vae_config.use_vae = False
    # Adjust observation space dimension since we're not using VAE latents (64D)
    task_config.observation_space_dim = 13 + 4  # root_state + action_dim (without latent_dims)
    print(f"VAE processing disabled: use_vae = {task_config.vae_config.use_vae}")
    print(f"Observation space dimension adjusted to: {task_config.observation_space_dim}")
    
    # Register the DCE navigation task (which has camera enabled)
    task_registry.register_task(
        task_name="dual_camera_demo_task",
        task_class=DCE_RL_Navigation_Task,
        task_config=task_config,
    )
    
    # Create the navigation task with camera enabled
    print("Creating navigation task with drone camera...")
    use_warp = True
    headless = False  # Enable Isaac Gym viewer to see the simulation
    
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
    
    # Create static environment cameras for comprehensive drone observation
    print("Creating multiple static environment cameras...")
    
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
        return config
    
    # 1. Primary overhead camera - center view INSIDE the middle chamber
    overhead1_config = create_camera_config()
    overhead1_camera = StaticEnvironmentCamera(
        camera_config=overhead1_config,
        num_envs=rl_task.num_envs,
        mesh_ids_array=mesh_ids,
        device=device
    )
    overhead1_camera.set_overhead_view(height=2.0)  # Much lower - closer to drone level
    overhead1_pixels = torch.zeros((rl_task.num_envs, 1, 480, 480), device=device, requires_grad=False)
    overhead1_segmentation = torch.zeros((rl_task.num_envs, 1, 480, 480), dtype=torch.int32, device=device, requires_grad=False)
    overhead1_camera.set_image_tensors(overhead1_pixels, overhead1_segmentation)
    
    static_cameras.append(overhead1_camera)
    camera_pixels.append(overhead1_pixels)
    camera_segmentations.append(overhead1_segmentation)
    camera_names.append("Overhead Center")
    
    # 2. Overhead camera - slightly off-center INSIDE the middle chamber  
    overhead2_config = create_camera_config()
    overhead2_camera = StaticEnvironmentCamera(
        camera_config=overhead2_config,
        num_envs=rl_task.num_envs,
        mesh_ids_array=mesh_ids,
        device=device
    )
    # Position offset from center, still inside the middle chamber
    positions2 = torch.zeros((rl_task.num_envs, 1, 3), device=device)
    positions2[:, 0, 0] = 1.5   # Small X offset to stay inside chamber
    positions2[:, 0, 1] = 1.5   # Small Y offset to stay inside chamber
    positions2[:, 0, 2] = 1.8   # Height much lower - closer to drone level
    overhead2_camera.set_camera_poses(positions2)  # Uses default downward orientation
    overhead2_pixels = torch.zeros((rl_task.num_envs, 1, 480, 480), device=device, requires_grad=False)
    overhead2_segmentation = torch.zeros((rl_task.num_envs, 1, 480, 480), dtype=torch.int32, device=device, requires_grad=False)
    overhead2_camera.set_image_tensors(overhead2_pixels, overhead2_segmentation)
    
    static_cameras.append(overhead2_camera)
    camera_pixels.append(overhead2_pixels)
    camera_segmentations.append(overhead2_segmentation)
    camera_names.append("Overhead Offset")
    
    # 3. Corner view - positioned INSIDE the middle chamber walls
    corner1_config = create_camera_config()
    corner1_camera = StaticEnvironmentCamera(
        camera_config=corner1_config,
        num_envs=rl_task.num_envs,
        mesh_ids_array=mesh_ids,
        device=device
    )
    corner1_camera.set_side_view(distance=2.0, height=1.5, angle_degrees=45.0)  # Much lower and closer to drone level
    corner1_pixels = torch.zeros((rl_task.num_envs, 1, 480, 480), device=device, requires_grad=False)
    corner1_segmentation = torch.zeros((rl_task.num_envs, 1, 480, 480), dtype=torch.int32, device=device, requires_grad=False)
    corner1_camera.set_image_tensors(corner1_pixels, corner1_segmentation)
    
    static_cameras.append(corner1_camera)
    camera_pixels.append(corner1_pixels)
    camera_segmentations.append(corner1_segmentation)
    camera_names.append("Corner View")
    
    # 4. Opposite corner view - positioned INSIDE the middle chamber walls
    corner2_config = create_camera_config()
    corner2_camera = StaticEnvironmentCamera(
        camera_config=corner2_config,
        num_envs=rl_task.num_envs,
        mesh_ids_array=mesh_ids,
        device=device
    )
    corner2_camera.set_side_view(distance=2.0, height=1.5, angle_degrees=225.0)  # Much lower and closer to drone level
    corner2_pixels = torch.zeros((rl_task.num_envs, 1, 480, 480), device=device, requires_grad=False)
    corner2_segmentation = torch.zeros((rl_task.num_envs, 1, 480, 480), dtype=torch.int32, device=device, requires_grad=False)
    corner2_camera.set_image_tensors(corner2_pixels, corner2_segmentation)
    
    static_cameras.append(corner2_camera)
    camera_pixels.append(corner2_pixels)
    camera_segmentations.append(corner2_segmentation)
    camera_names.append("Corner View 2")
    
    # 5. Mid-level overhead view - lower in the middle chamber
    overhead3_config = create_camera_config()
    overhead3_camera = StaticEnvironmentCamera(
        camera_config=overhead3_config,
        num_envs=rl_task.num_envs,
        mesh_ids_array=mesh_ids,
        device=device
    )
    overhead3_camera.set_overhead_view(height=1.2)  # Very low - almost at drone level
    overhead3_pixels = torch.zeros((rl_task.num_envs, 1, 480, 480), device=device, requires_grad=False)
    overhead3_segmentation = torch.zeros((rl_task.num_envs, 1, 480, 480), dtype=torch.int32, device=device, requires_grad=False)
    overhead3_camera.set_image_tensors(overhead3_pixels, overhead3_segmentation)
    
    static_cameras.append(overhead3_camera)
    camera_pixels.append(overhead3_pixels)
    camera_segmentations.append(overhead3_segmentation)
    camera_names.append("Overhead Close")
    
    # 6. Side view - positioned near chamber wall but looking inward
    wall_config = create_camera_config()
    wall_camera = StaticEnvironmentCamera(
        camera_config=wall_config,
        num_envs=rl_task.num_envs,
        mesh_ids_array=mesh_ids,
        device=device
    )
    wall_camera.set_side_view(distance=1.8, height=1.0, angle_degrees=0.0)  # Very low and close to center
    wall_pixels = torch.zeros((rl_task.num_envs, 1, 480, 480), device=device, requires_grad=False)
    wall_segmentation = torch.zeros((rl_task.num_envs, 1, 480, 480), dtype=torch.int32, device=device, requires_grad=False)
    wall_camera.set_image_tensors(wall_pixels, wall_segmentation)
    
    static_cameras.append(wall_camera)
    camera_pixels.append(wall_pixels)
    camera_segmentations.append(wall_segmentation)
    camera_names.append("Side View")
    
    print(f"Created {len(static_cameras)} static cameras for comprehensive drone observation")
    
    print("Static cameras created and configured!")
    
    # Add visual markers for static camera positions in the simulation
    print("Adding visual markers for camera positions...")
    
    print("Attempting to add camera markers to Isaac Gym viewer...")
    try:
        # Get Isaac Gym objects properly
        sim_env = rl_task.sim_env
        ige_manager = sim_env.IGE_env
        gym = ige_manager.gym
        sim = ige_manager.sim
        
        print(f"Isaac Gym objects acquired. Number of environments: {len(ige_manager.envs) if hasattr(ige_manager, 'envs') else 'Unknown'}")
        
        if hasattr(ige_manager, 'envs') and len(ige_manager.envs) > 0:
            env_handle = ige_manager.envs[0]
            print(f"Using environment handle: {env_handle}")
            
            # Known camera positions based on our settings - hardcoded for debugging
            marker_positions = [
                [0.0, 0.0, 2.0],      # Overhead Center
                [1.5, 1.5, 1.8],      # Overhead Offset
                [1.41, 1.41, 1.5],    # Corner View (45 degrees, 2.0 distance)
                [-1.41, -1.41, 1.5],  # Corner View 2 (225 degrees, 2.0 distance)
                [0.0, 0.0, 1.2],      # Overhead Close
                [1.8, 0.0, 1.0],      # Side View (0 degrees, 1.8 distance)
            ]
            
            # Bright, contrasting colors for camera markers
            camera_colors = [
                gym.Vec3(0.0, 1.0, 0.0),    # BRIGHT GREEN - Overhead Center
                gym.Vec3(1.0, 1.0, 0.0),    # BRIGHT YELLOW - Overhead Offset  
                gym.Vec3(1.0, 0.0, 0.0),    # BRIGHT RED - Corner View
                gym.Vec3(1.0, 0.0, 1.0),    # BRIGHT MAGENTA - Corner View 2
                gym.Vec3(0.0, 1.0, 1.0),    # BRIGHT CYAN - Overhead Close
                gym.Vec3(0.0, 0.0, 1.0),    # BRIGHT BLUE - Side View
            ]
            
            print(f"Creating {len(marker_positions)} camera markers with hardcoded positions...")
            for i, (pos, color, name) in enumerate(zip(marker_positions, camera_colors, camera_names)):
                print(f"Creating marker {i} ({name}) at position: {pos}")
                
                # CREATE HUGE VISIBLE SPHERES
                sphere_asset = gym.create_sphere(sim, 0.8, {})  # 80cm radius - very large
                pose = gym.Transform()
                pose.p = gym.Vec3(pos[0], pos[1], pos[2])
                actor = gym.create_actor(env_handle, sphere_asset, pose, f"camera_marker_{i}", 0, 0)
                gym.set_rigid_body_color(env_handle, actor, 0, gym.MESH_VISUAL, color)
                print(f"✅ Added {name} marker (80cm sphere) at: {pos}")
            
            # Also add a reference marker at origin
            origin_sphere = gym.create_sphere(sim, 0.3, {})
            origin_pose = gym.Transform()
            origin_pose.p = gym.Vec3(0.0, 0.0, 0.0)
            origin_actor = gym.create_actor(env_handle, origin_sphere, origin_pose, "origin_marker", 0, 0)
            gym.set_rigid_body_color(env_handle, origin_actor, 0, gym.MESH_VISUAL, gym.Vec3(1.0, 1.0, 1.0))  # White
            print("✅ Added WHITE origin reference marker at (0,0,0)")
            
            print("🎯 ALL CAMERA MARKERS CREATED SUCCESSFULLY!")
        else:
            print("❌ Warning: Could not access environment handles for camera markers")
    except Exception as e:
        print(f"❌ Error adding camera markers: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing without visual markers...")
    # Create output directory for captured images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"dual_camera_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Images will be saved to: {output_dir}")
    
    # Set up real-time visualization windows
    print("Setting up real-time visualization...")
    
    # Create windows for drone camera and all static cameras
    cv2.namedWindow("Drone Camera - Onboard View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Camera - Onboard View", 800, 400)
    
    # Create windows for each static camera
    static_camera_windows = []
    for i, name in enumerate(camera_names):
        window_name = f"Static Camera {i+1} - {name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 600, 300)  # Smaller windows for multiple cameras
        static_camera_windows.append(window_name)
    
    print("Real-time camera views enabled. Press 'q' to exit, 's' to save images.")
    print("Note: You'll see both the Isaac Gym viewer and multiple OpenCV camera windows.")
    print("- Isaac Gym viewer: Shows the 3D simulation with drone and BIG COLORED CAMERA MARKERS")
    print("- BIG Camera markers (50cm spheres): Green=Center, Yellow=Offset, Red=Corner1, Magenta=Corner2, Cyan=Close, Blue=Side")
    print(f"- OpenCV windows: {len(static_cameras)+1} total windows showing real-time depth+segmentation")
    print("- All cameras positioned INSIDE the middle chamber where drone operates")
    print("- Large bright colored spheres show exactly where each camera is positioned!")
    
    # Initialize simple movement commands  
    command_actions = torch.zeros((rl_task.num_envs, rl_task.task_config.action_space_dim), device=device)
    
    # Set initial hovering command to prevent early crashes
    command_actions[:, 0] = 0.0  # No forward movement initially
    command_actions[:, 1] = 0.0  # No lateral movement initially  
    command_actions[:, 2] = 0.0  # No vertical movement initially
    
    # Main visualization loop
    frame_count = 0
    save_images = False
    segmentation_warning_shown = False  # To avoid spamming warnings
    
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
            
            # Capture all static camera images
            for camera in static_cameras:
                camera.capture()
            
            # Get drone camera images (from obs_dict)
            drone_depth = rl_task.obs_dict["depth_range_pixels"][0, 0].cpu().numpy()
            
            # Check if segmentation is available, otherwise create simulated segmentation
            if "segmentation_pixels" in rl_task.obs_dict:
                drone_seg = rl_task.obs_dict["segmentation_pixels"][0, 0].cpu().numpy()
            else:
                if not segmentation_warning_shown:
                    print("Note: Creating simulated segmentation from depth data since segmentation_pixels not available")
                    segmentation_warning_shown = True
                # Create simulated segmentation based on depth for visualization
                # Different depth ranges get different segment IDs
                drone_seg = np.zeros_like(drone_depth)
                drone_seg[drone_depth < 0.3] = 1.0  # Close objects
                drone_seg[(drone_depth >= 0.3) & (drone_depth < 0.6)] = 2.0  # Medium distance
                drone_seg[(drone_depth >= 0.6) & (drone_depth < 1.0)] = 3.0  # Far objects
                drone_seg[drone_depth >= 1.0] = 0.5  # Background/walls
            
            # Create combined image for drone
            drone_combined = create_combined_image(drone_depth, drone_seg, "Drone Onboard")
            cv2.imshow("Drone Camera - Onboard View", drone_combined)
            
            # Process and display all static camera images
            for i, (pixels, segmentation, name, window_name) in enumerate(zip(camera_pixels, camera_segmentations, camera_names, static_camera_windows)):
                static_depth = pixels[0, 0].cpu().numpy()
                static_seg = segmentation[0, 0].cpu().numpy()
                static_combined = create_combined_image(static_depth, static_seg, name)
                cv2.imshow(window_name, static_combined)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User requested exit")
                break
            elif key == ord('s'):
                save_images = True
            
            # Save images every 30 frames or when 's' is pressed
            if frame_count % 30 == 0 or save_images:
                print(f"Saving frame {frame_count} from {len(static_cameras)+1} cameras...")
                
                # Save drone camera
                cv2.imwrite(
                    os.path.join(output_dir, f"drone_combined_frame_{frame_count:04d}.png"),
                    drone_combined
                )
                np.save(
                    os.path.join(output_dir, f"drone_depth_frame_{frame_count:04d}.npy"),
                    drone_depth
                )
                np.save(
                    os.path.join(output_dir, f"drone_seg_frame_{frame_count:04d}.npy"),
                    drone_seg
                )
                
                # Save all static cameras
                for i, (pixels, segmentation, name) in enumerate(zip(camera_pixels, camera_segmentations, camera_names)):
                    static_depth = pixels[0, 0].cpu().numpy()
                    static_seg = segmentation[0, 0].cpu().numpy()
                    static_combined = create_combined_image(static_depth, static_seg, name)
                    
                    # Clean filename from camera name
                    clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                    
                    cv2.imwrite(
                        os.path.join(output_dir, f"static_{i+1}_{clean_name}_frame_{frame_count:04d}.png"),
                        static_combined
                    )
                    np.save(
                        os.path.join(output_dir, f"static_{i+1}_{clean_name}_depth_frame_{frame_count:04d}.npy"),
                        static_depth
                    )
                    np.save(
                        os.path.join(output_dir, f"static_{i+1}_{clean_name}_seg_frame_{frame_count:04d}.npy"),
                        static_seg
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
        print(f"- Drone camera: Onboard depth + segmentation sensor")
        print(f"- Static cameras: Overhead (8m high) and corner view (inside chamber)")
        print(f"- All images saved to: {output_dir}")


if __name__ == "__main__":
    main() 