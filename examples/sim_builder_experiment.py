"""
Simple Gate Navigation with X500 Robot and D455 Camera using SimBuilder
=======================================================================

This script demonstrates navigation through a gate environment using:
1. X500 quadrotor robot with D455 depth camera sensor
2. Lee position controller for waypoint navigation
3. Gate environment with background trees
4. Real-time visualization of robot camera output with DCE RL Navigation processing

Key features:
- Uses SimBuilder for simple environment creation
- X500 quadrotor robot with onboard D455 camera
- Lee position controller for stable navigation
- Gate environment with trees for visual richness
- Waypoint navigation through the gate
- DCE RL Navigation depth and segmentation processing pipeline

Camera Processing Pipeline (DCE RL Navigation Method):
- Depth: Normalized [0,1] tensor → uint8 (×255) → JET colormap
- Segmentation: Raw segment IDs → 3-step DCE processing → Plasma colormap
  Step 1: Fix zero/negative values with min_positive
  Step 2: Normalize to [0,1] range
  Step 3: Apply plasma colormap for visualization
"""

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args
import cv2
import numpy as np
import time
import matplotlib.cm

# Isaac Gym imports for native camera API
from isaacgym import gymapi, gymtorch


def capture_robot_camera_images(env_manager):
    """Capture depth and segmentation images from robot camera using global tensor dictionary.
    
    Uses the exact same method as DCE RL navigation for consistency.
    """
    try:
        # Ensure sensors are rendered before capture (critical for proper data)
        env_manager.render(render_components="sensors")
        
        # Access camera data through global_tensor_dict (like DCE navigation and other examples)
        if not hasattr(env_manager, 'global_tensor_dict'):
            return None, None
        
        global_tensor_dict = env_manager.global_tensor_dict
        
        # Get depth and segmentation images from first environment
        depth_img = None
        seg_img = None
        
        if "depth_range_pixels" in global_tensor_dict and global_tensor_dict["depth_range_pixels"] is not None:
            # Get depth image from first environment, first camera
            depth_tensor = global_tensor_dict["depth_range_pixels"][0, 0]  # [env_idx, camera_idx]
            if depth_tensor is not None:
                # DCE Navigation Method: Convert normalized depth (0-1) to uint8 for visualization
                depth_img = (255.0 * depth_tensor.cpu().numpy()).astype(np.uint8)
        
        if "segmentation_pixels" in global_tensor_dict and global_tensor_dict["segmentation_pixels"] is not None:
            # Get segmentation image from first environment, first camera
            seg_tensor = global_tensor_dict["segmentation_pixels"][0, 0]  # [env_idx, camera_idx]
            if seg_tensor is not None:
                # DCE Navigation Method: Raw segmentation data (will be processed in create_combined_image)
                seg_img = seg_tensor.cpu().numpy()
        
        return depth_img, seg_img
        
    except Exception as e:
        logger.debug(f"Camera capture error: {e}")
        return None, None


def create_combined_image(depth_img, seg_img, title="X500 D455 Camera", collision_count=0, force_magnitude=0.0):
    """Create combined visualization of depth and segmentation images with collision status.
    
    Uses the exact same processing pipeline as DCE RL navigation for consistency:
    - Depth: Normalized [0,1] → uint8 → JET colormap
    - Segmentation: Raw segment IDs → 3-step DCE processing → Plasma colormap
    """
    if depth_img is None or seg_img is None:
        return None
    
    # === DEPTH PROCESSING (DCE Navigation Method) ===
    # Input: depth_img is already uint8 from capture function (255.0 * normalized_depth)
    # Apply JET colormap for depth visualization (same as DCE)
    depth_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
    
    # === SEGMENTATION PROCESSING (DCE Navigation 3-Step Method) ===
    seg_image_processed = seg_img.copy()
    
    # Step 1: Fix the error when there are no positive values (DCE method)
    if np.any(seg_image_processed > 0):
        min_positive = seg_image_processed[seg_image_processed > 0].min()
        seg_image_processed[seg_image_processed <= 0] = min_positive
    else:
        # If no positive values, set all to a small positive value
        seg_image_processed[:] = 0.1
    
    # Step 2: Normalize to [0,1] range (DCE method)
    seg_normalized = (seg_image_processed - seg_image_processed.min()) / (
        seg_image_processed.max() - seg_image_processed.min() + 1e-8
    )
    
    # Step 3: Apply plasma colormap (DCE method)
    seg_colored_float = matplotlib.cm.plasma(seg_normalized)
    seg_colored = (seg_colored_float[:, :, :3] * 255.0).astype(np.uint8)
    
    # === COMBINED VISUALIZATION ===
    # Create side-by-side layout
    h, w = depth_colored.shape[:2]
    combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined[:, :w] = depth_colored      # Left: Depth (JET colormap)
    combined[:, w:] = seg_colored        # Right: Segmentation (Plasma colormap)
    
    # Add simple labels only
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Depth", (10, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(combined, "Segmentation", (w + 10, 20), font, 0.5, (255, 255, 255), 1)
    
    return combined


def run_gate_navigation_with_camera(env_manager, duration_seconds=180.0):
    """Run waypoint navigation through the gate with camera visualization and collision detection."""
    logger.info(f"Starting gate navigation for {duration_seconds} seconds...")
    logger.info("Press ESC in camera window to exit early")
    logger.info("🛡️ Collision detection enabled - environment will reset on gate contact")
    
    # Define waypoints for navigation through the gate
    # Gate is positioned at ground level (Z=0), so drone should fly at reasonable height
    waypoints = [
        torch.tensor([-2.5, 0.0, 1.0], device=env_manager.device),  # Start position (1m height)
        # torch.tensor([0.0, 0.0, 0.5], device=env_manager.device),   # COLLISION TEST: Fly into gate at low height
        # torch.tensor([-1.0, 0.0, 1.2], device=env_manager.device),  # Approach gate (after reset)
        # torch.tensor([0.0, 0.0, 1.2], device=env_manager.device),   # Gate center (1.2m height to clear gate)
        # torch.tensor([1.0, 0.0, 1.2], device=env_manager.device),   # Exit gate
        # torch.tensor([2.5, 0.0, 1.0], device=env_manager.device),   # End position
        # torch.tensor([2.5, 2.0, 1.5], device=env_manager.device),   # Move to side and up
        # torch.tensor([0.0, 2.0, 1.8], device=env_manager.device),   # Above gate from side
        # torch.tensor([-2.5, 0.0, 1.0], device=env_manager.device),  # Return to start
    ]
    
    # Waypoint navigation parameters
    current_waypoint_idx = 0
    waypoint_reach_threshold = 0.5  # 50cm threshold
    waypoint_hold_time = 2.0  # Hold at each waypoint for 2 seconds
    waypoint_reached_time = None
    
    # Collision detection parameters
    collision_count = 0
    last_collision_time = 0
    collision_cooldown = 2.0  # Don't log collisions too frequently
    
    logger.info(f"🎯 Waypoint Navigation: X500 will navigate through {len(waypoints)} waypoints")
    logger.info("   Route: Start → Approach Gate → Through Gate → Exit → Side → Above → Return")
    
    start_time = time.time()
    step_count = 0
    
    # Reset environment
    env_manager.reset()
    
    while True:
        current_time = time.time()
        if current_time - start_time > duration_seconds:
            break
        
        # Get current robot position from environment state
        robot_positions = env_manager.global_tensor_dict["robot_position"]
        current_pos = robot_positions[0]  # First environment
        
        # Get current target waypoint
        current_target = waypoints[current_waypoint_idx]
        
        # Check if current waypoint is reached
        distance_to_waypoint = torch.norm(current_pos - current_target).item()
        
        # Safety mechanism: if drone is very far from current waypoint, find closest waypoint
        if distance_to_waypoint > 2.0:  # If more than 2 meters away
            closest_waypoint_idx = 0
            min_distance = float('inf')
            for i, waypoint in enumerate(waypoints):
                dist = torch.norm(current_pos - waypoint).item()
                if dist < min_distance:
                    min_distance = dist
                    closest_waypoint_idx = i
            
            if closest_waypoint_idx != current_waypoint_idx:
                logger.info(f"🔄 Drone too far from waypoint {current_waypoint_idx + 1}, switching to closest waypoint {closest_waypoint_idx + 1}")
                current_waypoint_idx = closest_waypoint_idx
                current_target = waypoints[current_waypoint_idx]
                waypoint_reached_time = None
                distance_to_waypoint = min_distance
        
        if distance_to_waypoint <= waypoint_reach_threshold:
            if waypoint_reached_time is None:
                waypoint_reached_time = current_time
                logger.info(f"🎯 Reached waypoint {current_waypoint_idx + 1}/{len(waypoints)}: [{current_target[0]:.2f}, {current_target[1]:.2f}, {current_target[2]:.2f}]")
            
            # Check if we've held at this waypoint long enough
            if current_time - waypoint_reached_time >= waypoint_hold_time:
                # Move to next waypoint
                current_waypoint_idx = (current_waypoint_idx + 1) % len(waypoints)
                waypoint_reached_time = None
                next_target = waypoints[current_waypoint_idx]
                logger.info(f"🚁 Moving to waypoint {current_waypoint_idx + 1}/{len(waypoints)}: [{next_target[0]:.2f}, {next_target[1]:.2f}, {next_target[2]:.2f}]")
        else:
            waypoint_reached_time = None  # Reset if we move away from waypoint
        
        # Debug control output every 300 steps (5 seconds at 60 FPS)
        if step_count % 300 == 0:
            pos_np = current_pos.cpu().numpy()
            target_np = current_target.cpu().numpy()
            error = current_target - current_pos
            distance = torch.norm(error).item()
            logger.info(f"Position: [{pos_np[0]:.2f}, {pos_np[1]:.2f}, {pos_np[2]:.2f}], Target: [{target_np[0]:.2f}, {target_np[1]:.2f}, {target_np[2]:.2f}]")
            logger.info(f"Error: [{error[0]:.2f}, {error[1]:.2f}, {error[2]:.2f}], Distance: {distance:.2f}m, Waypoint: {current_waypoint_idx + 1}/{len(waypoints)}")
            
            # Debug collision detection data
            try:
                if hasattr(env_manager, 'global_tensor_dict'):
                    # Check collision tensor status
                    if hasattr(env_manager, 'collision_tensor'):
                        collision_status = env_manager.collision_tensor[0].item()
                        logger.info(f"Collision tensor: {collision_status}")
                    
                    # Check contact forces
                    if 'robot_contact_force_tensor' in env_manager.global_tensor_dict:
                        contact_forces = env_manager.global_tensor_dict['robot_contact_force_tensor']
                        if contact_forces is not None and len(contact_forces) > 0:
                            contact_force_magnitude = torch.norm(contact_forces[0]).item()
                            logger.info(f"Contact force magnitude: {contact_force_magnitude:.4f} N")
                    
                    # Check force sensor data
                    if 'robot_force_sensor' in env_manager.global_tensor_dict:
                        force_data = env_manager.global_tensor_dict['robot_force_sensor']
                        if force_data is not None and len(force_data) > 0:
                            force_sensor_magnitude = torch.norm(force_data[0, :3]).item()
                            logger.info(f"Force sensor magnitude: {force_sensor_magnitude:.4f} N")
                    
                    # Show collision threshold
                    if hasattr(env_manager, 'cfg') and hasattr(env_manager.cfg, 'env'):
                        threshold = env_manager.cfg.env.collision_force_threshold
                        logger.info(f"Collision threshold: {threshold:.4f} N")
                        
            except Exception as e:
                logger.debug(f"Debug info error: {e}")
        
        # Create command actions for Lee controller: [x, y, z, yaw]
        actions = torch.zeros((env_manager.num_envs, 4), device=env_manager.device)
        actions[0, 0:3] = current_target  # Target position
        actions[0, 3] = 0.0  # Target yaw (face forward)
        
        # Step environment
        env_manager.step(actions=actions)
        
        # Manually trigger collision detection and environment reset
        try:
            # Call compute_observations to update collision tensor
            if hasattr(env_manager, 'compute_observations'):
                env_manager.compute_observations()
            
            # Check for collisions using the environment's collision detection system
            if hasattr(env_manager, 'collision_tensor'):
                collision_detected = env_manager.collision_tensor[0].item() > 0
                if collision_detected:
                    current_time = time.time()
                    if current_time - last_collision_time > collision_cooldown:
                        collision_count += 1
                        last_collision_time = current_time
                        pos_np = current_pos.cpu().numpy()
                        logger.warning(f"💥 COLLISION DETECTED! #{collision_count}")
                        logger.warning(f"   Position at collision: [{pos_np[0]:.2f}, {pos_np[1]:.2f}, {pos_np[2]:.2f}]")
                        logger.warning(f"   Target waypoint: {current_waypoint_idx + 1}/{len(waypoints)}")
                        logger.warning("   Environment will reset automatically...")
                        
                        # Reset waypoint to start position for safety after reset
                        current_waypoint_idx = 0
                        waypoint_reached_time = None
            
            # Call the environment's reset function for terminated environments
            if hasattr(env_manager, 'reset_terminated_and_truncated_envs'):
                envs_reset = env_manager.reset_terminated_and_truncated_envs()
                if len(envs_reset) > 0:
                    logger.info(f"🔄 Environment(s) {envs_reset.tolist()} reset due to collision/termination")
            
            # Check contact forces for display purposes
            current_force = 0.0
            if 'robot_contact_force_tensor' in env_manager.global_tensor_dict:
                contact_forces = env_manager.global_tensor_dict['robot_contact_force_tensor']
                if contact_forces is not None and len(contact_forces) > 0:
                    current_force = torch.norm(contact_forces[0]).item()
            
            # Alternative: Check force sensor data for display
            elif 'robot_force_sensor' in env_manager.global_tensor_dict:
                force_data = env_manager.global_tensor_dict['robot_force_sensor']
                if force_data is not None and len(force_data) > 0:
                    current_force = torch.norm(force_data[0, :3]).item()
        
        except Exception as e:
            if step_count % 300 == 0:  # Only log errors occasionally
                logger.debug(f"Collision detection error: {e}")
        
        # Capture and display camera images using Isaac Gym native API
        if step_count % 3 == 0:  # Every 3 steps for performance
            try:
                # Render sensors to update camera data (crucial step!)
                env_manager.render(render_components="sensors")
                
                # Use Isaac Gym native camera capture (inspired by reference file)
                depth_img, seg_img = capture_robot_camera_images(env_manager)
                
                if depth_img is not None and seg_img is not None:
                    # Use the current_force from collision detection above
                    # (current_force is already calculated in the collision detection section)
                    
                    # Create combined visualization with collision status
                    combined = create_combined_image(depth_img, seg_img, "X500 D455 Camera", 
                                                   collision_count, current_force)
                    
                    if combined is not None:
                        # Create named window with proper flags for visibility
                        cv2.namedWindow("X500 D455 Camera View", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("X500 D455 Camera View", 960, 270)  # D455 resolution scaled
                        cv2.imshow("X500 D455 Camera View", combined)
                        
                        # Print camera statistics periodically (DCE Navigation style)
                        if step_count % 300 == 0:
                            logger.info("📸 === DCE-Style Camera Data Analysis ===")
                            
                            # Depth analysis (on uint8 visualization data)
                            valid_depths = depth_img[depth_img > 0]  # Valid depth pixels
                            if len(valid_depths) > 0:
                                min_depth_vis = np.min(valid_depths)
                                max_depth_vis = np.max(valid_depths)
                                mean_depth_vis = np.mean(valid_depths)
                                valid_ratio = len(valid_depths) / depth_img.size * 100
                                logger.info(f"   Depth (uint8): range {min_depth_vis}-{max_depth_vis}, mean {mean_depth_vis:.1f}, valid {valid_ratio:.1f}%")
                                
                                # Convert back to actual depth values for reference
                                actual_min = min_depth_vis / 255.0 * 10.0  # Assuming 10m max range
                                actual_max = max_depth_vis / 255.0 * 10.0
                                actual_mean = mean_depth_vis / 255.0 * 10.0
                                logger.info(f"   Depth (actual): range {actual_min:.2f}-{actual_max:.2f}m, mean {actual_mean:.2f}m")
                            else:
                                logger.warning("   Depth: No valid pixels detected!")
                                
                            # Segmentation analysis (DCE method)
                            unique_segs = np.unique(seg_img)
                            logger.info(f"   Segmentation (raw): {len(unique_segs)} unique values, range {unique_segs.min():.1f} to {unique_segs.max():.1f}")
                            
                            # Show segment distribution
                            if len(unique_segs) <= 10:  # Only show if not too many segments
                                seg_counts = [(val, np.sum(seg_img == val)) for val in unique_segs]
                                logger.info(f"   Segment distribution: {seg_counts}")
                            
                            # Show processing pipeline status
                            pos_seg_count = np.sum(seg_img > 0)
                            zero_neg_count = np.sum(seg_img <= 0)
                            logger.info(f"   Segmentation processing: {pos_seg_count} positive pixels, {zero_neg_count} zero/negative pixels")
                            
                            # Expected segmentation with collision-enabled trees:
                            # - Gate: Should appear as distinct segments (cyan/turquoise in plasma)
                            # - Trees: Should now appear as tree segments (different plasma colors)
                            # - Ground: Background/ground plane segments (dark colors in plasma)
                            # - Sky/Empty: Zero/negative values (will be fixed by DCE processing)
                            
                            if zero_neg_count > 0:
                                logger.info("   → DCE Step 1: Will fix zero/negative values with min_positive")
                            logger.info("   → DCE Step 2: Normalizing to [0,1] range")
                            logger.info("   → DCE Step 3: Applying plasma colormap")
                            logger.info("   Expected segments: Gate + Trees + Walls + Ground (collision-enabled objects)")
                            logger.info("📸 === End Camera Analysis ===\n")
                    
                    # Check for ESC key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        logger.info("ESC pressed, exiting navigation")
                        break
                else:
                    # Show debug info when no camera data
                    if step_count % 300 == 0:
                        logger.warning("No camera images captured - camera may not be properly configured")
                            
            except Exception as e:
                if step_count % 300 == 0:  # Only log errors occasionally
                    logger.warning(f"Camera capture error: {e}")
        
        step_count += 1
    
    logger.info(f"\n✅ Gate navigation completed after {current_time - start_time:.1f} seconds")
    logger.info(f"📊 Navigation Statistics:")
    logger.info(f"   Total collisions detected: {collision_count}")
    logger.info(f"   Total simulation steps: {step_count}")
    if collision_count > 0:
        logger.info(f"   Collision rate: {collision_count / (current_time - start_time) * 60:.2f} collisions/minute")
    else:
        logger.info("   🏆 Perfect flight - no collisions detected!")


def main():
    """Main function."""
    try:
        args = get_args()
        
        logger.info("🚁 Starting Gate Navigation with X500 and D455 Camera using SimBuilder...")
        
        # Build environment using SimBuilder
        num_envs_requested = getattr(args, 'num_envs', 1)
        logger.info(f"🔧 Requesting {num_envs_requested} environments")
        
        env_manager = SimBuilder().build_env(
            sim_name="base_sim",
            env_name="gate_env",  # Use gate environment with trees
            robot_name="x500",    # X500 robot with D455 camera capability
            controller_name="lee_position_control",
            args=args,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            num_envs=num_envs_requested,
            headless=getattr(args, 'headless', False),
            use_warp=getattr(args, 'use_warp', False),
        )
        
        logger.info("✅ Environment built successfully using SimBuilder!")
        logger.info(f"Environment: gate_env, Robot: x500, Controller: lee_position_control")
        logger.info(f"Number of environments: {env_manager.num_envs}")
        logger.info(f"Device: {env_manager.device}")
        
        # Check if num_envs was properly set
        if env_manager.num_envs != num_envs_requested:
            logger.error(f"❌ Environment created {env_manager.num_envs} environments instead of requested {num_envs_requested}!")
            logger.error("This suggests the environment configuration is not properly overriding the default values.")
            logger.error("Attempting to continue with the actual number of environments...")
        else:
            logger.info(f"✅ Environment correctly created {num_envs_requested} environment(s)")
        
        # Run gate navigation with camera visualization
        run_gate_navigation_with_camera(env_manager, duration_seconds=180.0)  # 3 minutes
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        logger.info("👋 Goodbye!")


if __name__ == "__main__":
    main() 