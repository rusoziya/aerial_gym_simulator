"""
Simple Gate Navigation with X500 Robot and D455 Camera using SimBuilder
=======================================================================

This script demonstrates navigation through a gate environment using:
1. X500 quadrotor robot with D455 depth camera sensor
2. Lee position controller for waypoint navigation
3. Gate environment with background trees
4. Real-time visualization of robot camera output

Key features:
- Uses SimBuilder for simple environment creation
- X500 quadrotor robot with onboard D455 camera
- Lee position controller for stable navigation
- Gate environment with trees for visual richness
- Waypoint navigation through the gate
"""

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args
import cv2
import numpy as np
import time


def create_combined_image(depth_img, seg_img, title="X500 D455 Camera"):
        """Create combined visualization of depth and segmentation images."""
        if depth_img is None or seg_img is None:
            return None
        
        # Normalize depth image for visualization
        depth_norm = depth_img.copy()
        
        # Replace -inf with maximum depth value
        depth_norm[depth_norm == -np.inf] = 10.0
        
    # Take absolute value to handle negative depths
        depth_norm = np.abs(depth_norm)
        
        # Clip to reasonable range (0 to 10 meters for D455 range)
        depth_norm = np.clip(depth_norm, 0, 10.0)  
        depth_norm = (depth_norm / 10.0 * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        
        # Normalize segmentation image for visualization
        seg_norm = seg_img.copy()
        
        # If segmentation has limited values, enhance the contrast
        unique_segs = np.unique(seg_norm)
        if len(unique_segs) > 1:
            # Map each unique segment to a distinct value
            seg_enhanced = np.zeros_like(seg_norm)
            for i, val in enumerate(unique_segs):
                seg_enhanced[seg_norm == val] = i * (255 // len(unique_segs))
            seg_colored = cv2.applyColorMap(seg_enhanced.astype(np.uint8), cv2.COLORMAP_PLASMA)
        else:
            # If only one segment, create a depth-based pseudo-segmentation
            pseudo_seg = np.abs(depth_img).copy()
            pseudo_seg[pseudo_seg == np.inf] = 10.0
            pseudo_seg = np.clip(pseudo_seg, 0, 10.0)
            pseudo_seg = (pseudo_seg / 10.0 * 255).astype(np.uint8)
            seg_colored = cv2.applyColorMap(pseudo_seg, cv2.COLORMAP_PLASMA)
        
        # Create combined image (side by side)
        h, w = depth_colored.shape[:2]
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
        combined[:, :w] = depth_colored
        combined[:, w:] = seg_colored
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Depth", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Segmentation", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, title, (w // 2 - 50, h - 20), font, 0.5, (255, 255, 255), 1)
        
        return combined
    

def run_gate_navigation_with_camera(env_manager, duration_seconds=180.0):
    """Run waypoint navigation through the gate with camera visualization."""
    logger.info(f"Starting gate navigation for {duration_seconds} seconds...")
    logger.info("Press ESC in camera window to exit early")
        
        # Define waypoints for navigation through the gate
        waypoints = [
        torch.tensor([-2.5, 0.0, 0.8], device=env_manager.device),  # Start position
        torch.tensor([-1.0, 0.0, 1.2], device=env_manager.device),  # Approach gate
        torch.tensor([0.0, 0.0, 1.2], device=env_manager.device),   # Gate center
        torch.tensor([1.0, 0.0, 1.2], device=env_manager.device),   # Exit gate
        torch.tensor([2.5, 0.0, 0.8], device=env_manager.device),   # End position
        torch.tensor([2.5, 2.0, 1.5], device=env_manager.device),   # Move to side and up
        torch.tensor([0.0, 2.0, 1.8], device=env_manager.device),   # Above gate from side
        torch.tensor([-2.5, 0.0, 0.8], device=env_manager.device),  # Return to start
        ]
        
        # Waypoint navigation parameters
        current_waypoint_idx = 0
    waypoint_reach_threshold = 0.5  # 50cm threshold
    waypoint_hold_time = 2.0  # Hold at each waypoint for 2 seconds
        waypoint_reached_time = None
        
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
        
        # Create command actions for Lee controller: [x, y, z, yaw]
        actions = torch.zeros((env_manager.num_envs, 4), device=env_manager.device)
        actions[0, 0:3] = current_target  # Target position
        actions[0, 3] = 0.0  # Target yaw (face forward)
        
        # Step environment
        env_manager.step(actions=actions)
        
        # Capture and display camera images
        if hasattr(env_manager, 'get_camera_images') and step_count % 3 == 0:  # Every 3 steps for performance
            try:
                camera_images = env_manager.get_camera_images()
                if camera_images is not None and len(camera_images) > 0:
                    # Get depth and segmentation images from first environment
                    depth_img = camera_images[0].get('depth')
                    seg_img = camera_images[0].get('segmentation')
                    
                    if depth_img is not None and seg_img is not None:
                        # Convert tensors to numpy arrays
                        if torch.is_tensor(depth_img):
                            depth_img = depth_img.cpu().numpy()
                        if torch.is_tensor(seg_img):
                            seg_img = seg_img.cpu().numpy()
                        
                        # Create combined visualization
                        combined = create_combined_image(depth_img, seg_img, "X500 D455 Camera")
                        
                        if combined is not None:
                            cv2.imshow("X500 D455 Camera View", combined)
                            
                            # Print camera statistics periodically
                            if step_count % 300 == 0:
                                valid_depths = depth_img[depth_img != -np.inf]
                    if len(valid_depths) > 0:
                        abs_valid_depths = np.abs(valid_depths)
                        min_depth = np.min(abs_valid_depths)
                        max_depth = np.max(abs_valid_depths)
                        mean_depth = np.mean(abs_valid_depths)
                                    valid_ratio = len(valid_depths) / depth_img.size * 100
                                    logger.info(f"X500 D455 - Depth range: {min_depth:.2f}-{max_depth:.2f}m, Mean: {mean_depth:.2f}m, Valid: {valid_ratio:.1f}%")
                        
                        # Check for ESC key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
                            logger.info("ESC pressed, exiting navigation")
                            break
                
        except Exception as e:
                if step_count % 300 == 0:  # Only log errors occasionally
                    logger.warning(f"Camera capture error: {e}")
        
        step_count += 1
    
    logger.info(f"\n✅ Gate navigation completed after {current_time - start_time:.1f} seconds")


def main():
    """Main function."""
    try:
        args = get_args()
        
        logger.info("🚁 Starting Gate Navigation with X500 and D455 Camera using SimBuilder...")
        
        # Build environment using SimBuilder
        env_manager = SimBuilder().build_env(
            sim_name="base_sim",
            env_name="gate_env",  # Use gate environment with trees
            robot_name="x500",    # X500 robot with D455 camera capability
            controller_name="lee_position_control",
            args=args,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            num_envs=getattr(args, 'num_envs', 1),
            headless=getattr(args, 'headless', False),
            use_warp=getattr(args, 'use_warp', False),
        )
        
        logger.info("✅ Environment built successfully using SimBuilder!")
        logger.info(f"Environment: gate_env, Robot: x500, Controller: lee_position_control")
        logger.info(f"Number of environments: {env_manager.num_envs}")
        logger.info(f"Device: {env_manager.device}")
        
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