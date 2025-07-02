"""
Simple Gate Navigation with X500 Robot, D455 Camera, and Static Camera using SimBuilder
========================================================================================

This script demonstrates navigation through a gate environment using:
1. X500 quadrotor robot with D455 depth camera sensor
2. Static D455 camera positioned 3m in front of gate looking at gate center
3. Lee position controller for waypoint navigation
4. Gate environment with background trees
5. Real-time visualization of both camera outputs with DCE RL Navigation processing

Key features:
- Uses SimBuilder for simple environment creation
- X500 quadrotor robot with onboard D455 camera
- Static D455 camera with same specifications as robot camera
- Lee position controller for stable navigation
- Gate environment with trees for visual richness
- Waypoint navigation through the gate
- DCE RL Navigation depth and segmentation processing pipeline
- Two separate OpenCV windows for robot and static cameras

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


class StaticCameraManager:
    """Manages static camera setup and capture using Isaac Gym native API."""
    
    def __init__(self, env_manager):
        """Initialize static camera manager."""
        self.env_manager = env_manager
        self.gym = env_manager.IGE_env.gym
        self.sim = env_manager.IGE_env.sim
        self.env_handles = env_manager.IGE_env.env_handles
        self.camera_handles = []
        self.camera_setup_success = False
        
        # Setup static camera
        self._setup_static_camera()
    
    def _setup_static_camera(self):
        """Setup static camera using Isaac Gym native camera API with D455 specifications."""
        logger.info("Setting up static camera using Isaac Gym native API...")
        
        try:
            # Camera properties (D455 depth camera specifications)
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1280  # D455 depth resolution
            camera_props.height = 720  # D455 depth resolution
            camera_props.horizontal_fov = 87.0  # D455 FOV
            camera_props.near_plane = 0.4  # D455 minimum depth distance
            camera_props.far_plane = 20.0  # D455 maximum range
            camera_props.enable_tensors = True  # Enable GPU tensor access
            
            logger.info(f"Static camera properties (D455 specs): {camera_props.width}x{camera_props.height}, FOV: {camera_props.horizontal_fov}°")
            logger.info(f"Static camera depth range: {camera_props.near_plane}m - {camera_props.far_plane}m")
        
            # Create camera sensor in each environment
            self.camera_handles = []
            for i, env_handle in enumerate(self.env_handles):
                cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                self.camera_handles.append(cam_handle)
                logger.info(f"Created static camera sensor {i} in environment {i}")
            
            # Position camera to face the gate directly
            # Gate is positioned at ground level (Z=0), camera at 1.5m height looking at gate center
            camera_pos = gymapi.Vec3(0.0, -3.0, 1.5)  # 3m in front of gate, at gate center height
            camera_target = gymapi.Vec3(0.0, 0.0, 1.5)  # Look directly at gate center
            
            # Set camera transform for each environment using look_at
            for i, (env_handle, cam_handle) in enumerate(zip(self.env_handles, self.camera_handles)):
                # Use Isaac Gym's camera look_at functionality
                self.gym.set_camera_location(cam_handle, env_handle, camera_pos, camera_target)
                logger.info(f"Set static camera {i} to look from ({camera_pos.x}, {camera_pos.y}, {camera_pos.z}) toward ({camera_target.x}, {camera_target.y}, {camera_target.z})")
            
            logger.info("✓ Static cameras positioned to face gate directly")
            
            self.camera_setup_success = True
            
        except Exception as e:
            logger.error(f"❌ ERROR: Isaac Gym static camera setup failed: {e}")
            import traceback
            traceback.print_exc()
            self.camera_setup_success = False
    
    def capture_static_camera_images(self):
        """Capture depth and segmentation images from static camera."""
        if not self.camera_setup_success or len(self.camera_handles) == 0:
            return None, None
        
        try:
            # Step graphics and render all cameras
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            
            # Get images from first camera
            env_handle = self.env_handles[0]
            cam_handle = self.camera_handles[0]
            
            # Get depth image
            depth_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH
            )
            depth_img = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
            
            # Get segmentation image
            seg_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, env_handle, cam_handle, gymapi.IMAGE_SEGMENTATION
            )
            seg_img = gymtorch.wrap_tensor(seg_tensor).cpu().numpy()
            
            # End access to image tensors
            self.gym.end_access_image_tensors(self.sim)
            
            # Convert to DCE format for consistency with robot camera processing
            # Static camera gives raw depth values, need to normalize to [0,1] for DCE processing
            if depth_img is not None:
                # Replace -inf with maximum depth value
                depth_normalized = depth_img.copy()
                depth_normalized[depth_normalized == -np.inf] = 20.0  # Use far_plane value
                depth_normalized = np.abs(depth_normalized)  # Handle negative depths
                depth_normalized = np.clip(depth_normalized, 0.4, 20.0)  # Clip to camera range
                # Normalize to [0,1] range like DCE navigation expects
                depth_normalized = (depth_normalized - 0.4) / (20.0 - 0.4)
                # Convert to uint8 for DCE processing pipeline
                depth_img = (255.0 * depth_normalized).astype(np.uint8)
            
            return depth_img, seg_img
                
        except Exception as e:
            logger.debug(f"Error capturing static camera images: {e}")
            return None, None


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
        logger.debug(f"Robot camera capture error: {e}")
        return None, None


def create_combined_image(depth_img, seg_img, title="Camera"):
    """Create combined visualization of depth and segmentation images.
    
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


def run_gate_navigation_with_dual_cameras(env_manager, duration_seconds=180.0):
    """Run waypoint navigation through the gate with dual camera visualization and collision detection."""
    logger.info(f"Starting gate navigation with dual cameras for {duration_seconds} seconds...")
    logger.info("Press ESC in either camera window to exit early")
    logger.info("🛡️ Collision detection enabled - environment will reset on gate contact")
    
    # Initialize static camera manager
    static_camera = StaticCameraManager(env_manager)
    
    # Define waypoints for navigation through the gate
    # Gate is positioned at ground level (Z=0), so drone should fly at reasonable height
    waypoints = [
        torch.tensor([-2.5, 0.0, 1.0], device=env_manager.device),  # Start position (1m height)
        # Add more waypoints as needed
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
    
    start_time = time.time()
    step_count = 0
    
    # Create OpenCV windows for dual camera display
    cv2.namedWindow("X500 D455 Camera View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("X500 D455 Camera View", 960, 270)  # D455 resolution scaled
    
    if static_camera.camera_setup_success:
        cv2.namedWindow("Static D455 Camera View", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Static D455 Camera View", 2560, 720)  # D455 static camera resolution scaled
        cv2.resizeWindow("Static D455 Camera View", 960, 270)  # D455 static camera resolution scaled
    
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
        
        # Create command actions for Lee controller: [x, y, z, yaw]
        actions = torch.zeros((env_manager.num_envs, 4), device=env_manager.device)
        actions[0, 0:3] = current_target  # Target position
        actions[0, 3] = 0.0  # Target yaw (face forward)
        
        # Step environment
        env_manager.step(actions=actions)
        
        # Handle collision detection
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
                        logger.warning("   Environment will reset automatically...")
                        
                        # Reset waypoint to start position for safety after reset
                        current_waypoint_idx = 0
                        waypoint_reached_time = None
            
            # Call the environment's reset function for terminated environments
            if hasattr(env_manager, 'reset_terminated_and_truncated_envs'):
                envs_reset = env_manager.reset_terminated_and_truncated_envs()
                if len(envs_reset) > 0:
                    logger.info(f"🔄 Environment(s) {envs_reset.tolist()} reset due to collision/termination")
        
        except Exception as e:
            if step_count % 300 == 0:  # Only log errors occasionally
                logger.debug(f"Collision detection error: {e}")
        
        # Capture and display camera images
        if step_count % 3 == 0:  # Every 3 steps for performance
            try:
                # Capture robot camera images (DCE method)
                robot_depth, robot_seg = capture_robot_camera_images(env_manager)
                
                # Capture static camera images
                static_depth, static_seg = static_camera.capture_static_camera_images()
                
                # Display robot camera images
                if robot_depth is not None and robot_seg is not None:
                    robot_combined = create_combined_image(robot_depth, robot_seg, "X500 D455")
                    
                    if robot_combined is not None:
                        cv2.imshow("X500 D455 Camera View", robot_combined)
                
                # Display static camera images
                if static_depth is not None and static_seg is not None:
                    static_combined = create_combined_image(static_depth, static_seg, "Static D455")
                    
                    if static_combined is not None:
                        cv2.imshow("Static D455 Camera View", static_combined)
                
                # Print camera statistics periodically
                if step_count % 300 == 0:
                    logger.info("📸 === Dual Camera Data Analysis (DCE Processing) ===")
                    
                    # Robot camera analysis
                    if robot_depth is not None and robot_seg is not None:
                        logger.info("   Robot Camera (X500 D455):")
                        
                        # Depth analysis
                        valid_depths = robot_depth[robot_depth > 0]
                        if len(valid_depths) > 0:
                            min_depth_vis = np.min(valid_depths)
                            max_depth_vis = np.max(valid_depths)
                            mean_depth_vis = np.mean(valid_depths)
                            valid_ratio = len(valid_depths) / robot_depth.size * 100
                            logger.info(f"     Depth (uint8): range {min_depth_vis}-{max_depth_vis}, mean {mean_depth_vis:.1f}, valid {valid_ratio:.1f}%")
                        
                        # Segmentation analysis
                        unique_segs = np.unique(robot_seg)
                        logger.info(f"     Segmentation: {len(unique_segs)} unique values, range {unique_segs.min():.1f} to {unique_segs.max():.1f}")
                    
                    # Static camera analysis
                    if static_depth is not None and static_seg is not None:
                        logger.info("   Static Camera (D455):")
                        
                        # Depth analysis
                        valid_depths = static_depth[static_depth > 0]
                        if len(valid_depths) > 0:
                            min_depth_vis = np.min(valid_depths)
                            max_depth_vis = np.max(valid_depths)
                            mean_depth_vis = np.mean(valid_depths)
                            valid_ratio = len(valid_depths) / static_depth.size * 100
                            logger.info(f"     Depth (uint8): range {min_depth_vis}-{max_depth_vis}, mean {mean_depth_vis:.1f}, valid {valid_ratio:.1f}%")
                        
                        # Segmentation analysis
                        unique_segs = np.unique(static_seg)
                        logger.info(f"     Segmentation: {len(unique_segs)} unique values, range {unique_segs.min():.1f} to {unique_segs.max():.1f}")
                    
                    logger.info("📸 === End Dual Camera Analysis ===\n")
                
                # Check for ESC key press in either window
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    logger.info("ESC pressed, exiting navigation")
                    break
                    
            except Exception as e:
                if step_count % 300 == 0:  # Only log errors occasionally
                    logger.warning(f"Camera capture error: {e}")
        
        step_count += 1
    
    logger.info(f"\n✅ Dual camera gate navigation completed after {current_time - start_time:.1f} seconds")
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
        
        logger.info("🚁 Starting Gate Navigation with X500 D455 Camera and Static D455 Camera using SimBuilder...")
        
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
        
        # Run gate navigation with dual camera visualization
        run_gate_navigation_with_dual_cameras(env_manager, duration_seconds=180.0)  # 3 minutes
        
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