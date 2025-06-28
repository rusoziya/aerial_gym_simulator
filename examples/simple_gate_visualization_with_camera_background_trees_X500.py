"""
Simple Gate Visualization with Static Camera and X500 Robot using Isaac Gym Native API
=====================================================================================

This script creates a simple visualization of a gate environment with a static depth camera
and an X500 quadrotor robot using Isaac Gym's native camera API. This approach is simpler 
and more reliable than the Warp-based approach.

Key features:
- Gate environment with optional background trees
- X500 quadrotor robot for visual observation
- Static world-mounted camera positioned 3m from gate
- Real-time depth and segmentation image display
- Isaac Gym native camera API (no Warp dependency)
"""

import os
import sys
import time
import cv2
import numpy as np

# Isaac Gym must be imported first
from isaacgym import gymapi, gymtorch, gymutil

# Aerial Gym configs
from aerial_gym.config.asset_config.gate_asset_config import gate_asset_params
from aerial_gym.config.robot_config.x500_config import X500Cfg
import torch

class GateEnvironmentWithNativeCameraAndX500:
    """
    Gate environment with static camera and X500 robot using Isaac Gym native API.
    """
    
    def __init__(self, headless=False):
        """Initialize the environment."""
        print("🚁 Initializing Gate Environment with Native Isaac Gym Camera and X500 Robot...")
        
        self.headless = headless
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_envs = 1
        
        # Environment handles
        self.env_handles = []
        self.gate_handles = []
        self.tree_handles = []
        self.x500_handles = []  # Add X500 robot handles
        
        # Camera handles and setup
        self.camera_handles = []
        self.camera_setup_success = False
        
        # Initialize gym
        self._initialize_gym()
        self._create_sim()
        self._create_ground_plane()
        self._create_environments()
        self._setup_static_camera()
        
        if not self.headless:
            self._create_viewer()
        
        # Prepare simulation for rendering
        self.gym.prepare_sim(self.sim)
        
        print("✅ Environment initialization complete!")
    
    def _initialize_gym(self):
        """Initialize Isaac Gym."""
        print("Initializing Isaac Gym...")
        
        # Parse arguments
        custom_parameters = [
            {"name": "--headless", "action": "store_true", "help": "Run headless without viewer"},
        ]
        args = gymutil.parse_arguments(custom_parameters=custom_parameters)
        
        # Create gym instance
        self.gym = gymapi.acquire_gym()
        
        print("✓ Isaac Gym initialized")
    
    def _create_sim(self):
        """Create simulation."""
        print("Creating simulation...")
        
        # Simulation parameters
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.up_axis = gymapi.UP_AXIS_Z
        
        # Physics backend
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        
        # Create simulation
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise Exception("Failed to create simulation")
        
        print("✓ Simulation created")
    
    def _create_ground_plane(self):
        """Create ground plane."""
        print("Creating ground plane...")
        
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.0
        self.gym.add_ground(self.sim, plane_params)
    
        print("✓ Ground plane created")
    
    def _load_gate_asset(self):
        """Load gate asset."""
        print(f"Loading gate asset from: {gate_asset_params.file}")
        
        # Asset loading options
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.mesh_normal_mode = gymapi.FROM_ASSET
        
        # Load asset
        gate_asset = self.gym.load_asset(
            self.sim, 
            gate_asset_params.asset_folder, 
            gate_asset_params.file, 
            asset_options
        )
        
        if gate_asset is None:
            raise Exception(f"Failed to load gate asset: {gate_asset_params.file}")
        
        print("✓ Gate asset loaded successfully")
        return gate_asset
    
    def _load_x500_asset(self):
        """Load X500 robot asset."""
        print(f"Loading X500 robot asset from: {X500Cfg.robot_asset.file}")
        
        # Asset loading options for X500
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = X500Cfg.robot_asset.fix_base_link
        asset_options.collapse_fixed_joints = X500Cfg.robot_asset.collapse_fixed_joints
        asset_options.disable_gravity = X500Cfg.robot_asset.disable_gravity
        asset_options.replace_cylinder_with_capsule = X500Cfg.robot_asset.replace_cylinder_with_capsule
        asset_options.mesh_normal_mode = gymapi.FROM_ASSET
        asset_options.flip_visual_attachments = X500Cfg.robot_asset.flip_visual_attachments
        asset_options.density = X500Cfg.robot_asset.density
        asset_options.angular_damping = X500Cfg.robot_asset.angular_damping
        asset_options.linear_damping = X500Cfg.robot_asset.linear_damping
        asset_options.max_angular_velocity = X500Cfg.robot_asset.max_angular_velocity
        asset_options.max_linear_velocity = X500Cfg.robot_asset.max_linear_velocity
        asset_options.armature = X500Cfg.robot_asset.armature
        
        # Load X500 asset
        x500_asset = self.gym.load_asset(
            self.sim, 
            X500Cfg.robot_asset.asset_folder, 
            X500Cfg.robot_asset.file, 
            asset_options
        )
        
        if x500_asset is None:
            raise Exception(f"Failed to load X500 robot asset: {X500Cfg.robot_asset.file}")
        
        print("✓ X500 robot asset loaded successfully")
        return x500_asset
    
    def _create_environments(self):
        """Create environment with gate and optional trees."""
        print("Creating environments...")
        
        # Load gate asset
        gate_asset = self._load_gate_asset()
        
        # Load X500 asset
        x500_asset = self._load_x500_asset()
        
        # Environment bounds
        env_lower = gymapi.Vec3(-8.0, -8.0, 0.0)
        env_upper = gymapi.Vec3(8.0, 8.0, 8.0)
        
        # Create environment
        env_handle = self.gym.create_env(self.sim, env_lower, env_upper, 1)
        self.env_handles.append(env_handle)
        
        # Gate pose (center of environment)
        gate_pose = gymapi.Transform()
        gate_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)  # Gate on ground level
        
        # Rotate gate by 90 degrees around Z-axis so trees are directly behind it
        import math
        gate_pose.r = gymapi.Quat(0.0, 0.0, math.sin(math.pi/4), math.cos(math.pi/4))  # 90° rotation around Z-axis
        
        # Create gate actor
        gate_handle = self.gym.create_actor(
            env_handle,
            gate_asset,
            gate_pose,
            "gate",
            0,  # Collision group
            1   # Collision filter
        )
        self.gate_handles.append(gate_handle)
        
        # Set gate color for better visibility
        self.gym.set_rigid_body_color(
            env_handle, gate_handle, 0, 
            gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.2, 0.8)  # blue-ish
        )
        
        # Add some trees behind the gate for better visualization
        self._add_background_trees(env_handle)
        
        # Create X500 robot actor
        x500_pose = gymapi.Transform()
        x500_pose.p = gymapi.Vec3(-1.5, 0.0, 1.0)  # Position X500 to the left of gate, 1m above ground
        x500_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # Default orientation
        
        x500_handle = self.gym.create_actor(
            env_handle,
            x500_asset,
            x500_pose,
            "x500",
            0,  # Collision group
            1   # Collision filter
        )
        self.x500_handles.append(x500_handle)
        
        # Set X500 color for better visibility
        self.gym.set_rigid_body_color(
            env_handle, x500_handle, 0, 
            gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.8, 0.2)  # Yellow X500
        )
        
        print("✓ Environment created with gate and X500 robot!")
    
    def _add_background_trees(self, env_handle):
        """Add some trees behind the gate for better camera visualization."""
        print("Adding background trees...")
        
        try:
            # Load a tree asset from the existing tree models
            tree_asset_folder = f"{gate_asset_params.asset_folder}/../trees"
            available_trees = ["tree_0.urdf", "tree_1.urdf", "tree_2.urdf", "tree_3.urdf", "tree_4.urdf"]
            
            # Add trees at various positions behind the gate (negative Y from camera perspective)
            tree_positions = [
                (-2.0, 3.0, 0.0),   # Left behind gate
                (0.0, 4.0, 0.0),    # Directly behind gate
                (2.0, 3.0, 0.0),    # Right behind gate
                (-1.0, 5.0, 0.0),   # Further behind, left
                (1.0, 5.0, 0.0),    # Further behind, right
            ]
            
            for i, (x, y, z) in enumerate(tree_positions):
                if i >= len(available_trees):
                    break
                    
                tree_file = available_trees[i]
                tree_path = f"{tree_asset_folder}/{tree_file}"
                
                # Check if tree file exists
                if not os.path.exists(tree_path):
                    print(f"Tree file not found: {tree_path}, skipping...")
                    continue
                
                # Load tree asset
                tree_asset_options = gymapi.AssetOptions()
                tree_asset_options.fix_base_link = True
                tree_asset_options.collapse_fixed_joints = True
                tree_asset_options.disable_gravity = True
                tree_asset_options.replace_cylinder_with_capsule = True
                tree_asset_options.mesh_normal_mode = gymapi.FROM_ASSET
                
                tree_asset = self.gym.load_asset(
                    self.sim, 
                    tree_asset_folder, 
                    tree_file, 
                    tree_asset_options
                )
                
                if tree_asset is None:
                    print(f"Failed to load tree asset: {tree_file}")
                    continue
                
                # Create tree actor
                tree_pose = gymapi.Transform()
                tree_pose.p = gymapi.Vec3(x, y, z)
                tree_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                
                tree_handle = self.gym.create_actor(
                    env_handle,
                    tree_asset,
                    tree_pose,
                    f"tree_{i}",
                    0,  # Collision group
                    1   # Collision filter
                )
                
                self.tree_handles.append(tree_handle)
                
                # Set tree color for variety
                colors = [
                    gymapi.Vec3(0.2, 0.6, 0.2),  # Green
                    gymapi.Vec3(0.4, 0.8, 0.2),  # Light green
                    gymapi.Vec3(0.1, 0.4, 0.1),  # Dark green
                    gymapi.Vec3(0.3, 0.7, 0.3),  # Medium green
                    gymapi.Vec3(0.5, 0.9, 0.1),  # Yellow-green
                ]
                color = colors[i % len(colors)]
                self.gym.set_rigid_body_color(
                    env_handle, tree_handle, 0, 
                    gymapi.MESH_VISUAL, color
                )
                
                print(f"Added tree {i+1} at position ({x}, {y}, {z})")
                
        except Exception as e:
            print(f"Warning: Failed to add background trees: {e}")
            print("Continuing without trees...")
    
    def _setup_static_camera(self):
        """Setup static camera using Isaac Gym native camera API."""
        print("Setting up static camera using Isaac Gym native API...")
        
        try:
            # Camera properties (D455 depth camera specifications)
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1280  # D455 depth resolution
            camera_props.height = 720  # D455 depth resolution
            camera_props.horizontal_fov = 87.0  # D455 FOV
            camera_props.near_plane = 0.4  # D455 minimum depth distance
            camera_props.far_plane = 20.0  # D455 maximum range
            camera_props.enable_tensors = True  # Enable GPU tensor access
            
            print(f"Camera properties (D455 specs): {camera_props.width}x{camera_props.height}, FOV: {camera_props.horizontal_fov}°")
            print(f"Depth range: {camera_props.near_plane}m - {camera_props.far_plane}m")
        
            # Create camera sensor in each environment
            self.camera_handles = []
            for i, env_handle in enumerate(self.env_handles):
                cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                self.camera_handles.append(cam_handle)
                print(f"Created camera sensor {i} in environment {i}")
            
            # Position camera to face the gate directly (gate now rotated 90°)
            # Trees are positioned at: (-2,3,0), (0,4,0), (2,3,0), (-1,5,0), (1,5,0) - behind rotated gate
            # Gate is at (0,0,0) rotated 90°, so place camera in front of gate looking through it toward trees
            camera_pos = gymapi.Vec3(0.0, -3.0, 1.5)  # 3m in front of gate, at gate center height
            camera_target = gymapi.Vec3(0.0, 0.0, 1.5)  # Look directly at gate center
            
            # Set camera transform for each environment using look_at
            for i, (env_handle, cam_handle) in enumerate(zip(self.env_handles, self.camera_handles)):
                # Use Isaac Gym's camera look_at functionality
                self.gym.set_camera_location(cam_handle, env_handle, camera_pos, camera_target)
                print(f"Set camera {i} to look from ({camera_pos.x}, {camera_pos.y}, {camera_pos.z}) toward ({camera_target.x}, {camera_target.y}, {camera_target.z})")
            
            print(f"✓ Static cameras positioned to face gate directly")
            print(f"Camera position: (0, -3, 1.5)")
            print(f"Camera target: (0, 0, 1.5) - gate center")
            print(f"View: Looking through rotated gate toward trees behind it")
            
            self.camera_setup_success = True
            
        except Exception as e:
            print(f"❌ ERROR: Isaac Gym camera setup failed: {e}")
            import traceback
            traceback.print_exc()
            self.camera_setup_success = False
    
    def _create_viewer(self):
        """Create viewer."""
        if self.headless:
            return
        
        print("Creating viewer...")
        
        # Viewer camera properties
        cam_props = gymapi.CameraProperties()
        viewer = self.gym.create_viewer(self.sim, cam_props)
        
        if viewer is None:
            raise Exception("Failed to create viewer")
        
        # Set viewer camera
        cam_pos = gymapi.Vec3(-5.0, -5.0, 3.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
        
        self.viewer = viewer
        print("✓ Viewer created")
    
    def capture_camera_images(self):
        """Capture depth and segmentation images from static camera."""
        if not self.camera_setup_success or len(self.camera_handles) == 0:
            return None, None
        
        try:
            # Step graphics first to ensure everything is rendered
            self.gym.step_graphics(self.sim)
            
            # Render all camera sensors
            self.gym.render_all_camera_sensors(self.sim)
            
            # Start access to image tensors
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
            
            return depth_img, seg_img
                
        except Exception as e:
            print(f"Error capturing camera images: {e}")
            return None, None
            
    def create_combined_image(self, depth_img, seg_img, title="Static Camera"):
        """Create combined visualization of depth and segmentation images."""
        if depth_img is None or seg_img is None:
            return None
        
        # Normalize depth image for visualization
        depth_norm = depth_img.copy()
        
        # Replace -inf with maximum depth value
        depth_norm[depth_norm == -np.inf] = 10.0
        
        # Take absolute value to handle negative depths (coordinate system issue)
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
    
    def save_camera_image(self, combined_image, step_count):
        """Save camera image to file."""
        if combined_image is not None:
            filename = f"static_camera_combined_{step_count:04d}.jpg"
            cv2.imwrite(filename, combined_image)
            print(f"Saved image: {filename}")
    
    def run_visualization(self, duration=60.0, save_images=False):
        """Run the visualization loop."""
        print(f"Starting visualization for {duration} seconds...")
        print("Press ESC to exit early")
        
        start_time = time.time()
        step_count = 0
        
        # OpenCV window for camera display
        cv2.namedWindow("Static Camera View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Static Camera View", 2560, 720)  # Side-by-side 1280x720 images
        
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Check if duration exceeded
            if elapsed_time > duration:
                print(f"Visualization completed after {duration} seconds")
                break
            
            # Step simulation
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            # Capture camera images
            depth_img, seg_img = self.capture_camera_images()
            
            if depth_img is not None and seg_img is not None:
                # Create combined visualization
                combined_image = self.create_combined_image(depth_img, seg_img)
                
                if combined_image is not None:
                    # Display images
                    cv2.imshow("Static Camera View", combined_image)
                
                    # Note: Image saving functionality removed as requested
                    
                    # Print depth statistics periodically
                    if step_count % 60 == 0:  # Every second at 60 FPS
                        # Filter out -inf values for better statistics
                        valid_depths = depth_img[depth_img != -np.inf]
                        if len(valid_depths) > 0:
                            # Use absolute values for meaningful distance measurements
                            abs_valid_depths = np.abs(valid_depths)
                            min_depth = np.min(abs_valid_depths)
                            max_depth = np.max(abs_valid_depths)
                            mean_depth = np.mean(abs_valid_depths)
                            total_pixels = depth_img.size
                            valid_ratio = len(valid_depths) / total_pixels * 100
                            print(f"Step {step_count}: Depth range: {min_depth:.2f}-{max_depth:.2f}m, "
                                  f"Mean: {mean_depth:.2f}m, Valid pixels: {valid_ratio:.1f}%")
                        else:
                            print(f"Step {step_count}: No valid depth data (all -inf)")
                        
                        unique_segs = len(np.unique(seg_img))
                        print(f"  Segmentation: {unique_segs} unique values")
                
            # Handle viewer events
            if not self.headless:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if self.gym.query_viewer_has_closed(self.viewer):
                    break
            
            # Check for ESC key press in OpenCV window
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("ESC pressed, exiting visualization")
                break
            
            step_count += 1
        
        cv2.destroyAllWindows()
        print("Visualization loop completed")
    
    def print_environment_info(self):
        """Print information about the environment."""
        print("\n" + "="*50)
        print("ENVIRONMENT INFORMATION")
        print("="*50)
        print(f"Number of environments: {self.num_envs}")
        print(f"Device: {self.device}")
        print(f"Gate asset: {gate_asset_params.file}")
        print(f"X500 robot asset: {X500Cfg.robot_asset.file}")
        print(f"Camera setup success: {self.camera_setup_success}")
        print(f"Number of cameras: {len(self.camera_handles)}")
        print(f"Number of trees: {len(self.tree_handles)}")
        print(f"Number of X500 robots: {len(self.x500_handles)}")
        
        if self.camera_setup_success:
            print(f"Camera position: (0, -3, 1.5)")
            print(f"Camera target: (0, 0, 1.5) - gate center")
            print(f"Gate position: (0, 0, 0) - rotated 90°")
            print(f"X500 position: (-1.5, 0, 1.0) - left of gate, 1m above ground")
            print(f"Setup: Camera faces gate directly, X500 visible to left, trees behind gate")
        
        print("="*50)
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        
        cv2.destroyAllWindows()
        
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        
        if hasattr(self, 'sim') and self.sim is not None:
            self.gym.destroy_sim(self.sim)

        print("✓ Cleanup completed")

def main():
    """Main function."""
    try:
        print("🚁 Starting Gate Visualization with Native Isaac Gym Camera...")
        
        # Create environment
        env = GateEnvironmentWithNativeCameraAndX500(headless=False)
        
        # Print environment information
        env.print_environment_info()
        
        # Run visualization
        env.run_visualization(duration=120.0, save_images=False)
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'env' in locals():
            env.cleanup()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main() 