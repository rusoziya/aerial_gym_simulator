"""
Simple Gate Visualization Demo with Static Camera

This script creates and visualizes the gate environment using Isaac Gym with a static Warp-based camera.
It spawns a gate in the environment and uses NVIDIA Warp to render depth/RGB images from a static camera viewpoint.
"""

import sys
import os
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt

# Add the aerial_gym path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import Isaac Gym first (before PyTorch)
    import isaacgym
    from isaacgym import gymapi, gymtorch, gymutil
    
    # Import aerial gym components (still before PyTorch)
    from aerial_gym.config.env_config.gate_env import GateEnvCfg
    from aerial_gym.config.asset_config.gate_asset_config import gate_asset_params
    from aerial_gym.env_manager.asset_manager import AssetManager
    from aerial_gym.sim.sim_builder import SimBuilder
    from aerial_gym.sensors.warp.static_environment_camera import StaticEnvironmentCamera
    from aerial_gym.assets.warp_asset import WarpAsset
    from aerial_gym.env_manager.warp_env_manager import WarpEnv
    
    # Import PyTorch and Warp after Isaac Gym
    import torch
    import warp as wp
    
    print("✓ Successfully imported all required modules!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("This might be due to missing dependencies or Python environment issues.")
    sys.exit(1)


class GateEnvironmentWithCamera:
    """
    A gate environment visualizer with static Warp-based camera.
    """
    
    def __init__(self, headless=False):
        """
        Initialize the gate environment with camera.
        
        Args:
            headless (bool): Whether to run in headless mode (no graphics)
        """
        self.headless = headless
        self.gym = None
        self.sim = None
        self.viewer = None
        self.env_handles = []
        self.gate_handles = []
        self.warp_env = None
        self.static_camera = None
        
        # Environment parameters
        self.num_envs = 1  # Single environment for visualization
        self.env_spacing = 10.0  # Spacing between environments
        self.device = 'cuda:0'
        
        # Camera parameters
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fov = 80  # degrees - wider FOV for better coverage
        
        # Initialize Warp
        wp.init()
        wp.config.mode = "release"
        
        # Initialize the environment
        self._initialize_gym()
        self._create_sim()
        self._create_ground_plane()
        self._setup_warp_env()
        self._create_environments()
        self._setup_static_camera()
        
        if not self.headless:
            self._create_viewer()
            self._setup_camera()
    
    def _initialize_gym(self):
        """Initialize Isaac Gym."""
        print("Initializing Isaac Gym...")
        
        # Create gym instance
        self.gym = gymapi.acquire_gym()
        
        # Configure simulation parameters
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0  # 60 FPS
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        # Physics engine settings
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.num_threads = 0
        sim_params.physx.use_gpu = True
        
        # GPU settings
        sim_params.use_gpu_pipeline = True
        
        self.sim_params = sim_params
        
    def _create_sim(self):
        """Create the simulation."""
        print("Creating simulation...")
        
        # Create simulation
        self.sim = self.gym.create_sim(
            compute_device=0,
            graphics_device=0 if not self.headless else -1,
            type=gymapi.SIM_PHYSX,
            params=self.sim_params
        )
        
        if self.sim is None:
            raise RuntimeError("Failed to create simulation")
    
    def _create_ground_plane(self):
        """Create ground plane."""
        print("Creating ground plane...")
        
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.distance = 0
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        
        self.gym.add_ground(self.sim, plane_params)
    
    def _setup_warp_env(self):
        """Setup Warp environment for camera rendering."""
        print("Setting up Warp environment...")
        
        # Create a minimal global_sim_dict for WarpEnv
        global_sim_dict = {
            "num_envs": self.num_envs,
            "env_cfg": None  # We'll use minimal config
        }
        
        # Create WarpEnv for mesh management
        self.warp_env = WarpEnv(global_sim_dict, self.device)
        
        # Create environments in WarpEnv
        for i in range(self.num_envs):
            self.warp_env.create_env(i)
    
    def _load_gate_asset(self):
        """Load the gate asset for both Isaac Gym and Warp."""
        print("Loading gate asset...")
        
        # Isaac Gym asset loading options
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True  # Gate is fixed in place
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True  # Static object doesn't need gravity
        asset_options.replace_cylinder_with_capsule = True
        asset_options.mesh_normal_mode = gymapi.FROM_ASSET
        
        # Load gate asset for Isaac Gym
        gate_asset_path = f"{gate_asset_params.asset_folder}/{gate_asset_params.file}"
        
        if not os.path.exists(gate_asset_path):
            print(f"❌ Gate URDF not found at: {gate_asset_path}")
            print("Please ensure the gate URDF file exists!")
            sys.exit(1)
        
        print(f"Loading gate from: {gate_asset_path}")
        gate_asset = self.gym.load_asset(
            self.sim, 
            gate_asset_params.asset_folder, 
            gate_asset_params.file, 
            asset_options
        )
        
        if gate_asset is None:
            raise RuntimeError(f"Failed to load gate asset from {gate_asset_path}")
        
        # Create WarpAsset for camera rendering
        asset_file_path = f"{gate_asset_params.asset_folder}/{gate_asset_params.file}"
        
        # Create loading options from gate asset params
        loading_options = {
            'semantic_id': getattr(gate_asset_params, 'semantic_id', 1),
            'per_link_semantic': getattr(gate_asset_params, 'per_link_semantic', True),
            'semantic_masked_links': getattr(gate_asset_params, 'semantic_masked_links', {}),
            'use_collision_mesh_instead_of_visual': getattr(gate_asset_params, 'use_collision_mesh_instead_of_visual', False)
        }
        
        self.warp_gate_asset = WarpAsset(
            asset_name="gate",
            asset_file=asset_file_path,
            loading_options=loading_options
        )
        
        print("✓ Gate asset loaded successfully for both Isaac Gym and Warp!")
        return gate_asset
    
    def _create_environments(self):
        """Create the environments with gates."""
        print("Creating environments...")
        
        # Load gate asset
        gate_asset = self._load_gate_asset()
        
        # Environment bounds
        env_lower = gymapi.Vec3(-8.0, -8.0, 0.0)
        env_upper = gymapi.Vec3(8.0, 8.0, 8.0)
        
        # Create environment
        env_handle = self.gym.create_env(self.sim, env_lower, env_upper, 1)
        self.env_handles.append(env_handle)
        
        # Gate pose (center of environment)
        gate_pose = gymapi.Transform()
        gate_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)  # Gate on ground level
        gate_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # No rotation
        
        # Create gate actor in Isaac Gym
        gate_handle = self.gym.create_actor(
            env_handle,
            gate_asset,
            gate_pose,
            "gate",
            0,  # Collision group
            1   # Collision filter
        )
        self.gate_handles.append(gate_handle)
        
        # Add gate to Warp environment for camera rendering
        asset_info_dict = {
            "warp_asset": self.warp_gate_asset,
            "filename": gate_asset_params.file
        }
        
        # Add gate asset to environment with proper parameters
        self.warp_env.add_asset_to_env(
            asset_info_dict=asset_info_dict,
            env_id=0,
            global_asset_counter=0,
            segmentation_counter=1
        )
        
        # Add some trees behind the gate for better visualization and diagnosis
        self._add_background_trees()
        
        print("✓ Environment created with gate for both Isaac Gym and Warp!")
    
    def _add_background_trees(self):
        """Add some trees behind the gate for better camera visualization."""
        print("Adding background trees...")
        
        try:
            # Load a tree asset from the existing tree models
            tree_asset_folder = f"{gate_asset_params.asset_folder}/../trees"
            available_trees = ["tree_0.urdf", "tree_1.urdf", "tree_2.urdf", "tree_3.urdf", "tree_4.urdf"]
            
            asset_counter = 1  # Start from 1 since gate is 0
            
            # Add trees at various positions behind the gate (negative X from camera perspective)
            tree_positions = [
                (-3.0, -2.0, 0.0),  # Behind and left of gate
                (-4.0, 0.0, 0.0),   # Directly behind gate
                (-3.0, 2.0, 0.0),   # Behind and right of gate
                (-5.0, -1.0, 0.0),  # Further behind, left
                (-5.0, 1.0, 0.0),   # Further behind, right
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
                
                # Load tree asset for Isaac Gym
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
                
                # Create tree actor in Isaac Gym
                tree_pose = gymapi.Transform()
                tree_pose.p = gymapi.Vec3(x, y, z)
                tree_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                
                tree_handle = self.gym.create_actor(
                    self.env_handles[0],
                    tree_asset,
                    tree_pose,
                    f"tree_{i}",
                    0,  # Collision group
                    1   # Collision filter
                )
                
                # Create WarpAsset for camera rendering
                tree_loading_options = {
                    'semantic_id': 100 + i,  # Different semantic ID for trees
                    'per_link_semantic': False,
                    'semantic_masked_links': {},
                    'use_collision_mesh_instead_of_visual': False
                }
                
                warp_tree_asset = WarpAsset(
                    asset_name=f"tree_{i}",
                    asset_file=tree_path,
                    loading_options=tree_loading_options
                )
                
                # Add tree to Warp environment
                tree_asset_info_dict = {
                    "warp_asset": warp_tree_asset,
                    "filename": tree_file
                }
                
                self.warp_env.add_asset_to_env(
                    asset_info_dict=tree_asset_info_dict,
                    env_id=0,
                    global_asset_counter=asset_counter,
                    segmentation_counter=100 + i  # Different segmentation for trees
                )
                
                asset_counter += 1
                print(f"Added tree {i+1} at position ({x}, {y}, {z})")
                
        except Exception as e:
            print(f"Warning: Failed to add background trees: {e}")
            print("Continuing without trees...")
    
    def _setup_static_camera(self):
        """Setup static camera using Warp."""
        print("Setting up static camera...")
        
        # Create necessary tensors for WarpEnv
        # The WarpEnv expects asset state tensors for tracking transformations
        # Since our gate is static, we can create simple identity transforms
        num_assets = 1  # Just the gate
        
        # Create asset state tensor: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, scale, vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
        # We now have more assets: gate + trees
        total_assets = 6  # gate + up to 5 trees
        asset_state = torch.zeros((self.num_envs, total_assets, 13), device=self.device)
        
        # Gate at origin
        asset_state[:, 0, 0:3] = torch.tensor([0.0, 0.0, 0.0], device=self.device)  # Gate position
        asset_state[:, 0, 6] = 1.0  # quat_w (identity quaternion)
        asset_state[:, 0, 7] = 1.0  # scale
        
        # Trees behind gate (negative X)
        tree_positions = [
            (-3.0, -2.0, 0.0), (-4.0, 0.0, 0.0), (-3.0, 2.0, 0.0), 
            (-5.0, -1.0, 0.0), (-5.0, 1.0, 0.0)
        ]
        for i, (x, y, z) in enumerate(tree_positions):
            asset_state[:, i+1, 0:3] = torch.tensor([x, y, z], device=self.device)
            asset_state[:, i+1, 6] = 1.0  # quat_w (identity quaternion) 
            asset_state[:, i+1, 7] = 1.0  # scale
        
        global_tensor_dict = {
            'unfolded_env_asset_state_tensor': asset_state.view(-1, 13)
        }
        
        # Prepare Warp environment for simulation
        try:
            self.warp_env.prepare_for_simulation(global_tensor_dict)
            
            # Get mesh IDs for camera
            if self.warp_env.CONST_WARP_MESH_ID_LIST is not None:
                mesh_ids_array = wp.array(self.warp_env.CONST_WARP_MESH_ID_LIST, dtype=wp.uint64, device=self.device)
            else:
                print("Warning: No meshes found in WarpEnv, creating dummy mesh array")
                mesh_ids_array = wp.array([0], dtype=wp.uint64, device=self.device)
                
        except Exception as e:
            print(f"Warning: WarpEnv preparation failed: {e}")
            print("Continuing with simplified camera setup...")
            mesh_ids_array = wp.array([0], dtype=wp.uint64, device=self.device)
        
        # Camera configuration - create a simple config object
        class CameraConfig:
            def __init__(self, parent):
                self.width = parent.camera_width
                self.height = parent.camera_height
                self.horizontal_fov_deg = parent.camera_fov
                self.max_range = 20.0  # Reduced max range
                self.min_range = 0.1
                self.calculate_depth = True
                self.return_pointcloud = False
                self.pointcloud_in_world_frame = True
                self.segmentation_camera = True
                self.num_sensors = 1
                self.normalize_range = True
                self.near_out_of_range_value = 0.0
                self.far_out_of_range_value = 1.0
        
        camera_config = CameraConfig(self)
        
        # Create static environment camera
        try:
            self.static_camera = StaticEnvironmentCamera(
                camera_config=camera_config,
                num_envs=self.num_envs,
                mesh_ids_array=mesh_ids_array,
                device=self.device
            )
            
            # Set camera pose to look at the gate from the front
            # Position camera 8 meters in front of gate, 3 meters high for better viewing angle
            camera_positions = torch.tensor([[8.0, 0.0, 3.0]], device=self.device).repeat(self.num_envs, 1, 1)
            
            # Create orientation looking toward gate center 
            # The gate is at (0,0,0) and extends upward, so look slightly down from camera position
            import math
            from aerial_gym.utils.math import quat_from_euler_xyz_tensor
            
            # Calculate look-at direction manually:
            # Camera at (8, 0, 3) looking at gate center approximately (0, 0, 1.5)
            # This means: look in direction (-8, 0, -1.5) normalized
            
            # Use a simple downward-looking orientation with rotation toward negative X
            # Euler angles: roll=0, pitch=-10deg (look down slightly), yaw=180deg (face negative X direction)
            euler_angles = torch.tensor([0.0, -math.pi/18, math.pi], device=self.device)
            orientation_quat = quat_from_euler_xyz_tensor(euler_angles)
            camera_orientations = orientation_quat.unsqueeze(0).unsqueeze(0).repeat(self.num_envs, 1, 1)
            
            self.static_camera.set_camera_poses(camera_positions, camera_orientations)
            
            print(f"Camera positioned at: (8.0, 0.0, 3.0)")
            print(f"Camera orientation (roll, pitch, yaw): (0°, -10°, 180°)")
            print(f"Gate position: (0.0, 0.0, 0.0)")
            print(f"Expected objects in view: Gate at center, trees in background")
            
            # Set up image tensors for capturing
            self.depth_buffer = torch.zeros((self.num_envs, 1, self.camera_height, self.camera_width), 
                                            device=self.device, dtype=torch.float32)
            self.segmentation_buffer = torch.zeros((self.num_envs, 1, self.camera_height, self.camera_width), 
                                                   device=self.device, dtype=torch.int32)
            
            self.static_camera.set_image_tensors(self.depth_buffer, self.segmentation_buffer)
            
            print("✓ Static camera setup complete!")
            
        except Exception as e:
            print(f"Warning: Static camera setup failed: {e}")
            print("Continuing without camera functionality...")
            self.static_camera = None
    
    def _create_viewer(self):
        """Create the viewer for visualization."""
        if self.headless:
            return
        
        print("Creating viewer...")
        
        # Create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        
        if self.viewer is None:
            raise RuntimeError("Failed to create viewer")
    
    def _setup_camera(self):
        """Setup Isaac Gym viewer camera position."""
        if self.headless or self.viewer is None:
            return
        
        print("Setting up Isaac Gym viewer camera...")
        
        # Position viewer camera to see both the gate and the static camera position
        cam_pos = gymapi.Vec3(8.0, -8.0, 4.0)  # Position for angled view
        cam_target = gymapi.Vec3(0.0, 0.0, 1.5)  # Look at center of gate
        
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    
    def capture_camera_images(self):
        """Capture depth and segmentation images from the static camera."""
        if self.static_camera is None:
            return None, None
        
        try:
            # Capture images using the static camera
            depth_image = self.static_camera.capture()
            
            # Convert to numpy arrays for processing
            if depth_image is not None:
                depth_np = depth_image[0, 0].cpu().numpy()  # First environment, first camera
                # Normalize depth for visualization
                if depth_np.max() > 0:
                    depth_normalized = depth_np / depth_np.max()
                else:
                    depth_normalized = depth_np
                # Create a simple RGB representation from depth for visualization
                rgb_np = np.stack([depth_normalized, depth_normalized, depth_normalized], axis=-1)
                rgb_np = (rgb_np * 255).astype(np.uint8)
            else:
                rgb_np = None
                depth_np = None
                
            # Get segmentation if available
            if hasattr(self, 'segmentation_buffer'):
                seg_np = self.segmentation_buffer[0, 0].cpu().numpy()
            else:
                seg_np = None
            
            return rgb_np, depth_np
            
        except Exception as e:
            print(f"Warning: Camera capture failed: {e}")
            return None, None
    
    def save_camera_images(self, rgb_image, depth_image, step_count):
        """Save camera images to disk."""
        try:
            if rgb_image is not None:
                rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"gate_rgb_{step_count:04d}.jpg", rgb_bgr)
            
            if depth_image is not None:
                # Normalize depth for visualization
                if np.max(depth_image) > 0:
                    depth_vis = (depth_image / np.max(depth_image) * 255).astype(np.uint8)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    cv2.imwrite(f"gate_depth_{step_count:04d}.jpg", depth_vis)
        except Exception as e:
            print(f"Warning: Failed to save images at step {step_count}: {e}")
    
    def display_camera_images(self, rgb_image, depth_image):
        """Display camera images in OpenCV windows."""
        try:
            if rgb_image is not None:
                # Display RGB image
                cv2.imshow('Static Camera - RGB', rgb_image)
            
            if depth_image is not None:
                # Create colorized depth image for display
                if np.max(depth_image) > 0:
                    depth_normalized = depth_image / np.max(depth_image)
                    depth_vis = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    cv2.imshow('Static Camera - Depth', depth_vis)
                    
                    # Also show raw depth values
                    print(f"Depth range: {np.min(depth_image):.3f} to {np.max(depth_image):.3f}")
            
            # Check for ESC key to close windows
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                cv2.destroyAllWindows()
                
        except Exception as e:
            print(f"Warning: Failed to display images: {e}")
            
    def close_camera_windows(self):
        """Close camera display windows."""
        cv2.destroyAllWindows()
    
    def run_visualization(self, duration=60.0, save_images=True):
        """
        Run the visualization loop with camera capture.
        
        Args:
            duration (float): How long to run the visualization in seconds
            save_images (bool): Whether to save camera images to disk
        """
        print(f"Starting visualization with camera for {duration} seconds...")
        print("Controls:")
        print("  - Mouse: Look around")
        print("  - W/A/S/D: Move camera")
        print("  - Q/E: Move camera up/down")
        print("  - ESC: Exit")
        
        if save_images:
            print("Saving camera images to current directory...")
        
        # Prepare simulation
        self.gym.prepare_sim(self.sim)
        
        start_time = time.time()
        step_count = 0
        
        try:
            while not self.gym.query_viewer_has_closed(self.viewer):
                # Step simulation
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                
                # Capture camera images
                rgb_image, depth_image = self.capture_camera_images()
                
                # Display camera images in real-time
                if rgb_image is not None or depth_image is not None:
                    self.display_camera_images(rgb_image, depth_image)
                
                # Save images periodically
                if save_images and step_count % 30 == 0:  # Save every 30 steps (~0.5 seconds)
                    self.save_camera_images(rgb_image, depth_image, step_count)
                    if rgb_image is not None or depth_image is not None:
                        print(f"Step {step_count}: Camera captured images successfully!")
                    # Print depth stats every 60 steps to track what camera is seeing
                    if step_count % 60 == 0 and depth_image is not None:
                        print(f"  Depth stats - Min: {np.min(depth_image):.3f}, Max: {np.max(depth_image):.3f}, Mean: {np.mean(depth_image):.3f}")
                        unique_depths = np.unique(depth_image)
                        print(f"  Unique depth values: {len(unique_depths)} (showing first 5: {unique_depths[:5]})")
                
                # Update viewer
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                
                # Check if duration exceeded
                if time.time() - start_time > duration:
                    print(f"Visualization completed after {duration} seconds.")
                    break
                
                step_count += 1
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.016)  # ~60 FPS
                
        except KeyboardInterrupt:
            print("Visualization interrupted by user.")
        except Exception as e:
            print(f"Error during visualization loop: {e}")
            import traceback
            traceback.print_exc()
    
    def print_environment_info(self):
        """Print information about the created environment."""
        print("\n" + "="*60)
        print("GATE ENVIRONMENT WITH CAMERA INFORMATION")
        print("="*60)
        
        print(f"Number of environments: {len(self.env_handles)}")
        print(f"Number of gates: {len(self.gate_handles)}")
        print(f"Environment bounds: -8m to +8m in X,Y and 0m to +8m in Z")
        print(f"Gate position: Center of environment (0, 0, 0)")
        print(f"Gate dimensions: ~3m wide × 3m tall opening")
        
        if self.static_camera:
            print(f"Static camera: Active")
            print(f"Camera resolution: {self.camera_width}x{self.camera_height}")
            print(f"Camera FOV: {self.camera_fov} degrees")
            print(f"Camera position: (8.0, 0.0, 3.0)")
            print(f"Camera looking toward gate center")
            print(f"Camera will display live depth/RGB feeds in separate windows")
        else:
            print(f"Static camera: Failed to initialize")
        
        if self.viewer:
            print(f"Isaac Gym Viewer: Active (interactive)")
        else:
            print(f"Isaac Gym Viewer: Headless mode")
        
        print(f"Physics: Isaac Gym PhysX")
        print(f"Camera: NVIDIA Warp raycasting")
        print(f"Simulation timestep: {self.sim_params.dt:.4f}s")
        
        # Print Warp environment info
        if hasattr(self.warp_env, 'CONST_WARP_MESH_ID_LIST') and self.warp_env.CONST_WARP_MESH_ID_LIST:
            print(f"Warp meshes: {len(self.warp_env.CONST_WARP_MESH_ID_LIST)} loaded")
        else:
            print(f"Warp meshes: None loaded (camera may not work)")
        
        print("\nPress ESC in camera windows to close them.")
        print("The static camera should show depth/segmentation of the gate.")
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        
        # Close camera windows
        self.close_camera_windows()
        
        if self.static_camera:
            del self.static_camera
        
        if self.warp_env:
            del self.warp_env
        
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        
        if self.sim:
            self.gym.destroy_sim(self.sim)


def main():
    """Main function to run the gate visualization with camera."""
    
    print("GATE ENVIRONMENT WITH STATIC CAMERA VISUALIZATION")
    print("="*60)
    
    # Check if gate URDF exists
    gate_urdf_path = f"{gate_asset_params.asset_folder}/{gate_asset_params.file}"
    if not os.path.exists(gate_urdf_path):
        print(f"❌ Gate URDF file not found: {gate_urdf_path}")
        print("Please create the gate URDF file first!")
        return 1
    
    try:
        # Create visualizer with camera
        visualizer = GateEnvironmentWithCamera(headless=False)
        
        # Print environment information
        visualizer.print_environment_info()
        
        # Run visualization with camera capture
        visualizer.run_visualization(duration=300.0, save_images=True)  # 5 minutes
        
        # Cleanup
        visualizer.cleanup()
        
        print("✓ Gate visualization with camera completed successfully!")
        print("Camera images saved as gate_rgb_XXXX.jpg and gate_depth_XXXX.jpg")
        return 0
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 