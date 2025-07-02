"""
Simple Gate Visualization with Static Camera and X500 Robot with D455 Camera and Hover Control
=============================================================================================

This script creates a simple visualization of a gate environment with:
1. A static depth camera using Isaac Gym native API
2. An X500 quadrotor robot with D455 depth camera sensor
3. Lee position controller to make X500 hover in place
4. Real-time visualization of both camera outputs side-by-side

Key features:
- Gate environment with optional background trees
- X500 quadrotor robot with onboard D455 camera
- Lee position controller for hovering
- Static world-mounted camera positioned 3m from gate
- Real-time depth and segmentation image display from both cameras
- Dual camera system: Static + Agent camera with control
"""

import os
import sys
import time
import cv2
import numpy as np

# Isaac Gym must be imported first
from isaacgym import gymapi, gymtorch, gymutil

# Aerial Gym components
from aerial_gym.config.asset_config.gate_asset_config import gate_asset_params
from aerial_gym.config.robot_config.x500_config import X500Cfg
from aerial_gym.config.sensor_config.camera_config.d455_depth_config import RsD455Config
from aerial_gym.config.controller_config.lee_controller_config import control as lee_controller_config

# Controller imports
from aerial_gym.control.controllers.position_control import LeePositionController
from aerial_gym.control.control_allocation import ControlAllocator
from aerial_gym.utils.math import *

import torch

# Create X500 config with D455 camera enabled
class X500WithD455Cfg(X500Cfg):
    """X500 configuration with D455 camera sensor enabled."""
    
    class sensor_config(X500Cfg.sensor_config):
        enable_camera = True
        camera_config = RsD455Config
    
    class robot_asset(X500Cfg.robot_asset):
        name = "x500_with_d455"

class GateEnvironmentWithDualCamerasAndControl:
    """
    Gate environment with static camera and X500 robot with D455 camera and hover control.
    """
    
    def __init__(self, headless=False):
        """Initialize the environment."""
        print("🚁 Initializing Gate Environment with Dual Cameras and Hover Control...")
        
        self.headless = headless
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_envs = 1
        self.dt = 1.0 / 60.0  # 60 FPS simulation
        
        # Environment handles (Isaac Gym)
        self.env_handles = []
        self.gate_handles = []
        self.tree_handles = []
        self.x500_handles = []  # Add X500 robot handles
        
        # Static camera handles and setup
        self.camera_handles = []
        self.camera_setup_success = False
        
        # X500 camera handles
        self.x500_camera_handles = []
        
        # Control system
        self.controller = None
        self.control_allocator = None
        self.hover_position = torch.tensor([[-2.5, 0.0, 1.0]], device=self.device)  # Hover position
        self.hover_yaw = 0.0  # Hover yaw angle
        
        # Robot state tensors
        self.robot_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_orientation = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Initialize gym
        self._initialize_gym()
        self._create_sim()
        self._create_ground_plane()
        self._create_environments()
        self._setup_static_camera()
        self._setup_x500_with_camera()
        self._setup_controller()
        
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
    
    def _create_environments(self):
        """Create environment with gate and optional trees."""
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
        
        print("✓ Environment created with gate!")
    
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
            
            print(f"Static camera properties (D455 specs): {camera_props.width}x{camera_props.height}, FOV: {camera_props.horizontal_fov}°")
            print(f"Static camera depth range: {camera_props.near_plane}m - {camera_props.far_plane}m")
        
            # Create camera sensor in each environment
            self.camera_handles = []
            for i, env_handle in enumerate(self.env_handles):
                cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                self.camera_handles.append(cam_handle)
                print(f"Created static camera sensor {i} in environment {i}")
            
            # Position camera to face the gate directly (gate now rotated 90°)
            camera_pos = gymapi.Vec3(0.0, -3.0, 1.5)  # 3m in front of gate, at gate center height
            camera_target = gymapi.Vec3(0.0, 0.0, 1.5)  # Look directly at gate center
            
            # Set camera transform for each environment using look_at
            for i, (env_handle, cam_handle) in enumerate(zip(self.env_handles, self.camera_handles)):
                # Use Isaac Gym's camera look_at functionality
                self.gym.set_camera_location(cam_handle, env_handle, camera_pos, camera_target)
                print(f"Set static camera {i} to look from ({camera_pos.x}, {camera_pos.y}, {camera_pos.z}) toward ({camera_target.x}, {camera_target.y}, {camera_target.z})")
            
            print(f"✓ Static cameras positioned to face gate directly")
            
            self.camera_setup_success = True
            
        except Exception as e:
            print(f"❌ ERROR: Isaac Gym static camera setup failed: {e}")
            import traceback
            traceback.print_exc()
            self.camera_setup_success = False
    
    def _load_x500_asset(self):
        """Load X500 robot asset with D455 camera."""
        print(f"Loading X500 robot asset from: {X500WithD455Cfg.robot_asset.file}")
        
        # Asset loading options for X500
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = X500WithD455Cfg.robot_asset.fix_base_link
        asset_options.collapse_fixed_joints = X500WithD455Cfg.robot_asset.collapse_fixed_joints
        asset_options.disable_gravity = X500WithD455Cfg.robot_asset.disable_gravity
        asset_options.replace_cylinder_with_capsule = X500WithD455Cfg.robot_asset.replace_cylinder_with_capsule
        asset_options.mesh_normal_mode = gymapi.FROM_ASSET
        asset_options.flip_visual_attachments = X500WithD455Cfg.robot_asset.flip_visual_attachments
        asset_options.density = X500WithD455Cfg.robot_asset.density
        asset_options.angular_damping = X500WithD455Cfg.robot_asset.angular_damping
        asset_options.linear_damping = X500WithD455Cfg.robot_asset.linear_damping
        asset_options.max_angular_velocity = X500WithD455Cfg.robot_asset.max_angular_velocity
        asset_options.max_linear_velocity = X500WithD455Cfg.robot_asset.max_linear_velocity
        asset_options.armature = X500WithD455Cfg.robot_asset.armature
        
        # Load X500 asset
        x500_asset = self.gym.load_asset(
            self.sim, 
            X500WithD455Cfg.robot_asset.asset_folder, 
            X500WithD455Cfg.robot_asset.file, 
            asset_options
        )
        
        if x500_asset is None:
            raise Exception(f"Failed to load X500 robot asset: {X500WithD455Cfg.robot_asset.file}")
        
        print("✓ X500 robot asset loaded successfully")
        return x500_asset

    def _setup_x500_with_camera(self):
        """Setup X500 drone with D455 camera in the same environment."""
        print("Setting up X500 drone with D455 camera...")
        
        try:
            # Load X500 asset
            x500_asset = self._load_x500_asset()
            
            # Get the first environment handle
            env_handle = self.env_handles[0]
            
            # Create X500 robot actor
            x500_pose = gymapi.Transform()
            x500_pose.p = gymapi.Vec3(-2.5, 0.0, 1.0)  # Position X500 further left of gate, 1m above ground
            x500_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # Default orientation
            
            x500_handle = self.gym.create_actor(
                env_handle,
                x500_asset,
                x500_pose,
                "x500",
                0,  # Collision group
                1   # Collision filter
            )
            self.x500_handles = [x500_handle]
            
            # Set X500 color for better visibility
            self.gym.set_rigid_body_color(
                env_handle, x500_handle, 0, 
                gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.8, 0.2)  # Yellow X500
            )
            
            # Setup D455 camera on X500
            self._setup_x500_camera(env_handle, x500_handle)
            
            print("✓ X500 with D455 camera setup complete")
            
        except Exception as e:
            print(f"❌ ERROR: X500 with camera setup failed: {e}")
            import traceback
            traceback.print_exc()
            self.x500_handles = []
    
    def _setup_x500_camera(self, env_handle, x500_handle):
        """Setup D455 camera on X500 drone."""
        print("Setting up D455 camera on X500...")
        
        try:
            # Camera properties (D455 specifications from config)
            camera_props = gymapi.CameraProperties()
            camera_props.width = RsD455Config.width  # 480
            camera_props.height = RsD455Config.height  # 270
            camera_props.horizontal_fov = RsD455Config.horizontal_fov_deg  # 87.0
            camera_props.near_plane = RsD455Config.min_range  # 0.2
            camera_props.far_plane = RsD455Config.max_range  # 15.0
            camera_props.enable_tensors = True  # Enable GPU tensor access
            
            print(f"X500 D455 camera properties: {camera_props.width}x{camera_props.height}, FOV: {camera_props.horizontal_fov}°")
            print(f"X500 D455 depth range: {camera_props.near_plane}m - {camera_props.far_plane}m")
            
            # Create camera sensor attached to X500
            x500_cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            print(f"Debug: Created X500 camera sensor with handle: {x500_cam_handle}")
            
            if x500_cam_handle is None:
                raise Exception("Failed to create X500 camera sensor")
            
            # Attach camera to X500 base_link with forward-facing orientation
            # Camera position relative to X500 base_link (slightly forward and up)
            cam_pos = gymapi.Vec3(0.10, 0.0, 0.03)  # 10cm forward, 3cm up from center
            
            # Attach camera to X500
            self.gym.attach_camera_to_body(
                x500_cam_handle, env_handle, x500_handle,
                gymapi.Transform(p=cam_pos), gymapi.FOLLOW_TRANSFORM
            )
            print(f"Debug: Attached camera to X500 body at position ({cam_pos.x}, {cam_pos.y}, {cam_pos.z})")
            
            # Store camera handle
            self.x500_camera_handles = [x500_cam_handle]
            
            print("✓ D455 camera attached to X500")
            print(f"Debug: X500 camera handles stored: {self.x500_camera_handles}")
            
        except Exception as e:
            print(f"❌ ERROR: X500 camera setup failed: {e}")
            import traceback
            traceback.print_exc()
            self.x500_camera_handles = []
    
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
    
    def capture_all_camera_images(self):
        """Capture depth and segmentation images from both cameras."""
        static_depth, static_seg = None, None
        x500_depth, x500_seg = None, None
        
        try:
            # Step graphics and render all cameras once
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            
            env_handle = self.env_handles[0]
            
            # Capture static camera images
            if self.camera_setup_success and len(self.camera_handles) > 0:
                try:
                    cam_handle = self.camera_handles[0]
                    
                    # Get depth image
                    depth_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH
                    )
                    static_depth = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
                    
                    # Get segmentation image
                    seg_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env_handle, cam_handle, gymapi.IMAGE_SEGMENTATION
                    )
                    static_seg = gymtorch.wrap_tensor(seg_tensor).cpu().numpy()
                    
                except Exception as e:
                    print(f"Error capturing static camera: {e}")
            
            # Capture X500 camera images
            if len(self.x500_camera_handles) > 0:
                try:
                    x500_cam_handle = self.x500_camera_handles[0]
                    print(f"Debug: Attempting to capture from X500 camera handle: {x500_cam_handle}")
                    
                    # Get depth image
                    depth_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env_handle, x500_cam_handle, gymapi.IMAGE_DEPTH
                    )
                    x500_depth = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
                    print(f"Debug: X500 depth image shape: {x500_depth.shape}, range: {x500_depth.min():.2f} to {x500_depth.max():.2f}")
                    
                    # Get segmentation image
                    seg_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env_handle, x500_cam_handle, gymapi.IMAGE_SEGMENTATION
                    )
                    x500_seg = gymtorch.wrap_tensor(seg_tensor).cpu().numpy()
                    print(f"Debug: X500 seg image shape: {x500_seg.shape}, unique values: {len(np.unique(x500_seg))}")
                    
                except Exception as e:
                    print(f"Error capturing X500 camera: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Debug: No X500 camera handles available")
            
            # End access to image tensors
            self.gym.end_access_image_tensors(self.sim)
            
            return (static_depth, static_seg), (x500_depth, x500_seg)
                
        except Exception as e:
            print(f"Error in capture_all_camera_images: {e}")
            import traceback
            traceback.print_exc()
            return (None, None), (None, None)
    
    def create_combined_image(self, depth_img, seg_img, title="Camera"):
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
    
    def _setup_controller(self):
        """Setup Lee position controller for X500 hovering."""
        print("Setting up Lee position controller for X500...")
        
        try:
            # Initialize controller
            self.controller = LeePositionController(
                config=lee_controller_config,
                num_envs=self.num_envs,
                device=self.device
            )
            
            # Initialize control allocator
            self.control_allocator = ControlAllocator(
                num_envs=self.num_envs,
                dt=self.dt,
                config=X500WithD455Cfg.control_allocator_config,
                device=self.device
            )
            
            # Create additional required tensors for controller
            self.robot_euler_angles = torch.zeros((self.num_envs, 3), device=self.device)
            self.robot_body_angvel = torch.zeros((self.num_envs, 3), device=self.device) 
            self.robot_body_linvel = torch.zeros((self.num_envs, 3), device=self.device)
            self.robot_vehicle_orientation = torch.zeros((self.num_envs, 4), device=self.device)
            self.robot_vehicle_linvel = torch.zeros((self.num_envs, 3), device=self.device)
            self.robot_inertia = torch.eye(3, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1)
            self.robot_mass = torch.ones((self.num_envs,), device=self.device) * 1.5  # X500 mass ~1.5kg
            self.gravity = torch.tensor([0, 0, -9.81], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
            
            # Create global tensor dict for controller initialization
            global_tensor_dict = {
                "robot_position": self.robot_position,
                "robot_orientation": self.robot_orientation,
                "robot_euler_angles": self.robot_euler_angles,
                "robot_linvel": self.robot_linvel,
                "robot_angvel": self.robot_angvel,
                "robot_body_angvel": self.robot_body_angvel,
                "robot_body_linvel": self.robot_body_linvel,
                "robot_vehicle_orientation": self.robot_vehicle_orientation,
                "robot_vehicle_linvel": self.robot_vehicle_linvel,
                "robot_inertia": self.robot_inertia,
                "robot_mass": self.robot_mass,
                "gravity": self.gravity,
                "robot_force_tensor": torch.zeros((self.num_envs, 1, 3), device=self.device),
                "robot_torque_tensor": torch.zeros((self.num_envs, 1, 3), device=self.device),
            }
            
            # Initialize controller tensors
            self.controller.init_tensors(global_tensor_dict)
            
            print("✓ Lee position controller initialized")
            print(f"Hover target: {self.hover_position[0].cpu().numpy()}")
            
        except Exception as e:
            print(f"❌ ERROR: Controller setup failed: {e}")
            import traceback
            traceback.print_exc()
            self.controller = None
            self.control_allocator = None

    def _update_robot_state(self):
        """Update robot state from Isaac Gym simulation."""
        if len(self.x500_handles) == 0:
            return
            
        try:
            env_handle = self.env_handles[0]
            x500_handle = self.x500_handles[0]
            
            # Get robot state from Isaac Gym using root state tensor
            actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
            root_states = gymtorch.wrap_tensor(actor_root_state)
            
            # X500 is the second actor (index 1, gate is index 0)
            x500_state = root_states[1]  # [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, vel_x, vel_y, vel_z, angvel_x, angvel_y, angvel_z]
            
            # Update position and orientation
            self.robot_position[0] = x500_state[0:3]
            self.robot_orientation[0] = x500_state[3:7]  # [qx, qy, qz, qw]
            self.robot_linvel[0] = x500_state[7:10]
            self.robot_angvel[0] = x500_state[10:13]
            
            # Update derived quantities
            self.robot_vehicle_orientation[0] = self.robot_orientation[0]
            self.robot_vehicle_linvel[0] = self.robot_linvel[0]
            self.robot_body_angvel[0] = self.robot_angvel[0]
            self.robot_body_linvel[0] = self.robot_linvel[0]  # Body linear velocity same as world linear velocity for simplicity
            
            # Convert quaternion to euler angles
            from aerial_gym.utils.math import quat_to_euler_xyz
            self.robot_euler_angles[0] = quat_to_euler_xyz(self.robot_orientation[0:1])[0]
            
        except Exception as e:
            print(f"Error updating robot state: {e}")
            import traceback
            traceback.print_exc()

    def _apply_control(self):
        """Apply Lee position controller to make X500 hover."""
        if self.controller is None or self.control_allocator is None:
            return
            
        try:
            # Update robot state
            self._update_robot_state()
            
            # Create command actions for position control
            # [x, y, z, yaw] in world frame
            command_actions = torch.zeros((self.num_envs, 4), device=self.device)
            command_actions[0, 0:3] = self.hover_position[0]  # Target position
            command_actions[0, 3] = self.hover_yaw  # Target yaw
            
            # Compute control wrench using Lee controller
            wrench_command = self.controller.update(command_actions)
            
            # Allocate control to motors
            forces, torques = self.control_allocator.allocate_output(wrench_command, "wrench")
            
            # Apply forces and torques to X500 using direct tensor manipulation
            if len(self.x500_handles) > 0:
                try:
                    # Get force and torque tensors
                    force_tensor = self.gym.acquire_force_tensor(self.sim)
                    forces = gymtorch.wrap_tensor(force_tensor)
                    
                    # X500 is actor index 1 (gate is 0), base link is index 0
                    x500_actor_idx = 1
                    base_link_idx = 0
                    
                    # Apply wrench to base link
                    forces[x500_actor_idx, base_link_idx, 0:3] = wrench_command[0, 0:3]  # Force
                    forces[x500_actor_idx, base_link_idx, 3:6] = wrench_command[0, 3:6]  # Torque
                    
                except Exception as e:
                    print(f"Error applying forces via tensor: {e}")
                    # Fallback: try direct API calls
                    try:
                        env_handle = self.env_handles[0]
                        x500_handle = self.x500_handles[0]
                        
                        total_force = wrench_command[0, 0:3].cpu().numpy()
                        total_torque = wrench_command[0, 3:6].cpu().numpy()
                        
                        # Simple force application
                        force_vec = gymapi.Vec3(float(total_force[0]), float(total_force[1]), float(total_force[2]))
                        
                        # Apply upward force to counteract gravity
                        self.gym.apply_rigid_body_force_tensors(
                            env_handle, x500_handle, 0, force_vec, gymapi.ENV_SPACE
                        )
                        
                    except Exception as e2:
                        print(f"Error applying forces via API: {e2}")
                        # Print debug info
                        print(f"Debug: Wrench command: {wrench_command[0].cpu().numpy()}")
                        print(f"Debug: Position error: {(self.hover_position[0] - self.robot_position[0]).cpu().numpy()}")
                    
        except Exception as e:
            print(f"Error in control application: {e}")

    def run_visualization(self, duration=60.0):
        """Run the visualization loop with control."""
        print(f"Starting visualization with hover control for {duration} seconds...")
        print("Press ESC to exit early")
        
        start_time = time.time()
        step_count = 0
        
        # OpenCV windows for camera displays
        cv2.namedWindow("Static Camera View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Static Camera View", 2560, 720)  # Side-by-side 1280x720 images
        
        if len(self.x500_camera_handles) > 0:
            cv2.namedWindow("X500 D455 Camera View", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("X500 D455 Camera View", 960, 540)  # D455 resolution scaled
        
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Check if duration exceeded
            if elapsed_time > duration:
                print(f"Visualization completed after {duration} seconds")
                break
            
            # Apply control to make X500 hover
            self._apply_control()
            
            # Step simulation
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            # Capture all camera images at once
            (static_depth, static_seg), (x500_depth, x500_seg) = self.capture_all_camera_images()
            
            # Display static camera images
            if static_depth is not None and static_seg is not None:
                # Create combined visualization for static camera
                static_combined = self.create_combined_image(static_depth, static_seg, "Static Camera")
                
                if static_combined is not None:
                    # Display static camera images
                    cv2.imshow("Static Camera View", static_combined)
                    
                    # Print depth statistics periodically
                    if step_count % 60 == 0:  # Every second at 60 FPS
                        # Filter out -inf values for better statistics
                        valid_depths = static_depth[static_depth != -np.inf]
                        if len(valid_depths) > 0:
                            # Use absolute values for meaningful distance measurements
                            abs_valid_depths = np.abs(valid_depths)
                            min_depth = np.min(abs_valid_depths)
                            max_depth = np.max(abs_valid_depths)
                            mean_depth = np.mean(abs_valid_depths)
                            total_pixels = static_depth.size
                            valid_ratio = len(valid_depths) / total_pixels * 100
                            print(f"Step {step_count}: Static Camera - Depth range: {min_depth:.2f}-{max_depth:.2f}m, "
                                  f"Mean: {mean_depth:.2f}m, Valid pixels: {valid_ratio:.1f}%")
                            
                            # Print X500 position for control monitoring
                            if len(self.x500_handles) > 0:
                                pos = self.robot_position[0].cpu().numpy()
                                target = self.hover_position[0].cpu().numpy()
                                error = np.linalg.norm(pos - target)
                                print(f"          X500 Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                                      f"Target: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}), "
                                      f"Error: {error:.3f}m")
            
            # Display X500 camera images
            if x500_depth is not None and x500_seg is not None:
                # Create combined visualization for X500 camera
                x500_combined = self.create_combined_image(x500_depth, x500_seg, "X500 D455")
                
                if x500_combined is not None:
                    # Display X500 camera images
                    cv2.imshow("X500 D455 Camera View", x500_combined)
                    
                    # Print X500 camera statistics periodically
                    if step_count % 60 == 0:
                        valid_depths = x500_depth[x500_depth != -np.inf]
                        if len(valid_depths) > 0:
                            abs_valid_depths = np.abs(valid_depths)
                            min_depth = np.min(abs_valid_depths)
                            max_depth = np.max(abs_valid_depths)
                            mean_depth = np.mean(abs_valid_depths)
                            total_pixels = x500_depth.size
                            valid_ratio = len(valid_depths) / total_pixels * 100
                            print(f"          X500 D455 - Depth range: {min_depth:.2f}-{max_depth:.2f}m, "
                                  f"Mean: {mean_depth:.2f}m, Valid pixels: {valid_ratio:.1f}%")
            elif step_count % 60 == 0:
                print(f"          X500 D455 - No camera data available")
            
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
        print("DUAL CAMERA ENVIRONMENT WITH HOVER CONTROL")
        print("="*50)
        print(f"Number of environments: {self.num_envs}")
        print(f"Device: {self.device}")
        print(f"Gate asset: {gate_asset_params.file}")
        print(f"Static camera setup success: {self.camera_setup_success}")
        print(f"Number of static cameras: {len(self.camera_handles)}")
        print(f"Number of trees: {len(self.tree_handles)}")
        print(f"Number of X500 robots: {len(self.x500_handles)}")
        print(f"X500 with D455 setup: {'Success' if len(self.x500_handles) > 0 else 'Failed'}")
        print(f"Lee position controller: {'Active' if self.controller is not None else 'Failed'}")
        
        if self.camera_setup_success:
            print(f"Static camera position: (0, -3, 1.5)")
            print(f"Static camera target: (0, 0, 1.5) - gate center")
            print(f"Gate position: (0, 0, 0) - rotated 90°")
        
        if len(self.x500_handles) > 0:
            print(f"X500 initial position: (-2.5, 0, 1.0) - further left of gate, 1m above ground")
            print(f"X500 hover target: {self.hover_position[0].cpu().numpy()}")
            print(f"X500 with D455 camera: Active")
            print(f"X500 camera handles: {len(self.x500_camera_handles)} cameras")
            print(f"X500 camera specs: D455 (270x480, 87° FOV, 0.2-15m range)")
            if len(self.x500_camera_handles) > 0:
                print(f"X500 camera handle IDs: {self.x500_camera_handles}")
            else:
                print("⚠️  WARNING: X500 created but no camera handles found!")
        
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
        print("🚁 Starting Gate Visualization with Dual Cameras and Hover Control...")
        
        # Create environment
        env = GateEnvironmentWithDualCamerasAndControl(headless=False)
        
        # Print environment information
        env.print_environment_info()
        
        # Run visualization with control
        env.run_visualization(duration=120.0)
        
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