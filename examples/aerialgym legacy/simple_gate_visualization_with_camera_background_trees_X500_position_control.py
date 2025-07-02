"""
Simple Gate Visualization with Static Camera, X500 Robot with D455 Camera, and Position Control
===============================================================================================

This script creates a simple visualization of a gate environment with:
1. A static depth camera using Isaac Gym native API
2. An X500 quadrotor robot with D455 depth camera sensor
3. Lee position controller for X500 hovering at a target position
4. Real-time visualization of both camera outputs side-by-side

Key features:
- Gate environment with optional background trees
- X500 quadrotor robot with onboard D455 camera
- Static world-mounted camera positioned 3m from gate
- Lee position controller for stable hovering
- Real-time depth and segmentation image display from both cameras
- Dual camera system: Static + Agent camera
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
from aerial_gym.control.controllers.position_control import LeePositionController
from aerial_gym.control.control_allocation import ControlAllocator
from aerial_gym.utils.math import *

import torch

# Create X500 config with D455 camera enabled and fixed control allocation
class X500WithD455Cfg(X500Cfg):
    """X500 configuration with D455 camera sensor enabled."""
    
    class sensor_config(X500Cfg.sensor_config):
        enable_camera = True
        camera_config = RsD455Config
    
    class robot_asset(X500Cfg.robot_asset):
        name = "x500_with_d455"

class GateEnvironmentWithPositionControl:
    """
    Gate environment with static camera, X500 robot with D455 camera, and Lee position controller for hovering.
    """
    
    def __init__(self, headless=False):
        """Initialize the environment."""
        print("🚁 Initializing Gate Environment with Position Control...")
        
        self.headless = headless
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_envs = 1
        self.dt = 1.0 / 60.0  # 60 FPS simulation
        
        # Environment handles (Isaac Gym)
        self.env_handles = []
        self.gate_handles = []
        self.tree_handles = []
        self.x500_handles = []
        
        # Static camera handles
        self.camera_handles = []
        self.camera_setup_success = False
        
        # X500 camera handles
        self.x500_camera_handles = []
        
        # Control system
        self.lee_controller = None
        self.control_allocator = None
        
        # Set hover target (just slightly above starting position for testing)
        self.hover_target = torch.tensor([-2.5, 0.0, 1.2], device=self.device)  # 20cm above start
        self.current_target = self.hover_target.clone()
        
        # Robot state tensors (for Isaac Gym rigid body states)
        self.x500_rigid_body_states = None
        
        # Controller state tensors (derived from rigid body states)
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
        self._setup_lee_position_controller()
        
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
        sim_params.dt = self.dt
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
            
            # Set X500 initial position and orientation
            x500_start_pos = gymapi.Vec3(-2.5, 0.0, 1.0)  # Start at hover target position
            x500_start_rot = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # No rotation
            x500_pose = gymapi.Transform(x500_start_pos, x500_start_rot)
            
            # Create X500 robot actor
            x500_handle = self.gym.create_actor(
                env_handle, x500_asset, x500_pose, "x500", 0, 0
            )
            
            # Debug: Verify the actual spawn position
            actual_pose = self.gym.get_actor_rigid_body_states(env_handle, x500_handle, gymapi.STATE_POS)
            print(f"Debug: X500 actual spawn position: {actual_pose['pose']['p']}")
            print(f"Debug: X500 intended spawn position: (-2.5, 0.0, 1.0)")
            
            # Enable actor DOF for control
            self.gym.enable_actor_dof_force_sensors(env_handle, x500_handle)
            
            self.x500_handles = [x500_handle]
            
            # Set X500 color for better visibility
            self.gym.set_rigid_body_color(
                env_handle, x500_handle, 0, 
                gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.8, 0.2)  # Yellow X500
            )
            
            # Setup D455 camera on X500
            self._setup_x500_camera(env_handle, x500_handle)
            
            # Get rigid body state tensor for X500
            self._setup_x500_state_tensors()
            
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
    
    def _setup_x500_state_tensors(self):
        """Set up state tensors for X500 robot control."""
        try:
            print("Debug: Setting up X500 state tensors...")
            
            # Initialize state tensors on CPU (Isaac Gym physics runs on CPU)
            device = "cpu"  # Force CPU to match Isaac Gym physics
            
            # Robot state tensors (1 environment)
            self.robot_position = torch.zeros((1, 3), device=device, dtype=torch.float32)
            self.robot_orientation = torch.zeros((1, 4), device=device, dtype=torch.float32)
            self.robot_linvel = torch.zeros((1, 3), device=device, dtype=torch.float32)
            self.robot_angvel = torch.zeros((1, 3), device=device, dtype=torch.float32)
            self.robot_euler_angles = torch.zeros((1, 3), device=device, dtype=torch.float32)
            
            # Body frame velocities
            self.robot_body_linvel = torch.zeros((1, 3), device=device, dtype=torch.float32)
            self.robot_body_angvel = torch.zeros((1, 3), device=device, dtype=torch.float32)
            
            # Vehicle frame states
            self.robot_vehicle_orientation = torch.zeros((1, 4), device=device, dtype=torch.float32)
            self.robot_vehicle_linvel = torch.zeros((1, 3), device=device, dtype=torch.float32)
            
            # Get Isaac Gym actor root state tensor
            actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
            self.actor_root_states = gymtorch.wrap_tensor(actor_root_state)
            
            # CRITICAL: Refresh the tensor to get actual actor positions after simulation setup
            self.gym.refresh_actor_root_state_tensor(self.sim)
            
            print(f"Debug: Total actors in root state tensor: {len(self.actor_root_states)}")
            
            # Debug: Print all actor positions to identify X500
            for i in range(len(self.actor_root_states)):
                pos = self.actor_root_states[i][:3]
                print(f"Debug: Actor {i}: pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            
            # Find X500 actor by looking for the one closest to expected spawn position
            expected_pos = torch.tensor([-2.5, 0.0, 1.0], device="cpu")
            self.x500_actor_idx = None
            min_distance = float('inf')
            
            for i in range(len(self.actor_root_states)):
                actor_pos = self.actor_root_states[i][:3].cpu()
                distance = torch.norm(actor_pos - expected_pos).item()
                if distance < min_distance:
                    min_distance = distance
                    self.x500_actor_idx = i
            
            if self.x500_actor_idx is not None and min_distance < 1.0:  # Within 1m tolerance
                x500_pos = self.actor_root_states[self.x500_actor_idx][:3].cpu()
                print(f"✓ X500 actor found at index {self.x500_actor_idx}, position: {x500_pos}")
                self._update_robot_state()  # Initial state update
            else:
                print(f"❌ Could not find X500 actor (closest distance: {min_distance:.2f}m)")
                # Fallback: assume it's the last actor (often the case)
                self.x500_actor_idx = len(self.actor_root_states) - 1
                print(f"Using fallback X500 actor index: {self.x500_actor_idx}")
                self._update_robot_state()
                
        except Exception as e:
            print(f"❌ ERROR: X500 state tensor setup failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_robot_state(self):
        """Update robot state tensors from simulation."""
        try:
            # Refresh all relevant tensors from simulation
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            
            # Update robot state from root state tensor
            # Root state tensor format: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, vel_x, vel_y, vel_z, angvel_x, angvel_y, angvel_z]
            x500_state = self.actor_root_states[self.x500_actor_idx]
            
            # Update position, orientation, and velocities
            self.robot_position[0, :] = x500_state[0:3]
            self.robot_orientation[0, :] = x500_state[3:7]  # quaternion
            self.robot_linvel[0, :] = x500_state[7:10]
            self.robot_angvel[0, :] = x500_state[10:13]
            
            # Convert quaternion to euler angles for controller
            from aerial_gym.utils.math import get_euler_xyz_tensor
            self.robot_euler_angles[0] = get_euler_xyz_tensor(self.robot_orientation[0].unsqueeze(0))[0]
            
            # Convert global velocities to body frame (simplified approach)
            self.robot_body_linvel[0] = self.robot_linvel[0]
            self.robot_body_angvel[0] = self.robot_angvel[0]
            
            # Update vehicle orientation and velocity for controller
            self.robot_vehicle_orientation[0] = self.robot_orientation[0]
            self.robot_vehicle_linvel[0] = self.robot_linvel[0]
                
        except Exception as e:
            print(f"Error updating robot state: {e}")
            import traceback
            traceback.print_exc()
    
    def _step_physics_and_update_state(self):
        """Step physics simulation and update robot state - following DCE RL pattern."""
        try:
            # Step the physics simulation
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            # Update robot state from simulation (this is the key missing piece!)
            self._update_robot_state()
            
        except Exception as e:
            print(f"Error in physics step: {e}")
            import traceback
            traceback.print_exc()

    def _setup_lee_position_controller(self):
        """Set up Lee position controller for X500."""
        print("Setting up Lee position controller for X500...")
        
        try:
            # Import Lee position controller from the correct module
            from aerial_gym.control.controllers.position_control import LeePositionController
            from aerial_gym.control.control_allocation import ControlAllocator
            from aerial_gym.config.controller_config.lee_controller_config import control as lee_controller_config
            
            # Initialize controller with proper device
            device = "cpu"  # Match Isaac Gym physics device
            dt = 1.0/60.0  # 60 Hz simulation
            
            # Create controller config
            config = lee_controller_config()
            
            # Override controller gains for better waypoint tracking
            config.K_pos_tensor_max = [5.0, 5.0, 4.0]  # Increased position gains (was [3.0, 3.0, 2.0])
            config.K_vel_tensor_max = [4.0, 4.0, 4.0]  # Increased velocity gains (was [3.0, 3.0, 3.0])
            
            # Get X500 motor indices for application mask
            env_handle = self.env_handles[0]
            x500_handle = self.x500_handles[0]
            
            # X500 motor names in order
            motor_names = ["front_right_prop", "back_left_prop", "front_left_prop", "back_right_prop"]
            self.application_mask = []
            
            for motor_name in motor_names:
                try:
                    motor_idx = self.gym.find_actor_rigid_body_index(
                        env_handle, x500_handle, motor_name, gymapi.DOMAIN_ACTOR
                    )
                    self.application_mask.append(motor_idx)
                except:
                    print(f"Warning: Could not find motor '{motor_name}', using fallback index")
                    self.application_mask.append(len(self.application_mask) + 1)
            
            print(f"X500 motor application mask: {self.application_mask}")
            
            # Create control allocator config object (similar to X500Cfg.control_allocator_config)
            class ControlAllocatorConfig:
                def __init__(self):
                    self.num_motors = 4
                    self.force_application_level = "body_link"  # Apply forces at body level (more stable)
                    self.motor_directions = [1, 1, -1, -1]  # X500 motor directions
                    
                    # X500 allocation matrix (improved for better rank)
                    # Motor layout: 0=front_right, 1=back_left, 2=front_left, 3=back_right
                    arm_length = 0.175  # X500 arm length in meters
                    self.allocation_matrix = [
                        [0.0, 0.0, 0.0, 0.0],           # Fx (no lateral thrust)
                        [0.0, 0.0, 0.0, 0.0],           # Fy (no lateral thrust)  
                        [1.0, 1.0, 1.0, 1.0],           # Fz (all motors provide upward thrust)
                        [arm_length, -arm_length, -arm_length, arm_length],   # Mx (roll moment)
                        [arm_length, arm_length, -arm_length, -arm_length],   # My (pitch moment)
                        [-0.02, 0.02, -0.02, 0.02]      # Mz (yaw moment from prop direction)
                    ]
                    
                    # Motor model config
                    class MotorModelConfig:
                        def __init__(self):
                            self.use_rps = True
                            self.motor_thrust_constant_min = 8.54858e-6
                            self.motor_thrust_constant_max = 8.54858e-6
                            self.motor_time_constant_increasing_min = 0.0125
                            self.motor_time_constant_increasing_max = 0.0125
                            self.motor_time_constant_decreasing_min = 0.025
                            self.motor_time_constant_decreasing_max = 0.025
                            self.max_thrust = 20.0
                            self.min_thrust = 0.0
                            self.max_thrust_rate = 100000.0
                            self.thrust_to_torque_ratio = 0.025
                            self.use_discrete_approximation = False
                    
                    self.motor_model_config = MotorModelConfig()
            
            # Create control allocator config and set application mask
            control_alloc_config = ControlAllocatorConfig()
            control_alloc_config.application_mask = self.application_mask
            
            # Initialize Lee position controller
            self.lee_controller = LeePositionController(config, 1, device)  # config, num_envs, device
            
            # Set up target position tensor
            self.target_position = torch.zeros((1, 3), device=device, dtype=torch.float32)
            
            # Initialize control allocator for X500 with proper config
            self.control_allocator = ControlAllocator(
                num_envs=1,
                dt=dt,
                config=control_alloc_config,
                device=device
            )
            
            # Initialize robot mass and inertia for controller
            self.robot_mass = torch.tensor([1.5], device=device, dtype=torch.float32)  # X500 mass ~1.5kg
            
            # X500 inertia matrix (approximate values for quadrotor)
            self.robot_inertia = torch.tensor([
                [0.029, 0.0, 0.0],
                [0.0, 0.029, 0.0], 
                [0.0, 0.0, 0.055]
            ], device=device, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
            # Initialize controller tensors with global tensor dict
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
                "gravity": torch.tensor([0.0, 0.0, -9.81], device=device).expand(1, 3),
            }
            
            self.lee_controller.init_tensors(global_tensor_dict)
            
            print("✓ Lee position controller initialized")
            print(f"Controller gains - Position: {config.K_pos_tensor_max}")
            print(f"Controller gains - Velocity: {config.K_vel_tensor_max}")
            print(f"Current hover target: {self.target_position[0].cpu().numpy()}")
            
        except Exception as e:
            print(f"❌ Failed to setup Lee position controller: {e}")
            import traceback
            traceback.print_exc()
            raise

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
    
    def _apply_control(self):
        """Apply Lee position controller to make X500 follow waypoints."""
        # This method is now integrated into the main control loop in run_navigation
        # Remove this method to avoid confusion
        pass

    def _render_frame(self):
        """Render the current frame."""
        if not self.headless:
            # Step graphics
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            
            # Handle viewer events
            if self.gym.query_viewer_has_closed(self.viewer):
                return False
        return True

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
                    
                    # Get depth image
                    depth_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env_handle, x500_cam_handle, gymapi.IMAGE_DEPTH
                    )
                    x500_depth = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
                    
                    # Get segmentation image
                    seg_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env_handle, x500_cam_handle, gymapi.IMAGE_SEGMENTATION
                    )
                    x500_seg = gymtorch.wrap_tensor(seg_tensor).cpu().numpy()
                    
                except Exception as e:
                    print(f"Error capturing X500 camera: {e}")
            
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
    
    def run_navigation(self, duration_seconds=180.0):
        """Main navigation loop with waypoint navigation."""
        print(f"Starting navigation and visualization for {duration_seconds} seconds...")
        print("Press ESC to exit early")
        
        # Initialize step counter for debug output
        self.debug_step_count = 0
        
        # Define waypoints for navigation through the gate
        waypoints = [
            torch.tensor([-2.5, 0.0, 0.8], device="cpu"),  # Start position (closer to actual hover height)
            torch.tensor([-1.0, 0.0, 1.2], device="cpu"),  # Approach gate
            torch.tensor([0.0, 0.0, 1.2], device="cpu"),   # Gate center
            torch.tensor([1.0, 0.0, 1.2], device="cpu"),   # Exit gate
            torch.tensor([2.5, 0.0, 0.8], device="cpu"),   # End position
            torch.tensor([2.5, 2.0, 1.5], device="cpu"),   # Move to side and up
            torch.tensor([0.0, 2.0, 1.8], device="cpu"),   # Above gate from side
            torch.tensor([-2.5, 0.0, 0.8], device="cpu"),  # Return to start
        ]
        
        # Waypoint navigation parameters
        current_waypoint_idx = 0
        waypoint_reach_threshold = 0.5  # 50cm threshold to consider waypoint reached (increased from 30cm)
        waypoint_hold_time = 2.0  # Hold at each waypoint for 2 seconds (reduced from 3)
        waypoint_reached_time = None
        
        print(f"🎯 Waypoint Navigation: X500 will navigate through {len(waypoints)} waypoints")
        print(f"   Waypoints: Start → Approach Gate → Through Gate → Exit → Side → Above → Return")
        
        start_time = time.time()
        
        while not self.gym.query_viewer_has_closed(self.viewer):
            current_time = time.time()
            if current_time - start_time > duration_seconds:
                break
                
            # Get current robot position
            current_pos = self.robot_position[0].clone()
            
            # Get current target waypoint
            current_target = waypoints[current_waypoint_idx]
            
            # Store current target for display function
            self.current_navigation_target = current_target
            self.current_waypoint_idx = current_waypoint_idx
            
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
                    print(f"🔄 Drone too far from waypoint {current_waypoint_idx + 1}, switching to closest waypoint {closest_waypoint_idx + 1}")
                    current_waypoint_idx = closest_waypoint_idx
                    current_target = waypoints[current_waypoint_idx]
                    waypoint_reached_time = None
                    distance_to_waypoint = min_distance
            
            if distance_to_waypoint <= waypoint_reach_threshold:
                if waypoint_reached_time is None:
                    waypoint_reached_time = current_time
                    print(f"🎯 Reached waypoint {current_waypoint_idx + 1}/{len(waypoints)}: [{current_target[0]:.2f}, {current_target[1]:.2f}, {current_target[2]:.2f}]")
                
                # Check if we've held at this waypoint long enough
                if current_time - waypoint_reached_time >= waypoint_hold_time:
                    # Move to next waypoint
                    current_waypoint_idx = (current_waypoint_idx + 1) % len(waypoints)
                    waypoint_reached_time = None
                    next_target = waypoints[current_waypoint_idx]
                    print(f"🚁 Moving to waypoint {current_waypoint_idx + 1}/{len(waypoints)}: [{next_target[0]:.2f}, {next_target[1]:.2f}, {next_target[2]:.2f}]")
            else:
                waypoint_reached_time = None  # Reset if we move away from waypoint
            
            # Debug control output every 300 steps (5 seconds at 60 FPS)
            if self.debug_step_count % 300 == 0:
                pos_np = current_pos.cpu().numpy()
                target_np = current_target.cpu().numpy()
                error = current_target - current_pos
                distance = torch.norm(error).item()
                print(f"Debug Control - Pos: [{pos_np[0]:.2f}, {pos_np[1]:.2f}, {pos_np[2]:.2f}], Target: [{target_np[0]:.2f}, {target_np[1]:.2f}, {target_np[2]:.2f}]")
                print(f"Debug Control - Error: [{error[0]:.2f}, {error[1]:.2f}, {error[2]:.2f}], Distance: {distance:.2f}m, Waypoint: {current_waypoint_idx + 1}/{len(waypoints)}")
            
            # Run Lee position controller
            try:
                # Set target position to current waypoint
                self.target_position[0] = current_target
                
                # Create command actions for Lee controller: [x, y, z, yaw]
                command_actions = torch.zeros((1, 4), device="cpu")
                command_actions[0, 0:3] = current_target  # Target position
                command_actions[0, 3] = 0.0  # Target yaw (face forward)
                
                # Ensure Lee controller exists
                if self.lee_controller is None:
                    print("❌ Lee controller not initialized, skipping control step")
                    continue
                
                # Compute control wrench using Lee position controller
                wrench = self.lee_controller.update(command_actions)
                
                # Validate wrench
                if wrench is None or torch.isnan(wrench).any() or torch.isinf(wrench).any():
                    print(f"❌ Invalid wrench generated: {wrench}")
                    continue
                
                # Debug wrench every 300 steps (5 seconds at 60 FPS)
                if self.debug_step_count % 300 == 0:
                    wrench_np = wrench[0].cpu().numpy()
                    print(f"Debug Control - Wrench: [{wrench_np[0]:.2f}, {wrench_np[1]:.2f}, {wrench_np[2]:.2f}, {wrench_np[3]:.2f}, {wrench_np[4]:.2f}, {wrench_np[5]:.2f}]")
                
                # Allocate motor forces using control allocator
                try:
                    # Use the correct method name and parameters for ControlAllocator
                    # Since we have a wrench from Lee controller, use "wrench" mode to convert wrench -> motor thrusts -> motor forces
                    motor_forces, motor_torques = self.control_allocator.allocate_output(wrench, "wrench")
                    
                    # Validate motor forces
                    if motor_forces is None or torch.isnan(motor_forces).any() or torch.isinf(motor_forces).any():
                        print(f"❌ Invalid motor forces generated: {motor_forces}")
                        continue
                    
                    # Debug motor forces every 300 steps (5 seconds at 60 FPS)
                    if self.debug_step_count % 300 == 0:
                        motor_forces_np = motor_forces[0].cpu().numpy()
                        motor_torques_np = motor_torques[0].cpu().numpy()
                        print(f"Debug Motor Forces (N): {motor_forces_np}")
                        print(f"Debug Motor Torques (Nm): {motor_torques_np}")
                        
                        # Check if forces are reasonable magnitude
                        total_thrust = torch.sum(motor_forces[0, :, 2]).item()  # Sum Z forces
                        print(f"Debug Total Thrust: {total_thrust:.2f}N (need ~{1.5*9.81:.1f}N to hover)")
                        
                    # Apply motor forces to X500
                    self._apply_motor_forces(motor_forces)
                    
                except Exception as e:
                    print(f"Control allocation error: {e}")
                    import traceback
                    traceback.print_exc()
                
            except Exception as e:
                print(f"Lee controller error: {e}")
                import traceback
                traceback.print_exc()
            
            # Step physics and update state (this is the crucial missing piece!)
            self._step_physics_and_update_state()
            
            # Update cameras and visualization
            self._update_cameras()
            self._render_frame()
            
            self.debug_step_count += 1
        
        print(f"\n✅ Navigation completed after {current_time - start_time:.1f} seconds")
    
    def print_environment_info(self):
        """Print information about the environment."""
        print("\n" + "="*60)
        print("DUAL CAMERA POSITION CONTROL ENVIRONMENT INFORMATION")
        print("="*60)
        print(f"Number of environments: {self.num_envs}")
        print(f"Device: {self.device}")
        print(f"Gate asset: {gate_asset_params.file}")
        print(f"Static camera setup success: {self.camera_setup_success}")
        print(f"Number of static cameras: {len(self.camera_handles)}")
        print(f"Number of trees: {len(self.tree_handles)}")
        print(f"Number of X500 robots: {len(self.x500_handles)}")
        print(f"X500 with D455 setup: {'Success' if len(self.x500_handles) > 0 else 'Failed'}")
        print(f"Lee Position Controller: {'Active' if self.lee_controller is not None else 'Failed'}")  
        
        if hasattr(self, 'x500_actor_idx') and self.x500_actor_idx is not None:
            print(f"X500 actor index: {self.x500_actor_idx}")
            current_pos = self.robot_position[0].cpu().numpy()
            print(f"X500 current position: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}]")
        
        if self.camera_setup_success:
            print(f"Static camera position: (0, -3, 1.5)")
            print(f"Static camera target: (0, 0, 1.5) - gate center")
            print(f"Gate position: (0, 0, 0) - rotated 90°")
        
        if len(self.x500_handles) > 0:
            print(f"X500 start position: (-2.5, 0, 1.0)")
            print(f"X500 with D455 camera: Active")
            print(f"X500 camera handles: {len(self.x500_camera_handles)} cameras")
            print(f"X500 camera specs: D455 (270x480, 87° FOV, 0.2-15m range)")
            if len(self.x500_camera_handles) > 0:
                print(f"X500 camera handle IDs: {self.x500_camera_handles}")
            else:
                print("⚠️  WARNING: X500 created but no camera handles found!")
        
        if hasattr(self, 'lee_controller') and self.lee_controller is not None:
            print(f"Navigation mode: Waypoint Navigation")
            if hasattr(self, 'current_navigation_target'):
                target_pos = self.current_navigation_target.cpu().numpy()
                waypoint_num = getattr(self, 'current_waypoint_idx', 0) + 1
                print(f"Current target: [{target_pos[0]:6.2f}, {target_pos[1]:6.2f}, {target_pos[2]:6.2f}] (Waypoint {waypoint_num})")
            else:
                print(f"Navigation targets: Start → Gate → Exit → Side → Above → Return")
        
        print("="*60)
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        
        cv2.destroyAllWindows()
        
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        
        if hasattr(self, 'sim') and self.sim is not None:
            self.gym.destroy_sim(self.sim)

        print("✓ Cleanup completed")

    def _update_cameras(self):
        """Update camera images and capture data."""
        # Capture all camera images at once
        (static_depth, static_seg), (x500_depth, x500_seg) = self.capture_all_camera_images()
        
        # Store for display
        self.static_depth = static_depth
        self.static_seg = static_seg  
        self.x500_depth = x500_depth
        self.x500_seg = x500_seg
        
    def _update_display(self):
        """Update display and handle viewer events."""
        # Display static camera images
        if hasattr(self, 'static_depth') and self.static_depth is not None and self.static_seg is not None:
            # Create combined visualization for static camera
            static_combined = self.create_combined_image(self.static_depth, self.static_seg, "Static Camera")
            
            if static_combined is not None:
                # Display static camera images
                cv2.imshow("Static Camera View", static_combined)
                
                # Print depth statistics periodically
                if hasattr(self, 'debug_step_count') and self.debug_step_count % 300 == 0:  # Every 5 seconds at 60 FPS
                    # Filter out -inf values for better statistics
                    valid_depths = self.static_depth[self.static_depth != -np.inf]
                    if len(valid_depths) > 0:
                        # Use absolute values for meaningful distance measurements
                        abs_valid_depths = np.abs(valid_depths)
                        min_depth = np.min(abs_valid_depths)
                        max_depth = np.max(abs_valid_depths)
                        mean_depth = np.mean(abs_valid_depths)
                        total_pixels = self.static_depth.size
                        valid_ratio = len(valid_depths) / total_pixels * 100
                        
                        # Print robot position and target
                        current_pos = self.robot_position[0].cpu().numpy()
                        if hasattr(self, 'current_navigation_target'):
                            target_pos = self.current_navigation_target.cpu().numpy()
                            distance_to_target = torch.norm(self.robot_position[0] - self.current_navigation_target).item()
                            target_desc = f"Waypoint {getattr(self, 'current_waypoint_idx', 0) + 1}"
                        else:
                            target_pos = current_pos  # Fallback
                            distance_to_target = 0.0
                            target_desc = "Unknown"
                        
                        print(f"Step {self.debug_step_count}: Static Camera - Depth range: {min_depth:.2f}-{max_depth:.2f}m, "
                              f"Mean: {mean_depth:.2f}m, Valid pixels: {valid_ratio:.1f}%")
                        print(f"          X500 Pos: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}], "
                              f"Target: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}], "
                              f"Dist: {distance_to_target:.2f}m, {target_desc}")
        
        # Display X500 camera images
        if hasattr(self, 'x500_depth') and self.x500_depth is not None and self.x500_seg is not None:
            # Create combined visualization for X500 camera
            x500_combined = self.create_combined_image(self.x500_depth, self.x500_seg, "X500 D455")
            
            if x500_combined is not None:
                # Display X500 camera images
                cv2.imshow("X500 D455 Camera View", x500_combined)
                
                # Print X500 camera statistics periodically
                if hasattr(self, 'debug_step_count') and self.debug_step_count % 300 == 0:
                    valid_depths = self.x500_depth[self.x500_depth != -np.inf]
                    if len(valid_depths) > 0:
                        abs_valid_depths = np.abs(valid_depths)
                        min_depth = np.min(abs_valid_depths)
                        max_depth = np.max(abs_valid_depths)
                        mean_depth = np.mean(abs_valid_depths)
                        total_pixels = self.x500_depth.size
                        valid_ratio = len(valid_depths) / total_pixels * 100
                        print(f"          X500 D455 - Depth range: {min_depth:.2f}-{max_depth:.2f}m, "
                              f"Mean: {mean_depth:.2f}m, Valid pixels: {valid_ratio:.1f}%")
        elif hasattr(self, 'debug_step_count') and self.debug_step_count % 300 == 0:
            print(f"          X500 D455 - No camera data available")
        
        # Handle viewer events
        if not self.headless:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            if self.gym.query_viewer_has_closed(self.viewer):
                return False  # Signal to exit
        
        # Check for ESC key press in OpenCV window
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            print("ESC pressed, exiting navigation and visualization")
            return False  # Signal to exit
            
        return True  # Continue running

    def _apply_motor_forces(self, motor_forces):
        """Apply computed motor forces to X500 using Isaac Gym DOF force application."""
        try:
            # Use DOF force application instead of rigid body forces
            env_handle = self.env_handles[0]
            x500_handle = self.x500_handles[0]
            
            # Get DOF forces tensor
            dof_forces = torch.zeros(4, device="cpu", dtype=torch.float32)
            
            # Convert motor forces to DOF forces (take Z component for thrust)
            for i in range(min(4, motor_forces.shape[1])):
                # Extract thrust force (Z component) from motor force
                thrust_force = motor_forces[0, i, 2].cpu()  # Z component
                dof_forces[i] = thrust_force
            
            # Apply DOF forces directly to the X500 actor
            self.gym.apply_actor_dof_efforts(env_handle, x500_handle, dof_forces.numpy())
            
            # Debug output every 300 steps
            if hasattr(self, 'debug_step_count') and self.debug_step_count % 300 == 0:
                print(f"Debug DOF Forces applied: {dof_forces.numpy()}")
                
        except Exception as e:
            print(f"Error applying motor forces via DOF: {e}")
            # Fallback to direct force application on center of mass
            try:
                env_handle = self.env_handles[0]
                x500_handle = self.x500_handles[0]
                
                # Sum all motor forces to get net force
                net_force = torch.sum(motor_forces[0], dim=0).cpu()  # Sum over all motors
                
                # Apply net force to the base link (center of mass)
                force_vec = gymapi.Vec3(net_force[0].item(), net_force[1].item(), net_force[2].item())
                
                # Apply force at center of mass
                self.gym.apply_rigid_body_force_at_pos(
                    env_handle, x500_handle, 0,  # Base link index
                    force_vec, gymapi.Vec3(0, 0, 0),  # Force at center of mass
                    gymapi.ENV_SPACE
                )
                
                if hasattr(self, 'debug_step_count') and self.debug_step_count % 300 == 0:
                    print(f"Debug Fallback Force applied: [{net_force[0]:.2f}, {net_force[1]:.2f}, {net_force[2]:.2f}]")
                    
            except Exception as e2:
                print(f"Error in fallback force application: {e2}")
                import traceback
                traceback.print_exc()
    


def main():
    """Main function."""
    try:
        print("🚁 Starting Gate Navigation with Dual Cameras and Position Control...")
        
        # Create environment
        env = GateEnvironmentWithPositionControl(headless=False)
        
        # Print environment information
        env.print_environment_info()
        
        # Run navigation and visualization
        env.run_navigation(duration_seconds=180.0)  # 3 minutes
        
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