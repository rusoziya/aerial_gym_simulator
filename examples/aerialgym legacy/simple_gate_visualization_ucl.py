"""
Simple UCL Gate Visualization Demo

This script creates and visualizes the UCL-branded gate environment using Isaac Gym.
It spawns a gate with the UCL texture in the environment and allows you to view it interactively.
"""

import sys
import os
import numpy as np
import time

# Add the aerial_gym path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import Isaac Gym and Aerial Gym components
    import isaacgym
    from isaacgym import gymapi, gymtorch, gymutil
    
    # Import aerial gym components
    from aerial_gym.config.env_config.gate_env import GateEnvCfg
    from aerial_gym.config.asset_config.gate_asset_config import gate_asset_params
    from aerial_gym.env_manager.asset_manager import AssetManager
    from aerial_gym.sim.sim_builder import SimBuilder
    
    print("✓ Successfully imported all required modules!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("This might be due to missing dependencies or Python environment issues.")
    sys.exit(1)


class GateEnvironmentVisualizer:
    """
    A simple visualizer for the UCL gate environment.
    """
    
    def __init__(self, headless=False):
        """
        Initialize the gate environment visualizer.
        
        Args:
            headless (bool): Whether to run in headless mode (no graphics)
        """
        self.headless = headless
        self.gym = None
        self.sim = None
        self.viewer = None
        self.env_handles = []
        self.gate_handles = []
        
        # Environment parameters
        self.num_envs = 1  # Single environment for visualization
        self.env_spacing = 10.0  # Spacing between environments
        
        # Initialize the environment
        self._initialize_gym()
        self._create_sim()
        self._create_ground_plane()
        self._create_environments()
        
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
        self.device = 'cuda:0'
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
    
    def _load_gate_asset(self):
        """Load the UCL gate asset."""
        print("Loading UCL gate asset...")
        
        # Asset loading options - simplified for static gate
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True  # Gate is fixed in place
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True  # Static object doesn't need gravity
        asset_options.replace_cylinder_with_capsule = True
        asset_options.mesh_normal_mode = gymapi.FROM_ASSET  # Use normals from asset
        
        # Define the path to the UCL gate URDF
        gate_asset_file = "gate_ucl.urdf"
        gate_asset_folder = f"{os.path.dirname(os.path.abspath(__file__))}/../resources/models/environment_assets/objects"
        gate_asset_path = os.path.join(gate_asset_folder, gate_asset_file)
        
        # Pre-flight check: Verify that the texture file exists before loading
        texture_path = os.path.join(gate_asset_folder, "textures", "ucl_banner.png")
        if not os.path.exists(texture_path):
            print(f"❌ CRITICAL ERROR: Texture file not found at the expected path:")
            print(f"   {texture_path}")
            print("Please ensure 'ucl_banner.png' is inside a 'textures' folder next to the URDF file.")
            sys.exit(1)

        if not os.path.exists(gate_asset_path):
            print(f"❌ Gate URDF not found at: {gate_asset_path}")
            print("Please ensure the gate_ucl.urdf file exists!")
            sys.exit(1)
        
        print(f"Loading gate from: {gate_asset_path}")
        gate_asset = self.gym.load_asset(
            self.sim, 
            gate_asset_folder, 
            gate_asset_file, 
            asset_options
        )
        
        if gate_asset is None:
            raise RuntimeError(f"Failed to load gate asset from {gate_asset_path}")
        
        print("✓ UCL Gate asset loaded successfully!")
        return gate_asset
    
    def _create_environments(self):
        """Create the environments with the UCL gate."""
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
        gate_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)  # Center position
        gate_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # No rotation
        
        # Create gate actor
        gate_handle = self.gym.create_actor(
            env_handle,
            gate_asset,
            gate_pose,
            "gate_ucl",
            0,  # Collision group
            1   # Collision filter
        )
        self.gate_handles.append(gate_handle)
        
        # Note: Gate color is set by the material in the URDF
        
        print("✓ Environment created with UCL gate!")
    
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
        """Setup camera position for good gate viewing."""
        if self.headless or self.viewer is None:
            return
        
        print("Setting up camera...")
        
        # Position camera to get a good view of the gate
        cam_pos = gymapi.Vec3(8.0, -8.0, 4.0)  # Position for angled view
        cam_target = gymapi.Vec3(0.0, 0.0, 1.5)  # Look at center of gate
        
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    
    def run_visualization(self, duration=60.0, physics_mode=True):
        """
        Run the visualization loop.
        
        Args:
            duration (float): How long to run the visualization in seconds
            physics_mode (bool): Whether to run physics simulation or just render
        """
        print(f"Starting visualization for {duration} seconds...")
        print("Controls:")
        print("  - Mouse: Look around")
        print("  - W/A/S/D: Move camera")
        print("  - Q/E: Move camera up/down")
        print("  - ESC: Exit")
        
        if physics_mode:
            print("Running with physics simulation...")
            # Prepare simulation
            self.gym.prepare_sim(self.sim)
        else:
            print("Running in render-only mode (no physics)...")
        
        start_time = time.time()
        
        try:
            while not self.gym.query_viewer_has_closed(self.viewer):
                if physics_mode:
                    # Step simulation - use simpler approach for static objects
                    self.gym.simulate(self.sim)
                    self.gym.fetch_results(self.sim, True)
                
                # Update viewer (this works even without physics)
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                
                # Check if duration exceeded
                if time.time() - start_time > duration:
                    print(f"Visualization completed after {duration} seconds.")
                    break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.016)  # ~60 FPS
                
        except KeyboardInterrupt:
            print("Visualization interrupted by user.")
        except Exception as e:
            print(f"Error during visualization loop: {e}")
            print("This might be due to graphics driver or Isaac Gym compatibility issues.")
    
    def print_environment_info(self):
        """Print information about the created environment."""
        print("\n" + "="*60)
        print("UCL GATE ENVIRONMENT INFORMATION")
        print("="*60)
        
        print(f"Number of environments: {len(self.env_handles)}")
        print(f"Number of gates: {len(self.gate_handles)}")
        print(f"Environment bounds: -8m to +8m in X,Y and 0m to +8m in Z")
        print(f"Gate position: Center of environment (0, 0, 0)")
        print(f"Gate dimensions: ~2.5m wide × ~2.3m tall opening")
        
        if self.viewer:
            print(f"Viewer: Active (interactive)")
        else:
            print(f"Viewer: Headless mode")
        
        print(f"Physics: Isaac Gym PhysX")
        print(f"Simulation timestep: {self.sim_params.dt:.4f}s")
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        
        if self.sim:
            self.gym.destroy_sim(self.sim)


def main():
    """Main function to run the UCL gate visualization."""
    
    print("UCL GATE ENVIRONMENT VISUALIZATION")
    print("="*60)
    
    # Check if gate URDF exists
    gate_urdf_path = f"{os.path.dirname(os.path.abspath(__file__))}/../resources/models/environment_assets/objects/gate_ucl.urdf"
    if not os.path.exists(gate_urdf_path):
        print(f"❌ Gate URDF file not found: {gate_urdf_path}")
        print("Please create the gate_ucl.urdf file first!")
        return 1
    
    try:
        # Create visualizer
        visualizer = GateEnvironmentVisualizer(headless=False)
        
        # Print environment information
        visualizer.print_environment_info()
        
        # Run visualization
        visualizer.run_visualization(duration=300.0)  # 5 minutes
        
        # Cleanup
        visualizer.cleanup()
        
        print("✓ UCL Gate visualization completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 