"""
Simple Racing Gate Visualization Demo

This script visualizes the racing-patterned gate environment using Isaac Gym.
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
    
    print("✓ Successfully imported all required modules!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("This might be due to missing dependencies or Python environment issues.")
    sys.exit(1)


class GateEnvironmentVisualizer:
    """
    A simple visualizer for the racing gate environment.
    """
    
    def __init__(self, headless=False):
        self.headless = headless
        self.gym = None
        self.sim = None
        self.viewer = None
        self.env_handles = []
        self.gate_handles = []
        
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
        self.gym = gymapi.acquire_gym()
        
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.use_gpu = True
        
        self.device = 'cuda:0'
        sim_params.use_gpu_pipeline = True
        self.sim_params = sim_params
        
    def _create_sim(self):
        """Create the simulation."""
        print("Creating simulation...")
        self.sim = self.gym.create_sim(0, 0 if not self.headless else -1, gymapi.SIM_PHYSX, self.sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create simulation")
    
    def _create_ground_plane(self):
        """Create ground plane."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
    
    def _load_gate_asset(self):
        """Load the racing gate asset."""
        print("Loading racing gate asset...")
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.FROM_ASSET
        asset_options.disable_gravity = True

        gate_asset_file = "gate_racing.urdf"
        asset_root = f"{os.path.dirname(os.path.abspath(__file__))}/../resources/models/environment_assets"
        
        # Pre-flight check for texture and model files
        texture_path = os.path.join(asset_root, "objects", "textures", "racing_pattern.png")
        model_path = os.path.join(asset_root, "objects", "gate.obj")
        urdf_path = os.path.join(asset_root, "objects", gate_asset_file)

        if not os.path.exists(texture_path):
            print(f"❌ CRITICAL ERROR: Texture file not found at:\n   {texture_path}")
            sys.exit(1)
        if not os.path.exists(model_path):
            print(f"❌ CRITICAL ERROR: 3D model file not found at:\n   {model_path}")
            sys.exit(1)

        print(f"Loading URDF from: {urdf_path}")
        gate_asset = self.gym.load_asset(self.sim, asset_root, urdf_path, asset_options)
        if gate_asset is None:
            raise RuntimeError(f"Failed to load asset from {urdf_path}")
        
        print("✓ Racing Gate asset loaded successfully!")
        return gate_asset
    
    def _create_environments(self):
        """Create environments with the racing gate."""
        gate_asset = self._load_gate_asset()
        env_lower = gymapi.Vec3(-8.0, -8.0, 0.0)
        env_upper = gymapi.Vec3(8.0, 8.0, 8.0)
        
        env_handle = self.gym.create_env(self.sim, env_lower, env_upper, 1)
        self.env_handles.append(env_handle)
        
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        
        gate_handle = self.gym.create_actor(env_handle, gate_asset, pose, "gate_racing", 0, 1)
        self.gate_handles.append(gate_handle)
        
        print("✓ Environment created with racing gate!")
    
    def _create_viewer(self):
        """Create viewer."""
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise RuntimeError("Failed to create viewer")
    
    def _setup_camera(self):
        """Setup camera position."""
        cam_pos = gymapi.Vec3(5.0, -5.0, 3.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 1.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    
    def run_visualization(self, duration=300.0):
        """Run the visualization loop."""
        print("Starting visualization...")
        start_time = time.time()
        
        while not self.gym.query_viewer_has_closed(self.viewer):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            
            if time.time() - start_time > duration:
                break
            time.sleep(0.016)
    
    def cleanup(self):
        """Clean up resources."""
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        if self.sim:
            self.gym.destroy_sim(self.sim)

def main():
    print("RACING GATE ENVIRONMENT VISUALIZATION")
    print("="*60)
    try:
        visualizer = GateEnvironmentVisualizer()
        visualizer.run_visualization()
        visualizer.cleanup()
        print("✓ Visualization completed successfully!")
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 