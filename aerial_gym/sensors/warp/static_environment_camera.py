import warp as wp
import torch
import math
from aerial_gym.sensors.warp.warp_cam import WarpCam
from aerial_gym.utils.math import quat_from_euler_xyz_tensor
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("StaticEnvironmentCamera")

class StaticEnvironmentCamera:
    """
    A static environment camera sensor using Warp rendering pipeline.
    This camera is not attached to any robot and maintains a fixed world position.
    """
    
    def __init__(self, camera_config, num_envs, mesh_ids_array, device="cuda:0"):
        self.cfg = camera_config
        self.device = device
        self.num_envs = num_envs
        self.num_cameras = self.cfg.num_cameras if hasattr(self.cfg, 'num_cameras') else 1
        self.mesh_ids_array = mesh_ids_array
        
        # Initialize static camera positions and orientations
        self.static_positions = torch.zeros(
            (self.num_envs, self.num_cameras, 3), 
            device=self.device, 
            requires_grad=False
        )
        self.static_orientations = torch.zeros(
            (self.num_envs, self.num_cameras, 4), 
            device=self.device, 
            requires_grad=False
        )
        self.static_orientations[..., 3] = 1.0  # Initialize quaternions to identity
        
        # Create Warp camera sensor
        self.warp_camera = WarpCam(
            num_envs=self.num_envs,
            config=self.cfg,
            mesh_ids_array=self.mesh_ids_array,
            device=self.device
        )
        
        logger.info(f"Static environment camera initialized with {self.num_cameras} cameras per environment")
        
    def set_camera_poses(self, positions, orientations=None):
        """
        Set static camera poses in world coordinates.
        
        Args:
            positions: Tensor of shape (num_envs, num_cameras, 3) for world positions
            orientations: Optional tensor of shape (num_envs, num_cameras, 4) for quaternions
                         If None, cameras will look down the negative Z axis (typical overhead view)
        """
        self.static_positions[:] = positions
        
        if orientations is not None:
            self.static_orientations[:] = orientations
        else:
            # Default: cameras looking down (negative Z direction)
            # This creates an overhead view of the environment
            euler_angles = torch.tensor([math.pi, 0.0, 0.0], device=self.device)  # Look down
            default_quat = quat_from_euler_xyz_tensor(euler_angles)
            self.static_orientations[:] = default_quat.expand(self.num_envs, self.num_cameras, -1)
            
        # Update the Warp camera with static poses
        self.warp_camera.set_pose_tensor(self.static_positions, self.static_orientations)
        
        logger.debug(f"Updated static camera poses: positions={positions[0, 0]}, orientations={self.static_orientations[0, 0]}")
        
    def set_overhead_view(self, height=10.0, center_positions=None):
        """
        Convenience method to set up overhead view cameras.
        
        Args:
            height: Height above the environment center
            center_positions: Optional center positions for each environment
                            If None, uses (0, 0, height) for all environments
        """
        if center_positions is None:
            # Default to environment center
            positions = torch.zeros((self.num_envs, self.num_cameras, 3), device=self.device)
            positions[..., 2] = height  # Set height
        else:
            positions = center_positions.unsqueeze(1).expand(-1, self.num_cameras, -1)
            positions[..., 2] = height  # Override Z coordinate with height
            
        # Set up cameras looking straight down
        self.set_camera_poses(positions, orientations=None)
        
    def set_side_view(self, distance=15.0, height=5.0, angle_degrees=0.0):
        """
        Convenience method to set up side view cameras.
        
        Args:
            distance: Distance from environment center
            height: Height above ground
            angle_degrees: Rotation angle around Z axis (0 = looking from +X direction)
        """
        positions = torch.zeros((self.num_envs, self.num_cameras, 3), device=self.device)
        
        # Position cameras at distance from center
        angle_rad = math.radians(angle_degrees)
        positions[..., 0] = distance * math.cos(angle_rad)  # X position
        positions[..., 1] = distance * math.sin(angle_rad)  # Y position  
        positions[..., 2] = height  # Z position
        
        # Orient cameras to look toward center
        euler_angles = torch.tensor([0.0, -math.pi/6, angle_rad + math.pi], device=self.device)  # Slight downward tilt
        orientations = quat_from_euler_xyz_tensor(euler_angles)
        orientations = orientations.expand(self.num_envs, self.num_cameras, -1)
        
        self.set_camera_poses(positions, orientations)
        
    def set_image_tensors(self, pixels, segmentation_pixels=None):
        """Set image output tensors"""
        self.warp_camera.set_image_tensors(pixels, segmentation_pixels)
        
    def capture(self):
        """Capture images from static cameras"""
        return self.warp_camera.capture()
        
    def reset(self):
        """Reset cameras (no-op for static cameras, but maintain interface compatibility)"""
        pass
        
    def update(self):
        """Update cameras (no-op for static cameras since they don't move)"""
        pass


class StaticEnvironmentCameraConfig:
    """Configuration for static environment cameras"""
    
    def __init__(self):
        # Camera sensor type
        self.sensor_type = "camera"
        
        # Number of static cameras per environment
        self.num_cameras = 1
        self.num_sensors = 1  # Required by WarpCam - same as num_cameras for static cameras
        
        # Image dimensions
        self.width = 512
        self.height = 512
        
        # Camera parameters
        self.horizontal_fov_deg = 90.0
        self.max_range = 50.0
        self.min_range = 0.1
        
        # Rendering options
        self.calculate_depth = True
        self.return_pointcloud = False
        self.pointcloud_in_world_frame = True
        self.segmentation_camera = True
        
        # Static camera positions (will be set by user)
        self.camera_positions = [[0.0, 0.0, 10.0]]  # Default overhead view
        self.camera_orientations = None  # Will use default downward orientation
        
        # Noise and processing (typically disabled for static environment cameras)
        self.normalize_range = True
        self.near_out_of_range_value = 0.0
        self.far_out_of_range_value = 1.0
        
        # Sensor noise (typically disabled for static cameras)
        class sensor_noise:
            enable_sensor_noise = False
            std_a = 0.0
            std_b = 0.0
            std_c = 0.0
            mean_offset = 0.0
            pixel_dropout_prob = 0.0
            
        self.sensor_noise = sensor_noise()


# Example configurations
class OverheadCameraConfig(StaticEnvironmentCameraConfig):
    """Overhead view camera configuration"""
    
    def __init__(self):
        super().__init__()
        self.num_cameras = 1
        self.num_sensors = 1  # Required by WarpCam
        self.camera_positions = [[0.0, 0.0, 15.0]]  # 15m above center
        self.horizontal_fov_deg = 120.0  # Wide angle for full environment view


class SideViewCameraConfig(StaticEnvironmentCameraConfig):
    """Side view camera configuration"""
    
    def __init__(self):
        super().__init__()
        self.num_cameras = 1
        self.num_sensors = 1  # Required by WarpCam
        self.camera_positions = [[20.0, 0.0, 8.0]]  # Side view position
        self.horizontal_fov_deg = 60.0  # Narrower view


class MultiAngleCameraConfig(StaticEnvironmentCameraConfig):
    """Multiple static cameras from different angles"""
    
    def __init__(self):
        super().__init__()
        self.num_cameras = 4  # Four cameras around the environment
        self.num_sensors = 4  # Required by WarpCam - same as num_cameras
        self.camera_positions = [
            [15.0, 0.0, 8.0],    # East view
            [0.0, 15.0, 8.0],    # North view  
            [-15.0, 0.0, 8.0],   # West view
            [0.0, -15.0, 8.0],   # South view
        ]
        self.horizontal_fov_deg = 90.0 