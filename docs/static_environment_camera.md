# Static Environment Camera System

## Overview

The Static Environment Camera system allows you to create fixed-position cameras using the Warp rendering pipeline that are not attached to any robot. This is particularly useful for:

- **Multi-agent observation**: Getting bird's eye view of multi-drone formations
- **Training visualization**: Recording training progress from fixed viewpoints
- **Surveillance simulation**: Simulating static security cameras
- **Data collection**: Generating datasets from consistent camera positions

## Features

✅ **Multiple camera views**: Overhead, side view, multi-angle configurations  
✅ **Warp rendering**: High-performance GPU-accelerated ray-casting  
✅ **Depth + Segmentation**: Both depth images and semantic segmentation  
✅ **Static positioning**: Cameras maintain fixed world positions  
✅ **Easy integration**: Simple API for adding to existing environments  
✅ **Configurable**: Flexible camera parameters and positioning  

## Quick Start

### 1. Basic Usage

```python
from aerial_gym.sensors.warp.static_environment_camera import (
    StaticEnvironmentCamera,
    OverheadCameraConfig
)

# Create camera configuration
config = OverheadCameraConfig()

# Create static camera
camera = StaticEnvironmentCamera(
    camera_config=config,
    num_envs=1,
    mesh_ids_array=mesh_ids,  # From environment
    device="cuda"
)

# Set up overhead view
camera.set_overhead_view(height=15.0)

# Create image tensors
pixels = torch.zeros((1, 1, config.height, config.width), device="cuda")
segmentation = torch.zeros((1, 1, config.height, config.width), dtype=torch.int32, device="cuda")
camera.set_image_tensors(pixels, segmentation)

# Capture images
camera.capture()
```

### 2. Integration with Multi-Agent Environment

```python
from aerial_gym.envs.navigation_task.multi_agent_navigation_task_config import MultiAgentFormationTaskConfigLight
from aerial_gym.algorithms.multi_agent_wrapper import create_multi_agent_env

# Create environment with Warp enabled
def create_config():
    config = MultiAgentFormationTaskConfigLight()
    config.task_config["use_warp"] = True  # Required for static cameras
    config.sim_config["use_warp"] = True
    return config

env = create_multi_agent_env(task_config_class=create_config, ...)

# Get mesh IDs for Warp rendering
mesh_ids = env.task.sim_env.global_tensor_dict.get("CONST_WARP_MESH_ID_LIST")

# Create and use static camera as shown above
```

## Camera Configurations

### Pre-defined Configurations

#### OverheadCameraConfig
```python
config = OverheadCameraConfig()
# - 512x512 resolution
# - 120° horizontal FoV 
# - 15m above center by default
# - Wide angle for full environment view
```

#### SideViewCameraConfig  
```python
config = SideViewCameraConfig()
# - 512x512 resolution
# - 60° horizontal FoV
# - Side position by default
# - Narrower view for detailed observation
```

#### MultiAngleCameraConfig
```python
config = MultiAngleCameraConfig()
# - 4 cameras around environment
# - East, North, West, South positions
# - 90° horizontal FoV each
```

### Custom Configuration

```python
class MyCustomCameraConfig:
    def __init__(self):
        # Camera type
        self.sensor_type = "camera"
        self.num_cameras = 2
        
        # Image dimensions
        self.width = 1024
        self.height = 768
        
        # Camera parameters
        self.horizontal_fov_deg = 75.0
        self.max_range = 100.0
        self.min_range = 0.1
        
        # Rendering options
        self.calculate_depth = True
        self.return_pointcloud = False
        self.pointcloud_in_world_frame = True
        self.segmentation_camera = True
        
        # Processing options
        self.normalize_range = True
        self.near_out_of_range_value = 0.0
        self.far_out_of_range_value = 1.0
```

## Camera Positioning Methods

### 1. Overhead View
```python
# Simple overhead view
camera.set_overhead_view(height=20.0)

# Overhead view with custom center positions per environment
center_positions = torch.tensor([[[5.0, 3.0, 0.0]]], device=device)  # (num_envs, 3)
camera.set_overhead_view(height=15.0, center_positions=center_positions)
```

### 2. Side View
```python
# Side view from specific angle
camera.set_side_view(
    distance=25.0,      # Distance from center
    height=10.0,        # Height above ground
    angle_degrees=45.0  # Rotation around Z axis
)
```

### 3. Custom Positioning
```python
# Manual position and orientation setting
positions = torch.tensor([[[10.0, 5.0, 8.0]]], device=device)  # (num_envs, num_cameras, 3)
orientations = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=device)  # (num_envs, num_cameras, 4)

camera.set_camera_poses(positions, orientations)
```

## Image Capture and Processing

### Basic Capture
```python
# Capture images
camera.capture()

# Access image data
depth_image = pixels[0, 0].cpu().numpy()  # First env, first camera
segmentation_image = segmentation[0, 0].cpu().numpy()
```

### Save Images
```python
import cv2

# Convert and save depth image
depth_normalized = (depth_image * 255).astype(np.uint8)
cv2.imwrite("depth_frame.png", depth_normalized)

# Save segmentation image
seg_colored = (segmentation_image * 50).astype(np.uint8)  # Scale for visibility
cv2.imwrite("segmentation_frame.png", seg_colored)
```

### Create Videos
```python
# Capture sequence of images during simulation
for step in range(num_steps):
    # ... run simulation step ...
    
    camera.capture()
    
    if step % 10 == 0:  # Save every 10 steps
        depth_img = pixels[0, 0].cpu().numpy()
        depth_normalized = (depth_img * 255).astype(np.uint8)
        cv2.imwrite(f"frame_{step:04d}.png", depth_normalized)

# Create video with ffmpeg
# ffmpeg -r 10 -i frame_%04d.png -vcodec libx264 output_video.mp4
```

## Advanced Usage

### Multiple Static Cameras
```python
# Create multiple cameras with different views
cameras = []
configs = [OverheadCameraConfig(), SideViewCameraConfig()]

for i, config in enumerate(configs):
    camera = StaticEnvironmentCamera(config, num_envs, mesh_ids, device)
    
    if i == 0:  # Overhead
        camera.set_overhead_view(height=20.0)
    else:  # Side view
        camera.set_side_view(distance=30.0, height=12.0, angle_degrees=30.0)
    
    cameras.append(camera)
```

### Camera Array Around Environment
```python
# Create cameras in a circle around the environment
num_cameras = 8
camera_configs = []
cameras = []

for i in range(num_cameras):
    angle = i * (360.0 / num_cameras)
    
    config = SideViewCameraConfig()
    camera = StaticEnvironmentCamera(config, num_envs, mesh_ids, device)
    camera.set_side_view(distance=25.0, height=10.0, angle_degrees=angle)
    
    cameras.append(camera)
```

### Integration with Training Loop
```python
def training_loop_with_cameras():
    # Setup cameras
    overhead_camera = setup_overhead_camera()
    side_camera = setup_side_camera()
    
    for episode in range(num_episodes):
        obs = env.reset()
        
        for step in range(max_steps):
            # Get actions and step environment
            actions = get_actions(obs)
            obs, rewards, dones, info = env.step(actions)
            
            # Capture from static cameras
            overhead_camera.capture()
            side_camera.capture()
            
            # Log images periodically
            if step % 50 == 0:
                log_camera_images(overhead_camera, side_camera, episode, step)
            
            if any(dones):
                break
```

## Requirements

### System Requirements
- **CUDA GPU**: Required for Warp rendering
- **Warp**: NVIDIA Warp framework
- **PyTorch**: With CUDA support
- **OpenCV**: For image processing (optional)

### Environment Requirements
```python
# Environment must have Warp enabled
config.task_config["use_warp"] = True
config.sim_config["use_warp"] = True

# Environment must have mesh assets for rendering
# Verify mesh IDs are available:
mesh_ids = env.task.sim_env.global_tensor_dict.get("CONST_WARP_MESH_ID_LIST")
assert mesh_ids is not None, "Warp mesh IDs not available"
```

## Troubleshooting

### Common Issues

**1. "Warp mesh IDs not available"**
```python
# Solution: Enable Warp in configuration
config.task_config["use_warp"] = True
config.sim_config["use_warp"] = True
```

**2. "CUDA out of memory"**
```python
# Solution: Reduce image resolution or number of cameras
config.width = 256   # Instead of 512
config.height = 256  # Instead of 512
```

**3. "Black/empty images"**
```python
# Solution: Check camera positioning and range
camera.set_overhead_view(height=5.0)  # Lower height
config.max_range = 100.0  # Increase range
```

**4. "Camera looking in wrong direction"**
```python
# Solution: Adjust camera orientation manually
positions = torch.tensor([[[0.0, 0.0, 10.0]]], device=device)
# Create custom orientation quaternion
orientations = create_look_at_quaternion(target_position)
camera.set_camera_poses(positions, orientations)
```

### Performance Tips

1. **Use appropriate resolution**: Higher resolutions are slower
2. **Limit number of cameras**: Each camera adds computational cost
3. **Optimize capture frequency**: Don't capture every frame if not needed
4. **Use GPU memory efficiently**: Reuse image tensors when possible

## Examples

Complete working examples are available:

- **`examples/static_camera_example.py`**: Basic usage with multi-agent environment
- **`examples/multi_angle_recording.py`**: Multiple camera setup
- **`examples/training_with_cameras.py`**: Integration with training loop

## API Reference

### StaticEnvironmentCamera

#### Constructor
```python
StaticEnvironmentCamera(camera_config, num_envs, mesh_ids_array, device="cuda:0")
```

#### Methods
- `set_camera_poses(positions, orientations=None)`: Set manual camera poses
- `set_overhead_view(height=10.0, center_positions=None)`: Set overhead view
- `set_side_view(distance=15.0, height=5.0, angle_degrees=0.0)`: Set side view
- `set_image_tensors(pixels, segmentation_pixels=None)`: Set output tensors
- `capture()`: Capture images
- `reset()`: Reset cameras (no-op for static cameras)
- `update()`: Update cameras (no-op for static cameras)

### Camera Configuration Classes

#### StaticEnvironmentCameraConfig
Base configuration class with all standard parameters.

#### OverheadCameraConfig
Pre-configured for overhead/bird's eye view.

#### SideViewCameraConfig  
Pre-configured for side view observation.

#### MultiAngleCameraConfig
Pre-configured for multiple cameras around environment.

## Conclusion

The Static Environment Camera system provides a powerful and flexible way to add fixed-position cameras to Aerial Gym environments. It leverages the high-performance Warp rendering pipeline to provide real-time depth and segmentation imaging from any desired viewpoint, making it ideal for multi-agent observation, training visualization, and data collection. 