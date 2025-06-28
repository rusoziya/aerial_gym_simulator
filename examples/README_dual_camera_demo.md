# Dual Camera Demo

This demo showcases both a static environment camera and a drone with its onboard camera, displaying depth and segmentation outputs from both in real-time.

## Features

- **Single Environment**: Uses 1 environment instead of multiple for clear visualization
- **Dual Camera System**: 
  - Drone onboard camera (depth + segmentation)
  - Static environment cameras positioned inside the chamber:
    - Overhead view from 8m height (top-down monitoring)  
    - Corner view from 8m distance, 6m height (security-style perspective)
- **Dual Visualization**:
  - Isaac Gym 3D viewer showing the simulation environment with camera markers
  - Real-time OpenCV windows with side-by-side depth and segmentation images
- **Camera Markers**: Visual indicators in the 3D simulation:
  - Green sphere = Overhead camera position
  - Blue sphere = Side/corner camera position
- **No RL Training**: Pure visualization demo without any reinforcement learning

## What You'll See

The demo opens both a 3D simulation viewer and three camera windows:

**Isaac Gym 3D Viewer**: 
- Shows the full simulation environment with the drone flying
- Green sphere marker indicates overhead camera position
- Blue sphere marker indicates corner camera position
- You can navigate the 3D view to see the cameras and drone from different angles

**Camera Windows**:
1. **Drone Camera - Onboard View**: Shows depth and segmentation from the drone's onboard camera
2. **Overhead Camera - Top-Down View (8m high)**: Bird's eye view from inside the chamber, 8m above center
3. **Corner Camera - Side View (Corner of chamber)**: Side perspective from a corner inside the environment chamber

Each window displays:
- **Left side**: Depth information (grayscale)
- **Right side**: Segmentation information (colored with plasma colormap)

## Controls

- **q**: Quit the demo
- **s**: Save current frames to disk
- **ESC**: Also quits the demo

## Running the Demo

```bash
cd examples
python3 dual_camera_demo.py
```

Or make it executable and run directly:
```bash
chmod +x dual_camera_demo.py
./dual_camera_demo.py
```

## Output

The demo automatically saves images every 30 frames to a timestamped directory:
- Combined visualization images (depth + segmentation side by side)
- Raw depth data as .npy files
- Raw segmentation data as .npy files

## Technical Details

- Uses single environment configuration for focused observation
- Static cameras positioned inside the environment chamber for realistic indoor monitoring
- Uses the DCE Navigation Task configuration which includes camera sensors
- Implements static environment cameras using the Warp rendering pipeline
- Drone follows a simple circular movement pattern for demonstration
- All cameras provide both depth and segmentation information similar to the DCE navigation task

## Requirements

- CUDA-capable GPU (for Warp rendering)
- OpenCV for visualization
- Matplotlib for colormap processing
- All standard Aerial Gym dependencies 