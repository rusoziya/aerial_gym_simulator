# Random Color Functionality for Gate Navigation

This implementation adds random color functionality to the gate navigation task in Aerial Gym. The modification applies random colors to both the gate assets and the obstacle assets (objects_gate) without modifying the base Aerial Gym simulation files.

## Features

- **Random Colors for Gate Assets**: Each gate receives a random RGB color during initialization
- **Random Colors for Obstacle Assets**: Objects from the `objects_gate` folder receive random colors
- **Per-Episode Color Changes**: Colors are reapplied when environments are reset, ensuring each episode has different colors
- **Multi-Environment Support**: Each environment gets different random colors
- **Complete Asset Coverage**: Colors are applied to all rigid bodies of each asset

## Implementation Details

### Modified Files

1. **`aerial_gym/task/navigation_task_gate/navigation_task_gate.py`**
   - Added `apply_random_colors_to_gate_assets()` method
   - Added `apply_random_colors_to_reset_environments()` method
   - Modified `__init__()` to apply colors during initialization
   - Modified `reset_idx()` to reapply colors during reset

### Key Methods

#### `apply_random_colors_to_gate_assets()`
- Called during task initialization
- Applies random colors to all gate and obstacle assets across all environments
- Uses Isaac Gym's `set_rigid_body_color()` API
- Handles both gate assets and environment obstacle assets

#### `apply_random_colors_to_reset_environments(env_ids)`
- Called during environment reset
- Applies new random colors only to environments being reset
- Ensures each episode has different colors
- More efficient than reapplying to all environments

### Color Generation

- **RGB Values**: Random values between 0.0 and 1.0 for each color channel
- **Application**: Applied to visual mesh using `gymapi.MESH_VISUAL`
- **Coverage**: Applied to all rigid bodies of each asset

## Usage

### Basic Usage

```python
from aerial_gym.task.navigation_task_gate.navigation_task_gate import NavigationTaskGate
from aerial_gym.config.task_config.navigation_task_config_gate import task_config

# Create configuration
config = task_config()
config.num_envs = 4
config.headless = False  # Enable visualization to see colors

# Create task (colors applied automatically)
task = NavigationTaskGate(config)

# Run simulation
for episode in range(3):
    obs = task.reset()  # New colors applied during reset
    
    for step in range(100):
        actions = torch.zeros((config.num_envs, 4))
        obs, rewards, terminations, truncations, infos = task.step(actions)

task.close()
```

### Testing

Run the test script to verify functionality:

```bash
python test_random_colors.py
```

### Example

Run the example script to see the random colors in action:

```bash
python example_random_colors_gate_navigation.py
```

## Technical Implementation

### Asset Identification

The system identifies assets using:
- **Gate Assets**: First asset (index 0) or assets with "gate" in the name
- **Obstacle Assets**: Assets with "env_asset" in the name and index > 0

### Color Application Process

1. **Access Isaac Gym Environment**: Get gym object and environment handles
2. **Iterate Through Assets**: For each environment and asset
3. **Identify Asset Type**: Determine if asset is gate or obstacle
4. **Generate Random Color**: Create random RGB Vec3
5. **Apply to All Rigid Bodies**: Apply color to all parts of the asset
6. **Log Results**: Record successful color applications

### Error Handling

- Graceful handling of missing Isaac Gym environment
- Warning messages for unavailable handles
- Continue processing if individual assets fail
- Comprehensive error logging

## Benefits

1. **Visual Diversity**: Each episode has different colored assets
2. **Training Robustness**: Helps prevent overfitting to specific colors
3. **Easy Identification**: Different colors help distinguish environments
4. **Non-Intrusive**: Doesn't modify base Aerial Gym files
5. **Efficient**: Only applies colors when needed

## Requirements

- Aerial Gym Simulator
- Isaac Gym
- PyTorch
- NumPy

## Notes

- Colors are applied to visual meshes only (not collision meshes)
- Each environment gets different random colors
- Colors change each time environments are reset
- The implementation is safe and doesn't affect simulation physics
- All color applications are logged for debugging

## Troubleshooting

If colors don't appear:
1. Ensure `headless=False` in configuration
2. Check that Isaac Gym viewer is enabled
3. Verify that assets are being loaded correctly
4. Check logs for any error messages

If you see warnings about missing handles:
- This is normal if running in headless mode
- Colors will only be applied when Isaac Gym environment is available 