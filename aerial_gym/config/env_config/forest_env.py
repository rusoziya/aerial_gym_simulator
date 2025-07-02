from aerial_gym.config.asset_config.env_object_config import (
    tree_asset_params,
    object_asset_params,
    bottom_wall,
)
from aerial_gym.config.asset_config.base_asset import BaseAssetParams

import numpy as np
from aerial_gym.config.asset_config.env_object_config import AERIAL_GYM_DIRECTORY

# Create a custom tree configuration with more trees for a dense forest
class dense_tree_asset_params(tree_asset_params):
    num_assets = 6 # Increase from 1 to 8 trees for a denser forest
    
    # Spread trees more across the environment
    min_state_ratio = [
        0.05,  # Allow trees closer to edges (was 0.1)
        0.05,  # Allow trees closer to edges (was 0.1) 
        0.0,
        0,
        -tree_asset_params.min_state_ratio[4],  # Keep original rotation
        -tree_asset_params.min_state_ratio[5],
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.95,  # Allow trees closer to edges (was 0.9)
        0.95,  # Allow trees closer to edges (was 0.9)
        0.0,
        0,
        tree_asset_params.max_state_ratio[4],   # Keep original rotation
        tree_asset_params.max_state_ratio[5],
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

# Create individual camera marker assets - small colored boxes to mark each camera position
# North Camera Marker (Green) - positioned at (0, 3.5, 1.5), slightly closer than camera at (0, 4.0, 2.0)
class north_camera_marker(BaseAssetParams):
    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"
    file = "small_cube.urdf"
    
    # Position at (0, 3.0, 2.5) in environment bounds [-5,-5,-1] to [5,5,3] -> ratios (0.5, 0.8, 0.875)
    min_state_ratio = [0.5, 0.8, 0.875, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = [0.5, 0.8, 0.875, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    collision_mask = 1
    keep_in_env = True
    collapse_fixed_joints = True
    color = [0, 255, 0]  # Green

# South Camera Marker (Red) - positioned at (0, -3.5, 1.5), slightly closer than camera at (0, -4.0, 2.0)
class south_camera_marker(BaseAssetParams):
    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"
    file = "small_cube.urdf"
    
    # Position at (0, -3.0, 2.5) in environment bounds [-5,-5,-1] to [5,5,3] -> ratios (0.5, 0.2, 0.875)
    min_state_ratio = [0.5, 0.2, 0.875, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = [0.5, 0.2, 0.875, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    collision_mask = 1
    keep_in_env = True
    collapse_fixed_joints = True
    color = [255, 0, 0]  # Red

# East Camera Marker (Blue) - positioned at (3.5, 0, 1.5), slightly closer than camera at (4.0, 0, 2.0)
class east_camera_marker(BaseAssetParams):
    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"
    file = "small_cube.urdf"
    
    # Position at (3.0, 0, 2.5) in environment bounds [-5,-5,-1] to [5,5,3] -> ratios (0.8, 0.5, 0.875)
    min_state_ratio = [0.8, 0.5, 0.875, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = [0.8, 0.5, 0.875, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    collision_mask = 1
    keep_in_env = True
    collapse_fixed_joints = True
    color = [0, 0, 255]  # Blue

# West Camera Marker (Yellow) - positioned at (-3.5, 0, 1.5), slightly closer than camera at (-4.0, 0, 2.0)
class west_camera_marker(BaseAssetParams):
    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"
    file = "small_cube.urdf"
    
    # Position at (-3.0, 0, 2.5) in environment bounds [-5,-5,-1] to [5,5,3] -> ratios (0.2, 0.5, 0.875)
    min_state_ratio = [0.2, 0.5, 0.875, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = [0.2, 0.5, 0.875, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    collision_mask = 1
    keep_in_env = True
    collapse_fixed_joints = True
    color = [255, 255, 0]  # Yellow

import numpy as np


class ForestEnvCfg:
    class env:
        num_envs = 64
        num_env_actions = 4  # this is the number of actions handled by the environment
        # potentially some of these can be input from the RL agent for the robot and
        # some of them can be used to control various entities in the environment
        # e.g. motion of obstacles, etc.
        env_spacing = 5.0  # not used with heightfields/trimeshes

        num_physics_steps_per_env_step_mean = 10  # number of steps between camera renders mean
        num_physics_steps_per_env_step_std = 0  # number of steps between camera renders std

        render_viewer_every_n_steps = 1  # render the viewer every n steps
        reset_on_collision = (
            True  # reset environment when contact force on quadrotor is above a threshold
        )
        collision_force_threshold = 0.005  # collision force threshold [N]
        create_ground_plane = False  # create a ground plane
        sample_timestep_for_latency = True  # sample the timestep for the latency noise
        perturb_observations = True
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = False  # write to sim at every timestep

        use_warp = True
        lower_bound_min = [-5.0, -5.0, -1.0]  # lower bound for the environment space
        lower_bound_max = [-5.0, -5.0, -1.0]  # lower bound for the environment space
        upper_bound_min = [5.0, 5.0, 3.0]  # upper bound for the environment space
        upper_bound_max = [5.0, 5.0, 3.0]  # upper bound for the environment space

    class env_config:
        include_asset_type = {
            "trees": True,
            "objects": True,
            "bottom_wall": True,
            "north_camera_marker": True,  # Enable individual camera markers
            "south_camera_marker": True,
            "east_camera_marker": True,
            "west_camera_marker": True,
        }

        # maps the above names to the classes defining the assets. They can be enabled and disabled above in include_asset_type
        asset_type_to_dict_map = {
            "trees": dense_tree_asset_params,  # Use dense tree configuration instead of default
            "objects": object_asset_params,
            "bottom_wall": bottom_wall,
            "north_camera_marker": north_camera_marker,  # Add individual camera markers
            "south_camera_marker": south_camera_marker,
            "east_camera_marker": east_camera_marker,
            "west_camera_marker": west_camera_marker,
        }
