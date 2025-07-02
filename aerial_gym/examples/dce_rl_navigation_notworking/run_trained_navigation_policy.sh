#!/usr/bin/env bash

# Set environment variables to configure the drone model and environment
export SIM_NAME="base_sim"
export ENV_NAME="env_with_obstacles"
export ROBOT_NAME="base_quadrotor_with_stereo_camera"
export CONTROLLER_NAME="lee_velocity_control"

# Run the navigation script with the required parameters
python3 dce_nn_navigation.py --train_dir=$(pwd)/selected_network --experiment=selected_network --env=test --obs_key="observations" --load_checkpoint_kind=best
