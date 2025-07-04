#!/usr/bin/env bash

python3 dce_nn_navigation.py --train_dir=$(pwd)/selected_network --experiment=selected_network --env=test --obs_key="observations" --load_checkpoint_kind=best --sim_name="base_sim" --env_name="env_with_obstacles" --robot_name="base_quadrotor_with_stereo_camera" --controller_name="lee_velocity_control"
