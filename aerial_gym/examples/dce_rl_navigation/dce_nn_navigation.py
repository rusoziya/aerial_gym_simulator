from __future__ import annotations

import time
import isaacgym

# isort: on
import torch
from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import (
    parse_aerialgym_cfg,
)
from aerial_gym.utils import get_args
from aerial_gym.registry.task_registry import task_registry


from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import DCE_RL_Navigation_Task
from aerial_gym.examples.dce_rl_navigation.sf_inference_class import NN_Inference_Class

import matplotlib
import numpy as np
from PIL import Image


def sample_command(args) -> None:
    use_warp = True
    # Enable viewing by default for inference - user can override with --headless
    headless = getattr(args, 'headless', False)  # Default to False (viewing enabled)
    print(f"DCE Inference - Headless mode: {headless}")
    
    # seg_frames = []
    # depth_frames = []
    # merged_image_frames = []

    rl_task = task_registry.make_task(
        "dce_navigation_task", seed=42, use_warp=use_warp, headless=headless
    )
    print("Number of environments", rl_task.num_envs)
    command_actions = torch.zeros((rl_task.num_envs, rl_task.task_config.action_space_dim))
    command_actions[:, 0] = 1.5
    command_actions[:, 1] = 0.0
    command_actions[:, 2] = 0.0
    nn_model = get_network(rl_task.num_envs)
    nn_model.eval()
    nn_model.reset(torch.arange(rl_task.num_envs))
    rl_task.reset()
    for i in range(0, 50000):
        start_time = time.time()
        obs, rewards, termination, truncation, infos = rl_task.step(command_actions)

        obs["obs"] = obs["observations"]
        action = nn_model.get_action(obs)
        action = torch.tensor(action).expand(rl_task.num_envs, -1)
        command_actions[:] = action

        reset_ids = (termination + truncation).nonzero(as_tuple=True)
        if torch.any(termination):
            terminated_envs = termination.nonzero(as_tuple=True)
            print(f"Resetting environments {terminated_envs} due to Termination")
        if torch.any(truncation):
            truncated_envs = truncation.nonzero(as_tuple=True)
            print(f"Resetting environments {truncated_envs} due to Timeout")
        nn_model.reset(reset_ids)


def get_network(num_envs) -> None:
    """Script entry point."""
    cfg = parse_aerialgym_cfg(evaluation=True)
    print("CFG is:", cfg)
    nn_model = NN_Inference_Class(num_envs, 3, 81, cfg)
    return nn_model


if __name__ == "__main__":
    task_registry.register_task(
        task_name="dce_navigation_task",
        task_class=DCE_RL_Navigation_Task,
        task_config=task_registry.get_task_config(
            "navigation_task"
        ),  # same config as navigation task
    )
    args = get_args()
    sample_command(args)
