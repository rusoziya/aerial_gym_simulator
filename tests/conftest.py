"""Shared fixtures for aerial_gym unit tests.

CRITICAL: isaacgym must be imported before torch in every test file.
"""

import isaacgym  # noqa: F401 — must be imported before torch
import torch
import pytest


@pytest.fixture
def device():
    """Return CUDA device if available, else CPU."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def num_envs():
    """Default number of environments for mock tensors."""
    return 16


@pytest.fixture
def mock_obs_dict(device, num_envs):
    """Build a minimal mock observation dict matching the 150D gate observation layout.

    Layout (150D total):
        [0:3]     drone_position          (3D)
        [3:6]     static_camera_position  (3D)
        [6:9]     static_camera_orient    (3D)
        [9:12]    drone_orientation        (3D)
        [12:15]   linear_velocity         (3D)
        [15:18]   angular_velocity        (3D)
        [18:22]   prev_actions            (4D)
        [22:86]   drone_vae_latent        (64D)
        [86:150]  static_vae_latent       (64D)
    """
    return {
        "drone_position": torch.zeros(num_envs, 3, device=device),
        "static_camera_position": torch.zeros(num_envs, 3, device=device),
        "static_camera_orient": torch.zeros(num_envs, 3, device=device),
        "drone_orientation": torch.zeros(num_envs, 3, device=device),
        "linear_velocity": torch.zeros(num_envs, 3, device=device),
        "angular_velocity": torch.zeros(num_envs, 3, device=device),
        "prev_actions": torch.zeros(num_envs, 4, device=device),
        "drone_vae_latent": torch.zeros(num_envs, 64, device=device),
        "static_vae_latent": torch.zeros(num_envs, 64, device=device),
    }
