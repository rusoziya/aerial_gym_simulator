"""Shared fixtures for behavior-capture tests.
CRITICAL: isaacgym must be imported before torch.
"""
import isaacgym  # noqa: F401
import torch
import pytest

@pytest.fixture
def device():
    return "cpu"

@pytest.fixture
def num_envs():
    return 4
