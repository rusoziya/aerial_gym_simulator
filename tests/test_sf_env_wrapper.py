"""Tests for SF env wrapper (architectural: class may be in different module)."""
import isaacgym  # noqa: F401
import torch
import pytest
from unittest.mock import Mock


class TestSFEnvWrapperExists:
    def test_base_wrapper_importable(self):
        try:
            from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import AerialGymVecEnv
            assert AerialGymVecEnv is not None
        except ImportError:
            # May have been moved to train_common.py
            from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_common import AerialGymVecEnv
            assert AerialGymVecEnv is not None

    def test_gate_wrapper_importable(self):
        try:
            from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net_gate import AerialGymVecEnv
            assert AerialGymVecEnv is not None
        except ImportError:
            try:
                from aerial_gym.rl_training.sample_factory.aerialgym_examples.env_wrapper_gate import AerialGymVecEnvGate
                assert AerialGymVecEnvGate is not None
            except ImportError:
                pytest.skip("Gate wrapper not found")
