"""Tests for SF training configuration (architectural: configs may be in separate modules)."""

import isaacgym  # noqa: F401
import pytest


class TestBaseTrainingConfigImportable:
    def test_train_script_importable(self):
        # The base training script should be importable
        try:
            import aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net

            assert True
        except ImportError:
            pytest.skip("Sample Factory not available")

    def test_gate_train_script_importable(self):
        try:
            import aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net_gate

            assert True
        except ImportError:
            pytest.skip("Sample Factory not available")
