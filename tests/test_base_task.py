"""Tests for aerial_gym.task.base_task — BaseTask ABC."""
import isaacgym  # noqa: F401
import torch
import pytest
from unittest.mock import MagicMock


class TestBaseTask:
    def setup_method(self):
        from aerial_gym.task.base_task import BaseTask
        self.BaseTask = BaseTask

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            self.BaseTask(MagicMock(seed=-1))

    def test_concrete_subclass_works(self):
        class ConcreteTask(self.BaseTask):
            def render(self, mode="human"): pass
            def reset(self): pass
            def reset_idx(self, env_ids): pass
            def step(self, action): pass
            def close(self): pass

        cfg = MagicMock(seed=42)
        task = ConcreteTask(cfg)
        assert task.task_config is cfg

    def test_seed_sets_determinism(self):
        class ConcreteTask(self.BaseTask):
            def render(self, mode="human"): pass
            def reset(self): pass
            def reset_idx(self, env_ids): pass
            def step(self, action): pass
            def close(self): pass

        cfg = MagicMock(seed=42)
        task = ConcreteTask(cfg)
        r1 = torch.randn(5)
        task.seed(42)
        r2 = torch.randn(5)
        task.seed(42)
        r3 = torch.randn(5)
        assert torch.allclose(r2, r3)
