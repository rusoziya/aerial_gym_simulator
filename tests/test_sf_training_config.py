"""Behavior tests for Sample Factory training configuration values."""
import isaacgym  # noqa: F401
import pytest


class TestBaseTrainingDefaults:
    """Capture the exact training hyperparameters for quad_with_obstacles."""

    def setup_method(self):
        # Read env_configs from the training script
        import ast
        with open('aerial_gym/rl_training/sample_factory/aerialgym_examples/train_aerialgym_custom_net.py') as f:
            src = f.read()
        # Find env_configs dict
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'env_configs':
                        self.configs = eval(compile(ast.Expression(node.value), '<test>', 'eval'))
                        return
        self.configs = {}

    def test_quad_with_obstacles_exists(self):
        assert "quad_with_obstacles" in self.configs

    def test_quad_uses_rnn(self):
        assert self.configs["quad_with_obstacles"]["use_rnn"] is True

    def test_quad_rnn_size_64(self):
        assert self.configs["quad_with_obstacles"]["rnn_size"] == 64

    def test_quad_rnn_type_gru(self):
        assert self.configs["quad_with_obstacles"]["rnn_type"] == "gru"

    def test_quad_env_agents_16(self):
        assert self.configs["quad_with_obstacles"]["env_agents"] == 16

    def test_quad_batch_size(self):
        assert self.configs["quad_with_obstacles"]["batch_size"] == 2048

    def test_quad_learning_rate(self):
        assert self.configs["quad_with_obstacles"]["learning_rate"] == pytest.approx(3e-4)

    def test_quad_gamma(self):
        assert self.configs["quad_with_obstacles"]["gamma"] == 0.98

    def test_quad_reward_scale(self):
        assert self.configs["quad_with_obstacles"]["reward_scale"] == 0.1

    def test_quad_action_dim_3(self):
        assert self.configs["quad_with_obstacles"]["action_space_dim"] == 3

    def test_position_setpoint_exists(self):
        assert "position_setpoint_task" in self.configs

    def test_navigation_task_exists(self):
        assert "navigation_task" in self.configs

    def test_nav_uses_rnn(self):
        assert self.configs["navigation_task"]["use_rnn"] is True


class TestGateTrainingDefaults:
    """Capture gate-specific training parameters."""

    def setup_method(self):
        import ast
        with open('aerial_gym/rl_training/sample_factory/aerialgym_examples/train_aerialgym_custom_net_gate.py') as f:
            src = f.read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'env_configs':
                        self.configs = eval(compile(ast.Expression(node.value), '<test>', 'eval'))
                        return
        self.configs = {}

    def test_quad_gate_exists(self):
        assert "quad_with_obstacles_gate" in self.configs

    def test_gate_uses_rnn(self):
        assert self.configs["quad_with_obstacles_gate"]["use_rnn"] is True

    def test_gate_rnn_size(self):
        assert self.configs["quad_with_obstacles_gate"]["rnn_size"] in (64, 128)

    def test_gate_env_agents(self):
        agents = self.configs["quad_with_obstacles_gate"]["env_agents"]
        assert agents in (16, 128, 256)

    def test_gate_4d_actions(self):
        assert self.configs["quad_with_obstacles_gate"]["action_space_dim"] == 4
