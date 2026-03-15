"""Behavior-capture tests for the registry system (gate_config_10).

Verifies that all expected tasks, robots, controllers, envs, and sims
are registered with the correct names and classes.
"""

import isaacgym  # noqa: F401


class TestTaskRegistry:
    def setup_method(self):
        from aerial_gym.registry.task_registry import task_registry

        self.registry = task_registry

    def test_navigation_task_registered(self):
        cls = self.registry.get_task_class("navigation_task")
        assert cls is not None

    def test_navigation_task_gate_registered(self):
        cls = self.registry.get_task_class("navigation_task_gate")
        assert cls is not None

    def test_position_setpoint_task_registered(self):
        cls = self.registry.get_task_class("position_setpoint_task")
        assert cls is not None

    def test_navigation_task_config_exists(self):
        cfg = self.registry.get_task_config("navigation_task")
        assert cfg is not None

    def test_navigation_task_gate_config_exists(self):
        cfg = self.registry.get_task_config("navigation_task_gate")
        assert cfg is not None


class TestRobotRegistry:
    def setup_method(self):
        from aerial_gym.registry.robot_registry import robot_registry

        self.registry = robot_registry

    def test_lmf2_registered(self):
        cfg = self.registry.get_robot_config("lmf2")
        assert cfg is not None

    def test_base_quadrotor_registered(self):
        cfg = self.registry.get_robot_config("base_quadrotor")
        assert cfg is not None


class TestControllerRegistry:
    def setup_method(self):
        from aerial_gym.registry.controller_registry import controller_registry

        self.registry = controller_registry

    def test_lmf2_velocity_control_registered(self):
        cfg = self.registry.get_controller_config("lmf2_velocity_control")
        assert cfg is not None


class TestEnvRegistry:
    def setup_method(self):
        from aerial_gym.registry.env_registry import env_config_registry

        self.registry = env_config_registry

    def test_gate_env_registered(self):
        cfg = self.registry.get_env_config("gate_env")
        assert cfg is not None

    def test_env_with_obstacles_registered(self):
        cfg = self.registry.get_env_config("env_with_obstacles")
        assert cfg is not None


class TestSimRegistry:
    def setup_method(self):
        from aerial_gym.registry.sim_registry import sim_config_registry

        self.registry = sim_config_registry

    def test_base_sim_registered(self):
        cfg = self.registry.get_sim_config("base_sim")
        assert cfg is not None
