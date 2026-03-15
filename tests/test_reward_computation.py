"""Behavior-capture tests for gate reward computation (gate_config_10).

Tests the compute_gate_reward @torch.jit.script function and reward
pipeline behavior to detect drift after refactoring.
"""

import isaacgym  # noqa: F401
import pytest
import torch


@pytest.fixture
def reward_params():
    """Actual gate_config_10 reward parameters as tensors."""
    from aerial_gym.config.task_config.navigation_task_config_gate import task_config

    params = {}
    for k, v in task_config.reward_parameters.items():
        params[k] = torch.tensor(v, device="cpu")
    return params


@pytest.fixture
def num_envs():
    return 4


@pytest.fixture
def zero_state(num_envs):
    """State where drone is at target with zero actions."""
    return {
        "pos_error": torch.zeros(num_envs, 3),
        "prev_pos_error": torch.zeros(num_envs, 3),
        "crashes": torch.zeros(num_envs, dtype=torch.bool),
        "action": torch.zeros(num_envs, 4),
        "prev_action": torch.zeros(num_envs, 4),
        "robot_position": torch.zeros(num_envs, 3),
        "robot_orientation": torch.tensor([[0, 0, 0, 1.0]] * num_envs),
        "gate_position": torch.zeros(num_envs, 3),
        "gate_passed": torch.zeros(num_envs, dtype=torch.bool),
        "gate_width": torch.full((num_envs,), 2.5),
        "gate_height": torch.full((num_envs,), 2.4),
        "gate_center_height": torch.full((num_envs,), 1.2),
        "boundary_violation": torch.zeros(num_envs, dtype=torch.bool),
    }


class TestComputeGateReward:
    """Test the JIT-compiled gate reward function."""

    def setup_method(self):
        from aerial_gym.task.navigation_task_gate.reward_functions import compute_gate_reward

        self.compute = compute_gate_reward

    def test_returns_three_tensors(self, zero_state, reward_params, num_envs):
        result = self.compute(
            zero_state["pos_error"],
            zero_state["prev_pos_error"],
            zero_state["crashes"],
            zero_state["action"],
            zero_state["prev_action"],
            zero_state["robot_position"],
            zero_state["robot_orientation"],
            zero_state["gate_position"],
            zero_state["gate_passed"],
            0.0,
            reward_params,
            zero_state["gate_width"],
            zero_state["gate_height"],
            zero_state["gate_center_height"],
            zero_state["boundary_violation"],
        )
        assert len(result) == 3
        rewards, crashes, alignment = result
        assert rewards.shape == (num_envs,)
        assert crashes.shape == (num_envs,)
        assert alignment.shape == (num_envs,)

    def test_collision_gives_collision_penalty(self, zero_state, reward_params, num_envs):
        zero_state["crashes"] = torch.ones(num_envs, dtype=torch.bool)
        rewards, _, _ = self.compute(
            zero_state["pos_error"],
            zero_state["prev_pos_error"],
            zero_state["crashes"],
            zero_state["action"],
            zero_state["prev_action"],
            zero_state["robot_position"],
            zero_state["robot_orientation"],
            zero_state["gate_position"],
            zero_state["gate_passed"],
            0.0,
            reward_params,
            zero_state["gate_width"],
            zero_state["gate_height"],
            zero_state["gate_center_height"],
            zero_state["boundary_violation"],
        )
        assert torch.all(rewards == -100.0)

    def test_no_crash_reward_is_finite(self, zero_state, reward_params, num_envs):
        rewards, _, _ = self.compute(
            zero_state["pos_error"],
            zero_state["prev_pos_error"],
            zero_state["crashes"],
            zero_state["action"],
            zero_state["prev_action"],
            zero_state["robot_position"],
            zero_state["robot_orientation"],
            zero_state["gate_position"],
            zero_state["gate_passed"],
            0.0,
            reward_params,
            zero_state["gate_width"],
            zero_state["gate_height"],
            zero_state["gate_center_height"],
            zero_state["boundary_violation"],
        )
        assert torch.all(torch.isfinite(rewards))

    def test_closer_to_target_gives_higher_reward(self, zero_state, reward_params, num_envs):
        far = zero_state.copy()
        far["pos_error"] = torch.tensor([[5.0, 0, 0]] * num_envs)
        far["prev_pos_error"] = torch.tensor([[5.0, 0, 0]] * num_envs)

        close = zero_state.copy()
        close["pos_error"] = torch.tensor([[0.5, 0, 0]] * num_envs)
        close["prev_pos_error"] = torch.tensor([[0.5, 0, 0]] * num_envs)

        args = lambda s: (
            s["pos_error"],
            s["prev_pos_error"],
            s["crashes"],
            s["action"],
            s["prev_action"],
            s["robot_position"],
            s["robot_orientation"],
            s["gate_position"],
            s["gate_passed"],
            0.0,
            reward_params,
            s["gate_width"],
            s["gate_height"],
            s["gate_center_height"],
            s["boundary_violation"],
        )

        r_far, _, _ = self.compute(*args(far))
        r_close, _, _ = self.compute(*args(close))
        assert r_close.mean().item() > r_far.mean().item()

    def test_curriculum_fraction_scales_reward(self, zero_state, reward_params, num_envs):
        args = lambda frac: (
            zero_state["pos_error"],
            zero_state["prev_pos_error"],
            zero_state["crashes"],
            zero_state["action"],
            zero_state["prev_action"],
            zero_state["robot_position"],
            zero_state["robot_orientation"],
            zero_state["gate_position"],
            zero_state["gate_passed"],
            frac,
            reward_params,
            zero_state["gate_width"],
            zero_state["gate_height"],
            zero_state["gate_center_height"],
            zero_state["boundary_violation"],
        )

        r_low, _, _ = self.compute(*args(0.0))
        r_high, _, _ = self.compute(*args(1.0))
        assert not torch.allclose(r_low, r_high)


class TestRewardParameterConsistency:
    """Verify reward parameters are internally consistent."""

    def test_collision_penalty_is_negative(self, reward_params):
        assert reward_params["collision_penalty"].item() < 0

    def test_gate_passage_is_positive(self, reward_params):
        assert reward_params["gate_passage_reward_magnitude"].item() > 0

    def test_penalties_are_nonnegative_magnitude(self, reward_params):
        for key in reward_params:
            if "penalty_magnitude" in key:
                assert reward_params[key].item() >= 0, f"{key} should be >= 0"

    def test_all_params_are_finite(self, reward_params):
        for key, val in reward_params.items():
            assert torch.isfinite(val).all(), f"{key} is not finite"


@pytest.mark.parametrize(
    "level,expected_min,expected_max",
    [
        (3, 3, 3),
        (13, 3, 10),
        (23, 10, 10),
    ],
)
def test_obstacle_count_bounds(level, expected_min, expected_max):
    from aerial_gym.config.task_config.navigation_task_config_gate import task_config

    count = task_config.curriculum.get_obstacle_count_behind_gate(level)
    assert expected_min <= count <= expected_max
