"""Exact numerical behavior tests for reward computation.

These tests lock down the EXACT numerical outputs of reward functions
for specific inputs. Any refactoring that changes these values indicates
behavioral drift.
"""
import isaacgym  # noqa: F401
import torch
import pytest


@pytest.fixture(scope="module")
def gate_reward_params():
    from aerial_gym.config.task_config.navigation_task_config_gate import task_config
    return {k: torch.tensor(v) for k, v in task_config.reward_parameters.items()}


@pytest.fixture(scope="module")
def nav_reward_params():
    from aerial_gym.config.task_config.navigation_task_config import task_config
    return {k: torch.tensor(v) for k, v in task_config.reward_parameters.items()}


class TestGateRewardExactOutputs:
    """Lock down exact reward values for specific input scenarios."""

    def setup_method(self):
        from aerial_gym.task.navigation_task_gate.navigation_task_gate import compute_gate_reward
        self.compute = compute_gate_reward

    def _make_state(self, N=4):
        return dict(
            pos_error=torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [2.0, 2.0, 2.0]]),
            prev_pos_error=torch.tensor([[1.1, 0, 0], [0, 1.1, 0], [0, 0, 1.1], [2.1, 2.1, 2.1]]),
            crashes=torch.zeros(N, dtype=torch.bool),
            action=torch.zeros(N, 4),
            prev_action=torch.zeros(N, 4),
            robot_pos=torch.tensor([[0, -1.5, 1.5]] * N),
            robot_orient=torch.tensor([[0, 0, 0, 1.0]] * N),
            gate_pos=torch.zeros(N, 3),
            gate_passed=torch.zeros(N, dtype=torch.bool),
            gate_width=torch.full((N,), 2.5),
            gate_height=torch.full((N,), 2.4),
            gate_center=torch.full((N,), 1.2),
            bv=torch.zeros(N, dtype=torch.bool),
        )

    def _call(self, s, rp, frac=0.5):
        return self.compute(
            s["pos_error"], s["prev_pos_error"], s["crashes"], s["action"], s["prev_action"],
            s["robot_pos"], s["robot_orient"], s["gate_pos"], s["gate_passed"],
            frac, rp, s["gate_width"], s["gate_height"], s["gate_center"], s["bv"])

    def test_approach_rewards_exact(self, gate_reward_params):
        s = self._make_state()
        r, _, _ = self._call(s, gate_reward_params)
        assert r[0].item() == pytest.approx(3.783727, abs=1e-3)
        assert r[3].item() == pytest.approx(3.538029, abs=1e-3)

    def test_collision_exact(self, gate_reward_params):
        s = self._make_state()
        s["crashes"] = torch.ones(4, dtype=torch.bool)
        r, _, _ = self._call(s, gate_reward_params)
        assert all(r[i].item() == pytest.approx(-100.0, abs=1e-3) for i in range(4))

    def test_at_gate_center_high_reward(self, gate_reward_params):
        s = self._make_state()
        s["robot_pos"] = torch.tensor([[0.0, 0.01, 1.2]] * 4)
        s["pos_error"] = torch.tensor([[0.01, 0, 0]] * 4)
        s["prev_pos_error"] = s["pos_error"] + 0.5
        r, _, _ = self._call(s, gate_reward_params)
        assert r[0].item() == pytest.approx(259.897, abs=1.0)

    def test_far_from_gate_low_reward(self, gate_reward_params):
        s = self._make_state()
        s["robot_pos"] = torch.tensor([[3.0, -3.0, 1.5]] * 4)
        s["pos_error"] = torch.tensor([[5.0, 0, 0]] * 4)
        s["prev_pos_error"] = s["pos_error"]
        r, _, _ = self._call(s, gate_reward_params)
        assert r[0].item() == pytest.approx(0.01476, abs=0.01)

    def test_big_actions_reduce_reward(self, gate_reward_params):
        s = self._make_state()
        r_zero, _, _ = self._call(s, gate_reward_params)
        s["action"] = torch.ones(4, 4) * 0.9
        r_big, _, _ = self._call(s, gate_reward_params)
        assert r_big[0].item() < r_zero[0].item()

    def test_curriculum_fraction_0_vs_1(self, gate_reward_params):
        s = self._make_state()
        r0, _, _ = self._call(s, gate_reward_params, frac=0.0)
        r1, _, _ = self._call(s, gate_reward_params, frac=1.0)
        assert r0[0].item() == pytest.approx(3.026982, abs=1e-2)
        assert r1[0].item() == pytest.approx(4.540473, abs=1e-2)
        assert r1[0].item() > r0[0].item()

    def test_closer_gets_higher_reward(self, gate_reward_params):
        s = self._make_state()
        s["pos_error"] = torch.tensor([[3.0, 0, 0]] * 4)
        s["prev_pos_error"] = torch.tensor([[3.0, 0, 0]] * 4)
        r_far, _, _ = self._call(s, gate_reward_params)
        s["pos_error"] = torch.tensor([[0.5, 0, 0]] * 4)
        s["prev_pos_error"] = torch.tensor([[0.5, 0, 0]] * 4)
        r_close, _, _ = self._call(s, gate_reward_params)
        assert r_close[0].item() > r_far[0].item()


class TestBaseNavRewardExactOutputs:
    """Lock down base navigation reward values."""

    def setup_method(self):
        from aerial_gym.task.navigation_task.navigation_task import compute_reward
        self.compute = compute_reward

    def test_approach_exact(self, nav_reward_params):
        pe = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [2.0, 2.0, 2.0]])
        r, _ = self.compute(pe, pe + 0.1, torch.zeros(4, dtype=torch.bool),
                            torch.zeros(4, 4), torch.zeros(4, 4), 0.5, nav_reward_params)
        assert r[0].item() == pytest.approx(8.093, abs=0.1)

    def test_collision_exact(self, nav_reward_params):
        pe = torch.tensor([[1.0, 0, 0]] * 4)
        r, _ = self.compute(pe, pe, torch.ones(4, dtype=torch.bool),
                            torch.zeros(4, 4), torch.zeros(4, 4), 0.5, nav_reward_params)
        assert all(r[i].item() == pytest.approx(-100.0) for i in range(4))

    def test_at_target_high_reward(self, nav_reward_params):
        r, _ = self.compute(torch.zeros(4, 3), torch.ones(4, 3) * 0.1,
                            torch.zeros(4, dtype=torch.bool),
                            torch.zeros(4, 4), torch.zeros(4, 4), 0.5, nav_reward_params)
        assert r[0].item() == pytest.approx(15.915, abs=0.1)
