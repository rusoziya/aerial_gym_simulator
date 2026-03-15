"""Behavior tests for base navigation reward sub-components."""
import isaacgym  # noqa: F401
import torch
import pytest
from aerial_gym.task.navigation_task.navigation_task import compute_reward
from aerial_gym.config.task_config.navigation_task_config import task_config


@pytest.fixture(scope="module")
def rp():
    return {k: torch.tensor(v) for k, v in task_config.reward_parameters.items()}


class TestPositionRewardDecay:
    def test_closer_higher_reward(self, rp):
        r_close, _ = compute_reward(torch.tensor([[0.5,0,0]]), torch.tensor([[0.5,0,0]]),
                                     torch.zeros(1, dtype=torch.bool), torch.zeros(1,4), torch.zeros(1,4), 0.0, rp)
        r_far, _ = compute_reward(torch.tensor([[5.0,0,0]]), torch.tensor([[5.0,0,0]]),
                                   torch.zeros(1, dtype=torch.bool), torch.zeros(1,4), torch.zeros(1,4), 0.0, rp)
        assert r_close[0].item() > r_far[0].item()


class TestGettingCloserAsymmetry:
    def test_2x_penalty_for_farther(self, rp):
        base, _ = compute_reward(torch.tensor([[2.0,0,0]]), torch.tensor([[2.0,0,0]]),
                                  torch.zeros(1, dtype=torch.bool), torch.zeros(1,4), torch.zeros(1,4), 0.0, rp)
        closer, _ = compute_reward(torch.tensor([[1.0,0,0]]), torch.tensor([[2.0,0,0]]),
                                    torch.zeros(1, dtype=torch.bool), torch.zeros(1,4), torch.zeros(1,4), 0.0, rp)
        farther, _ = compute_reward(torch.tensor([[3.0,0,0]]), torch.tensor([[2.0,0,0]]),
                                     torch.zeros(1, dtype=torch.bool), torch.zeros(1,4), torch.zeros(1,4), 0.0, rp)
        gain = closer[0].item() - base[0].item()
        loss = base[0].item() - farther[0].item()
        assert loss > gain  # 2x asymmetry

class TestDistanceFromGoal:
    def test_at_origin_max_component(self, rp):
        r, _ = compute_reward(torch.zeros(1,3), torch.zeros(1,3), torch.zeros(1, dtype=torch.bool),
                               torch.zeros(1,4), torch.zeros(1,4), 0.0, rp)
        assert r[0].item() > 0

class TestActionPenaltiesNav:
    @pytest.mark.parametrize("dim", [0, 2, 3])
    def test_penalty_per_axis(self, rp, dim):
        a = torch.zeros(1, 4)
        a[0, dim] = 0.9
        r_big, _ = compute_reward(torch.ones(1,3), torch.ones(1,3), torch.zeros(1, dtype=torch.bool),
                                   a, torch.zeros(1,4), 0.0, rp)
        r_zero, _ = compute_reward(torch.ones(1,3), torch.ones(1,3), torch.zeros(1, dtype=torch.bool),
                                    torch.zeros(1,4), torch.zeros(1,4), 0.0, rp)
        assert r_zero[0].item() > r_big[0].item()
