"""Exact numerical behavior tests for action transformation.

The gate task uses a 4D velocity controller: actions in [-1,1] are scaled
to velocity commands with dimension-specific scaling factors.
"""
import isaacgym  # noqa: F401
import torch
import pytest


@pytest.fixture(scope="module")
def transform():
    from aerial_gym.config.task_config.navigation_task_config_gate import task_config
    return task_config.action_transformation_function


class TestActionTransformExactOutputs:
    """Lock down exact transformation values for each axis."""

    def test_x_axis_scaling(self, transform):
        out = transform(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        assert out[0, 0].item() == pytest.approx(0.6, abs=1e-4)
        assert out[0, 1].item() == pytest.approx(0.0, abs=1e-4)
        assert out[0, 2].item() == pytest.approx(0.0, abs=1e-4)
        assert out[0, 3].item() == pytest.approx(0.0, abs=1e-4)

    def test_y_axis_scaling(self, transform):
        out = transform(torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
        assert out[0, 1].item() == pytest.approx(0.6, abs=1e-4)

    def test_z_axis_scaling(self, transform):
        out = transform(torch.tensor([[0.0, 0.0, 1.0, 0.0]]))
        assert out[0, 2].item() == pytest.approx(0.4, abs=1e-4)

    def test_yaw_axis_scaling(self, transform):
        out = transform(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
        assert out[0, 3].item() == pytest.approx(0.5, abs=1e-4)

    def test_full_negative(self, transform):
        out = transform(torch.tensor([[-1.0, -1.0, -1.0, -1.0]]))
        assert out[0, 0].item() == pytest.approx(-0.6, abs=1e-4)
        assert out[0, 1].item() == pytest.approx(-0.6, abs=1e-4)
        assert out[0, 2].item() == pytest.approx(-0.4, abs=1e-4)
        assert out[0, 3].item() == pytest.approx(-0.5, abs=1e-4)

    def test_half_inputs(self, transform):
        out = transform(torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
        assert out[0, 0].item() == pytest.approx(0.3, abs=1e-4)
        assert out[0, 1].item() == pytest.approx(0.3, abs=1e-4)
        assert out[0, 2].item() == pytest.approx(0.2, abs=1e-4)
        assert out[0, 3].item() == pytest.approx(0.25, abs=1e-4)

    def test_linearity(self, transform):
        """Transform should be linear: tf(2x) = 2*tf(x)."""
        x = torch.tensor([[0.3, 0.4, 0.2, 0.1]])
        t1 = transform(x)
        t2 = transform(2 * x)
        assert torch.allclose(t2, 2 * t1, atol=1e-5)

    def test_batch_consistency(self, transform):
        """Same input repeated should give same output."""
        single = torch.tensor([[0.5, -0.3, 0.7, -0.1]])
        batch = single.repeat(8, 1)
        out = transform(batch)
        for i in range(8):
            assert torch.allclose(out[i], out[0], atol=1e-6)
