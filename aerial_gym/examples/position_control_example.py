from __future__ import annotations

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
import torch

from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.helpers import get_args

if __name__ == "__main__":
    args = get_args()
    logger.warning("This example demonstrates the use of geometric controllers for a quadrotor.")
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="base_quadrotor",
        controller_name="lee_position_control",
        args=None,
        device="cuda:0",
        num_envs=args.num_envs,
        headless=args.headless,
        use_warp=args.use_warp,
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    env_manager.reset()
    for i in range(10000):
        if i % 1000 == 0:
            logger.info(f"Step {i}, changing target setpoint.")
            env_manager.reset()
        env_manager.step(actions=actions)
