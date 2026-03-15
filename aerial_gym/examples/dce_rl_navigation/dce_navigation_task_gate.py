from __future__ import annotations

import os

from aerial_gym.task.navigation_task_gate.navigation_task_gate import NavigationTaskGate
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)


class DCE_RL_Navigation_Task_Gate(NavigationTaskGate):
    def __init__(self, task_config: object, **kwargs: object) -> None:
        # Honor headless coming from SF or CLI; default to False (viewer enabled)
        sf_headless = os.environ.get("SF_HEADLESS", None)
        if sf_headless is not None:
            task_config.headless = sf_headless.lower() == "true"
        elif task_config.headless is None:
            task_config.headless = False

        # Allow Sample Factory to override env count via SF_ENV_AGENTS
        sf_env_agents = os.environ.get("SF_ENV_AGENTS", None)
        if sf_env_agents is not None:
            try:
                env_agents_override = int(sf_env_agents)
                if env_agents_override > 0:
                    task_config.num_envs = env_agents_override
                    logger.info(f"Using SF_ENV_AGENTS={env_agents_override}")
            except ValueError:
                pass

        super().__init__(task_config=task_config, **kwargs)

        # Ensure static camera manager has a device attribute
        device = self.device
        if self.static_camera_manager is not None:
            self.static_camera_manager.device = device
