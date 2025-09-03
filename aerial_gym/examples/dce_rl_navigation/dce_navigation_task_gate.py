from aerial_gym.task.navigation_task_gate.navigation_task_gate import NavigationTaskGate
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)

import os
import torch
from aerial_gym.utils.math import quat_rotate_inverse
from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net_gate import (
    parse_aerialgym_cfg,
)


class DCE_RL_Navigation_Task_Gate(NavigationTaskGate):
    def __init__(self, task_config, **kwargs):
        # Honor headless coming from SF or CLI; default to False (viewer enabled)
        sf_headless = os.environ.get("SF_HEADLESS", None)
        if sf_headless is not None:
            task_config.headless = sf_headless.lower() == "true"
            logger.info(
                f"DCE Gate Navigation Task - Using SF_HEADLESS environment variable: {task_config.headless}"
            )
        elif not hasattr(task_config, "headless") or task_config.headless is None:
            task_config.headless = False
            logger.info(
                "DCE Gate Navigation Task - Using default headless=False for visualization"
            )
        else:
            logger.info(
                f"DCE Gate Navigation Task - Using pre-configured headless: {task_config.headless}"
            )

        # Allow Sample Factory to override env count via SF_ENV_AGENTS for viewer/enjoy
        env_agents_override = None
        try:
            if "SF_ENV_AGENTS" in os.environ:
                env_agents_override = int(os.environ["SF_ENV_AGENTS"])
                logger.info(
                    f"Found SF_ENV_AGENTS environment variable: {env_agents_override}"
                )
        except Exception:
            pass

        if env_agents_override is not None and env_agents_override > 0:
            logger.info(
                f"Detected env_agents={env_agents_override} from environment - setting environment count."
            )
            task_config.num_envs = env_agents_override

        super().__init__(task_config=task_config, **kwargs)

        # Backward-compat: expose a parsed SF cfg for older entrypoints
        try:
            self.sf_cfg = parse_aerialgym_cfg(evaluation=True)
        except Exception:
            self.sf_cfg = None

        # Hotfix: some static camera managers may miss a 'device' attribute used by orientation updates
        try:
            dev = self.device if hasattr(self, "device") else (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )
            scm = getattr(self, "static_camera_manager", None)
            if scm is not None and not hasattr(scm, "device"):
                setattr(scm, "device", dev)
            # Also patch sim_env manager if present
            if hasattr(self, "sim_env"):
                scm2 = getattr(self.sim_env, "static_camera_manager", None)
                if scm2 is not None and not hasattr(scm2, "device"):
                    setattr(scm2, "device", dev)
        except Exception:
            pass

    # Subclass no longer overrides relative pose; use base implementation in NavigationTaskGate
    pass
