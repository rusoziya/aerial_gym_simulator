from aerial_gym.task.navigation_task.navigation_task import NavigationTask
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)

from aerial_gym.utils.math import quat_rotate_inverse, get_euler_xyz_tensor
import torch


class DCE_RL_Navigation_Task(NavigationTask):
    def __init__(self, task_config, **kwargs):
        task_config.action_space_dim = 3
        task_config.curriculum.min_level = 36
        
        # Enable visualization by default for DCE navigation training
        task_config.headless = False
        logger.info(f"DCE Navigation Task - Headless mode: {task_config.headless}")
        
        # Check for Sample Factory env_agents parameter to force specific environment count  
        # This handles rollout worker subprocesses that don't go through registration
        env_agents_override = None
        try:
            # Try to access the global Sample Factory config if available
            import os
            if 'SF_ENV_AGENTS' in os.environ:
                env_agents_override = int(os.environ['SF_ENV_AGENTS'])
                logger.info(f"Found SF_ENV_AGENTS environment variable: {env_agents_override}")
        except:
            pass
        
        # Force specific environment count if env_agents is specified
        if env_agents_override is not None and env_agents_override > 0:
            logger.info(f"Detected env_agents={env_agents_override} from environment - setting environment count.")
            task_config.num_envs = env_agents_override
        elif task_config.num_envs <= 16:
            logger.info(f"Using {task_config.num_envs} environments as configured.")
        else:
            logger.critical("Hardcoding number of envs to 16 if it is greater than that.")
            task_config.num_envs = 16
            
        super().__init__(task_config=task_config, **kwargs)

    # just changing how the observations are returned for the code to work
    # This was the original code.

    def process_obs_for_task(self):
        vec_to_target = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            (self.target_position - self.obs_dict["robot_position"]),
        )
        dist_to_tgt = torch.norm(vec_to_target, dim=1)
        self.task_obs["observations"][:, 0:3] = vec_to_target / dist_to_tgt.unsqueeze(1)
        self.task_obs["observations"][:, 3] = dist_to_tgt / 5.0
        euler_angles = ssa(get_euler_xyz_tensor(self.obs_dict["robot_vehicle_orientation"]))
        self.task_obs["observations"][:, 4:6] = euler_angles[:, 0:2]
        self.task_obs["observations"][:, 6] = 0.0
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 13:17] = self.obs_dict["robot_actions"]
        self.task_obs["observations"][:, 17:81] = self.image_latents


@torch.jit.script
def ssa(a: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle"""
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi
