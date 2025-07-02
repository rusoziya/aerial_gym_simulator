from aerial_gym.task.navigation_task.navigation_task import NavigationTask
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)

from aerial_gym.utils.math import quat_rotate_inverse, get_euler_xyz_tensor
import torch


class DCE_RL_Navigation_Task(NavigationTask):
    def __init__(self, task_config, **kwargs):
        task_config.action_space_dim = 3
        task_config.curriculum.min_level = 30  # DCE curriculum starts from level 30
        task_config.curriculum.max_level = 50
        
        # Handle headless setting from Sample Factory command line parameters
        # Check if Sample Factory passed headless setting via environment or direct config
        import os
        sf_headless = os.environ.get('SF_HEADLESS', None)
        if sf_headless is not None:
            task_config.headless = sf_headless.lower() == 'true'
            logger.info(f"DCE Navigation Task - Using SF_HEADLESS environment variable: {task_config.headless}")
        elif not hasattr(task_config, 'headless') or task_config.headless is None:
            task_config.headless = False  # Default to visualization enabled
            logger.info(f"DCE Navigation Task - Using default headless=False for visualization")
        else:
            logger.info(f"DCE Navigation Task - Using pre-configured headless: {task_config.headless}")
        
        logger.info(f"DCE Navigation Task - Final headless mode: {task_config.headless}")
        
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
        else:
            logger.info(f"Using {task_config.num_envs} environments as configured.")
            
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
