from aerial_gym.task.navigation_task_gate.navigation_task_gate import NavigationTaskGate
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)

from aerial_gym.utils.math import quat_rotate_inverse, get_euler_xyz_tensor
import torch


class DCE_RL_Navigation_Task_Gate(NavigationTaskGate):
    def __init__(self, task_config, **kwargs):
        task_config.action_space_dim = 4  # 4D action space [x_vel, y_vel, z_vel, yaw_rate] for direct velocity control
        task_config.curriculum.min_level = 3  # Gate task starts from level 3 (matches gate environment obstacles)
        task_config.curriculum.max_level = 23  # Gate task goes up to level 23
        
        # Handle headless setting from Sample Factory command line parameters
        # Check if Sample Factory passed headless setting via environment variable
        import os
        sf_headless = os.environ.get('SF_HEADLESS', None)
        if sf_headless is not None:
            task_config.headless = sf_headless.lower() == 'true'
            logger.info(f"DCE Gate Navigation Task - Using SF_HEADLESS environment variable: {task_config.headless}")
        elif not hasattr(task_config, 'headless') or task_config.headless is None:
            task_config.headless = False  # Default to visualization enabled for gate navigation
            logger.info(f"DCE Gate Navigation Task - Using default headless=False for visualization")
        else:
            logger.info(f"DCE Gate Navigation Task - Using pre-configured headless: {task_config.headless}")
        
        logger.info(f"DCE Gate Navigation Task - Final headless mode: {task_config.headless}")
        
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

    # ===== RESTORED: Enhanced observation processing for gate navigation with dual cameras =====
    # 145D total: 4D target guidance + 13D basic state + 64D drone VAE + 64D static camera VAE (target guidance restored)

    def process_obs_for_task(self):
        # RESTORED: Target direction and distance observations are now ENABLED
        # This DCE task now matches the base navigation task with full target guidance
        
        # Calculate target info for both rewards AND observations
        vec_to_target = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            (self.target_position - self.obs_dict["robot_position"]),
        )
        dist_to_tgt = torch.norm(vec_to_target, dim=1)
        
        # ===== RESTORED: Target guidance observations =====
        # UNCOMMENTED: These lines are now active for full navigation assistance
        self.task_obs["observations"][:, 0:3] = vec_to_target / dist_to_tgt.unsqueeze(1)
        self.task_obs["observations"][:, 3] = dist_to_tgt / 5.0
        # ====================================================
        
        # Basic state observations (17D) - RESTORED to original positions
        euler_angles = ssa(get_euler_xyz_tensor(self.obs_dict["robot_vehicle_orientation"]))
        self.task_obs["observations"][:, 4:6] = euler_angles[:, 0:2]  # RESTORED: was [0:2]
        self.task_obs["observations"][:, 6] = 0.0  # RESTORED: was [2]
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]  # RESTORED: was [3:6]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]  # RESTORED: was [6:9]
        # 4D actions [x_vel, y_vel, z_vel, yaw_rate] for velocity controller
        self.task_obs["observations"][:, 13:17] = self.obs_dict["robot_actions"]  # RESTORED: was [9:13]
        
        # Enhanced observation space with dual VAE latents (145D total) - RESTORED to original positions
        self.task_obs["observations"][:, 17:81] = self.image_latents  # Drone camera VAE (64D) - RESTORED: was [13:77]
        self.task_obs["observations"][:, 81:145] = self.static_image_latents  # Static camera VAE (64D) - FIXED: was static_camera_latents


@torch.jit.script
def ssa(a: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle"""
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi