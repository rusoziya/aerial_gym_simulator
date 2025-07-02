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

        # logger.critical("Hardcoding number of envs to 16 if it is greater than that.")
        # task_config.num_envs = 4 if task_config.num_envs > 16 else task_config.num_envs
        logger.critical("Setting number of environments to 1.")
        task_config.num_envs = 1  # Single environment
        task_config.sim_name = "base_sim"
        task_config.env_name = "env_with_obstacles"
        # task_config.env_name = "forest_env"

        # Configure the drone model to match save_camera_stream.py
        # task_config.robot_name = "base_quadrotor_with_stereo_camera"
        # task_config.controller_name = "lmf2_velocity_control" 

        # task_config.controller_name = "lee_velocity_control"
        task_config.robot_name = "lmf2"
        task_config.controller_name = "lmf2_velocity_control"
        
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
        # Only use image latents if VAE is enabled
        if hasattr(self, 'image_latents') and self.task_config.vae_config.use_vae:
            self.task_obs["observations"][:, 17:81] = self.image_latents


@torch.jit.script
def ssa(a: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle"""
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi
