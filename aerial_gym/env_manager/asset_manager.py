from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("asset_manager")
logger.setLevel("DEBUG")


class AssetManager:
    def __init__(self, global_tensor_dict, num_keep_in_env):
        self.init_tensors(global_tensor_dict, num_keep_in_env)

    def init_tensors(self, global_tensor_dict, num_keep_in_env):
        self.env_asset_state_tensor = global_tensor_dict["env_asset_state_tensor"]
        self.asset_min_state_ratio = global_tensor_dict["asset_min_state_ratio"]
        self.asset_max_state_ratio = global_tensor_dict["asset_max_state_ratio"]
        self.env_bounds_min = (
            global_tensor_dict["env_bounds_min"]
            .unsqueeze(1)
            .expand(-1, self.env_asset_state_tensor.shape[1], -1)
        )
        self.env_bounds_max = (
            global_tensor_dict["env_bounds_max"]
            .unsqueeze(1)
            .expand(-1, self.env_asset_state_tensor.shape[1], -1)
        )
        self.num_keep_in_env = num_keep_in_env

    def prepare_for_sim(self):
        self.reset(self.num_keep_in_env)
        logger.warning(f"Number of obstacles to be kept in the environment: {self.num_keep_in_env}")

    def pre_physics_step(self, actions):
        pass

    def post_physics_step(self):
        pass

    def step(self, actions):
        pass
        # Implement this function if needed.
        # this functionality can do speciic things with the environment assets on stepping.
        # nothing really needs to be done for static environments.
        # if force needs to be applied, it should be done in the other classes and it's
        # better to leave this class to manipulate the state tensors.

    def reset(self, num_obstacles_per_env):
        self.reset_idx(torch.arange(self.env_asset_state_tensor.shape[0]), num_obstacles_per_env)

    def reset_idx(self, env_ids, num_obstacles_per_env=0):
        # logger.warning(f"[OBSTACLE_DEBUG] AssetManager.reset_idx called with env_ids={env_ids.tolist() if hasattr(env_ids, 'tolist') else env_ids}, num_obstacles_per_env={num_obstacles_per_env}")
        # logger.warning(f"[OBSTACLE_DEBUG] num_keep_in_env={self.num_keep_in_env}, total_asset_tensor_shape={self.env_asset_state_tensor.shape}")
        
        if num_obstacles_per_env < self.num_keep_in_env:
            # logger.warning(f"[OBSTACLE_DEBUG] Requested obstacles ({num_obstacles_per_env}) < minimum ({self.num_keep_in_env}), using minimum")
            logger.info(
                "Number of obstacles required in the environment by the \
                  code is lesser than the minimum number of obstacles that the environment configuration specifies."
            )
            num_obstacles_per_env = self.num_keep_in_env

        # logger.warning(f"[OBSTACLE_DEBUG] Final num_obstacles_per_env to keep visible: {num_obstacles_per_env}")
        # logger.warning(f"[OBSTACLE_DEBUG] Assets that will be VISIBLE: indices 0 to {num_obstacles_per_env-1}")
        # logger.warning(f"[OBSTACLE_DEBUG] Assets that will be HIDDEN: indices {num_obstacles_per_env} to {self.env_asset_state_tensor.shape[1]-1}")

        sampled_asset_state_ratio = torch_rand_float_tensor(
            self.asset_min_state_ratio, self.asset_max_state_ratio
        )
        self.env_asset_state_tensor[env_ids, :, 0:3] = torch_interpolate_ratio(
            min=self.env_bounds_min,
            max=self.env_bounds_max,
            ratio=sampled_asset_state_ratio[..., 0:3],
        )[env_ids, :, 0:3]
        self.env_asset_state_tensor[env_ids, :, 3:7] = quat_from_euler_xyz_tensor(
            sampled_asset_state_ratio[env_ids, :, 3:6]
        )
        
        # put those obstacles not needed in the environment outside
        # Avoid advanced indexing shape issues on some PyTorch/CUDA combos by iterating per-env
        # logger.warning(f"[OBSTACLE_DEBUG] Moving assets {num_obstacles_per_env}..end to position (-1000, -1000, -1000)")
        try:
            env_list = env_ids.tolist() if hasattr(env_ids, 'tolist') else list(env_ids)
        except Exception:
            env_list = [int(env_ids)]
        for eid in env_list:
            self.env_asset_state_tensor[eid, num_obstacles_per_env:, 0:3] = -1000.0
        
        # Count visible obstacles per environment for debugging
        # for env_id in (env_ids.tolist() if hasattr(env_ids, 'tolist') else [env_ids]):
        #     visible_count = 0
        #     hidden_count = 0
        #     for asset_idx in range(self.env_asset_state_tensor.shape[1]):
        #         pos = self.env_asset_state_tensor[env_id, asset_idx, 0:3]
        #         if pos[0] > -500:  # Not hidden
        #             visible_count += 1
        #         else:
        #             hidden_count += 1
        #     logger.warning(f"[OBSTACLE_DEBUG] Env {env_id}: {visible_count} visible assets, {hidden_count} hidden assets")
