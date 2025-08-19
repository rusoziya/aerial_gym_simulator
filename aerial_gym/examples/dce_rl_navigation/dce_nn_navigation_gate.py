import time
import isaacgym

# isort: on
import torch
from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net_gate import (
    parse_aerialgym_cfg,
)
from aerial_gym.utils import get_args
from aerial_gym.registry.task_registry import task_registry


from aerial_gym.examples.dce_rl_navigation.dce_navigation_task_gate import DCE_RL_Navigation_Task_Gate
from aerial_gym.examples.dce_rl_navigation.sf_inference_class_gate import NN_Inference_Class

import matplotlib
import numpy as np
from PIL import Image


def sample_command(args):
    use_warp = True
    # Enable viewing by default for inference - user can override with --headless
    headless = getattr(args, 'headless', False)  # Default to False (viewing enabled)
    print(f"DCE Gate Inference - Headless mode: {headless}")
    
    # seg_frames = []
    # depth_frames = []
    # merged_image_frames = []

    rl_task = task_registry.make_task(
        "dce_navigation_task_gate", seed=42, use_warp=use_warp, headless=headless
    )
    print("Number of environments", rl_task.num_envs)
    
    # 4D action space [x_vel, y_vel, z_vel, yaw_rate] for velocity controller
    command_actions = torch.zeros((rl_task.num_envs, rl_task.task_config.action_space_dim), device=rl_task.device)
    
    # Initialize inference model
    nn_model = get_network(rl_task.num_envs)
    nn_model.eval()
    
    # Optionally load checkpoint (prefer explicit env var to avoid SF arg conflicts)
    import os, glob
    env_model = os.environ.get('DCE_MODEL', '').strip()
    if env_model:
        print(f"[Inference] DCE_MODEL={env_model}")
    model_path = env_model or getattr(args, 'load_checkpoint', None) or getattr(args, 'checkpoint', None)
    loaded = False
    if model_path:
        try:
            nn_model.load_model(model_path)
            loaded = True
        except Exception as e:
            print(f"[Inference] Warning: failed to load model '{model_path}': {e}")
    # Auto-detect from Sample Factory cfg if not explicitly provided
    if not loaded:
        try:
            cfg = getattr(nn_model, 'cfg', None)
            td = getattr(cfg, 'train_dir', None)
            ex = getattr(cfg, 'experiment', None)
            kind = getattr(cfg, 'load_checkpoint_kind', 'best')
            if td and ex:
                ckpt_dir = os.path.join(str(td), str(ex), 'checkpoint_p0')
                pattern = 'best*.pth' if str(kind) == 'best' else '*.pth'
                candidates = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
                if not candidates and str(kind) != 'best':
                    candidates = sorted(glob.glob(os.path.join(ckpt_dir, 'best*.pth')))
                if candidates:
                    auto_ckpt = candidates[-1]
                    print(f"[Inference] Auto-selected checkpoint: {auto_ckpt}")
                    nn_model.load_model(auto_ckpt)
                    loaded = True
                else:
                    print(f"[Inference] No checkpoints found in {ckpt_dir}")
        except Exception as e:
            print(f"[Inference] Auto-detect failed: {e}")
    if not loaded:
        print("[Inference] ERROR: No model loaded. Set DCE_MODEL to a valid .pth or provide train_dir/experiment with saved checkpoints.")
        raise SystemExit(1)
    
    nn_model.reset_rnn_states()
    rl_task.reset()
    
    for i in range(0, 50000):
        start_time = time.time()
        obs, rewards, termination, truncation, infos = rl_task.step(command_actions)

        # Build batched obs for policy (150D)
        obs_batch = obs["observations"]  # [N, 150]
        
        # Get batched actions from policy
        try:
            actions_np = nn_model.get_action_batched({"obs": obs_batch})
        except Exception:
            actions_np = nn_model.get_action_batched(obs_batch)
        actions = torch.tensor(actions_np, device=rl_task.device)
        command_actions[:] = actions

        reset_ids = (termination + truncation).nonzero(as_tuple=True)
        if torch.any(termination):
            terminated_envs = termination.nonzero(as_tuple=True)
            print(f"Resetting environments {terminated_envs} due to Termination")
        if torch.any(truncation):
            truncated_envs = truncation.nonzero(as_tuple=True)
            print(f"Resetting environments {truncated_envs} due to Timeout")
        nn_model.reset(reset_ids)

    # # Uncomment the below lines to save the frames from an episode as a GIF
    #     # save obs to file as a .gif
    #     image1 = (
    #         255.0 * rl_task.obs_dict["depth_range_pixels"][0, 0].cpu().numpy()
    #     ).astype(np.uint8)
    #     seg_image1 = rl_task.obs_dict["segmentation_pixels"][0, 0].cpu().numpy()
    #     seg_image1[seg_image1 <= 0] = seg_image1[seg_image1 > 0].min()
    #     seg_image1_normalized = (seg_image1 - seg_image1.min()) / (
    #         seg_image1.max() - seg_image1.min()
    #     )

    #     # set colormap to plasma in matplotlib
    #     seg_image1_normalized_plasma = matplotlib.cm.plasma(seg_image1_normalized)
    #     seg_image1 = Image.fromarray((seg_image1_normalized_plasma * 255.0).astype(np.uint8))

    #     depth_image1 = Image.fromarray(image1)
    #     image_4d = np.zeros((image1.shape[0], image1.shape[1], 4))
    #     image_4d[:, :, 0] = image1
    #     image_4d[:, :, 1] = image1
    #     image_4d[:, :, 2] = image1
    #     image_4d[:, :, 3] = 255.0
    #     merged_image = np.concatenate((image_4d, seg_image1_normalized_plasma * 255.0), axis=0)
    #     # save frames to array:
    #     seg_frames.append(seg_image1)
    #     depth_frames.append(depth_image1)
    #     merged_image_frames.append(Image.fromarray(merged_image.astype(np.uint8)))
    # if termination[0] or truncation[0]:
    #     print("i", i)
    #     rl_task.reset()
    #     # save frames as a gif:
    #     seg_frames[0].save(
    #         f"seg_frames_{i}.gif",
    #         save_all=True,
    #         append_images=seg_frames[1:],
    #         duration=100,
    #         loop=0,
    #     )
    #     depth_frames[0].save(
    #         f"depth_frames_{i}.gif",
    #         save_all=True,
    #         append_images=depth_frames[1:],
    #         duration=100,
    #         loop=0,
    #     )
    #     merged_image_frames[0].save(
    #         f"merged_image_frames_{i}.gif",
    #         save_all=True,
    #         append_images=merged_image_frames[1:],
    #         duration=100,
    #         loop=0,
    #     )
    #     seg_frames = []
    #     depth_frames = []
    #     merged_image_frames = []


def get_network(num_envs):
    """Script entry point."""
    from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net_gate import register_aerialgym_custom_components
    register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg(evaluation=True)
    print("CFG is:", cfg)
    # ENHANCED: 4D action space and 150D observation space for position-aware gate navigation
    nn_model = NN_Inference_Class(num_envs, 4, 150, cfg)  # 4D action, 150D observation (position-aware)
    return nn_model


if __name__ == "__main__":
    task_registry.register_task(
        task_name="dce_navigation_task_gate",
        task_class=DCE_RL_Navigation_Task_Gate,
        task_config=task_registry.get_task_config(
            "navigation_task_gate"
        ),  # use gate navigation task config
    )
    args = get_args()
    sample_command(args) 