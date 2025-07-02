import time
import isaacgym

# isort: on
import torch
from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym import (
    parse_aerialgym_cfg,
)
from aerial_gym.utils import get_args
from aerial_gym.registry.task_registry import task_registry


from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import DCE_RL_Navigation_Task
from aerial_gym.examples.dce_rl_navigation.sf_inference_class import NN_Inference_Class

import matplotlib
import numpy as np
from PIL import Image
import os
import cv2  # Import OpenCV for real-time display
from datetime import datetime
from typing import List


def sample_command(args):
    use_warp = True
    headless = False  # Force viewer to be enabled to see 3D simulation
    
    # Flag to enable real-time visualization
    show_realtime = not headless
    
    # 1) pick an output folder (side-by-side with your script)
    output_dir = os.path.join(os.getcwd(), "gifs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 2) your frame buffers
    # seg_frames: List[Image.Image] = []
    # depth_frames: List[Image.Image] = []
    merged_image_frames: List[Image.Image] = []

    # Create task with minimal parameters
    rl_task = task_registry.make_task(
        "dce_navigation_task", 
        seed=42, 
        use_warp=use_warp, 
        headless=headless
    )
    print("Number of environments", rl_task.num_envs)
    command_actions = torch.zeros((rl_task.num_envs, rl_task.task_config.action_space_dim))
    command_actions[:, 0] = 1.5
    command_actions[:, 1] = 0.0
    command_actions[:, 2] = 0.0
    nn_model = get_network(rl_task.num_envs)
    nn_model.eval()
    nn_model.reset(torch.arange(rl_task.num_envs))
    rl_task.reset()
    
    # Create window for real-time display if not headless
    if show_realtime:
        # cv2.namedWindow("Depth Camera", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Segmentation Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Combined View", cv2.WINDOW_NORMAL)
        # Set window sizes
        # cv2.resizeWindow("Depth Camera", 480, 270)
        # cv2.resizeWindow("Segmentation Camera", 480, 270)
        cv2.resizeWindow("Combined View", 480, 540)
        print("Real-time camera view enabled. Press 'q' to exit.")
        
    for i in range(0, 50000):
        start_time = time.time()
        obs, rewards, termination, truncation, infos = rl_task.step(command_actions)

        obs["obs"] = obs["observations"]
        # print(obs["observations"].shape)
        action = nn_model.get_action(obs)
        # print("Action", action, action.shape)
        action = torch.as_tensor(action, device=rl_task.device).expand(rl_task.num_envs, -1)
        command_actions[:] = action

        reset_ids = (termination + truncation).nonzero(as_tuple=True)
        if torch.any(termination):
            terminated_envs = termination.nonzero(as_tuple=True)
            print(f"Resetting environments {terminated_envs} due to Termination")
        if torch.any(truncation):
            truncated_envs = truncation.nonzero(as_tuple=True)
            print(f"Resetting environments {truncated_envs} due to Timeout")
        nn_model.reset(reset_ids)

        # Capture frames for visualization
        image1 = (
            255.0 * rl_task.obs_dict["depth_range_pixels"][0, 0].cpu().numpy()
        ).astype(np.uint8)
        seg_image1 = rl_task.obs_dict["segmentation_pixels"][0, 0].cpu().numpy()
        
        # Fix the error when there are no positive values in the segmentation image
        if np.any(seg_image1 > 0):
            min_positive = seg_image1[seg_image1 > 0].min()
            seg_image1[seg_image1 <= 0] = min_positive
        else:
            # If no positive values, set all to a small positive value
            seg_image1[:] = 0.1
            
        seg_image1_normalized = (seg_image1 - seg_image1.min()) / (
            seg_image1.max() - seg_image1.min() + 1e-8
        )

        # set colormap to plasma in matplotlib
        seg_image1_normalized_plasma = matplotlib.cm.plasma(seg_image1_normalized)
        seg_image1_pil = Image.fromarray((seg_image1_normalized_plasma * 255.0).astype(np.uint8))

        depth_image1_pil = Image.fromarray(image1)
        image_4d = np.zeros((image1.shape[0], image1.shape[1], 4))
        image_4d[:, :, 0] = image1
        image_4d[:, :, 1] = image1
        image_4d[:, :, 2] = image1
        image_4d[:, :, 3] = 255.0
        merged_image = np.concatenate((image_4d, seg_image1_normalized_plasma * 255.0), axis=0)
        
        # Save frames to array for GIF creation - only keeping merged image
        # seg_frames.append(seg_image1_pil)
        # depth_frames.append(depth_image1_pil)
        merged_image_frames.append(Image.fromarray(merged_image.astype(np.uint8)))
        
        # Display frames in real-time windows
        if show_realtime:
            # Convert PIL images to OpenCV format (RGB to BGR)
            # depth_cv = cv2.cvtColor(np.array(depth_image1_pil), cv2.COLOR_RGB2BGR)
            # seg_cv = cv2.cvtColor(np.array(seg_image1_pil), cv2.COLOR_RGBA2BGR)
            merged_cv = cv2.cvtColor(merged_image[:,:,0:3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Display images
            # cv2.imshow("Depth Camera", depth_cv)
            # cv2.imshow("Segmentation Camera", seg_cv)
            cv2.imshow("Combined View", merged_cv)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested exit")
                break
        
        # 3) detect end of episode
        if termination[0] or truncation[0]:
            # build a unique file-stem (episode index + timestamp)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = f"epi{i}_{ts}"
            
            # 4) save only the merged GIF
            # seg_frames[0].save(
            #     os.path.join(output_dir, f"{stem}_seg.gif"),
            #     save_all=True,
            #     append_images=seg_frames[1:],
            #     duration=100,
            #     loop=0,
            # )
            # depth_frames[0].save(
            #     os.path.join(output_dir, f"{stem}_depth.gif"),
            #     save_all=True,
            #     append_images=depth_frames[1:],
            #     duration=100,
            #     loop=0,
            # )
            merged_image_frames[0].save(
                os.path.join(output_dir, f"{stem}_merged.gif"),
                save_all=True,
                append_images=merged_image_frames[1:],
                duration=100,
                loop=0,
            )
            
            # 5) reset buffers and environments
            # seg_frames.clear()
            # depth_frames.clear()
            merged_image_frames.clear()
            rl_task.reset()

    # Clean up OpenCV windows when done
    if show_realtime:
        cv2.destroyAllWindows()


def get_network(num_envs):
    """Script entry point."""
    # register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg(evaluation=True)
    print("CFG is:", cfg)
    nn_model = NN_Inference_Class(num_envs, 3, 81, cfg)
    return nn_model


if __name__ == "__main__":
    # Modified task registration
    task_registry.register_task(
        task_name="dce_navigation_task",
        task_class=DCE_RL_Navigation_Task,
        task_config=task_registry.get_task_config("navigation_task"),
    )
    args = get_args()
    sample_command(args)
