from __future__ import annotations

import numpy as np
import torch

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.task.navigation_task_gate.camera_pose_and_visibility import CameraPoseAndVisibility

logger = CustomLogger("navigation_task_gate_camera")


class CameraObservations:
    def __init__(self, task: object) -> None:
        self.task = task
        self._pose_vis = CameraPoseAndVisibility(task)

    def _compute_visibility_metrics(self, infos_to_return: dict[str, torch.Tensor]) -> None:
        self._pose_vis._compute_visibility_metrics(infos_to_return)

    def _get_static_camera_pose_relative_to_drone(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._pose_vis._get_static_camera_pose_relative_to_drone()

    def process_image_observation(self) -> None:
        """Process drone camera depth image: apply curriculum noise, frame dropout, and VAE encode."""
        image_obs = self.task.obs_dict["depth_range_pixels"].squeeze(1)

        noised_image_obs = self._apply_drone_camera_noise(image_obs)
        noised_image_obs = self._apply_drone_frame_dropout(noised_image_obs)

        self.task.obs_dict["depth_range_pixels_noised"] = noised_image_obs.unsqueeze(1)

        if self.task.task_config.vae_config.use_vae:
            self._encode_drone_vae(noised_image_obs)

    def _apply_drone_camera_noise(self, image_obs: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise and pixel dropout to drone depth image."""
        noised = image_obs.clone()
        camera_noise_disabled = bool(
            self.task.sim_env.global_tensor_dict.get("camera_randomization/noise_disabled", False)
        )
        drone_noise_override = bool(
            self.task.sim_env.global_tensor_dict.get(
                "camera_randomization/drone_noise_disabled", False
            )
        )
        if not self.task.task_config.curriculum.enable_camera_noise:
            return noised

        if not camera_noise_disabled and not drone_noise_override:
            gaussian_std, dropout_rate = self.task.task_config.curriculum.get_camera_noise(
                self.task.curriculum_level
            )
        else:
            gaussian_std, dropout_rate = self.task.task_config.curriculum.get_camera_noise(3)

        if gaussian_std > 0:
            noised = noised + torch.randn_like(noised) * gaussian_std
        if dropout_rate > 0:
            dropout_mask = torch.rand_like(noised) < dropout_rate
            noised = noised.masked_fill(dropout_mask, 1.0)
        return torch.clamp(noised, 0.0, 1.0)

    def _apply_drone_frame_dropout(self, noised: torch.Tensor) -> torch.Tensor:
        """Apply entire-frame blank/freeze dropout to drone depth image."""
        frame_dropout_disabled = bool(
            self.task.sim_env.global_tensor_dict.get(
                "camera_randomization/frame_dropout_disabled", False
            )
        )
        drone_fd_override = bool(
            self.task.sim_env.global_tensor_dict.get(
                "camera_randomization/drone_frame_dropout_disabled", False
            )
        )

        if self.task.task_config.curriculum.enable_camera_frame_dropout:
            if not frame_dropout_disabled and not drone_fd_override:
                fd = self.task.task_config.curriculum.get_camera_frame_dropout(
                    self.task.curriculum_level
                )
            else:
                fd = self.task.task_config.curriculum.get_camera_frame_dropout(3)
            p_blank = fd.get("drone_blank", 0.0)
            p_freeze = fd.get("drone_freeze", 0.0)

            if self.task._prev_drone_depth is None:
                self.task._prev_drone_depth = noised.clone()

            if p_blank > 0.0:
                blank_mask = (torch.rand(noised.shape[0], device=noised.device) < p_blank).view(
                    -1, 1, 1
                )
                noised = torch.where(blank_mask, torch.ones_like(noised), noised)
            if p_freeze > 0.0:
                freeze_mask = (torch.rand(noised.shape[0], device=noised.device) < p_freeze).view(
                    -1, 1, 1
                )
                apply_freeze = freeze_mask if p_blank == 0.0 else (freeze_mask & (~blank_mask))
                noised = torch.where(apply_freeze, self.task._prev_drone_depth, noised)
            self.task._prev_drone_depth = noised.clone()
        else:
            self.task._prev_drone_depth = noised.clone()

        return noised

    def _encode_drone_vae(self, noised_image_obs: torch.Tensor) -> None:
        """Encode drone depth image through VAE to get latent representation."""
        try:
            img = noised_image_obs.to(self.task.device, dtype=torch.float32).contiguous()
            if img.dim() == 2:
                img = img.unsqueeze(0).unsqueeze(0).expand(self.task.sim_env.num_envs, -1, -1)
            elif img.dim() == 3:
                img = img.unsqueeze(1)
            self.task.image_latents[:] = self.task.shared_vae_model.encode(img)
        except RuntimeError as e:
            logger.warning(f"VAE encoding of drone camera failed: {e}")
            self.task.image_latents.zero_()

    def process_static_camera_observation(self) -> None:
        """Capture static camera depth, apply curriculum noise/dropout, and VAE encode."""
        try:
            static_depth, static_seg = self.task.static_camera_manager.capture_images(batched=True)

            if static_depth is not None and self.task.task_config.vae_config.use_vae:
                self._process_static_depth(static_depth, static_seg)
            else:
                if not self.task._no_static_logged:
                    self.task._no_static_logged = True
                    if static_depth is None:
                        logger.warning("Static camera data is None - capture failed")
                    elif not self.task.task_config.vae_config.use_vae:
                        logger.warning("VAE disabled - static camera latents will be zeros")
                self.task.static_image_latents.fill_(0.0)

        except RuntimeError as e:
            logger.error(f"Static camera processing error: {e}")
            self.task.static_image_latents.fill_(0.0)

    def _process_static_depth(
        self, static_depth: torch.Tensor | np.ndarray, static_seg: object
    ) -> None:
        """Apply noise, frame dropout, and VAE encoding to static camera depth."""
        if isinstance(static_depth, np.ndarray):
            clean = static_depth.copy()
        else:
            clean = static_depth.clone()

        noised = self._apply_static_camera_noise(clean)
        noised = self._apply_static_frame_dropout(noised)

        self.task.obs_dict["static_depth_clean"] = clean[0]
        self.task.obs_dict["static_depth_noised"] = noised[0] if noised.ndim == 3 else noised
        self.task.obs_dict["static_seg"] = static_seg

        self._encode_static_vae(noised)

    def _apply_static_camera_noise(
        self, depth: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Apply Gaussian noise and pixel dropout to static camera depth."""
        if isinstance(depth, np.ndarray):
            noised = depth.copy()
        else:
            noised = depth.clone()

        if not self.task.task_config.curriculum.enable_camera_noise:
            return noised

        static_noise_override = bool(
            self.task.sim_env.global_tensor_dict.get(
                "camera_randomization/static_noise_disabled", False
            )
        )
        global_noise_disabled = bool(
            self.task.sim_env.global_tensor_dict.get("camera_randomization/noise_disabled", False)
        )

        if not global_noise_disabled and not static_noise_override:
            gaussian_std, dropout_rate = self.task.task_config.curriculum.get_camera_noise(
                self.task.curriculum_level
            )
        else:
            gaussian_std, dropout_rate = self.task.task_config.curriculum.get_camera_noise(3)

        if isinstance(noised, np.ndarray):
            if gaussian_std > 0:
                noised = noised + np.random.normal(0.0, gaussian_std, size=noised.shape)
            if dropout_rate > 0:
                mask = np.random.rand(*noised.shape) < dropout_rate
                noised[mask] = 1.0
            return np.clip(noised, 0.0, 1.0)
        else:
            if gaussian_std > 0:
                noised = noised + torch.randn_like(noised) * gaussian_std
            if dropout_rate > 0:
                mask = torch.rand_like(noised) < dropout_rate
                noised = noised.masked_fill(mask, 1.0)
            return torch.clamp(noised, 0.0, 1.0)

    def _apply_static_frame_dropout(
        self, noised: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Apply entire-frame blank/freeze dropout to static camera depth."""
        if not self.task.task_config.curriculum.enable_camera_frame_dropout:
            return noised

        static_fd_override = bool(
            self.task.sim_env.global_tensor_dict.get(
                "camera_randomization/static_frame_dropout_disabled", False
            )
        )
        global_fd_disabled = bool(
            self.task.sim_env.global_tensor_dict.get(
                "camera_randomization/frame_dropout_disabled", False
            )
        )

        if not global_fd_disabled and not static_fd_override:
            fd = self.task.task_config.curriculum.get_camera_frame_dropout(
                self.task.curriculum_level
            )
        else:
            fd = self.task.task_config.curriculum.get_camera_frame_dropout(3)
        p_blank = fd.get("static_blank", 0.0)
        p_freeze = fd.get("static_freeze", 0.0)

        if self.task._prev_static_depth is None:
            if isinstance(noised, np.ndarray):
                self.task._prev_static_depth = noised.copy()
            else:
                self.task._prev_static_depth = noised.clone()

        if isinstance(noised, np.ndarray):
            if p_blank > 0.0 and np.random.rand() < p_blank:
                noised[...] = 1.0
            elif p_freeze > 0.0 and np.random.rand() < p_freeze:
                noised = self.task._prev_static_depth.copy()
            self.task._prev_static_depth = noised.copy()
        else:
            do_blank = torch.rand(1, device=noised.device).item() < p_blank
            if do_blank:
                noised = torch.ones_like(noised)
            else:
                do_freeze = torch.rand(1, device=noised.device).item() < p_freeze
                if do_freeze:
                    noised = self.task._prev_static_depth.clone()
            self.task._prev_static_depth = noised.clone()

        return noised

    def _encode_static_vae(self, noised: torch.Tensor | np.ndarray) -> None:
        """Convert static camera depth to tensor and encode through shared VAE."""
        try:
            if isinstance(noised, np.ndarray):
                depth_tensor = torch.from_numpy(noised).float().to(self.task.device)
            else:
                depth_tensor = noised

            if depth_tensor.dim() == 2:
                depth_tensor = depth_tensor.unsqueeze(0).expand(self.task.sim_env.num_envs, -1, -1)
            elif depth_tensor.dim() == 3 and depth_tensor.shape[0] != self.task.sim_env.num_envs:
                n = depth_tensor.shape[0]
                if n < self.task.sim_env.num_envs:
                    reps = (self.task.sim_env.num_envs + n - 1) // n
                    depth_tensor = depth_tensor.repeat(reps, 1, 1)[: self.task.sim_env.num_envs]
                else:
                    depth_tensor = depth_tensor[: self.task.sim_env.num_envs]

            self.task.static_image_latents[:] = self.task.shared_vae_model.encode(depth_tensor)
        except RuntimeError as e:
            logger.warning(f"VAE encoding of static camera failed: {e}")
