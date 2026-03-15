from __future__ import annotations

import os
import torch
import numpy as np

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import *  # noqa: F401,F403

logger = CustomLogger("navigation_task_gate_camera")


class CameraObservations:
    def __init__(self, task: object) -> None:
        self.task = task

    def process_image_observation(self) -> None:
        """Process drone camera observations with D455 curriculum-dependent noise."""
        # Get the drone's depth image (normalized 0.0–1.0)
        image_obs = self.task.obs_dict["depth_range_pixels"].squeeze(1)  # shape: (num_envs, H, W)
        # DEBUG: Compare per-env drone camera images to ensure diversity
        if (not self.task._drone_cam_debug_last) or (self.task.num_task_steps % 200 == 0):
            ne = int(image_obs.shape[0])
            def _mean_env(idx) -> None:
                return float(image_obs[idx].mean().item()) if idx < ne else float('nan')
            def _same(idx) -> None:
                return (idx < ne) and bool(torch.allclose(image_obs[0], image_obs[idx]))
            envs_to_check = [5]  # reduced debug output: only env5
            means = {i: _mean_env(i) for i in envs_to_check}
            sames = {i: _same(i) for i in [1, 5, 8, 12]}
            self.task._drone_cam_debug_last = self.task.num_task_steps

        # Apply D455 camera noise if enabled and not ablated
        noised_image_obs = image_obs.clone()  # Start with clean image
        camera_noise_disabled = False
        try:
            camera_noise_disabled = bool(self.task.sim_env.global_tensor_dict.get('camera_randomization/noise_disabled', False))
        except (KeyError, TypeError):
            camera_noise_disabled = bool(self.task.disable_camera_noise_randomization)
        # Per-camera override: if set, apply to drone camera processing
        drone_noise_override = bool(self.task.sim_env.global_tensor_dict.get('camera_randomization/drone_noise_disabled', False)) if hasattr(self.task.sim_env, 'global_tensor_dict') else False
        if self.task.task_config.curriculum.enable_camera_noise:
            # Use current level when enabled; otherwise force minimum schedule (level 3)
            if not camera_noise_disabled and not drone_noise_override:
                gaussian_std, dropout_rate = self.task.task_config.curriculum.get_camera_noise(self.task.curriculum_level)
            else:
                gaussian_std, dropout_rate = self.task.task_config.curriculum.get_camera_noise(3)

            # Gaussian noise: add N(0, gaussian_std) to each pixel (depth measurement uncertainty)
            if gaussian_std > 0:
                noise = torch.randn_like(noised_image_obs) * gaussian_std
                noised_image_obs = noised_image_obs + noise

            # Pixel dropout: set a fraction of pixels to 1.0 (missing depth readings)
            if dropout_rate > 0:
                dropout_mask = torch.rand_like(noised_image_obs) < dropout_rate
                noised_image_obs = noised_image_obs.masked_fill(dropout_mask, 1.0)  # 1.0 = max depth (no reading)

            # Clamp values to valid range [0, 1]
            noised_image_obs = torch.clamp(noised_image_obs, 0.0, 1.0)

        # Entire-frame dropout (curriculum-driven)
        frame_dropout_disabled = False
        try:
            frame_dropout_disabled = bool(self.task.sim_env.global_tensor_dict.get('camera_randomization/frame_dropout_disabled', False))
        except (KeyError, TypeError):
            frame_dropout_disabled = bool(self.task.disable_camera_frame_dropout_randomization)
        drone_fd_override = bool(self.task.sim_env.global_tensor_dict.get('camera_randomization/drone_frame_dropout_disabled', False)) if hasattr(self.task.sim_env, 'global_tensor_dict') else False
        if self.task.task_config.curriculum.enable_camera_frame_dropout:
            # Use current level unless frame-dropout is disabled; noise flag should not affect frame dropout
            if not frame_dropout_disabled and not drone_fd_override:
                fd = self.task.task_config.curriculum.get_camera_frame_dropout(self.task.curriculum_level)
            else:
                fd = self.task.task_config.curriculum.get_camera_frame_dropout(3)
            p_blank = fd.get("drone_blank", 0.0)
            p_freeze = fd.get("drone_freeze", 0.0)

            # Ensure buffer exists
            if not self.task._prev_drone_depth is not None:
                self.task._prev_drone_depth = noised_image_obs.clone()

            # Apply frame dropout effects
            if p_blank > 0.0:
                blank_mask = (torch.rand(noised_image_obs.shape[0], device=noised_image_obs.device) < p_blank).view(-1, 1, 1)
                noised_image_obs = torch.where(blank_mask, torch.ones_like(noised_image_obs), noised_image_obs)
            if p_freeze > 0.0:
                freeze_mask = (torch.rand(noised_image_obs.shape[0], device=noised_image_obs.device) < p_freeze).view(-1, 1, 1)
                # Apply freeze only where not already blanked
                apply_freeze = freeze_mask if p_blank == 0.0 else (freeze_mask & (~blank_mask))
                noised_image_obs = torch.where(apply_freeze, self.task._prev_drone_depth, noised_image_obs)
            # Update previous buffer after potential dropout
            self.task._prev_drone_depth = noised_image_obs.clone()
        else:
            # Maintain previous buffer if feature disabled
            if not self.task._prev_drone_depth is not None:
                self.task._prev_drone_depth = noised_image_obs.clone()
            else:
                self.task._prev_drone_depth = noised_image_obs.clone()

        # Store noised drone camera image for GIF generation (add channel dimension back)
        self.task.obs_dict["depth_range_pixels_noised"] = noised_image_obs.unsqueeze(1)  # shape: (num_envs, 1, H, W)

        # Encode the (potentially noisy) image using VAE
        if self.task.task_config.vae_config.use_vae:
            # Ensure tensor is float32, contiguous, on correct device, with channel dim
            try:
                img = noised_image_obs
                if isinstance(img, torch.Tensor):
                    img = img.to(self.task.device, dtype=torch.float32).contiguous()
                    if img.dim() == 2:
                        img = img.unsqueeze(0).unsqueeze(0).expand(self.task.sim_env.num_envs, -1, -1)
                    elif img.dim() == 3:
                        img = img.unsqueeze(1)
                else:
                    # Fallback: convert numpy to tensor
                    if isinstance(img, np.ndarray):
                        img = torch.from_numpy(img).to(self.task.device, dtype=torch.float32).contiguous()
                        if img.ndim == 2:
                            img = img.unsqueeze(0).unsqueeze(0).expand(self.task.sim_env.num_envs, -1, -1)
                        elif img.ndim == 3:
                            img = img.unsqueeze(1)
                self.task.image_latents[:] = self.task.shared_vae_model.encode(img)
            except RuntimeError as e:
                logger.warning(f"VAE encoding of drone camera failed: {e}")
                self.task.image_latents.zero_()
            # DEBUG: Compare per-env drone VAE latents
            if (not self.task._drone_vae_debug_last) or (self.task.num_task_steps % 200 == 0):
                z = self.task.image_latents
                ne = int(z.shape[0])
                def _absmean_env(idx) -> None:
                    return float(torch.mean(torch.abs(z[idx])).item()) if idx < ne else float('nan')
                def _same(idx) -> None:
                    return (idx < ne) and bool(torch.allclose(z[0], z[idx]))
                envs_to_check = [5]  # reduced debug output: only env5
                means = {i: _absmean_env(i) for i in envs_to_check}
                sames = {i: _same(i) for i in [1, 5, 8, 12]}
                self.task._drone_vae_debug_last = self.task.num_task_steps

    def process_static_camera_observation(self) -> None:
        """Process static camera observations with D455 curriculum-dependent noise."""
        try:
            # Request batched capture so each env gets its own image for VAE, while
            # GIF/debug paths will still use env0 via non-batched calls where needed
            static_depth, static_seg = self.task.static_camera_manager.capture_images(batched=True)

            # CRITICAL DEBUG: Log static camera capture success/failure
            if not self.task._static_debug_logged:
                self.task._static_debug_logged = True
                if static_depth is not None:
                    logger.warning(f"✅ Static camera capture successful: shape={static_depth.shape if hasattr(static_depth, 'shape') else 'N/A'}, type={type(static_depth)}")
                else:
                    logger.warning("❌ Static camera capture failed: static_depth is None")
            # Periodic per-env capture stats to confirm diversity
            if (not self.task._static_cam_debug_last) or (self.task.num_task_steps % 200 == 0):
                if hasattr(static_depth, 'shape') and getattr(static_depth, 'ndim', 0) == 3:
                    x = static_depth  # (N,H,W)
                    ne = int(x.shape[0])
                    def _mean_env(idx) -> None:
                        return float(x[idx].mean().item()) if idx < ne else float('nan')
                    def _same(idx) -> None:
                        return (idx < ne) and bool(np.allclose(x[0], x[idx])) if isinstance(x, np.ndarray) else bool(torch.allclose(x[0], x[idx]))
                    envs_to_check = [5]  # reduced debug output: only env5
                    means = {i: _mean_env(i) for i in envs_to_check}
                    # sames calculation omitted for brevity
                    self.task._static_cam_debug_last = self.task.num_task_steps

            if static_depth is not None and self.task.task_config.vae_config.use_vae:
                # Store clean static camera image (batched) and env0 view for GIF/debug
                if isinstance(static_depth, np.ndarray):
                    static_depth_clean_batched = static_depth.copy()
                    static_depth_clean_env0 = static_depth_clean_batched[0]
                else:
                    static_depth_clean_batched = static_depth.clone()
                    static_depth_clean_env0 = static_depth_clean_batched[0]

                # Apply D455 camera noise if enabled and not ablated (operate on batched copy)
                static_depth_noised = static_depth_clean_batched.copy() if isinstance(static_depth_clean_batched, np.ndarray) else static_depth_clean_batched.clone()
                static_noise_override = bool(self.task.sim_env.global_tensor_dict.get('camera_randomization/static_noise_disabled', False)) if hasattr(self.task.sim_env, 'global_tensor_dict') else False
                global_noise_disabled = bool(self.task.sim_env.global_tensor_dict.get('camera_randomization/noise_disabled', False)) if hasattr(self.task.sim_env, 'global_tensor_dict') else False
                if self.task.task_config.curriculum.enable_camera_noise:
                    # Current level unless disabled -> then use level 3 minimum
                    if not global_noise_disabled and not static_noise_override:
                        gaussian_std, dropout_rate = self.task.task_config.curriculum.get_camera_noise(self.task.curriculum_level)
                    else:
                        gaussian_std, dropout_rate = self.task.task_config.curriculum.get_camera_noise(3)

                    # Handle numpy array case
                    if isinstance(static_depth_noised, np.ndarray):
                        if gaussian_std > 0:
                            noise = np.random.normal(0.0, gaussian_std, size=static_depth_noised.shape)
                            static_depth_noised = static_depth_noised + noise
                        if dropout_rate > 0:
                            dropout_mask = np.random.rand(*static_depth_noised.shape) < dropout_rate
                            static_depth_noised[dropout_mask] = 1.0
                        static_depth_noised = np.clip(static_depth_noised, 0.0, 1.0)
                    else:
                        if gaussian_std > 0:
                            noise = torch.randn_like(static_depth_noised) * gaussian_std
                            static_depth_noised = static_depth_noised + noise
                        if dropout_rate > 0:
                            dropout_mask = torch.rand_like(static_depth_noised) < dropout_rate
                            static_depth_noised = static_depth_noised.masked_fill(dropout_mask, 1.0)
                        static_depth_noised = torch.clamp(static_depth_noised, 0.0, 1.0)

                # Entire-frame dropout (curriculum-driven)
                static_fd_override = bool(self.task.sim_env.global_tensor_dict.get('camera_randomization/static_frame_dropout_disabled', False)) if hasattr(self.task.sim_env, 'global_tensor_dict') else False
                global_fd_disabled = bool(self.task.sim_env.global_tensor_dict.get('camera_randomization/frame_dropout_disabled', False)) if hasattr(self.task.sim_env, 'global_tensor_dict') else False
                if self.task.task_config.curriculum.enable_camera_frame_dropout:
                    # Decouple from noise flag: only frame-dropout flags control this schedule
                    fd = self.task.task_config.curriculum.get_camera_frame_dropout(self.task.curriculum_level) if (not global_fd_disabled and not static_fd_override) else self.task.task_config.curriculum.get_camera_frame_dropout(3)
                    p_blank = fd.get("static_blank", 0.0)
                    p_freeze = fd.get("static_freeze", 0.0)
                    # Initialize previous static buffer
                    if not self.task._prev_static_depth is not None:
                        if isinstance(static_depth_noised, np.ndarray):
                            self.task._prev_static_depth = static_depth_noised.copy()
                        else:
                            self.task._prev_static_depth = static_depth_noised.clone()
                    # Apply blank then freeze
                    if isinstance(static_depth_noised, np.ndarray):
                        if p_blank > 0.0 and (np.random.rand() < p_blank):
                            static_depth_noised[...] = 1.0
                        elif p_freeze > 0.0 and (np.random.rand() < p_freeze):
                            static_depth_noised = self.task._prev_static_depth.copy()
                    else:
                        do_blank = (torch.rand(1, device=static_depth_noised.device).item() < p_blank)
                        if do_blank:
                            static_depth_noised = torch.ones_like(static_depth_noised)
                        else:
                            do_freeze = (torch.rand(1, device=static_depth_noised.device).item() < p_freeze)
                            if do_freeze:
                                static_depth_noised = self.task._prev_static_depth.clone()
                    # update buffer
                    if isinstance(static_depth_noised, np.ndarray):
                        self.task._prev_static_depth = static_depth_noised.copy()
                    else:
                        self.task._prev_static_depth = static_depth_noised.clone()

                # Store env0-only static camera images for GIF/debug (keep pipeline unchanged)
                self.task.obs_dict["static_depth_clean"] = static_depth_clean_env0
                # If numpy/tensor batched, select env0 for GIF/debug without altering VAE input
                self.task.obs_dict["static_depth_noised"] = static_depth_noised[0] if (hasattr(static_depth_noised, 'ndim') and (getattr(static_depth_noised, 'ndim') == 3)) else static_depth_noised
                self.task.obs_dict["static_seg"] = static_seg

                try:
                    # Convert to tensor and process through VAE (use noised version for training)
                    if isinstance(static_depth_noised, np.ndarray):
                        static_depth_tensor = torch.from_numpy(static_depth_noised).float().to(self.task.device)
                    else:
                        static_depth_tensor = static_depth_noised

                    # Ensure shape is (num_envs, H, W). If single image (H, W), broadcast to all envs
                    if static_depth_tensor.dim() == 2:
                        static_depth_tensor = static_depth_tensor.unsqueeze(0).expand(self.task.sim_env.num_envs, -1, -1)
                    elif static_depth_tensor.dim() == 3 and static_depth_tensor.shape[0] != self.task.sim_env.num_envs:
                        # Safe fallback: pad/trim to num_envs
                        n, h, w = static_depth_tensor.shape
                        if n < self.task.sim_env.num_envs:
                            reps = (self.task.sim_env.num_envs + n - 1) // n
                            static_depth_tensor = static_depth_tensor.repeat(reps, 1, 1)[:self.task.sim_env.num_envs]
                        else:
                            static_depth_tensor = static_depth_tensor[:self.task.sim_env.num_envs]

                    # Periodic static camera depth summary (match DroneCam style)
                    if (not self.task._static_cam_depth_logged) or (self.task.num_task_steps % 200 == 0):
                        self.task._static_cam_depth_logged = True
                        depth = static_depth_tensor
                        ne = int(depth.shape[0])
                        def _mean_env(idx) -> None:
                            return float(torch.mean(depth[idx]).item()) if idx < ne else float('nan')
                        def _same(idx) -> None:
                            return (idx < ne) and bool(torch.allclose(depth[0], depth[idx]))
                        envs_to_check = [0, 1, 5, 8, 12]
                        means = {i: _mean_env(i) for i in envs_to_check}
                        sames = {i: _same(i) for i in [1, 5, 8, 12]}

                    # CRITICAL DEBUG: Log VAE encoding attempt (once)
                    if not self.task._vae_debug_logged:
                        self.task._vae_debug_logged = True
                        logger.warning(f"🔧 VAE encoding static camera: input_shape={static_depth_tensor.shape}, device={static_depth_tensor.device}")

                    encoded_latents = self.task.shared_vae_model.encode(static_depth_tensor)
                    self.task.static_image_latents[:] = encoded_latents

                    # CRITICAL DEBUG: Verify VAE output periodically and compare across envs
                    if (not self.task._vae_output_logged) or (self.task.num_task_steps % 200 == 0):
                        self.task._vae_output_logged = True
                        z = encoded_latents
                        ne = int(z.shape[0])
                        def _absmean_env(idx) -> None:
                            return float(torch.mean(torch.abs(z[idx])).item()) if idx < ne else float('nan')
                        def _same(idx) -> None:
                            return (idx < ne) and bool(torch.allclose(z[0], z[idx]))
                        envs_to_check = [5]  # reduced debug output: only env5
                        means = {i: _absmean_env(i) for i in envs_to_check}
                except RuntimeError as e:
                    logger.warning(f"VAE encoding of static camera failed: {e}")
            else:
                # No static camera data or VAE disabled
                if not self.task._no_static_logged:
                    self.task._no_static_logged = True
                    if static_depth is None:
                        logger.warning("❌ Static camera data is None - camera capture failed")
                    elif not self.task.task_config.vae_config.use_vae:
                        logger.warning("❌ VAE disabled in config - static camera latents will be zeros")

                # Fill with zeros if no data
                self.task.static_image_latents.fill_(0.0)

        except RuntimeError as e:
            logger.error(f"❌ Static camera processing error: {e}")
            # Fallback to zeros on any error
            self.task.static_image_latents.fill_(0.0)

    def _compute_visibility_metrics(self, infos_to_return: dict[str, torch.Tensor]) -> None:
        """Compute geometric gate visibility and static FOV metrics (non-reward, for logging)."""
        # Geometric gate visibility metric (pose-only, no pixels)
        # Disabled by default; enable with SF_ENABLE_GEOM_VISIBILITY, static_visibility/enable, or VISIBILITY_DEBUG
        try:
            gtd = self.task.sim_env.global_tensor_dict
            _flag_env = _os.environ.get('SF_ENABLE_GEOM_VISIBILITY', '').strip().lower() in ('1', 'true', 'yes', 'y')
            _flag_gtd = bool(gtd.get('static_visibility/enable', False))
            _flag_dbg = _os.environ.get('VISIBILITY_DEBUG', '').strip().lower() in ('1', 'true', 'yes', 'y')
            if _flag_env or _flag_gtd or _flag_dbg:
                # Grid resolution (defaults 30x30)
                try:
                    N = int(_os.environ.get('SF_STATIC_VIS_N', gtd.get('static_visibility/N', 30)))
                except (ValueError, TypeError):
                    N = 30
                try:
                    M = int(_os.environ.get('SF_STATIC_VIS_M', gtd.get('static_visibility/M', 30)))
                except (ValueError, TypeError):
                    M = 30
                N = max(4, int(N)); M = max(4, int(M))

                # Camera pose per env from StaticCameraManager debug caches
                scm = self.task.static_camera_manager
                have_cam = (scm is not None and hasattr(scm, 'last_camera_pos') and hasattr(scm, 'last_camera_target')
                            and len(scm.last_camera_pos) >= self.task.num_envs and len(scm.last_camera_target) >= self.task.num_envs)
                if have_cam:
                    cam_pos = torch.tensor(scm.last_camera_pos, dtype=torch.float32, device=self.task.device)
                    cam_tgt = torch.tensor(scm.last_camera_target, dtype=torch.float32, device=self.task.device)
                else:
                    cam_pos = None
                # Fallback: synthesize camera pose from base_y/base_z and gate center if caches missing
                if cam_pos is not None:
                    # Camera basis (right, up, forward)
                    up_world = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.task.device).view(1, 3).expand(self.task.num_envs, 3)
                    fwd = cam_tgt - cam_pos
                    fwd = fwd / (torch.norm(fwd, dim=1, keepdim=True) + 1e-8)
                    right = torch.cross(fwd, up_world)
                    right = right / (torch.norm(right, dim=1, keepdim=True) + 1e-8)
                    up = torch.cross(right, fwd)

                    # Gate grid in world coordinates for each env
                    W = self.task.gate_width.view(self.task.num_envs, 1, 1)
                    H = self.task.gate_height.view(self.task.num_envs, 1, 1)
                    gx = self.task.gate_position[:, 0].view(self.task.num_envs, 1, 1)
                    gy = self.task.gate_position[:, 1].view(self.task.num_envs, 1, 1)
                    gz = self.task.gate_position[:, 2].view(self.task.num_envs, 1, 1)
                    # Grid coordinates (center-sampled)
                    xi = (torch.arange(N, device=self.task.device, dtype=torch.float32) + 0.5) / float(N) - 0.5  # [-0.5, 0.5]
                    zj = (torch.arange(M, device=self.task.device, dtype=torch.float32) + 0.5) / float(M)         # [0, 1]
                    # Broadcast-friendly grid centers
                    X = gx + (W * xi.view(1, N, 1))            # (E,N,1)
                    Z = gz + (H * zj.view(1, 1, M))            # (E,1,M)
                    Y = gy.view(self.task.num_envs, 1, 1)           # (E,1,1)
                    # Expand to (E,N,M)
                    X = X.expand(self.task.num_envs, N, M)
                    Z = Z.expand(self.task.num_envs, N, M)
                    Y = Y.expand(self.task.num_envs, N, M)
                    # Assemble world points (E,N,M,3)
                    cam_pos_ = cam_pos.view(self.task.num_envs, 1, 1, 3)
                    Pw = torch.stack([X, Y, Z], dim=3) - cam_pos_
                    # Project to camera basis
                    rx = right.view(self.task.num_envs, 1, 1, 3)
                    upv = up.view(self.task.num_envs, 1, 1, 3)
                    fv = fwd.view(self.task.num_envs, 1, 1, 3)
                    x_c = torch.sum(Pw * rx, dim=3)
                    y_c = torch.sum(Pw * upv, dim=3)
                    z_c = torch.sum(Pw * fv, dim=3)

                    # Frustum test using symmetric FOV (D455 ~87° horiz, use same for vert)
                    half_angle = _math.radians(87.0 * 0.5)
                    tan_half = _math.tan(half_angle)
                    near = 0.4; far = 20.0
                    z_ok = (z_c > near) & (z_c < far)
                    nx = torch.abs(x_c) / torch.clamp(z_c, min=1e-6)
                    ny = torch.abs(y_c) / torch.clamp(z_c, min=1e-6)
                    h_ok = nx <= tan_half
                    v_ok = ny <= tan_half
                    frustum_mask = z_ok & h_ok & v_ok

                    # Occlusion by drone (sphere test via closest-point on segment)
                    drone_pos = self.task.obs_dict.get('robot_position', None)
                    if isinstance(drone_pos, torch.Tensor) and drone_pos.shape[0] == self.task.num_envs:
                        cpos = cam_pos_
                        qpts = torch.stack([X, Y, Z], dim=3)
                        seg = qpts - cpos
                        vv = torch.sum(seg * seg, dim=3)  # |v|^2
                        w = drone_pos.view(self.task.num_envs, 1, 1, 3) - cpos
                        t = torch.sum(w * seg, dim=3) / torch.clamp(vv, min=1e-6)
                        t = torch.clamp(t, 0.0, 1.0).unsqueeze(3)
                        p_closest = cpos + t * seg
                        d2 = torch.sum((p_closest - drone_pos.view(self.task.num_envs, 1, 1, 3)) ** 2, dim=3)
                        try:
                            r = float(_os.environ.get('SF_DRONE_OCCLUSION_RADIUS_M', gtd.get('static_visibility/drone_radius_m', 0.25)))
                        except (ValueError, TypeError):
                            r = 0.25
                        occluded = d2 <= (r * r)
                    else:
                        occluded = torch.zeros_like(frustum_mask)

                    visible_mask = frustum_mask & (~occluded)
                    total_cells = float(N * M)
                    vis_frac = torch.sum(visible_mask, dim=(1, 2)).to(torch.float32) / total_cells
                    frustum_frac = torch.sum(frustum_mask, dim=(1, 2)).to(torch.float32) / total_cells
                    eff = torch.where(frustum_frac > 1e-6, vis_frac / torch.clamp(frustum_frac, min=1e-6), torch.zeros_like(frustum_frac))

                    # Export to infos for logging/aggregation
                    self.task.infos["static_visibility/abs"] = vis_frac.detach()
                    self.task.infos["static_visibility/frustum"] = frustum_frac.detach()
                    self.task.infos["static_visibility/eff"] = eff.detach()
                    # Ensure values also appear in infos_to_return (used by step() callers)
                    infos_to_return["static_visibility/abs"] = vis_frac.detach()
                    infos_to_return["static_visibility/frustum"] = frustum_frac.detach()
                    infos_to_return["static_visibility/eff"] = eff.detach()

                    # Optional env0 debug print (every 5 steps) when VISIBILITY_DEBUG or ABLATE_DEBUG
                    try:
                        debug_on = (
                            _os.environ.get('VISIBILITY_DEBUG', '').strip().lower() in ('1', 'true', 'yes', 'y') or
                            _os.environ.get('ABLATE_DEBUG', '').strip().lower() in ('1', 'true', 'yes', 'y')
                        )
                        step_i = int(self.task.num_task_steps)
                        # Use env0's current episode-relative step for display; clamp to be non-decreasing within episode
                        try:
                            step_ep0 = int(self.task.episode_lengths[0].item())
                        except (ValueError, TypeError):
                            step_ep0 = step_i
                        disp_step = step_ep0
                        if debug_on and self.task.num_envs > 0:
                            e0 = 0
                            # Episode index for env0 inferred from number of times it reset; approximate via completed episodes count for env0
                            try:
                                ep_idx0 = int(self.task._ep_count_env0.item())  # if maintained elsewhere
                            except (ValueError, TypeError):
                                # Fallback: count resets via episode_lengths reset to 0 transitions is non-trivial here; approximate using len(completed_episodes)
                                ep_idx0 = max(0, len(self.task.completed_episodes) - 1)
                            cx, cy, cz = float(cam_pos[e0, 0].item()), float(cam_pos[e0, 1].item()), float(cam_pos[e0, 2].item())
                            gx0 = float(self.task.gate_position[e0, 0].item()); gy0 = float(self.task.gate_position[e0, 1].item()); gz0 = float(self.task.gate_position[e0, 2].item())
                            w0 = float(self.task.gate_width[e0].item()); h0 = float(self.task.gate_height[e0].item())
                            vf = float(vis_frac[e0].item()); ff = float(frustum_frac[e0].item()); ef = float(eff[e0].item())
                    except (ValueError, TypeError):
                        pass
                else:
                    # No fallback; require real camera pose caches for visibility
                    pass
        except (ValueError, TypeError) as _e_vis:
            # Keep visibility diagnostics non-fatal
            try:
                if _os.environ.get('VISIBILITY_DEBUG', '').strip().lower() in ('1', 'true', 'yes', 'y') or _os.environ.get('ABLATE_DEBUG', '').strip().lower() in ('1', 'true', 'yes', 'y'):
                    logger.warning(f"[VIS] Geometric visibility computation skipped due to: {_e_vis}")
                else:
                    logger.debug(f"[VIS] Geometric visibility computation skipped: {_e_vis}")
            except (KeyError, TypeError):
                logger.debug(f"[VIS] Geometric visibility computation skipped: {_e_vis}")

        # Static FOV metric (non-reward): carry over the reward formula strictly as a metric
        # Does NOT modify rewards; only logs per-env metrics to infos, and optional env0 debug under VISIBILITY_DEBUG
        try:
            scm = self.task.static_camera_manager
            have_cam = (scm is not None and hasattr(scm, 'last_camera_pos') and hasattr(scm, 'last_camera_target')
                        and len(scm.last_camera_pos) >= self.task.num_envs and len(scm.last_camera_target) >= self.task.num_envs)
            if have_cam:
                cam_pos = torch.tensor(scm.last_camera_pos, dtype=torch.float32, device=self.task.device)
                cam_tgt = torch.tensor(scm.last_camera_target, dtype=torch.float32, device=self.task.device)
                up_world = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.task.device).view(1, 3).expand(self.task.num_envs, 3)
                fwd = cam_tgt - cam_pos
                fwd = fwd / (torch.norm(fwd, dim=1, keepdim=True) + 1e-8)
                right = torch.cross(fwd, up_world)
                right = right / (torch.norm(right, dim=1, keepdim=True) + 1e-8)
                up = torch.cross(right, fwd)
                # Vector from camera to drone (robot)
                robot_position = self.task.obs_dict.get('robot_position', torch.zeros((self.task.num_envs, 3), device=self.task.device))
                pw = robot_position - cam_pos
                x_c = torch.sum(pw * right, dim=1)
                y_c = torch.sum(pw * up, dim=1)
                z_c = torch.sum(pw * fwd, dim=1)
                half_fov_rad = _math.radians(87.0 * 0.5)
                horiz_angle = torch.atan2(torch.abs(x_c), torch.clamp(z_c, min=1e-6))
                vert_angle = torch.atan2(torch.abs(y_c), torch.clamp(z_c, min=1e-6))
                visible = (z_c > 0.1) & (horiz_angle <= half_fov_rad) & (vert_angle <= half_fov_rad)
                h_norm = torch.clamp(horiz_angle / half_fov_rad, 0.0, 1.0)
                v_norm = torch.clamp(vert_angle / half_fov_rad, 0.0, 1.0)
                m_norm = torch.maximum(h_norm, v_norm)
                try:
                    fov_alpha = float(self.task.task_config.reward_parameters.get("static_fov_visibility_exponent", 2.0))
                except (ValueError, TypeError):
                    fov_alpha = 2.0
                fov_score = torch.pow(torch.clamp(1.0 - m_norm, min=0.0), fov_alpha)
                # Export purely as metrics (no reward changes)
                self.task.infos["static_fov/visible"] = visible.float()
                self.task.infos["static_fov/horiz_angle_rad"] = horiz_angle
                self.task.infos["static_fov/vert_angle_rad"] = vert_angle
                self.task.infos["static_fov/score"] = fov_score
                # Mirror into infos_to_return so inference sees them
                infos_to_return["static_fov/visible"] = visible.float()
                infos_to_return["static_fov/horiz_angle_rad"] = horiz_angle
                infos_to_return["static_fov/vert_angle_rad"] = vert_angle
                infos_to_return["static_fov/score"] = fov_score
                # Optional env0 debug
                if _os.environ.get('VISIBILITY_DEBUG', '').strip().lower() in ('1', 'true', 'yes', 'y') and self.task.num_envs > 0:
                    e0 = 0
                    step_ep0 = int(self.task.episode_lengths[0].item())
        except (ValueError, TypeError):
            pass

    def _get_static_camera_pose_relative_to_drone(self) -> torch.Tensor:
        """Compute per-environment static camera pose and orientation relative to the drone.

        Position: camera_world - robot_world, rotated into drone/body frame (obs[3:6]).
        Orientation: Euler XYZ of q_rel = q_drone^-1 ⊗ q_cam (obs[6:9]).
        Camera world pose reflects either static base placement (with yaw sweep/randomization)
        or dynamic following when enabled.
        """
        device = self.task.device
        num_envs = self.task.num_envs

        # Base Y from task_config first, then env var, else default
        try:
            base_y = float(self.task.task_config.static_camera_base_y)
        except (ValueError, TypeError):
            base_y = -3.0

        # Base Z can be numeric or 'adaptive'
        adaptive_z = False
        base_z_value = 1.5
        cfg_base_z = self.task.task_config.static_camera_base_z
        if cfg_base_z is not None:
            if isinstance(cfg_base_z, str) and cfg_base_z.strip().lower() == 'adaptive':
                adaptive_z = True
            else:
                base_z_value = float(cfg_base_z)
        else:
            env_base_z = os.environ.get('SF_STATIC_CAMERA_BASE_Z', None)
            if env_base_z is not None and str(env_base_z).strip().lower() == 'adaptive':
                adaptive_z = True
            elif env_base_z is not None:
                base_z_value = float(env_base_z)

        # Resolve per-env Z when adaptive
        if adaptive_z:
            try:
                gate_center_z = self.task.gate_center_height
                if gate_center_z is None:
                    gate_center_z = torch.full((num_envs,), 1.5, device=device, dtype=torch.float32)
                elif not torch.is_tensor(gate_center_z):
                    gate_center_z = torch.full((num_envs,), float(gate_center_z), device=device, dtype=torch.float32)
                else:
                    gate_center_z = gate_center_z.to(device=device, dtype=torch.float32).view(-1)
            except (ValueError, TypeError):
                gate_center_z = torch.full((num_envs,), 1.5, device=device, dtype=torch.float32)
        else:
            gate_center_z = torch.full((num_envs,), float(base_z_value), device=device, dtype=torch.float32)

        # Determine if dynamic camera following is effective (enabled and not disabled)
        dynamic_enabled = bool(self.task.task_config.curriculum.enable_dynamic_camera_following)
        try:
            dyn_dis = bool(self.task.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False)) if hasattr(self.task.sim_env, 'global_tensor_dict') else False
        except (KeyError, TypeError):
            dyn_dis = False
        # Arc-follow takes precedence if enabled
        try:
            arc_follow_enabled = bool(self.task.sim_env.global_tensor_dict.get('static_camera/arc_follow_enabled', False))
        except (KeyError, TypeError):
            arc_follow_enabled = False
        dynamic_effective = bool((dynamic_enabled and not dyn_dis) and not arc_follow_enabled)

        # Camera world positions for each env
        cam_world = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        if dynamic_effective:
            # Follow the drone with fixed offset
            try:
                from aerial_gym.config.task_config.navigation_task_config_gate import task_config as _tc
                x_off, y_off, z_off = _tc.curriculum.get_dynamic_camera_follow_offset()
            except ImportError:
                x_off, y_off, z_off = 0.0, -1.0, 0.0
            try:
                robot_pos_world = self.task.obs_dict['robot_position'].to(device=device, dtype=torch.float32)
            except (KeyError, TypeError):
                robot_pos_world = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
            cam_world[:, 0] = robot_pos_world[:, 0] + float(x_off)
            cam_world[:, 1] = robot_pos_world[:, 1] + float(y_off)
            cam_world[:, 2] = robot_pos_world[:, 2] + float(z_off)
        else:
            # Static base placement (x=0, y=base_y, z=gate_center or fixed)
            cam_world[:, 0] = 0.0
            cam_world[:, 1] = float(base_y)
            cam_world[:, 2] = gate_center_z

        # Robot world pose tensors
        try:
            robot_pos = self.task.obs_dict['robot_position'].to(device=device, dtype=torch.float32)
        except (KeyError, TypeError):
            robot_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        # Prefer vehicle (yaw-only) orientation for stable horizontal frame; fallback to body
        q = None
        try:
            q = self.task.obs_dict.get('robot_vehicle_orientation', None)
        except (KeyError, TypeError):
            q = None
        if q is None:
            try:
                q = self.task.obs_dict.get('robot_orientation', None)
            except (KeyError, TypeError):
                q = None
        if q is None:
            q = torch.zeros((num_envs, 4), device=device, dtype=torch.float32)
            q[:, 3] = 1.0
        else:
            q = q.to(device=device, dtype=torch.float32)

        # Relative position in world then rotate into the drone/body frame
        rel_world = cam_world - robot_pos
        rel_pos_body = quat_rotate_inverse(q, rel_world)

        # Compute camera world orientation as look-at towards a per-env target
        # Target is the adaptive gate center when static base or dynamic following
        gate_pos_world = self.task.gate_position
        gcz = self.task.gate_center_height
        target_world = gate_pos_world.clone()
        target_world[:, 2] = gate_pos_world[:, 2] + gcz

        # When static base+ yaw sweep/randomization is active, adjust target X/Y using current yaw offset if available
        if not dynamic_effective:
            try:
                scm = self.task.static_camera_manager
                have_angles = (scm is not None) and hasattr(scm, 'current_camera_angles') and (len(scm.current_camera_angles) >= num_envs)
            except Exception:
                have_angles = False
            if have_angles:
                # Build per-env target consistent with update_camera_positions
                target_distance = torch.abs(cam_world[:, 1])  # |base_y|
                angles_deg = torch.tensor(scm.current_camera_angles[:num_envs], dtype=torch.float32, device=device)
                ang = angles_deg * (3.141592653589793 / 180.0)
                target_world[:, 0] = cam_world[:, 0] + target_distance * torch.sin(ang)
                target_world[:, 1] = cam_world[:, 1] + target_distance * torch.cos(ang)

        # Camera forward vector and Euler in world
        fwd = target_world - cam_world
        fwd = fwd / (torch.norm(fwd, dim=1, keepdim=True) + 1e-8)
        # Yaw: angle in XY toward +Y; Pitch: elevation toward +Z; Roll=0 (world up maintained)
        fx, fy, fz = fwd[:, 0], fwd[:, 1], fwd[:, 2]
        yaw_cam = torch.atan2(fx, torch.clamp(fy, min=1e-8))
        hyp = torch.sqrt(torch.clamp(fx * fx + fy * fy, min=1e-8))
        pitch_cam = torch.atan2(fz, hyp)
        roll_cam = torch.zeros_like(yaw_cam)
        eul_cam = torch.stack([roll_cam, pitch_cam, yaw_cam], dim=1)
        q_cam = quat_from_euler_xyz_tensor(eul_cam)

        # Relative orientation q_rel = q_drone^-1 ⊗ q_cam; then Euler in drone/body frame
        q_drone = q
        q_drone_conj = torch.stack([-q_drone[:, 0], -q_drone[:, 1], -q_drone[:, 2], q_drone[:, 3]], dim=1)
        q_rel = quat_mul(q_drone_conj, q_cam)
        rel_orient_euler = ssa(get_euler_xyz_tensor(q_rel))
        drone_eul_world = ssa(get_euler_xyz_tensor(q_drone))
        # Stash debug state for printing after obs assembly
        self.task._debug_cam_world = cam_world.detach().clone()
        self.task._debug_rel_pos = rel_pos_body.detach().clone()
        self.task._debug_rel_eul = rel_orient_euler.detach().clone()
        self.task._debug_cam_eul = eul_cam.detach().clone()
        self.task._debug_drone_eul = drone_eul_world.detach().clone()
        return rel_pos_body, rel_orient_euler


