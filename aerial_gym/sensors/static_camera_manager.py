from __future__ import annotations

import math
import numpy as np
import torch

from aerial_gym.task.task_config_protocol import TaskConfig
from aerial_gym.env_manager.env_manager import EnvManager
from isaacgym import gymapi

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("static_camera_manager")


class StaticCameraManager:
    """Manages static camera for gate navigation using Isaac Gym native API."""
    
    def __init__(self, env_manager: EnvManager, task_config: TaskConfig) -> None:
        self.env_manager = env_manager
        self.task_config = task_config
        self.gym = env_manager.IGE_env.gym
        self.sim = env_manager.IGE_env.sim
        self.env_handles = env_manager.IGE_env.env_handles
        self.camera_handles = []
        self.camera_setup_success = False
        self.use_synthetic_camera = False  # Initialize synthetic camera flag
        # Ensure device exists for tensor ops (fix jitter sampling fallback)
        try:
            self.device = getattr(env_manager, 'device', None)
            if self.device is None:
                self.device = getattr(task_config, 'device', 'cpu')
        except Exception:
            self.device = 'cpu'
        
        # Gate position (will be updated dynamically based on actual gate positions)
        self.gate_position = [0.0, 0.0, 0.0]  # Default, overridden by adaptive positioning
        self.env_bounds = [[-4.0, -4.0, 0.0], [4.0, 4.0, 4.0]]  # Updated for gate_env bounds
        
        # Per-environment camera angle tracking - FIXED during each episode
        self.num_envs = len(self.env_handles)
        self.current_camera_angles = [0.0] * self.num_envs  # Track angle per environment
        # Debug tracking of last camera placement and target per env
        self.last_camera_pos = [(0.0, -3.0, 1.5) for _ in range(self.num_envs)]
        self.last_camera_target = [(0.0, 0.0, 1.5) for _ in range(self.num_envs)]
        self.last_angle_deg = [0.0 for _ in range(self.num_envs)]
        # Per-env placement randomization (translation + small Euler jitter)
        # Defaults are no jitter; can be enabled/configured via task_config
        self.static_cam_randomize = bool(getattr(task_config, 'static_camera_randomize_placement', False))
        self.static_cam_min_t = list(getattr(task_config, 'static_camera_min_translation', [0.0, 0.0, 0.0]))
        self.static_cam_max_t = list(getattr(task_config, 'static_camera_max_translation', [0.0, 0.0, 0.0]))
        self.static_cam_min_euler = list(getattr(task_config, 'static_camera_min_euler_deg', [0.0, 0.0, 0.0]))
        self.static_cam_max_euler = list(getattr(task_config, 'static_camera_max_euler_deg', [0.0, 0.0, 0.0]))
        # Storage for per-env jitter, refreshed on setup and per-episode resets
        self._trans_jitter = [(0.0, 0.0, 0.0) for _ in range(self.num_envs)]
        self._euler_jitter_deg = [(0.0, 0.0, 0.0) for _ in range(self.num_envs)]
        
        self._setup_static_camera()
    
    def get_average_camera_angle(self) -> float:
        """Get average camera angle across all environments for logging."""
        if not hasattr(self, 'current_camera_angles') or not self.current_camera_angles:
            return 0.0
        return sum(self.current_camera_angles) / len(self.current_camera_angles)
    
    def _setup_static_camera(self) -> None:
        """Setup static camera using Isaac Gym native camera API with D455 specifications."""
        logger.info("Setting up static camera for gate navigation...")
        
        # Headless mode is supported for camera sensors; proceed with GPU camera setup
        if self.task_config.headless:
            logger.info("Headless mode detected - proceeding with static camera GPU setup")
        
        try:
            # Camera properties (D455 depth camera specifications - match working example)
            camera_props = gymapi.CameraProperties()
            camera_props.width = 240  # Reduced depth resolution width
            camera_props.height = 135  # Reduced depth resolution height
            camera_props.horizontal_fov = 87.0  # D455 FOV
            camera_props.near_plane = 0.4  # D455 minimum depth distance
            camera_props.far_plane = 20.0  # D455 maximum range
            camera_props.enable_tensors = True  # Enable GPU tensor access
            
            logger.info(f"Static camera properties (D455 specs): {camera_props.width}x{camera_props.height}, FOV: {camera_props.horizontal_fov}°")
            logger.info(f"Static camera depth range: {camera_props.near_plane}m - {camera_props.far_plane}m")
        
            # Create camera sensor in each environment
            self.camera_handles = []
            for i, env_handle in enumerate(self.env_handles):
                cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                if cam_handle >= 0:  # Valid camera handle
                    self.camera_handles.append(cam_handle)
                    logger.info(f"Created static camera sensor {i} in environment {i}")
                else:
                    logger.warning(f"Failed to create camera for environment {i}, handle: {cam_handle}")
                    # Do not fall back to synthetic or return; proceed and mark setup incomplete
            
            # Respect CLI/task_config base_y/base_z (with 'adaptive' support) for initial placement
            try:
                base_y = float(getattr(self.task_config, 'static_camera_base_y', float(os.environ.get('SF_STATIC_CAMERA_BASE_Y', -3.0))))
            except Exception:
                base_y = -3.0
            # Determine base Z spawning mode: numeric or 'adaptive' to gate center per env
            adaptive_z = False
            base_z_value = 1.5
            try:
                cfg_base_z = getattr(self.task_config, 'static_camera_base_z', None)
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
            except Exception:
                pass

            # Try to read per-env adaptive gate center Z from global tensors (if available)
            try:
                gtd = self.env_manager.IGE_env.global_tensor_dict
                gate_center_per_env = gtd.get('gate/center_height_per_env', None)
            except Exception:
                gate_center_per_env = None

            # Set camera transform for each environment using configured positioning
            for i, (env_handle, cam_handle) in enumerate(zip(self.env_handles, self.camera_handles)):
                # Initialize per-env jitter for the first episode
                if self.static_cam_randomize:
                    try:
                        # Sample translation jitter uniformly between per-axis bounds
                        u = torch.rand(3, device=self.device)
                        tmin = torch.tensor(self.static_cam_min_t, device=self.device, dtype=torch.float32)
                        tmax = torch.tensor(self.static_cam_max_t, device=self.device, dtype=torch.float32)
                        t = (tmin + u * (tmax - tmin)).tolist()
                        # Sample Euler jitter (roll, pitch, yaw) in degrees
                        v = torch.rand(3, device=self.device)
                        emin = torch.tensor(self.static_cam_min_euler, device=self.device, dtype=torch.float32)
                        emax = torch.tensor(self.static_cam_max_euler, device=self.device, dtype=torch.float32)
                        e = (emin + v * (emax - emin)).tolist()
                        self._trans_jitter[i] = (float(t[0]), float(t[1]), float(t[2]))
                        self._euler_jitter_deg[i] = (float(e[0]), float(e[1]), float(e[2]))
                    except Exception:
                        self._trans_jitter[i] = (0.0, 0.0, 0.0)
                        self._euler_jitter_deg[i] = (0.0, 0.0, 0.0)
                if adaptive_z and gate_center_per_env is not None and i < len(gate_center_per_env):
                    env_base_z = float(gate_center_per_env[i].item())
                else:
                    env_base_z = float(base_z_value)
                # Apply translation jitter
                jx, jy, jz = self._trans_jitter[i]
                camera_pos = gymapi.Vec3(0.0 + jx, float(base_y) + jy, env_base_z + jz)
                # Look directly at the gate center height for that env
                camera_target = gymapi.Vec3(0.0, 0.0, env_base_z)
                self.gym.set_camera_location(cam_handle, env_handle, camera_pos, camera_target)
                logger.info(
                    f"Set static camera {i} to look from ({camera_pos.x}, {camera_pos.y}, {camera_pos.z}) toward ({camera_target.x}, {camera_target.y}, {camera_target.z})"
                )
                # Store for debug
                try:
                    self.last_camera_pos[i] = (float(camera_pos.x), float(camera_pos.y), float(camera_pos.z))
                    self.last_camera_target[i] = (float(camera_target.x), float(camera_target.y), float(camera_target.z))
                    self.last_angle_deg[i] = 0.0
                except Exception:
                    pass

            # Cache base values into global dict for downstream updates
            try:
                if hasattr(self.env_manager.IGE_env, 'global_tensor_dict'):
                    self.env_manager.IGE_env.global_tensor_dict['static_camera/base_y'] = float(base_y)
                    self.env_manager.IGE_env.global_tensor_dict['static_camera/base_z'] = ('adaptive' if adaptive_z else float(base_z_value))
            except Exception:
                pass

            if len(self.camera_handles) == len(self.env_handles):
                logger.info("✓ Static camera setup complete with configured positioning (base_y/base_z)")
                self.camera_setup_success = True
                self.use_synthetic_camera = False
            else:
                logger.error(f"Static camera setup incomplete: created {len(self.camera_handles)}/{len(self.env_handles)} cameras")
                self.camera_setup_success = False
                self.use_synthetic_camera = False
            
        except Exception as e:
            logger.warning(f"Static camera setup failed: {e}")
            self.camera_setup_success = False
            self.use_synthetic_camera = False
    
    def update_camera_positions(self, curriculum_level: int, env_ids: torch.Tensor) -> None:
        """Update static camera orientation ONLY for resetting environments."""
        if hasattr(self, 'use_synthetic_camera') and self.use_synthetic_camera:
            # In synthetic mode, update orientation per resetting env with spawn-aware logic
            from aerial_gym.config.task_config.navigation_task_config_gate import task_config
            max_angle_range, _, _ = task_config.curriculum.get_static_camera_difficulty(curriculum_level)
            # Read flags and robot positions
            try:
                parent = getattr(self, 'env_manager', None)
                disable_flag = False
                rp = None
                if parent is not None and hasattr(parent, 'global_tensor_dict'):
                    gtd = parent.global_tensor_dict
                    disable_flag = bool(gtd.get('static_camera_randomization/orientation_disabled', False))
                    rp = gtd.get('robot_position', None)
            except Exception:
                disable_flag = False
                rp = None
            import random, math
            horizontal_fov = 87.0
            half_fov = horizontal_fov * 0.5
            margin = 5.0
            for env_idx in env_ids:
                if env_idx < len(self.current_camera_angles):
                    if disable_flag or max_angle_range <= 0:
                        ang = 0.0
                    else:
                        # Camera at (0,-3) looks toward +Y (0°). Keep both gate (0°) and drone in FOV.
                        if rp is not None and env_idx < rp.shape[0]:
                            cam_x, cam_y = 0.0, -3.0
                            dx = float(rp[env_idx, 0].item()) - cam_x
                            dy = float(rp[env_idx, 1].item()) - cam_y
                            theta_r = math.degrees(math.atan2(dx, dy))
                            gate_low, gate_high = -half_fov + margin, half_fov - margin
                            rob_low, rob_high = theta_r - (half_fov - margin), theta_r + (half_fov - margin)
                            low = max(gate_low, rob_low, -max_angle_range)
                            high = min(gate_high, rob_high, max_angle_range)
                            if high > low:
                                ang = random.uniform(low, high)
                            else:
                                target = max(min(theta_r, gate_high), gate_low)
                                ang = max(-max_angle_range, min(max_angle_range, target))
                        else:
                            ang = random.uniform(-max_angle_range, max_angle_range)
                    self.current_camera_angles[env_idx] = ang
            logger.debug(f"Synthetic camera mode - updated angles for envs {env_ids.tolist()}")
            return
            
        if not self.camera_setup_success or len(self.camera_handles) == 0:
            return
        
        # Get maximum angle range. For yaw sweep feature we use fixed ±30° (curriculum-independent)
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        _max_angle_range, _, _ = task_config.curriculum.get_static_camera_difficulty(curriculum_level)
        FIXED_SWEEP_MAX_DEG = 15.0
        # Honor ablation flag from parent task/global dict and read robot positions
        try:
            parent = getattr(self, 'env_manager', None)
            disable_flag = False
            rp = None
            if parent is not None and hasattr(parent, 'global_tensor_dict'):
                gtd = parent.global_tensor_dict
                disable_flag = bool(gtd.get('static_camera_randomization/orientation_disabled', False))
                rp = gtd.get('robot_position', None)
        except Exception:
            disable_flag = False
            rp = None
        
        try:
            # Fixed camera base position with optional CLI overrides
            try:
                parent = getattr(self, 'env_manager', None)
                gtd = parent.global_tensor_dict if (parent is not None and hasattr(parent, 'global_tensor_dict')) else {}
            except Exception:
                gtd = {}
            try:
                base_y = float(os.environ.get('SF_STATIC_CAMERA_BASE_Y', gtd.get('static_camera/base_y', -3.0)))
            except Exception:
                base_y = -3.0
            # Determine base Z spawning mode: numeric or 'adaptive' to gate center
            try:
                base_z_env = os.environ.get('SF_STATIC_CAMERA_BASE_Z', None)
                if base_z_env is None:
                    base_z_env = gtd.get('static_camera/base_z', 1.5)
                # Accept string 'adaptive' (case-insensitive) to enable adaptive Z per env
                if isinstance(base_z_env, str) and base_z_env.strip().lower() == 'adaptive':
                    base_z = None  # Means adaptive per env
                else:
                    base_z = float(base_z_env)
            except Exception:
                base_z = 1.5
            # If base_z is None (adaptive per env), use a numeric placeholder for Vec3; per-env Z resolved below
            try:
                base_z_for_vec = 1.5 if base_z is None else float(base_z)
            except Exception:
                base_z_for_vec = 1.5
            base_camera_pos = gymapi.Vec3(0.0, base_y, base_z_for_vec)
            
            import math
            import random
            
            # Ensure per-env randomized sweep parameters exist
            try:
                if not hasattr(self, 'sweep_phase_offsets') or (len(getattr(self, 'sweep_phase_offsets', [])) != len(self.env_handles)):
                    self.sweep_phase_offsets = [0.0 for _ in range(len(self.env_handles))]
                    self.sweep_directions = [1.0 for _ in range(len(self.env_handles))]
            except Exception:
                pass
            
            # Update camera orientation ONLY for the specified environments (those resetting)
            for env_idx in env_ids:
                if env_idx >= len(self.env_handles) or env_idx >= len(self.camera_handles):
                    continue
                # Re-sample per-env jitter on reset for fresh episodes
                if self.static_cam_randomize:
                    try:
                        u = torch.rand(3, device=self.device)
                        tmin = torch.tensor(self.static_cam_min_t, device=self.device, dtype=torch.float32)
                        tmax = torch.tensor(self.static_cam_max_t, device=self.device, dtype=torch.float32)
                        t = (tmin + u * (tmax - tmin)).tolist()
                        v = torch.rand(3, device=self.device)
                        emin = torch.tensor(self.static_cam_min_euler, device=self.device, dtype=torch.float32)
                        emax = torch.tensor(self.static_cam_max_euler, device=self.device, dtype=torch.float32)
                        e = (emin + v * (emax - emin)).tolist()
                        self._trans_jitter[env_idx] = (float(t[0]), float(t[1]), float(t[2]))
                        self._euler_jitter_deg[env_idx] = (float(e[0]), float(e[1]), float(e[2]))
                    except Exception:
                        self._trans_jitter[env_idx] = (0.0, 0.0, 0.0)
                        self._euler_jitter_deg[env_idx] = (0.0, 0.0, 0.0)
                    
                # Optional: constant yaw sweep (±30°), curriculum-independent
                try:
                    parent = getattr(self, 'env_manager', None)
                    gtd = parent.global_tensor_dict if (parent is not None and hasattr(parent, 'global_tensor_dict')) else {}
                    sweep_enabled = str(gtd.get('static_camera/yaw_sweep_enabled', 'false')).lower() == 'true'
                    sweep_speed_deg = float(gtd.get('static_camera/yaw_sweep_speed_deg', 10.0))
                except Exception:
                    sweep_enabled = False
                    sweep_speed_deg = 10.0

                if sweep_enabled:
                    # Compute time-based angle: A(level)*sin(omega*t + phase).
                    # Linear amplitude schedule: 2° at level 3 → 19° at level end_level; clamp outside.
                    start_level = 3
                    # Honor evaluation stretch: extend beyond 23 up to eval_end when enabled
                    try:
                        from aerial_gym.config.task_config.navigation_task_config_gate import task_config as _tc_eval
                        # Detect eval-stretch (prefer global_tensor_dict, fallback to env var)
                        try:
                            parent = getattr(self, 'env_manager', None)
                            gtd_local = parent.global_tensor_dict if (parent is not None and hasattr(parent, 'global_tensor_dict')) else {}
                        except Exception:
                            gtd_local = {}
                        eval_en = bool(gtd_local.get('eval_stretch_enabled', False))
                        if not eval_en:
                            import os as _os
                            eval_en = _os.environ.get("EVAL_STRETCH_ENABLED", "0").strip() in ("1", "true", "True")
                        try:
                            eval_end = int(gtd_local.get('eval_stretch_end_level', getattr(_tc_eval.curriculum, 'eval_stretch_end_level', 23)))
                        except Exception:
                            eval_end = int(getattr(_tc_eval.curriculum, 'eval_stretch_end_level', 23))
                        end_level = int(eval_end) if eval_en else 23
                    except Exception:
                        end_level = 23
                    A_min = 2.0
                    A_max = 19.0
                    if curriculum_level <= start_level:
                        A = A_min
                    elif curriculum_level >= end_level:
                        A = A_max
                    else:
                        frac = float(curriculum_level - start_level) / max(1.0, float(end_level - start_level))
                        A = A_min + frac * (A_max - A_min)
                    dt = 1.0/60.0
                    # Keep peak angular speed similar to baseline A0=50° when changing amplitude.
                    # For theta(t)=A*sin(ωt), peak speed = A*ω. Compensate ω by (A0/A).
                    A0 = 50.0
                    comp = (A0 / max(A, 1e-6))
                    # Additionally, increase sweep speed with curriculum level (1.0x -> 2.0x)
                    try:
                        from aerial_gym.config.task_config.navigation_task_config_gate import task_config as _tc2
                        # Respect eval stretch when enabled (eval only), otherwise cap at training max
                        try:
                            parent = getattr(self, 'env_manager', None)
                            gtd_local = parent.global_tensor_dict if (parent is not None and hasattr(parent, 'global_tensor_dict')) else {}
                        except Exception:
                            gtd_local = {}
                        eval_en = bool(gtd_local.get('eval_stretch_enabled', False))
                        min_lvl = int(getattr(_tc2.curriculum, 'min_level', 1))
                        max_lvl_cfg = int(getattr(_tc2.curriculum, 'max_level', min_lvl))
                        max_lvl_eval = int(getattr(_tc2.curriculum, 'eval_stretch_end_level', max_lvl_cfg))
                        max_lvl = max_lvl_eval if eval_en else max_lvl_cfg
                        level_clamped = max(min(curriculum_level, max_lvl), min_lvl)
                        denom = max(1, max_lvl - min_lvl)
                        level_frac = float(level_clamped - min_lvl) / float(denom)
                        speed_scale = 1.0 + level_frac
                    except Exception:
                        speed_scale = 1.0
                    sweep_speed_eff = sweep_speed_deg * speed_scale * comp
                    omega = (sweep_speed_eff * 3.14159 / 180.0) * dt
                    # Use global sim step as t and per-env small phase to desynchronize
                    sim_steps = 0
                    try:
                        steps_obj = gtd.get('sim_steps', 0)
                        # Support torch tensors or plain ints
                        if hasattr(steps_obj, 'shape') or hasattr(steps_obj, 'ndim'):
                            # Torch tensor
                            try:
                                if getattr(steps_obj, 'ndim', 0) == 0:
                                    sim_steps = int(steps_obj.item())
                                else:
                                    idx = env_idx if env_idx < steps_obj.shape[0] else 0
                                    sim_steps = int(steps_obj[idx].item())
                            except Exception:
                                sim_steps = 0
                        else:
                            sim_steps = int(steps_obj)
                    except Exception:
                        sim_steps = 0
                    # Per-env randomized phase and direction so starting angle and direction vary
                    # Re-randomize on first step after reset for each env (sim_steps == 0)
                    try:
                        if sim_steps == 0:
                            # Use seeded torch RNG for determinism across runs with the same --seed
                            rand_phase = float(torch.rand(1, device=self.device).item())  # [0,1)
                            self.sweep_phase_offsets[env_idx] = -math.pi + (2.0 * math.pi * rand_phase)
                            dir_flag = int(torch.randint(low=0, high=2, size=(1,), device=self.device).item())
                            self.sweep_directions[env_idx] = 1.0 if dir_flag == 1 else -1.0
                        phi0 = self.sweep_phase_offsets[env_idx]
                        direction = self.sweep_directions[env_idx]
                    except Exception:
                        phi0 = 0.0
                        direction = 1.0
                    angle_offset_degrees = A * math.sin(direction * (omega * sim_steps) + phi0)
                    debug_max_range = A
                    # [YawSweep DEBUG DISABLED]
                else:
                    # Spawn-aware angle selection: keep both gate (0°) and drone inside FOV; or 0 if disabled
                    # Extend the allowable angle range using the same per-level formula as yaw sweep:
                    # A = max(0, atan2(x_half, |y_center - base_y|) - 43.5) + 2.5 (deg)
                    # Also honor evaluation stretch during inference when enabled.
                    max_angle_range = _max_angle_range
                    try:
                        from aerial_gym.config.task_config.navigation_task_config_gate import task_config as _tc_fix
                        # Detect eval-stretch
                        try:
                            parent = getattr(self, 'env_manager', None)
                            gtd_local = parent.global_tensor_dict if (parent is not None and hasattr(parent, 'global_tensor_dict')) else {}
                        except Exception:
                            gtd_local = {}
                        eval_en = False
                        try:
                            eval_en = bool(gtd_local.get('eval_stretch_enabled', False))
                        except Exception:
                            eval_en = False
                        if not eval_en:
                            try:
                                import os as _os
                                eval_en = _os.environ.get("EVAL_STRETCH_ENABLED", "0").strip() in ("1", "true", "True")
                            except Exception:
                                eval_en = False
                        try:
                            eval_end = int(gtd_local.get('eval_stretch_end_level', getattr(_tc_fix.curriculum, 'eval_stretch_end_level', 23)))
                        except Exception:
                            eval_end = int(getattr(_tc_fix.curriculum, 'eval_stretch_end_level', 23))
                        eff_level = min(curriculum_level, eval_end) if eval_en else curriculum_level
                        sr_fix = _tc_fix.curriculum.get_spawn_ranges(eff_level)
                        x_half_fix = float(sr_fix.get('x_half_span_m', 0.5))
                        y_center_fix = float(sr_fix.get('y_center_m', -1.5))
                        dy_fix = abs(y_center_fix - float(base_y))
                        half_fov_fix = 87.0 * 0.5
                        margin_fix = 2.5
                        alpha_fix = math.degrees(math.atan2(x_half_fix, max(1e-6, dy_fix)))
                        sweep_like_max = max(0.0, alpha_fix - half_fov_fix) + margin_fix
                        max_angle_range = max(max_angle_range, sweep_like_max)
                    except Exception:
                        pass
                    # When sweep is disabled, honor the orientation randomization disable flag
                    if disable_flag or max_angle_range <= 0:
                        angle_offset_degrees = 0.0
                    else:
                        horizontal_fov = 87.0
                        half_fov = horizontal_fov * 0.5
                        margin = 2.5
                        # Extend allowable randomization range further under eval-stretch (levels beyond 23)
                        try:
                            from aerial_gym.config.task_config.navigation_task_config_gate import task_config as _tc_ext
                            try:
                                parent = getattr(self, 'env_manager', None)
                                gtd_local = parent.global_tensor_dict if (parent is not None and hasattr(parent, 'global_tensor_dict')) else {}
                            except Exception:
                                gtd_local = {}
                            eval_en = bool(gtd_local.get('eval_stretch_enabled', False))
                            if not eval_en:
                                import os as _os
                                eval_en = _os.environ.get("EVAL_STRETCH_ENABLED", "0").strip() in ("1", "true", "True")
                            eval_end = int(gtd_local.get('eval_stretch_end_level', getattr(_tc_ext.curriculum, 'eval_stretch_end_level', 23)))
                            if eval_en and curriculum_level > 23:
                                # Scale max_angle_range slightly up to eval_end to keep randomization non-zero
                                # e.g., +25% headroom when at eval_end
                                frac = float(min(curriculum_level, eval_end) - 23) / max(1.0, float(eval_end - 23))
                                max_angle_range = max_angle_range * (1.0 + 0.25 * frac)
                        except Exception:
                            pass
                        if rp is not None and env_idx < rp.shape[0]:
                            cam_x, cam_y = base_camera_pos.x, base_camera_pos.y
                            dx = float(rp[env_idx, 0].item()) - cam_x
                            dy = float(rp[env_idx, 1].item()) - cam_y
                            theta_r = math.degrees(math.atan2(dx, dy))  # 0° points to +Y
                            gate_low, gate_high = -half_fov + margin, half_fov - margin
                            rob_low, rob_high = theta_r - (half_fov - margin), theta_r + (half_fov - margin)
                            low = max(gate_low, rob_low, -max_angle_range)
                            high = min(gate_high, rob_high, max_angle_range)
                            if high > low:
                                # Seeded torch RNG for deterministic selection
                                u = float(torch.rand(1, device=self.device).item())
                                angle_offset_degrees = low + u * (high - low)
                            else:
                                target = max(min(theta_r, gate_high), gate_low)
                                angle_offset_degrees = max(-max_angle_range, min(max_angle_range, target))
                        else:
                            # Seeded torch RNG for deterministic selection
                            u = float(torch.rand(1, device=self.device).item())
                            angle_offset_degrees = -max_angle_range + u * (2.0 * max_angle_range)
                        debug_max_range = max_angle_range
                
                # Store the angle for this environment
                if env_idx < len(self.current_camera_angles):
                    self.current_camera_angles[env_idx] = angle_offset_degrees
                
                # Convert to radians and update camera
                angle_offset_radians = angle_offset_degrees * (3.14159 / 180.0)
                # Euler jitter policy: avoid yaw jitter if curriculum yaw/sweep active. Apply only pitch (small tilt)
                jitter_roll_deg, jitter_pitch_deg, jitter_yaw_deg = self._euler_jitter_deg[env_idx] if (0 <= env_idx < len(self._euler_jitter_deg)) else (0.0, 0.0, 0.0)
                if sweep_enabled or (not disable_flag and max_angle_range > 0):
                    # Curriculum yaw active: zero yaw jitter
                    jitter_yaw_deg = 0.0
                # Apply pitch jitter as a small vertical target offset; roll is not supported via set_camera_location
                pitch_rad = jitter_pitch_deg * (3.14159 / 180.0)
                
                # Resolve per-env base Z (adaptive to gate center if requested)
                try:
                    env_base_z = base_z
                    if env_base_z is None:
                        gh = gtd.get('gate/center_height_per_env', None)
                        env_base_z = float(gh[env_idx].item()) if gh is not None else 1.5
                except Exception:
                    env_base_z = 1.5
                base_camera_env_pos = gymapi.Vec3(base_camera_pos.x, base_camera_pos.y, env_base_z)
                # Apply per-env translation jitter sampled at setup/reset
                try:
                    jx, jy, jz = self._trans_jitter[env_idx]
                except Exception:
                    jx, jy, jz = 0.0, 0.0, 0.0
                base_camera_env_pos = gymapi.Vec3(base_camera_env_pos.x + jx, base_camera_env_pos.y + jy, base_camera_env_pos.z + jz)

                # Calculate offset target position based on randomized angle for this environment
                target_distance = abs(base_camera_env_pos.y)  # Keep look-at distance consistent with base Y
                # Apply additional small yaw jitter around the curriculum yaw (if allowed)
                yaw_total = angle_offset_radians + (jitter_yaw_deg * (3.14159 / 180.0))
                target_x = base_camera_env_pos.x + target_distance * math.sin(yaw_total)
                target_y = base_camera_env_pos.y + target_distance * math.cos(yaw_total)
                # Look at gate adaptive center height, to keep camera pitched to the center
                try:
                    gh = gtd.get('gate/center_height_per_env', None)
                    target_z = float(gh[env_idx].item()) if gh is not None else env_base_z
                except Exception:
                    target_z = env_base_z
                # Apply pitch jitter as small vertical offset in look-at target
                target_z = target_z + math.tan(pitch_rad) * target_distance
                new_target = gymapi.Vec3(target_x, target_y, target_z)
                
                # Update ONLY this environment's camera
                env_handle = self.env_handles[env_idx]
                cam_handle = self.camera_handles[env_idx]
                self.gym.set_camera_location(cam_handle, env_handle, base_camera_env_pos, new_target)
                # Update debug caches
                try:
                    self.last_camera_pos[env_idx] = (float(base_camera_env_pos.x), float(base_camera_env_pos.y), float(base_camera_env_pos.z))
                    self.last_camera_target[env_idx] = (float(new_target.x), float(new_target.y), float(new_target.z))
                    self.last_angle_deg[env_idx] = float(angle_offset_degrees)
                except Exception:
                    pass
                # Debug only for env 0 to avoid spam
                if env_idx == 0:
                    pass
                
                # [YawSweep DEBUG DISABLED] logger.warning(f"[YawSweep] Updated static camera for env {env_idx} - Level {curriculum_level}: {angle_offset_degrees:.1f}° (max range: ±{debug_max_range:.1f}°)")
            
            # [YawSweep DEBUG DISABLED] logger.warning(f"[YawSweep] Updated static camera orientation for {len(env_ids)} environments")
            
        except Exception as e:
            logger.warning(f"Failed to update static camera orientation: {e}")
            # Fall back to fixed positioning if update fails
            logger.debug(f"Static camera orientation update failed - using fixed positioning")
            return
    
    def update_dynamic_camera_following(self, robot_positions: torch.Tensor, gate_positions: torch.Tensor, gate_center_heights: torch.Tensor) -> None:
        """Reimplemented dynamic follow: keep camera 1 m behind the drone (−Y),
        same height, and primarily look at the drone. If the gate is far outside
        the view, minimally steer the look target toward the gate while keeping
        the drone in frame.

        Args:
            robot_positions: (N,3) drone world positions
            gate_positions:  (N,3) gate world positions
            gate_center_heights: (N,) gate center Z per env
        """
        if hasattr(self, 'use_synthetic_camera') and self.use_synthetic_camera:
            return
        if not self.camera_setup_success or len(self.camera_handles) == 0:
            return
        try:
            from aerial_gym.task.task_config_protocol import TaskConfig
from aerial_gym.env_manager.env_manager import EnvManager
from isaacgym import gymapi
            import math
            # Fixed offsets in world frame (Y offset can be overridden via global tensor dict)
            try:
                gtd = self.env_manager.IGE_env.global_tensor_dict
                y_off = float(gtd.get('dynamic_camera_following/offset_y_m', -1.0))
            except Exception:
                y_off = -1.0
            x_off, z_off = 0.0, 0.0
            half_fov = 87.0 * 0.5
            margin = 5.0
            for env_idx in range(min(len(self.env_handles), len(self.camera_handles), robot_positions.shape[0])):
                drone = robot_positions[env_idx]
                cam_x = float(drone[0].item()) + x_off
                cam_y = float(drone[1].item()) + y_off
                cam_z = float(drone[2].item()) + z_off
                camera_pos = gymapi.Vec3(cam_x, cam_y, cam_z)

                # Default look target: the drone itself
                target_drone = gymapi.Vec3(float(drone[0].item()), float(drone[1].item()), float(drone[2].item()))

                # Check gate visibility and minimally bias toward it if necessary
                gate = gate_positions[env_idx]
                gate_cz = float(gate_center_heights[env_idx].item())
                # yaw to drone and gate (0° toward +Y)
                yaw_d = math.degrees(math.atan2(target_drone.x - cam_x, target_drone.y - cam_y))
                yaw_g = math.degrees(math.atan2(float(gate[0].item()) - cam_x, float(gate[1].item()) - cam_y))
                delta = yaw_g - yaw_d
                while delta > 180.0:
                    delta -= 360.0
                while delta < -180.0:
                    delta += 360.0
                # If gate is outside FOV when centered on drone, optionally blend target
                try:
                    disable_blend = bool(self.env_manager.IGE_env.global_tensor_dict.get('dynamic_camera_following/disable_gate_blending', False))
                except Exception:
                    disable_blend = False
                if not disable_blend and abs(delta) > (half_fov - margin):
                    w = 0.2  # small bias toward gate
                    tgx = (1.0 - w) * target_drone.x + w * float(gate[0].item())
                    tgy = (1.0 - w) * target_drone.y + w * float(gate[1].item())
                    tgz = (1.0 - w) * target_drone.z + w * gate_cz
                    camera_target = gymapi.Vec3(tgx, tgy, tgz)
                else:
                    camera_target = target_drone

                cam_handle = self.camera_handles[env_idx]
                env_handle = self.env_handles[env_idx]
                self.gym.set_camera_location(cam_handle, env_handle, camera_pos, camera_target)
                # Update debug caches so visibility/FOV metrics use the correct dynamic camera pose
                try:
                    self.last_camera_pos[env_idx] = (
                        float(camera_pos.x), float(camera_pos.y), float(camera_pos.z)
                    )
                    self.last_camera_target[env_idx] = (
                        float(camera_target.x), float(camera_target.y), float(camera_target.z)
                    )
                    self.last_angle_deg[env_idx] = 0.0
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Failed to update dynamic camera following: {e}")
            return

    def update_arc_follow(self, robot_positions: torch.Tensor, gate_positions: torch.Tensor, gate_center_heights: torch.Tensor, radius_m: float = 2.0) -> None:
        """Arc-follow: constrain camera to a circular arc of fixed radius around the gate center
        (in X–Y), oscillating along the arc but always looking at a blend of drone and gate.

        Args:
            robot_positions: (N,3) drone world positions
            gate_positions:  (N,3) gate world positions
            gate_center_heights: (N,) gate center Z per env
            radius_m: arc radius from gate center (default 2.0 m)
        """
        if hasattr(self, 'use_synthetic_camera') and self.use_synthetic_camera:
            return
        if not self.camera_setup_success or len(self.camera_handles) == 0:
            return
        try:
            from aerial_gym.task.task_config_protocol import TaskConfig
from aerial_gym.env_manager.env_manager import EnvManager
from isaacgym import gymapi
            import math
            # Read global sim steps for a smooth arc oscillation per env
            try:
                gtd = self.env_manager.IGE_env.global_tensor_dict
                sim_steps_obj = gtd.get('sim_steps', 0)
                if hasattr(sim_steps_obj, 'shape') or hasattr(sim_steps_obj, 'ndim'):
                    # tensor: take env-specific index when available
                    get_step = lambda idx: int(sim_steps_obj[idx].item()) if getattr(sim_steps_obj, 'ndim', 0) > 0 and idx < sim_steps_obj.shape[0] else int(sim_steps_obj.item())
                else:
                    step_val = int(sim_steps_obj)
                    get_step = lambda idx: step_val
            except Exception:
                get_step = lambda idx: 0

            # Oscillation parameters (slow arc motion)
            omega = 2.0 * math.pi / 600.0  # one full cycle ~600 frames (~10s at 60Hz)
            phase_per_env = 0.0
            for env_idx in range(min(len(self.env_handles), len(self.camera_handles), robot_positions.shape[0])):
                steps = get_step(env_idx)
                # Gate center (x,y) and desired arc point
                gx = float(gate_positions[env_idx, 0].item())
                gy = float(gate_positions[env_idx, 1].item())
                gz_center = float(gate_center_heights[env_idx].item())

                # Arc angle oscillates with time; small per-env phase for de-sync
                theta = omega * steps + (0.17 * env_idx)
                arc_x = gx + radius_m * math.sin(theta)
                arc_y = gy - radius_m * math.cos(theta)  # "behind" gate is negative Y direction

                # Keep camera near gate center height (or slightly above)
                cam_z = gz_center
                camera_pos = gymapi.Vec3(arc_x, arc_y, cam_z)

                # Blend look target between drone and gate to keep both in frame
                drone = robot_positions[env_idx]
                w = 0.3  # bias toward gate to stabilize
                tgt_x = (1.0 - w) * float(drone[0].item()) + w * gx
                tgt_y = (1.0 - w) * float(drone[1].item()) + w * gy
                tgt_z = (1.0 - w) * float(drone[2].item()) + w * gz_center
                camera_target = gymapi.Vec3(tgt_x, tgt_y, tgt_z)

                cam_handle = self.camera_handles[env_idx]
                env_handle = self.env_handles[env_idx]
                self.gym.set_camera_location(cam_handle, env_handle, camera_pos, camera_target)
        except Exception as e:
            logger.warning(f"Failed to update arc-follow camera: {e}")
            return

    def update_locked_follow(self, robot_positions: torch.Tensor) -> None:
        """Keep camera position fixed; rotate to always center the drone.

        Args:
            robot_positions: Tensor (num_envs, 3) with drone positions in world coords.
        """
        if hasattr(self, 'use_synthetic_camera') and self.use_synthetic_camera:
            return
        if not self.camera_setup_success or len(self.camera_handles) == 0:
            return
        try:
            # Base camera position for all envs
            from aerial_gym.task.task_config_protocol import TaskConfig
from aerial_gym.env_manager.env_manager import EnvManager
from isaacgym import gymapi
            base_y = float(os.environ.get('SF_STATIC_CAMERA_BASE_Y', -3.0))
            base_z_env = os.environ.get('SF_STATIC_CAMERA_BASE_Z', '1.5')
            if isinstance(base_z_env, str) and base_z_env.strip().lower() == 'adaptive':
                # Try to read per-env adaptive Z from global tensors; fallback to 1.5
                try:
                    gtd = self.env_manager.IGE_env.global_tensor_dict
                    gate_center_per_env = gtd.get('gate/center_height_per_env', None)
                except Exception:
                    gate_center_per_env = None
            else:
                gate_center_per_env = None
            for env_idx in range(min(len(self.env_handles), len(self.camera_handles), robot_positions.shape[0])):
                cam_x = 0.0
                cam_y = base_y
                if gate_center_per_env is not None and env_idx < len(gate_center_per_env):
                    cam_z = float(gate_center_per_env[env_idx].item())
                else:
                    try:
                        cam_z = float(base_z_env)
                    except Exception:
                        cam_z = 1.5
                # Build camera and target
                camera_pos = gymapi.Vec3(cam_x, cam_y, cam_z)
                drone = robot_positions[env_idx]
                target = gymapi.Vec3(float(drone[0].item()), float(drone[1].item()), float(drone[2].item()))
                cam_handle = self.camera_handles[env_idx]
                env_handle = self.env_handles[env_idx]
                self.gym.set_camera_location(cam_handle, env_handle, camera_pos, target)
        except Exception as e:
            logger.warning(f"Failed to update locked-follow camera: {e}")
            return
    
    def capture_images(self, batched: bool = False) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Capture depth and segmentation images from static camera.
        Args:
            batched: when True, return stacked depth of shape (num_envs, H, W) and seg for env0;
                     when False (default), return only env0 depth (H, W) and seg for compatibility
                     with GIF/debug pipelines.
        """
        if hasattr(self, 'use_synthetic_camera') and self.use_synthetic_camera:
            logger.error("Static camera synthetic mode disabled; no images will be generated.")
            return None, None
        
        if not self.camera_setup_success or len(self.camera_handles) == 0:
            # One-time lazy re-initialization attempt
            try:
                if (not hasattr(self, '_lazy_setup_attempted')) or (self._lazy_setup_attempted is False):
                    self._lazy_setup_attempted = True
                    logger.warning("Static camera not set up; attempting one-time lazy initialization")
                    try:
                        self._setup_static_camera()
                    except Exception as e:
                        logger.warning(f"Lazy static camera setup attempt failed: {e}")
                    # Re-check after lazy init
                    if not self.camera_setup_success or len(self.camera_handles) == 0:
                        logger.error("Static camera still not set up after lazy init; returning no images.")
                        return None, None
                else:
                    logger.error("Static camera not set up; returning no images.")
                    return None, None
            except Exception:
                logger.error("Static camera not set up; returning no images.")
                return None, None
        
        try:
            # Step graphics and render all cameras
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            
            if batched:
                # Get images from all environments so each env has its own view
                depth_imgs = []
                seg_imgs = []
                for i, (env_handle, cam_handle) in enumerate(zip(self.env_handles, self.camera_handles)):
                    # Depth image
                    depth_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH
                    )
                    depth_img = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
                    depth_imgs.append(depth_img)
                    # Segmentation image (used only for GIF/debug, keep env0 for compatibility)
                    seg_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env_handle, cam_handle, gymapi.IMAGE_SEGMENTATION
                    )
                    seg_img = gymtorch.wrap_tensor(seg_tensor).cpu().numpy()
                    seg_imgs.append(seg_img)
                
                # End access to image tensors
                self.gym.end_access_image_tensors(self.sim)
                
                # Stack per-env depth and normalize to [0, 1] like DCE expects
                if len(depth_imgs) > 0 and depth_imgs[0] is not None:
                    depth_stack = np.stack(depth_imgs, axis=0)
                    depth_stack[depth_stack == -np.inf] = 20.0  # Use far_plane value
                    depth_stack = np.abs(depth_stack)
                    depth_stack = np.clip(depth_stack, 0.4, 20.0)
                    depth_stack = (depth_stack - 0.4) / (20.0 - 0.4)
                    depth_stack = depth_stack.astype(np.float32)
                else:
                    depth_stack = None
                
                # Keep only env0 segmentation for downstream GIF/debug compatibility
                seg_img0 = seg_imgs[0] if len(seg_imgs) > 0 else None
                
                return depth_stack, seg_img0
            else:
                # Get images from camera 0 (single env for GIF/debug)
                env_handle = self.env_handles[0]
                cam_handle = self.camera_handles[0]
                
                # Get depth image
                depth_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH
                )
                depth_img = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
                
                # Get segmentation image
                seg_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_handle, cam_handle, gymapi.IMAGE_SEGMENTATION
                )
                seg_img = gymtorch.wrap_tensor(seg_tensor).cpu().numpy()
                
                # End access to image tensors
                self.gym.end_access_image_tensors(self.sim)
                
                # Normalize depth to [0,1]
                if depth_img is not None:
                    depth_normalized = depth_img.copy()
                    depth_normalized[depth_normalized == -np.inf] = 20.0
                    depth_normalized = np.abs(depth_normalized)
                    depth_normalized = np.clip(depth_normalized, 0.4, 20.0)
                    depth_normalized = (depth_normalized - 0.4) / (20.0 - 0.4)
                    depth_img = depth_normalized.astype(np.float32)
                
                return depth_img, seg_img
                
        except Exception as e:
            logger.error(f"Static camera capture error: {e}")
            return None, None
    def _generate_synthetic_camera_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic camera data for headless training."""
        try:
            # Create synthetic depth image (240x135) with reasonable gate-like features
            height, width = 135, 240
            depth_img = np.full((height, width), 0.5, dtype=np.float32)  # Mid-range depth
            
            # Add gate-like features to the synthetic depth (scaled to new resolution)
            gate_w = max(1, width // 4)   # ~60 px at 240 width
            gate_h = max(1, height // 3)  # ~45 px at 135 height
            gate_x_start = width // 2 - gate_w // 2
            gate_x_end = width // 2 + gate_w // 2
            gate_y_start = height // 2 - gate_h // 2
            gate_y_end = height // 2 + gate_h // 2
            
            # Gate opening (closer depth)
            depth_img[gate_y_start:gate_y_end, gate_x_start:gate_x_end] = 0.8
            
            # Gate frame (farther depth)
            frame_thickness = max(1, min(width, height) // 24)  # scale thickness
            # Top and bottom frame
            depth_img[gate_y_start-frame_thickness:gate_y_start, gate_x_start-frame_thickness:gate_x_end+frame_thickness] = 0.2
            depth_img[gate_y_end:gate_y_end+frame_thickness, gate_x_start-frame_thickness:gate_x_end+frame_thickness] = 0.2
            # Left and right frame
            depth_img[gate_y_start:gate_y_end, gate_x_start-frame_thickness:gate_x_start] = 0.2
            depth_img[gate_y_start:gate_y_end, gate_x_end:gate_x_end+frame_thickness] = 0.2
            
            # Add some noise for realism
            noise = np.random.normal(0, 0.02, (height, width)).astype(np.float32)
            depth_img = np.clip(depth_img + noise, 0.0, 1.0)
            
            # Create synthetic segmentation image
            seg_img = np.zeros((height, width), dtype=np.uint8)
            seg_img[gate_y_start:gate_y_end, gate_x_start:gate_x_end] = 1  # Gate opening
            
            return depth_img, seg_img
            
        except Exception as e:
            logger.debug(f"Synthetic camera data generation error: {e}")
            # Return zero arrays as fallback
            return np.zeros((135, 240), dtype=np.float32), np.zeros((135, 240), dtype=np.uint8)

