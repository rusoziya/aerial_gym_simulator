from __future__ import annotations

import os
import torch

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.env_flag_utils import read_env_bool

logger = CustomLogger("navigation_task_gate_curriculum")


class NavigationTaskGateCurriculumMixin:
    """Curriculum management methods extracted from NavigationTaskGate."""

    def setup_curriculum_logging(self) -> None:
        """Setup separate curriculum logging file in train_dir."""
        try:
            # Try to determine train_dir path from Sample Factory environment or working directory
            
            # Get train_dir from environment variable or use current directory/train_dir
            train_dir = os.environ.get('SF_TRAIN_DIR', './train_dir')
            
            # If train_dir doesn't exist, create it
            os.makedirs(train_dir, exist_ok=True)
            
            # Create curriculum log filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            curriculum_log_filename = f"curriculum_gate_navigation_{timestamp}.log"
            curriculum_log_path = os.path.join(train_dir, curriculum_log_filename)
            
            # Open curriculum log file in UTF-8 encoding
            self.curriculum_log_file = open(curriculum_log_path, 'w', encoding='utf-8')
            
            # Log initial setup
            init_message = f"=== CURRICULUM LOGGING STARTED ===\nTimestamp: {timestamp}\nLog file: {curriculum_log_path}\n"
            self.curriculum_log_file.write(init_message)
            self.curriculum_log_file.flush()
            
            logger.info(f"Curriculum logging setup successful: {curriculum_log_path}")
            
        except OSError as e:
            # If curriculum logging setup fails, continue without it
            logger.warning(f"Failed to setup curriculum logging: {e}")
            logger.warning("Continuing without curriculum file logging (console logging still active)")
            self.curriculum_log_file = None

    def log_curriculum_update(self, message: str) -> None:
        """Log curriculum update messages to both console and curriculum log file."""
        try:
            # Always log to console
            logger.warning(message)
            
            # Try to log to file if available
            if self.curriculum_log_file:
                try:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.curriculum_log_file.write(f"[{timestamp}] {message}\n")
                    self.curriculum_log_file.flush()  # Ensure immediate write
                except OSError as e:
                    # If file logging fails, continue without it
                    logger.debug(f"Failed to write to curriculum log file: {e}")
        except OSError as e:
            # If anything fails, just log to console
            logger.warning(f"Curriculum update: {message}")
            logger.debug(f"Curriculum logging error: {e}")

    def check_and_update_curriculum_level(self, successes: torch.Tensor, crashes: torch.Tensor, timeouts: torch.Tensor) -> None:
        """
        COMPREHENSIVE MULTI-ASPECT CURRICULUM LEARNING SYSTEM
        
        Updates curriculum level and applies changes to multiple difficulty aspects:
        1. Obstacle count behind gate (increases by 1 per level, cap at 10)
        2. Drone spawning difficulty (angle and distance from gate)
        3. Drone orientation randomization (progressive random orientations)
        4. Static camera positioning (progressive angle and distance changes)
        """
        # Early exit for testing/eval with forced level: lock level and skip progression
        forced = os.environ.get('SF_FORCE_CURRICULUM_LEVEL', None)
        if forced is None:
            forced = self.task_config.force_curriculum_level
        if forced is not None and str(forced).lower() != 'no':
            eval_stretch_enabled = bool(self.task_config.curriculum.eval_stretch_enabled)
            effective_max = (
                int(self.task_config.curriculum.eval_stretch_end_level)
                if eval_stretch_enabled else self.task_config.curriculum.max_level
            )
            self.curriculum_level = int(forced)
            self.curriculum_level = min(
                max(self.curriculum_level, self.task_config.curriculum.min_level),
                effective_max,
            )
            if hasattr(self.sim_env, 'global_tensor_dict'):
                self.sim_env.global_tensor_dict["curriculum_level"] = int(self.curriculum_level)
            self.obs_dict["curriculum_level"] = self.curriculum_level
            self.max_curriculum_level_reached = max(self.max_curriculum_level_reached, self.curriculum_level)
            return
        self.success_aggregate += torch.sum(successes)
        self.crashes_aggregate += torch.sum(crashes)
        self.timeouts_aggregate += torch.sum(timeouts)

        instances = self.success_aggregate + self.crashes_aggregate + self.timeouts_aggregate
        
        # Remove excessive debugging as requested by user

        if instances >= self.task_config.curriculum.check_after_log_instances:
            success_rate = self.success_aggregate / instances
            crash_rate = self.crashes_aggregate / instances
            timeout_rate = self.timeouts_aggregate / instances
            
            old_level = self.curriculum_level
            self.log_curriculum_update(f"[CURRICULUM UPDATE] EVALUATING curriculum after {instances} instances:")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Success rate: {success_rate:.3f}")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Crash rate: {crash_rate:.3f}")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Timeout rate: {timeout_rate:.3f}")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Current level: {old_level} (max reached: {self.max_curriculum_level_reached})")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Thresholds: increase>{self.task_config.curriculum.success_rate_for_increase:.3f}, decrease<{self.task_config.curriculum.success_rate_for_decrease:.3f}")
            # Track cooldown state
            if not True: self._curriculum_cooldown = 0
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Cooldown windows remaining: {self._curriculum_cooldown}")
            # Maintain per-window success history (trim to last 3 windows)
            try:
                sr_float = float(success_rate.item()) if hasattr(success_rate, 'item') else float(success_rate)
            except (ValueError, TypeError):
                sr_float = float(success_rate)
            if not True:
                self._success_window_history = []
            self._success_window_history.append(sr_float)
            if len(self._success_window_history) > 3:
                self._success_window_history.pop(0)
            # Compute current-window success and 3-window average (including current)
            s_t = sr_float
            if len(self._success_window_history) >= 3:
                avg3 = sum(self._success_window_history[-3:]) / 3.0
            else:
                # Use available windows until 3 are accumulated
                denom = max(1, len(self._success_window_history))
                avg3 = sum(self._success_window_history) / denom
            self.infos["curriculum/success_window_s_t"] = torch.tensor(s_t, dtype=torch.float32)
            self.infos["curriculum/success_avg3"] = torch.tensor(avg3, dtype=torch.float32)

            action_msg = "LEVEL UNCHANGED"
            # Respect cooldown
            if self._curriculum_cooldown > 0:
                self._curriculum_cooldown -= 1
                action_msg = f"LEVEL HOLD (cooldown {self._curriculum_cooldown} left)"
            else:
                # Check only at cooldown boundary
                inc_threshold = float(self.task_config.curriculum.success_rate_for_increase)
                avg3_threshold = float(self.task_config.curriculum.avg3_success_for_increase)
                if (len(self._success_window_history) >= 3) and (s_t >= inc_threshold) and (avg3 >= avg3_threshold):
                    self.curriculum_level += self.task_config.curriculum.increase_step
                    self.max_curriculum_level_reached = max(self.max_curriculum_level_reached, self.curriculum_level)
                    self._curriculum_cooldown = self.task_config.curriculum.cooldown_windows
                    action_msg = (
                        f"LEVEL INCREASED: {old_level} -> {self.curriculum_level} "
                        f"(s_t {s_t:.3f} >= {inc_threshold:.2f} and avg3 {avg3:.3f} >= {avg3_threshold:.2f})"
                    )
                elif success_rate < self.task_config.curriculum.success_rate_for_decrease and self.curriculum_level > self.task_config.curriculum.min_level:
                    self.curriculum_level -= self.task_config.curriculum.decrease_step
                    self._curriculum_cooldown = self.task_config.curriculum.cooldown_windows
                    action_msg = f"LEVEL DECREASED: {old_level} -> {self.curriculum_level} (SR {success_rate:.3f} < threshold)"
            # Apply optional maximum cap without changing per-level scaling
            cap_env = os.environ.get('SF_MAX_CURRICULUM_LEVEL', None)
            cap_cfg = self.task_config.max_curriculum_level
            cap = int(cap_env) if cap_env is not None else (int(cap_cfg) if cap_cfg is not None else None)
            if cap is not None:
                if self.curriculum_level > cap:
                    self.curriculum_level = cap
                    action_msg = f"LEVEL CAPPED at {cap} (progression halted above cap)"
            # Apply optional minimum start level (training only; no effect in inference)
            min_env = os.environ.get('SF_MIN_CURRICULUM_LEVEL', None)
            if min_env is not None:
                min_cap = int(min_env)
                if self.curriculum_level < min_cap:
                    self.curriculum_level = min_cap
                    action_msg = f"LEVEL RAISED to start min {min_cap}"
            # Honor forced curriculum level: override and freeze progression
            forced = os.environ.get('SF_FORCE_CURRICULUM_LEVEL', None)
            if forced is None:
                forced = self.task_config.force_curriculum_level
            if forced is not None:
                self.curriculum_level = int(forced)
                action_msg = f"LEVEL FORCED: {self.curriculum_level} (progression disabled)"
                # Reset aggregates to avoid immediate re-evaluation noise
                self.success_aggregate = 0; self.crashes_aggregate = 0; self.timeouts_aggregate = 0
            self.log_curriculum_update(f"[CURRICULUM UPDATE] {action_msg}")

            # Clamp curriculum_level to valid range (honor eval stretch end level if enabled)
            eval_stretch_enabled = bool(self.task_config.curriculum.eval_stretch_enabled)
            effective_max = (
                int(self.task_config.curriculum.eval_stretch_end_level)
                if eval_stretch_enabled else self.task_config.curriculum.max_level
            )
            self.curriculum_level = min(
                max(self.curriculum_level, self.task_config.curriculum.min_level),
                effective_max,
            )
            self.obs_dict["curriculum_level"] = self.curriculum_level
            
            # Propagate curriculum level to env manager for gate unlocking
            if hasattr(self.sim_env, 'global_tensor_dict'):
                # Only update the value; gate selection will occur on reset_idx
                self.sim_env.global_tensor_dict["curriculum_level"] = int(self.curriculum_level)
            
            
            # 1. OBSTACLE COUNT PROGRESSION: Apply new obstacle count behind gate
            try:
                obs_dis = bool(self.sim_env.global_tensor_dict.get('obstacles_randomization/disabled', False))
            except (KeyError, TypeError):
                obs_dis = False
            if obs_dis:
                try:
                    obstacles_behind_gate = int(self.sim_env.global_tensor_dict.get('obstacles_randomization/fixed_count', 0))
                except (ValueError, TypeError):
                    obstacles_behind_gate = 0
            else:
                obstacles_behind_gate = self.task_config.curriculum.get_obstacle_count_behind_gate(self.curriculum_level)
            
            # FIXED CALCULATION: Account for visible assets only (not all loaded assets)
            # Even though 11 gate variants are loaded, only 1 is visible at any time
            visible_gates = 1  # Only 1 gate visible at a time (others hidden by gate selection system)
            walls = 6  # 6 boundary walls  
            robot = 0  # Robot is NOT part of env_asset_state_tensor (handled separately)
            fixed_assets_visible = visible_gates + walls  # = 7 visible fixed assets
                
            total_obstacles_in_env = fixed_assets_visible + obstacles_behind_gate
            self.obs_dict["num_obstacles_in_env"] = total_obstacles_in_env
            
            # CRITICAL: Also update the environment manager's global tensor dict for asset management
            # This ensures the asset manager gets the updated obstacle count when environments reset
            if hasattr(self.sim_env, 'global_tensor_dict'):
                old_count = self.sim_env.global_tensor_dict.get("num_obstacles_in_env", 0)
                self.sim_env.global_tensor_dict["num_obstacles_in_env"] = total_obstacles_in_env
            
            # The asset manager may be caching the initial obstacle count, so we need to force it to update
            if hasattr(self.sim_env, 'asset_manager'):
                try:
                    # Try to directly update the asset manager's obstacle count
                    if hasattr(self.sim_env.asset_manager, 'num_obstacles_per_env'):
                        old_count = getattr(self.sim_env.asset_manager, 'num_obstacles_per_env', 'unknown')
                        self.sim_env.asset_manager.num_obstacles_per_env = total_obstacles_in_env
                        self.log_curriculum_update(f"[CRITICAL FIX] Direct asset manager update: {old_count} → {total_obstacles_in_env}")
                        
                    # NOTE: Asset manager changes will be applied when environments naturally reset
                    self.log_curriculum_update(f"[CRITICAL FIX] Asset manager updated - changes will apply on next environment reset")
                    
                except Exception as e:
                    self.log_curriculum_update(f"[CRITICAL FIX] Warning: Failed to directly update asset manager: {e}")
            
            # ALTERNATIVE: Try to access environment configuration directly
            if hasattr(self.sim_env, 'env_config'):
                try:
                    if hasattr(self.sim_env.env_config, 'num_obstacles'):
                        old_env_count = getattr(self.sim_env.env_config, 'num_obstacles', 'unknown')
                        self.sim_env.env_config.num_obstacles = total_obstacles_in_env
                        self.log_curriculum_update(f"[CRITICAL FIX] Environment config update: {old_env_count} → {total_obstacles_in_env}")
                except Exception as e:
                    self.log_curriculum_update(f"[CRITICAL FIX] Warning: Failed to update environment config: {e}")
            
            # 2. STATIC CAMERA DIFFICULTY: Update camera parameters for NEW episodes only
            # Update max camera angle for logging (affects new episodes only)
            self.max_camera_angle, self.camera_height_offset, self.camera_distance_offset = self.task_config.curriculum.get_static_camera_difficulty(self.curriculum_level)
            
            # DON'T update camera positions here - only update on episode reset
            # This ensures camera orientation stays fixed during each episode
            self.log_curriculum_update(f"[CAMERA UPDATE] Camera max angle updated for NEW episodes: ±{self.max_camera_angle:.1f}° (existing episodes unchanged)")

            # Calculate curriculum progress fraction
            self.curriculum_progress_fraction = (
                self.curriculum_level - self.task_config.curriculum.min_level
            ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

            self._log_curriculum_details(success_rate, crash_rate, timeout_rate, obstacles_behind_gate, total_obstacles_in_env)

            self._populate_curriculum_infos(success_rate, crash_rate, timeout_rate, obstacles_behind_gate, total_obstacles_in_env)
            
            self.log_curriculum_update(f"[CURRICULUM UPDATE] RESETTING counters for next evaluation period")
            self.success_aggregate = 0
            self.crashes_aggregate = 0
            self.timeouts_aggregate = 0

