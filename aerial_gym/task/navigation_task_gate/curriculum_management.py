from __future__ import annotations

import os
import torch

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.env_flag_utils import read_env_bool

logger = CustomLogger("navigation_task_gate_curriculum")


def setup_curriculum_logging(task) -> None:
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
        task.curriculum_log_file = open(curriculum_log_path, 'w', encoding='utf-8')
        
        # Log initial setup
        init_message = f"=== CURRICULUM LOGGING STARTED ===\nTimestamp: {timestamp}\nLog file: {curriculum_log_path}\n"
        task.curriculum_log_file.write(init_message)
        task.curriculum_log_file.flush()
        
        logger.info(f"Curriculum logging setup successful: {curriculum_log_path}")
        
    except OSError as e:
        # If curriculum logging setup fails, continue without it
        logger.warning(f"Failed to setup curriculum logging: {e}")
        logger.warning("Continuing without curriculum file logging (console logging still active)")
        task.curriculum_log_file = None

def log_curriculum_update(task, message: str) -> None:
    """Log curriculum update messages to both console and curriculum log file."""
    try:
        # Always log to console
        logger.warning(message)
        
        # Try to log to file if available
        if task.curriculum_log_file:
            try:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                task.curriculum_log_file.write(f"[{timestamp}] {message}\n")
                task.curriculum_log_file.flush()  # Ensure immediate write
            except OSError as e:
                # If file logging fails, continue without it
                logger.debug(f"Failed to write to curriculum log file: {e}")
    except OSError as e:
        # If anything fails, just log to console
        logger.warning(f"Curriculum update: {message}")
        logger.debug(f"Curriculum logging error: {e}")

def check_and_update_curriculum_level(task, successes: torch.Tensor, crashes: torch.Tensor, timeouts: torch.Tensor) -> None:
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
        forced = task.task_config.force_curriculum_level
    if forced is not None and str(forced).lower() != 'no':
        eval_stretch_enabled = bool(task.task_config.curriculum.eval_stretch_enabled)
        effective_max = (
            int(task.task_config.curriculum.eval_stretch_end_level)
            if eval_stretch_enabled else task.task_config.curriculum.max_level
        )
        task.curriculum_level = int(forced)
        task.curriculum_level = min(
            max(task.curriculum_level, task.task_config.curriculum.min_level),
            effective_max,
        )
        if hasattr(task.sim_env, 'global_tensor_dict'):
            task.sim_env.global_tensor_dict["curriculum_level"] = int(task.curriculum_level)
        task.obs_dict["curriculum_level"] = task.curriculum_level
        task.max_curriculum_level_reached = max(task.max_curriculum_level_reached, task.curriculum_level)
        return
    task.success_aggregate += torch.sum(successes)
    task.crashes_aggregate += torch.sum(crashes)
    task.timeouts_aggregate += torch.sum(timeouts)

    instances = task.success_aggregate + task.crashes_aggregate + task.timeouts_aggregate
    
    # Remove excessive debugging as requested by user

    if instances >= task.task_config.curriculum.check_after_log_instances:
        success_rate = task.success_aggregate / instances
        crash_rate = task.crashes_aggregate / instances
        timeout_rate = task.timeouts_aggregate / instances
        
        old_level = task.curriculum_level
        task.log_curriculum_update(f"[CURRICULUM UPDATE] EVALUATING curriculum after {instances} instances:")
        task.log_curriculum_update(f"[CURRICULUM UPDATE]   Success rate: {success_rate:.3f}")
        task.log_curriculum_update(f"[CURRICULUM UPDATE]   Crash rate: {crash_rate:.3f}")
        task.log_curriculum_update(f"[CURRICULUM UPDATE]   Timeout rate: {timeout_rate:.3f}")
        task.log_curriculum_update(f"[CURRICULUM UPDATE]   Current level: {old_level} (max reached: {task.max_curriculum_level_reached})")
        task.log_curriculum_update(f"[CURRICULUM UPDATE]   Thresholds: increase>{task.task_config.curriculum.success_rate_for_increase:.3f}, decrease<{task.task_config.curriculum.success_rate_for_decrease:.3f}")
        # Track cooldown state
        if not True: task._curriculum_cooldown = 0
        task.log_curriculum_update(f"[CURRICULUM UPDATE]   Cooldown windows remaining: {task._curriculum_cooldown}")
        # Maintain per-window success history (trim to last 3 windows)
        try:
            sr_float = float(success_rate.item()) if hasattr(success_rate, 'item') else float(success_rate)
        except (ValueError, TypeError):
            sr_float = float(success_rate)
        if not True:
            task._success_window_history = []
        task._success_window_history.append(sr_float)
        if len(task._success_window_history) > 3:
            task._success_window_history.pop(0)
        # Compute current-window success and 3-window average (including current)
        s_t = sr_float
        if len(task._success_window_history) >= 3:
            avg3 = sum(task._success_window_history[-3:]) / 3.0
        else:
            # Use available windows until 3 are accumulated
            denom = max(1, len(task._success_window_history))
            avg3 = sum(task._success_window_history) / denom
        task.infos["curriculum/success_window_s_t"] = torch.tensor(s_t, dtype=torch.float32)
        task.infos["curriculum/success_avg3"] = torch.tensor(avg3, dtype=torch.float32)

        action_msg = "LEVEL UNCHANGED"
        # Respect cooldown
        if task._curriculum_cooldown > 0:
            task._curriculum_cooldown -= 1
            action_msg = f"LEVEL HOLD (cooldown {task._curriculum_cooldown} left)"
        else:
            # Check only at cooldown boundary
            inc_threshold = float(task.task_config.curriculum.success_rate_for_increase)
            avg3_threshold = float(task.task_config.curriculum.avg3_success_for_increase)
            if (len(task._success_window_history) >= 3) and (s_t >= inc_threshold) and (avg3 >= avg3_threshold):
                task.curriculum_level += task.task_config.curriculum.increase_step
                task.max_curriculum_level_reached = max(task.max_curriculum_level_reached, task.curriculum_level)
                task._curriculum_cooldown = task.task_config.curriculum.cooldown_windows
                action_msg = (
                    f"LEVEL INCREASED: {old_level} -> {task.curriculum_level} "
                    f"(s_t {s_t:.3f} >= {inc_threshold:.2f} and avg3 {avg3:.3f} >= {avg3_threshold:.2f})"
                )
            elif success_rate < task.task_config.curriculum.success_rate_for_decrease and task.curriculum_level > task.task_config.curriculum.min_level:
                task.curriculum_level -= task.task_config.curriculum.decrease_step
                task._curriculum_cooldown = task.task_config.curriculum.cooldown_windows
                action_msg = f"LEVEL DECREASED: {old_level} -> {task.curriculum_level} (SR {success_rate:.3f} < threshold)"
        # Apply optional maximum cap without changing per-level scaling
        cap_env = os.environ.get('SF_MAX_CURRICULUM_LEVEL', None)
        cap_cfg = task.task_config.max_curriculum_level
        cap = int(cap_env) if cap_env is not None else (int(cap_cfg) if cap_cfg is not None else None)
        if cap is not None:
            if task.curriculum_level > cap:
                task.curriculum_level = cap
                action_msg = f"LEVEL CAPPED at {cap} (progression halted above cap)"
        # Apply optional minimum start level (training only; no effect in inference)
        min_env = os.environ.get('SF_MIN_CURRICULUM_LEVEL', None)
        if min_env is not None:
            min_cap = int(min_env)
            if task.curriculum_level < min_cap:
                task.curriculum_level = min_cap
                action_msg = f"LEVEL RAISED to start min {min_cap}"
        # Honor forced curriculum level: override and freeze progression
        forced = os.environ.get('SF_FORCE_CURRICULUM_LEVEL', None)
        if forced is None:
            forced = task.task_config.force_curriculum_level
        if forced is not None:
            task.curriculum_level = int(forced)
            action_msg = f"LEVEL FORCED: {task.curriculum_level} (progression disabled)"
            # Reset aggregates to avoid immediate re-evaluation noise
            task.success_aggregate = 0; task.crashes_aggregate = 0; task.timeouts_aggregate = 0
        task.log_curriculum_update(f"[CURRICULUM UPDATE] {action_msg}")

        # Clamp curriculum_level to valid range (honor eval stretch end level if enabled)
        eval_stretch_enabled = bool(task.task_config.curriculum.eval_stretch_enabled)
        effective_max = (
            int(task.task_config.curriculum.eval_stretch_end_level)
            if eval_stretch_enabled else task.task_config.curriculum.max_level
        )
        task.curriculum_level = min(
            max(task.curriculum_level, task.task_config.curriculum.min_level),
            effective_max,
        )
        task.obs_dict["curriculum_level"] = task.curriculum_level
        
        # Propagate curriculum level to env manager for gate unlocking
        if hasattr(task.sim_env, 'global_tensor_dict'):
            # Only update the value; gate selection will occur on reset_idx
            task.sim_env.global_tensor_dict["curriculum_level"] = int(task.curriculum_level)
        
        
        # 1. OBSTACLE COUNT PROGRESSION: Apply new obstacle count behind gate
        try:
            obs_dis = bool(task.sim_env.global_tensor_dict.get('obstacles_randomization/disabled', False))
        except (KeyError, TypeError):
            obs_dis = False
        if obs_dis:
            try:
                obstacles_behind_gate = int(task.sim_env.global_tensor_dict.get('obstacles_randomization/fixed_count', 0))
            except (ValueError, TypeError):
                obstacles_behind_gate = 0
        else:
            obstacles_behind_gate = task.task_config.curriculum.get_obstacle_count_behind_gate(task.curriculum_level)
        
        # FIXED CALCULATION: Account for visible assets only (not all loaded assets)
        # Even though 11 gate variants are loaded, only 1 is visible at any time
        visible_gates = 1  # Only 1 gate visible at a time (others hidden by gate selection system)
        walls = 6  # 6 boundary walls  
        robot = 0  # Robot is NOT part of env_asset_state_tensor (handled separately)
        fixed_assets_visible = visible_gates + walls  # = 7 visible fixed assets
            
        total_obstacles_in_env = fixed_assets_visible + obstacles_behind_gate
        task.obs_dict["num_obstacles_in_env"] = total_obstacles_in_env
        
        # CRITICAL: Also update the environment manager's global tensor dict for asset management
        # This ensures the asset manager gets the updated obstacle count when environments reset
        if hasattr(task.sim_env, 'global_tensor_dict'):
            old_count = task.sim_env.global_tensor_dict.get("num_obstacles_in_env", 0)
            task.sim_env.global_tensor_dict["num_obstacles_in_env"] = total_obstacles_in_env
        
        # The asset manager may be caching the initial obstacle count, so we need to force it to update
        if hasattr(task.sim_env, 'asset_manager'):
            try:
                # Try to directly update the asset manager's obstacle count
                if hasattr(task.sim_env.asset_manager, 'num_obstacles_per_env'):
                    old_count = getattr(task.sim_env.asset_manager, 'num_obstacles_per_env', 'unknown')
                    task.sim_env.asset_manager.num_obstacles_per_env = total_obstacles_in_env
                    task.log_curriculum_update(f"[CRITICAL FIX] Direct asset manager update: {old_count} → {total_obstacles_in_env}")
                    
                # NOTE: Asset manager changes will be applied when environments naturally reset
                task.log_curriculum_update(f"[CRITICAL FIX] Asset manager updated - changes will apply on next environment reset")
                
            except Exception as e:
                task.log_curriculum_update(f"[CRITICAL FIX] Warning: Failed to directly update asset manager: {e}")
        
        # ALTERNATIVE: Try to access environment configuration directly
        if hasattr(task.sim_env, 'env_config'):
            try:
                if hasattr(task.sim_env.env_config, 'num_obstacles'):
                    old_env_count = getattr(task.sim_env.env_config, 'num_obstacles', 'unknown')
                    task.sim_env.env_config.num_obstacles = total_obstacles_in_env
                    task.log_curriculum_update(f"[CRITICAL FIX] Environment config update: {old_env_count} → {total_obstacles_in_env}")
            except Exception as e:
                task.log_curriculum_update(f"[CRITICAL FIX] Warning: Failed to update environment config: {e}")
        
        # 2. STATIC CAMERA DIFFICULTY: Update camera parameters for NEW episodes only
        # Update max camera angle for logging (affects new episodes only)
        task.max_camera_angle, task.camera_height_offset, task.camera_distance_offset = task.task_config.curriculum.get_static_camera_difficulty(task.curriculum_level)
        
        # DON'T update camera positions here - only update on episode reset
        # This ensures camera orientation stays fixed during each episode
        task.log_curriculum_update(f"[CAMERA UPDATE] Camera max angle updated for NEW episodes: ±{task.max_camera_angle:.1f}° (existing episodes unchanged)")

        # Calculate curriculum progress fraction
        task.curriculum_progress_fraction = (
            task.curriculum_level - task.task_config.curriculum.min_level
        ) / (task.task_config.curriculum.max_level - task.task_config.curriculum.min_level)

        task._log_curriculum_details(success_rate, crash_rate, timeout_rate, obstacles_behind_gate, total_obstacles_in_env)

        task._populate_curriculum_infos(success_rate, crash_rate, timeout_rate, obstacles_behind_gate, total_obstacles_in_env)
        
        task.log_curriculum_update(f"[CURRICULUM UPDATE] RESETTING counters for next evaluation period")
        task.success_aggregate = 0
        task.crashes_aggregate = 0
        task.timeouts_aggregate = 0

