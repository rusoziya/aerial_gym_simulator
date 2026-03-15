from __future__ import annotations

import math
import os

import torch

from sample_factory.algo.learning.learner import Learner
from sample_factory.train import run_rl
from sample_factory.utils.typing import Config


def run_with_influence_tracking(cfg: Config) -> None:
    """Enhanced training with complete observation influence tracking."""
    
    # Import the complete observation influence tracker and gradient attribution
    try:
        from aerial_gym.utils.gradient_monitor import create_influence_tracker, INFLUENCE_MONITOR_AVAILABLE
        from aerial_gym.utils.gradient_attribution import create_gradient_tracker, GRAD_ATTR_AVAILABLE
    except ImportError:
        print("❌ Influence/gradient trackers not available")
        INFLUENCE_MONITOR_AVAILABLE = False
        GRAD_ATTR_AVAILABLE = False
    
    if not INFLUENCE_MONITOR_AVAILABLE:
        print("❌ Complete observation influence tracker not available - falling back to standard training")
        return run_rl(cfg)

    print("🔬 Complete observation influence tracking ENABLED - analyzing ALL 150D observation components")
    print("   📊 Log interval: {} steps".format(cfg.gradient_log_interval))
    print("   📋 Print interval: {} steps".format(cfg.gradient_print_interval))
    print("✅ Complete observation influence tracker ready")
    print("🔍 Will analyze ALL 150D observation components for neural network influence")
    
    # Store original wandb.log if wandb is used
    original_wandb_log = None
    if cfg.with_wandb:
        try:
            import wandb
            original_wandb_log = wandb.log
        except ImportError:
            pass

    # Create tracker instances (attached to model later)
    influence_tracker = None
    grad_tracker = None

    # Cache last non-empty obs_grad metrics to mirror every step
    _last_obsgrad_from_influence = {}
    _last_obsgrad_from_grad = {}
    # Continuous curriculum logging: remember last known values and emit every step
    CURRICULUM_KEYS = [
        'curriculum/total_timeouts',
        'curriculum/total_successes',
        'curriculum/total_crashes',
        'curriculum/total_resets',
        'curriculum/timeout_rate',
        'curriculum/success_rate',
        'curriculum/state_noise_static_pos_std_m',
        'curriculum/state_noise_static_orient_std_deg',
        'curriculum/state_noise_drone_pos_std_m',
        'curriculum/state_noise_drone_orient_std_deg',
        'curriculum/progress',
        'curriculum/obstacles_behind_gate',
        'curriculum/max_level_reached',
        'curriculum/level',
        'curriculum/crash_rate',
        'curriculum/camera_max_angle',
        'curriculum/camera_gaussian_std',
        'curriculum/camera_frame_freeze_static',
        'curriculum/camera_frame_freeze_drone',
        'curriculum/camera_frame_dropout_static_total',
        'curriculum/camera_frame_dropout_drone_total',
        'curriculum/camera_frame_blank_static',
        'curriculum/camera_frame_blank_drone',
        'curriculum/camera_dropout_rate',
        'curriculum/camera_current_angle',
    ]
    _last_curriculum = {k: 0.0 for k in CURRICULUM_KEYS}

    def enhanced_wandb_log(metrics, **kwargs) -> None:
        """Enhanced wandb logging that includes influence monitoring metrics"""
        nonlocal influence_tracker
        nonlocal _last_obsgrad_from_influence
        nonlocal _last_obsgrad_from_grad
        # Ensure a dict
        if metrics is None:
            metrics = {}
        else:
            metrics = dict(metrics)
        
        # Inject step from cfg if not provided
        step_key = None
        if 'step' in kwargs:
            step_key = 'step'
        elif 'commit' in kwargs and isinstance(kwargs.get('commit'), bool):
            # no explicit step in kwargs
            pass
        
        # Prefer Sample Factory train_step as global step
        global_step = None
        from sample_factory.algo.learning.learner import Learner as _L
        
        # Fallback: attach frames/env_steps from cfg if available
        frames = None
        if hasattr(cfg, 'train_step') and isinstance(cfg.train_step, (int, float)):
            frames = int(cfg.train_step)
        elif hasattr(cfg, 'env_steps') and isinstance(cfg.env_steps, (int, float)):
            frames = int(cfg.env_steps)
        if frames is not None:
            metrics.setdefault('frames', frames)
            kwargs.setdefault('step', frames)
        
        # Merge influence and grad metrics
        if influence_tracker:
            if not influence_tracker.should_log():
                print("[W&B_DEBUG][obs_grad] influence_tracker present but not scheduled to log at this step (forcing minimal log header)")
            influence_metrics = influence_tracker.get_logging_metrics()
            # Cast to plain floats
            for k, v in list(influence_metrics.items()):
                try:
                    influence_metrics[k] = float(v)
                except (ValueError, TypeError):
                    del influence_metrics[k]
            metrics.update(influence_metrics)
            # Update cache if we received any obs/influence keys
            had = any(
                isinstance(k, str) and k.startswith(('obs_grad/', 'influence/', 'grad_attr/', 'obs_influence/'))
                for k in influence_metrics.keys()
            )
            if had:
                _last_obsgrad_from_influence = dict(influence_metrics)
            # Also mirror per-slice obs_grad/influence summaries under episode_extra_stats for dashboard grouping
            try:
                episode_extra = {}
                # Expected keys provided by tracker (examples):
                #  - obs_grad/slice/..., influence/slice/...
                #  - obs_grad/total_recent, influence/total_recent, ...
                source_metrics = influence_metrics if len(influence_metrics) > 0 else _last_obsgrad_from_influence
                for name, val in list(source_metrics.items()):
                    if not isinstance(name, str):
                        continue
                    if name.startswith(('obs_grad/', 'influence/', 'grad_attr/', 'obs_influence/')):
                        prefix_removed = name.split('/', 1)[1] if '/' in name else name
                        new_key = 'episode_extra_stats/obs_grad/' + prefix_removed
                        episode_extra[new_key] = float(val)
                if len(episode_extra) > 0:
                    metrics.update(episode_extra)
                    #     pass
                # Derived metrics: camera/state shares and per-slice magnitudes + shares
                # Collect slice values by suffix (ignore totals/backward_passes)
                SLICE_NAMES_STATE = {
                    'drone_position', 'static_camera_pos', 'static_camera_orient',
                    'drone_orientation',
                    'linear_vel', 'angular_vel', 'actions',
                    'drone_linear_vel', 'drone_angular_vel', 'drone_actions'
                }
                SLICE_NAMES_CAMERA = {'drone_camera_vae', 'static_camera_vae'}
                # Modality groupings for visual/kinematic/spatial shares
                SLICE_NAMES_VISUAL = {'drone_camera_vae', 'static_camera_vae'}
                SLICE_NAMES_KINEMATIC = {'drone_linear_vel', 'drone_angular_vel', 'drone_actions', 'linear_vel', 'angular_vel', 'actions'}
                SLICE_NAMES_SPATIAL = {'drone_position', 'static_camera_pos', 'static_camera_orient', 'drone_orientation'}
                total_val = 0.0
                camera_val = 0.0
                state_val = 0.0
                visual_val = 0.0
                kinematic_val = 0.0
                spatial_val = 0.0
                slice_values = {}
                for name, val in list(source_metrics.items()):
                    if not isinstance(name, str):
                        continue
                    if not (name.startswith(('obs_grad/', 'influence/', 'grad_attr/', 'obs_influence/'))):
                        continue
                    parts = name.split('/')
                    # Extract slice label after 'slice_pct'/'slice_mag' if present
                    label = parts[-1]
                    if 'slice_pct' in parts:
                        idx = parts.index('slice_pct')
                        if idx + 1 < len(parts):
                            label = parts[idx + 1]
                    elif 'slice_mag' in parts:
                        idx = parts.index('slice_mag')
                        if idx + 1 < len(parts):
                            label = parts[idx + 1]
                    suffix = parts[-1]
                    if suffix.startswith('total_') or suffix == 'backward_passes':
                        continue
                    try:
                        scalar = float(val)
                    except (ValueError, TypeError):
                        continue
                    total_val += scalar
                    # Normalize label by removing common postfixes
                    base = label
                    for post in ['_mean_norm_recent', '_mean_norm_overall', '_recent', '_overall', '_mean_norm', '_mean', '_norm']:
                        if base.endswith(post):
                            base = base[: -len(post)]
                            break
                    slice_values[base] = scalar
                    # Camera bucket
                    if base in SLICE_NAMES_CAMERA or 'camera_vae' in base:
                        camera_val += scalar
                    # State bucket
                    elif (base in SLICE_NAMES_STATE or
                          base.startswith('drone_position') or base.startswith('drone_orientation') or
                          base.startswith('drone_linear_vel') or base.startswith('drone_angular_vel') or
                          base.startswith('drone_actions') or
                          base.startswith('static_camera_pos') or base.startswith('static_camera_orient')):
                        state_val += scalar
                    # Modality buckets
                    if base in SLICE_NAMES_VISUAL or 'camera_vae' in base:
                        visual_val += scalar
                    if (base in SLICE_NAMES_KINEMATIC or
                        base.startswith('drone_linear_vel') or base.startswith('drone_angular_vel') or base.startswith('drone_actions')):
                        kinematic_val += scalar
                    if (base in SLICE_NAMES_SPATIAL or
                        base.startswith('drone_position') or base.startswith('drone_orientation') or
                        base.startswith('static_camera_pos') or base.startswith('static_camera_orient')):
                        spatial_val += scalar
                    else:
                        # Unknown slices count toward total but not camera/state buckets
                        pass
                if total_val > 0.0:
                    camera_share = camera_val / total_val
                    # Fallback if state set did not match tracker naming: infer as residual
                    if state_val <= 0.0 and camera_val > 0.0:
                        state_val = max(total_val - camera_val, 0.0)
                    state_share = state_val / total_val
                    # Ratios [0,1]
                    metrics['episode_extra_stats/obs_grad/camera_share'] = float(camera_share)
                    metrics['episode_extra_stats/obs_grad/state_share'] = float(state_share)
                    # Percentages [0,100]
                    metrics['episode_extra_stats/obs_grad/camera_share_pct'] = float(camera_share * 100.0)
                    metrics['episode_extra_stats/obs_grad/state_share_pct'] = float(state_share * 100.0)
                    # Modality shares (visual/kinematic/spatial)
                    visual_share = visual_val / total_val
                    kinematic_share = kinematic_val / total_val
                    spatial_share = spatial_val / total_val
                    metrics['episode_extra_stats/obs_grad/visual_share'] = float(visual_share)
                    metrics['episode_extra_stats/obs_grad/kinematic_share'] = float(kinematic_share)
                    metrics['episode_extra_stats/obs_grad/spatial_share'] = float(spatial_share)
                    metrics['episode_extra_stats/obs_grad/visual_share_pct'] = float(visual_share * 100.0)
                    metrics['episode_extra_stats/obs_grad/kinematic_share_pct'] = float(kinematic_share * 100.0)
                    metrics['episode_extra_stats/obs_grad/spatial_share_pct'] = float(spatial_share * 100.0)
                    # Per-slice magnitudes and percentages
                    for base, sval in slice_values.items():
                        metrics['episode_extra_stats/obs_grad/slice_mag/' + base] = float(sval)
                        metrics['episode_extra_stats/obs_grad/slice_pct/' + base] = float((sval / total_val) * 100.0)

                    # Per-observation percentages (recent/overall) with sum/residual guardrails
                    try:
                        base_recent = {}
                        base_overall = {}
                        for name, val in list(source_metrics.items()):
                            if not isinstance(name, str) or not name.startswith(('obs_grad/', 'influence/', 'grad_attr/')):
                                continue
                            parts = name.split('/')
                            suffix = parts[-1]
                            # map to base
                            base = suffix
                            for post in ['_mean_norm_recent', '_mean_norm_overall', '_recent', '_overall', '_mean_norm', '_mean', '_norm']:
                                if base.endswith(post):
                                    base = base[: -len(post)]
                                    break
                            try:
                                scalar = float(val)
                            except (ValueError, TypeError):
                                continue
                            if suffix.endswith('_mean_norm_recent') or suffix.endswith('_recent'):
                                base_recent[base] = base_recent.get(base, 0.0) + scalar
                            if suffix.endswith('_mean_norm_overall') or suffix.endswith('_overall'):
                                base_overall[base] = base_overall.get(base, 0.0) + scalar
                        # recent
                        tot_r = float(sum(base_recent.values()))
                        if tot_r > 0.0:
                            sum_pct_r = 0.0
                            for b, v in base_recent.items():
                                p = 100.0 * float(v) / tot_r
                                metrics[f'episode_extra_stats/obs_grad/obs_pct_recent/{b}'] = float(p)
                                sum_pct_r += p
                            metrics['episode_extra_stats/obs_grad/obs_pct_recent/_sum'] = float(sum_pct_r)
                            metrics['episode_extra_stats/obs_grad/obs_pct_recent/_residual'] = float(100.0 - sum_pct_r)
                        # overall
                        tot_o = float(sum(base_overall.values()))
                        if tot_o > 0.0:
                            sum_pct_o = 0.0
                            for b, v in base_overall.items():
                                p = 100.0 * float(v) / tot_o
                                metrics[f'episode_extra_stats/obs_grad/obs_pct_overall/{b}'] = float(p)
                                sum_pct_o += p
                            metrics['episode_extra_stats/obs_grad/obs_pct_overall/_sum'] = float(sum_pct_o)
                            metrics['episode_extra_stats/obs_grad/obs_pct_overall/_residual'] = float(100.0 - sum_pct_o)
                    except (ValueError, TypeError):
                        pass
                    #     captured = ','.join(sorted(list(slice_values.keys()))[:8])
                    #     pass
            except (ValueError, TypeError):
                pass
            if influence_tracker.should_log():
                influence_tracker.step()
                if hasattr(influence_tracker, 'step_count'):
                    if influence_tracker.step_count % cfg.gradient_print_interval == 0:
                        influence_tracker.print_analysis_summary()
        else:
            pass
        if grad_tracker and grad_tracker.should_log():
            grad_metrics = grad_tracker.get_logging_metrics()
            for k, v in list(grad_metrics.items()):
                try:
                    grad_metrics[k] = float(v)
                except (ValueError, TypeError):
                    del grad_metrics[k]
            metrics.update(grad_metrics)
            # Mirror obs_grad from gradient attribution tracker as well
            mirrored = {}
            for name, val in list(grad_metrics.items()):
                if isinstance(name, str) and name.startswith(('obs_grad/', 'influence/', 'grad_attr/')):
                    prefix_removed = name.split('/', 1)[1] if '/' in name else name
                    mirrored['episode_extra_stats/obs_grad/' + prefix_removed] = float(val)
            if len(mirrored) > 0:
                metrics.update(mirrored)
                _last_obsgrad_from_grad = dict(grad_metrics)
        else:
            pass
        
        # Cast any remaining tensor/np values to Python scalars
        for k, v in list(metrics.items()):
            try:
                metrics[k] = float(v)
            except (ValueError, TypeError):
                try:
                    metrics[k] = int(v)
                except (ValueError, TypeError):
                    # Drop non-loggable values
                    del metrics[k]

        # Drop non-finite episode_extra_stats/* entries to prevent NaN/Inf propagation in W&B
        for k in list(metrics.keys()):
            if isinstance(k, str) and k.startswith('episode_extra_stats/'):
                v = metrics[k]
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    del metrics[k]
        
        # Remove any top-level curriculum/* metrics to avoid logging them (keep episode_extra_stats/*)
        for k in list(metrics.keys()):
            if isinstance(k, str) and k.startswith('curriculum/'):
                del metrics[k]
        
        # Define metrics with step mapping once
        import wandb
        if hasattr(wandb, 'define_metric'):
            wandb.define_metric('frames')
            # Namespace common custom groups (exclude top-level curriculum/*)
            for name in list(metrics.keys()):
                if name.startswith(('obs_grad/', 'influence/', 'gpu/', 'reward_breakdown/', 'episode_extra_stats/obs_grad/', 'episode_extra_stats/curriculum/')):
                    wandb.define_metric(name, step_metric='frames')
            # Ensure episode_extra_stats trajectory keys are tracked against frames
            for key in (
                'episode_extra_stats/path_efficiency',
                'episode_extra_stats/time_to_gate_steps',
                'episode_extra_stats/min_gate_distance',
                'episode_extra_stats/center_offset_success',
                'episode_extra_stats/height_offset_success',
            ):
                wandb.define_metric(key, step_metric='frames')
        
        if original_wandb_log:
            original_wandb_log(metrics, **kwargs)

    # Store original functions before monkey patching
    from sample_factory.algo.learning.learner import Learner
    original_learner_init = Learner.init
    original_learner_train = Learner.train

    # Tracker configuration
    tracker_config = {
        'log_interval': cfg.gradient_log_interval,
        'print_interval': cfg.gradient_print_interval,
    }
    grad_config = {
        'log_interval': cfg.gradient_log_interval,
        'print_interval': cfg.gradient_print_interval,
    }

    def enhanced_learner_init(self) -> None:
        """Enhanced learner init that attaches influence tracker to the model"""
        nonlocal influence_tracker
        nonlocal grad_tracker
        result = original_learner_init(self)
        
        print(f"🔧 Enhanced learner init called")
        print(f"🔧 actor_critic type: {type(self.actor_critic)}")
        print(f"🔧 actor_critic is None: {self.actor_critic is None}")
        
        if hasattr(self, 'actor_critic') and self.actor_critic is not None:
            print("🔧 Creating influence tracker with actual model...")
            print(f"🔧 Model structure: {self.actor_critic}")
            
            try:
                # Respect both CLI flag and environment override for enabling influence tracker
                import os as _os
                _env_inf = _os.getenv('SF_ENABLE_INFLUENCE_TRACKER')
                _enable_influence = bool(cfg.enable_gradient_monitoring)
                if _env_inf is not None:
                    _enable_influence = (str(_env_inf).lower() == 'true')
                if _enable_influence:
                    influence_tracker = create_influence_tracker(self.actor_critic, tracker_config)
                    print(f"🔧 Created influence tracker: {type(influence_tracker)}")
                    print(f"🔧 Influence tracker enabled: {influence_tracker.enabled if influence_tracker else 'None'}")
                    if influence_tracker and influence_tracker.enabled:
                        print("✅ Influence tracker successfully attached to model")
                        print(f"   🔍 Monitoring ALL observation components based on 150D observation structure:")
                        print(f"      • [0:3] Drone position | [3:9] Static camera pose | [9:12] Drone orientation")
                        print(f"      • [12:18] Velocities | [18:22] Actions | [22:86] Drone camera | [86:150] Static camera")
                        print(f"   📊 Complete 150D observation breakdown:")
                        print(f"      🎯 Spatial: 15D (position + pose + orientation)")
                        print(f"      ⚡ Kinematic: 10D (velocities + actions)")
                        print(f"      📹 Visual: 128D (dual camera VAE latents)")
                        print(f"   📊 Logging every {cfg.gradient_log_interval} steps")
                        print(f"   📺 Analysis summary every {cfg.gradient_print_interval} steps")
                    else:
                        print("❌ Failed to attach influence tracker")
                else:
                    influence_tracker = None
                    print("🚫 Influence tracker DISABLED (via flag/env)")
            except (ValueError, TypeError) as e:
                print(f"❌ Error creating influence tracker: {e}")
                influence_tracker = None
            
            # Optionally create gradient attribution tracker
            try:
                import os as _os
                _env_grad = _os.getenv('SF_ENABLE_GRAD_ATTR')
                _enable_grad = bool(cfg.enable_grad_attribution)
                if _env_grad is not None:
                    _enable_grad = (str(_env_grad).lower() == 'true')
                if _enable_grad:
                    grad_tracker = create_gradient_tracker(self.actor_critic, grad_config)
                    if grad_tracker and grad_tracker.enabled:
                        print("✅ Gradient attribution tracker successfully attached")
                    else:
                        print("❌ Failed to attach gradient attribution tracker")
                else:
                    grad_tracker = None
                    print("🚫 Gradient attribution DISABLED (via flag/env)")
            except (ValueError, TypeError) as e:
                print(f"❌ Error creating gradient attribution tracker: {e}")
                grad_tracker = None

            # Attach a small forward hook to the actor_critic input path to mirror the obs tensor and capture its grad
            try:
                tracker_ref = grad_tracker
                if tracker_ref and tracker_ref.enabled:
                    def _ac_forward_hook(mod, inp) -> None:
                        # Try to locate the 150D obs inside Sample Factory's normalized_obs_dict
                        try:
                            arg = inp[0] if isinstance(inp, tuple) and len(inp) > 0 else inp
                            # Case 1: dict input with key 'obs'
                            if isinstance(arg, dict):
                                t = arg.get('obs', None)
                                if torch.is_tensor(t) and t.dim() == 2 and t.shape[1] == 150:
                                    # Sanitize normalized obs before encoder to kill any non-finite values
                                    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                                    t = t.clamp_(-1e6, 1e6)
                                    x = t.detach().requires_grad_(True)
                                    if hasattr(self, '_grad_tracker') and self._grad_tracker:
                                        x.register_hook(lambda g: self._grad_tracker.consume_grad(g))
                                    # Stash proxy for backward hook
                                    mod._obs_proxy = x
                                    # Replace in a shallow-copied dict to avoid in-place side-effects
                                    new_arg = dict(arg)
                                    new_arg['obs'] = x
                                    if isinstance(inp, tuple):
                                        return (new_arg,) + tuple(inp[1:])
                                    else:
                                        return new_arg
                                return None
                            # Case 2: raw tensor input
                            if torch.is_tensor(arg) and arg.dim() == 2 and arg.shape[1] == 150:
                                t = arg
                                t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                                t = t.clamp_(-1e6, 1e6)
                                x = t.detach().requires_grad_(True)
                                if hasattr(self, '_grad_tracker') and self._grad_tracker:
                                    x.register_hook(lambda g: self._grad_tracker.consume_grad(g))
                                mod._obs_proxy = x
                                if isinstance(inp, tuple):
                                    lst = list(inp)
                                    lst[0] = x
                                    return tuple(lst)
                                else:
                                    return x
                            return None
                        except (KeyError, TypeError):
                            return None
                    # Prefer encoder if not scripted, else fall back to actor_critic root
                    target = getattr(self.actor_critic, 'encoder', None)
                    if target is None or hasattr(target, '_c') or 'ScriptModule' in str(type(target)):
                        target = self.actor_critic
                    self._grad_attr_forward_handle = target.register_forward_pre_hook(_ac_forward_hook)

                    def _ac_backward_hook(mod, grad_in, grad_out) -> None:
                        x = getattr(mod, '_obs_proxy', None)
                        if x is not None and x.grad is not None and hasattr(self, '_grad_tracker') and self._grad_tracker:
                            self._grad_tracker.consume_grad(x.grad)
                    self._grad_attr_backward_handle = target.register_full_backward_hook(_ac_backward_hook)
            except (KeyError, TypeError) as e:
                print(f"❌ Failed to attach actor_critic grad mirror hooks: {e}")
        else:
            print("🔧 Cannot create influence/gradient trackers - model not available")
            if hasattr(self, 'actor_critic'):
                print(f"   • actor_critic exists but is None: {self.actor_critic is None}")
            else:
                print("   • actor_critic attribute doesn't exist")
        
        # Store tracker references on learner for access in train method
        self._influence_tracker = influence_tracker
        self._grad_tracker = grad_tracker

        # One-time: emit initial curriculum keys only under episode_extra_stats/* to avoid top-level curriculum/*
        import wandb
        frames0 = int(self.train_step)
        curriculum_keys = [
            'curriculum/level','curriculum/progress','curriculum/success_rate','curriculum/crash_rate','curriculum/timeout_rate',
            'curriculum/obstacles_behind_gate','curriculum/total_assets','curriculum/max_level_reached',
            'curriculum/camera_gaussian_std','curriculum/camera_dropout_rate',
            'curriculum/camera_frame_dropout_drone_total','curriculum/camera_frame_dropout_static_total',
            'curriculum/camera_frame_freeze_drone','curriculum/camera_frame_blank_drone',
            'curriculum/camera_frame_freeze_static','curriculum/camera_frame_blank_static',
            'curriculum/camera_max_angle','curriculum/camera_current_angle',
            'curriculum/state_noise_drone_pos_std_m','curriculum/state_noise_drone_orient_std_deg',
            'curriculum/state_noise_static_pos_std_m','curriculum/state_noise_static_orient_std_deg',
            'curriculum/total_successes','curriculum/total_crashes','curriculum/total_timeouts',
        ]
        boot = {'frames': frames0}
        for k in curriculum_keys:
            boot[f'episode_extra_stats/{k}'] = 0.0
        wandb.log(boot, step=frames0)
        return result

    def enhanced_train(self, *args, **kwargs) -> None:
        """Enhanced train method that updates influence tracker"""
        # Log when training method is called
        current_step_before = self.train_step
        print(f"🔧 enhanced_train() called - current step BEFORE: {current_step_before}")
        
        result = original_learner_train(self, *args, **kwargs)
        
        current_step_after = self.train_step
        print(f"🔧 enhanced_train() finished - current step AFTER: {current_step_after}")
        
        # Learner-side W&B logging of curriculum level if present in episode stats
        try:
            import wandb
            frames = int(current_step_after)
            if hasattr(self, 'all_episodic_stats'):
                # Sample Factory aggregates episode stats; we can pull our injected keys if present
                # latest_stats is a dict of lists; take last value for curriculum/level if available
                latest = self.last_episodic_stats
                curr_level = None
                curr_level_minus_1 = None
                path_efficiency = None
                time_to_gate_steps = None
                min_gate_distance = None
                center_offset_success = None
                height_offset_success = None
                if isinstance(latest, dict):
                    # common aggregation stores arrays; try several namespaces
                    for key in ('curriculum/level', 'curriculum_level', 'episode_extra_stats/curriculum_level'):
                        if key in latest:
                            v = latest[key]
                            if isinstance(v, (list, tuple)) and len(v) > 0:
                                curr_level = float(v[-1])
                            elif isinstance(v, (int, float)):
                                curr_level = float(v)
                            break
                    # Also try to read the minus_1 variant
                    for key in ('curriculum/level_minus_1', 'curriculum_level_minus_1', 'episode_extra_stats/curriculum_level_minus_1'):
                        if key in latest:
                            v = latest[key]
                            if isinstance(v, (list, tuple)) and len(v) > 0:
                                curr_level_minus_1 = float(v[-1])
                            elif isinstance(v, (int, float)):
                                curr_level_minus_1 = float(v)
                            break
                    # Fetch episode_extra_stats trajectory metrics if available
                    def _get_last(key_name) -> None:
                        try:
                            if key_name in latest:
                                v = latest[key_name]
                                if isinstance(v, (list, tuple)) and len(v) > 0:
                                    return float(v[-1])
                                elif isinstance(v, (int, float)):
                                    return float(v)
                        except (ValueError, TypeError):
                            return None
                        return None
                    path_efficiency = _get_last('episode_extra_stats/path_efficiency') or _get_last('path_efficiency')
                    time_to_gate_steps = _get_last('episode_extra_stats/time_to_gate_steps') or _get_last('time_to_gate_steps')
                    min_gate_distance = _get_last('episode_extra_stats/min_gate_distance') or _get_last('min_gate_distance')
                    center_offset_success = _get_last('episode_extra_stats/center_offset_success') or _get_last('center_offset_success')
                    height_offset_success = _get_last('episode_extra_stats/height_offset_success') or _get_last('height_offset_success')
                # Do not log continuous curriculum/current_* or derived level values anymore
                curr_level = None
                curr_level_minus_1 = None
                try:
                    # Helper that tries multiple namespaces to find the latest value
                    def _get_last_with_prefixes(key_name: str) -> None:
                        v0 = _get_last(key_name)
                        if v0 is not None:
                            return v0
                        v1 = _get_last('episode_extra_stats/' + key_name)
                        if v1 is not None:
                            return v1
                        if key_name.startswith('curriculum/'):
                            bare = key_name.split('/', 1)[1]
                            v2 = _get_last(bare)
                            if v2 is not None:
                                return v2
                        return None
                    curriculum_keys = CURRICULUM_KEYS
                    cur_payload = {}
                    if isinstance(latest, dict):
                        for k in curriculum_keys:
                            v = _get_last_with_prefixes(k)
                            if v is not None:
                                # Log only under episode namespace
                                cur_payload[f'episode_extra_stats/{k}'] = v
                        # Derive total_resets if components available
                        def _pl_get(name: str) -> None:
                            if name in cur_payload:
                                return cur_payload[name]
                            if f'episode_extra_stats/{name}' in cur_payload:
                                return cur_payload[f'episode_extra_stats/{name}']
                            return None
                        ts = _pl_get('curriculum/total_successes')
                        tc = _pl_get('curriculum/total_crashes')
                        tt = _pl_get('curriculum/total_timeouts')
                        if ts is not None and tc is not None and tt is not None:
                            cur_payload['episode_extra_stats/curriculum/total_resets'] = float(ts + tc + tt)
                    if len(cur_payload) > 0:
                        # Remove any continuous current_* keys before logging
                        for k in list(cur_payload.keys()):
                            if isinstance(k, str) and (k.endswith('curriculum/current_level') or k.endswith('curriculum/current_level_minus_1') or k.endswith('curriculum/current_progress')):
                                del cur_payload[k]
                        if len(cur_payload) > 0:
                            cur_payload['frames'] = frames
                            wandb.log(cur_payload, step=frames)
                        first_keys = list(cur_payload.keys())[:6]
                        print(f"[W&B_DEBUG][learner] logged curriculum keys: {first_keys} ... (total {len(cur_payload)})")
                    # Update last-known curriculum values
                    for k in curriculum_keys:
                        if k in cur_payload:
                            _last_curriculum[k] = float(cur_payload[k])
                    # Emit continuous curriculum series each step, carrying forward last-known values
                    forward_payload = {'frames': frames}
                    for k in curriculum_keys:
                        if k in ('curriculum/current_level','curriculum/current_level_minus_1','curriculum/current_progress'):
                            continue
                        forward_payload[f'episode_extra_stats/{k}'] = float(_last_curriculum.get(k, 0.0))
                    if len(forward_payload) > 1:
                        wandb.log(forward_payload, step=frames)
                except RuntimeError:
                    pass
                # Log trajectory metrics if present (NaN will be ignored by W&B)
                traj_payload = {}
                # Prefer running means if available; fall back to base keys
                def _pref(keys) -> None:
                    for k in keys:
                        val = _get_last(k)
                        if val is not None:
                            return val
                    return None
                pe_out = _pref(['episode_extra_stats/path_efficiency_running_mean','path_efficiency_running_mean','episode_extra_stats/path_efficiency'])
                ttg_out = _pref(['episode_extra_stats/time_to_gate_running_mean','time_to_gate_running_mean','episode_extra_stats/time_to_gate_steps'])
                mgd_out = _pref(['episode_extra_stats/min_gate_distance_running_mean','min_gate_distance_running_mean','episode_extra_stats/min_gate_distance'])
                co_out  = _pref(['episode_extra_stats/center_offset_running_mean','center_offset_running_mean','episode_extra_stats/center_offset_success'])
                ho_out  = _pref(['episode_extra_stats/height_offset_running_mean','height_offset_running_mean','episode_extra_stats/height_offset_success'])
                pass_rate = _get_last('episode_extra_stats/gate_pass_rate')
                ep_total  = _get_last('episode_extra_stats/episodes_total')
                ep_cross  = _get_last('episode_extra_stats/episodes_crossed')
                if pe_out is not None:
                    traj_payload['episode_extra_stats/path_efficiency'] = pe_out
                # Only forward time_to_gate when it's defined (success-conditioned)
                if ttg_out is not None and math.isfinite(ttg_out):
                    traj_payload['episode_extra_stats/time_to_gate_steps'] = ttg_out
                if mgd_out is not None:
                    traj_payload['episode_extra_stats/min_gate_distance'] = mgd_out
                if co_out is not None:
                    traj_payload['episode_extra_stats/center_offset_success'] = co_out
                if ho_out is not None:
                    traj_payload['episode_extra_stats/height_offset_success'] = ho_out
                if pass_rate is not None:
                    traj_payload['episode_extra_stats/gate_pass_rate'] = pass_rate
                if ep_total is not None:
                    traj_payload['episode_extra_stats/episodes_total'] = ep_total
                if ep_cross is not None:
                    traj_payload['episode_extra_stats/episodes_crossed'] = ep_cross
                # Also forward VAE latent diagnostics if present in latest infos
                def _get_last_any(names) -> None:
                    for nm in names:
                        v = _get_last(nm)
                        if v is not None:
                            return v
                    return None
                for name in (
                    'episode_extra_stats/vae/drone_mean',
                    'episode_extra_stats/vae/static_mean',
                    'episode_extra_stats/vae/drone_std',
                    'episode_extra_stats/vae/static_std',
                    'episode_extra_stats/vae/drone_dim_std_mean',
                    'episode_extra_stats/vae/static_dim_std_mean',
                    'episode_extra_stats/vae/static_to_drone_norm_ratio',
                ):
                    v = _get_last_any([name, name.replace('episode_extra_stats/','')])
                    if v is not None:
                        traj_payload[name] = float(v)

                if len(traj_payload) > 0:
                    traj_payload['frames'] = frames
                    wandb.log(traj_payload, step=frames)
                    print(f"[W&B_DEBUG][learner] logged traj metrics keys: {list(traj_payload.keys())}")
        except RuntimeError:
            pass
        
        if hasattr(self, '_influence_tracker') and self._influence_tracker and self._influence_tracker.enabled:
            tracker = self._influence_tracker
            print(f"🔧 Calling tracker.step() - Sample Factory training step {current_step_after}")
            
            # Update tracker to use Sample Factory's actual step count
            tracker.step_count = current_step_after
            
            # Show data collection progress
            if hasattr(tracker, 'activation_history'):
                influence_samples = len(tracker.activation_history.get('drone_camera_vae', {}).get('correlations', []))
                print(f"🔧 Tracker synced to step {current_step_after} - collected {influence_samples} influence samples")
            
            # Print analysis at specified intervals
            if current_step_after > 0 and current_step_after % cfg.gradient_print_interval == 0:
                print(f"🔧 Influence analysis at training step {current_step_after}")
                tracker.print_analysis_summary()
        else:
            print(f"🔧 No influence tracker available for step {current_step_after}")
        
        # Optionally print gradient summary at intervals
        if hasattr(self, '_grad_tracker') and self._grad_tracker and self._grad_tracker.enabled:
            if current_step_after > 0 and current_step_after % cfg.gradient_print_interval == 0:
                print(f"🔧 Gradient attribution analysis at training step {current_step_after}")
                self._grad_tracker.print_gradient_summary()
        
        # Explicit obs-grad logging to W&B under episode_extra_stats (same path as other episodic metrics)
        try:
            import wandb
            frames = int(current_step_after)
            # Prefer influence tracker; fall back to grad tracker
            metric_sources = []
            if hasattr(self, '_influence_tracker') and self._influence_tracker and self._influence_tracker.enabled:
                metric_sources.append(self._influence_tracker.get_logging_metrics())
            if hasattr(self, '_grad_tracker') and self._grad_tracker and self._grad_tracker.enabled:
                metric_sources.append(self._grad_tracker.get_logging_metrics())
            merged = {}
            for src in metric_sources:
                for k, v in list(src.items()):
                    merged[k] = float(v)
            # Build payload like curriculum block
            obs_payload = {}
            # Mirror raw slice metrics
            for name, val in merged.items():
                if isinstance(name, str) and name.startswith(('obs_grad/', 'influence/', 'grad_attr/')):
                    key_tail = name.split('/', 1)[1] if '/' in name else name
                    obs_payload['episode_extra_stats/obs_grad/' + key_tail] = float(val)
            # Derived shares
            SLICE_NAMES_STATE = {'drone_position','static_camera_pos','static_camera_orient','drone_orientation','linear_vel','angular_vel','actions','drone_linear_vel','drone_angular_vel','drone_actions'}
            SLICE_NAMES_CAMERA = {'drone_camera_vae','static_camera_vae'}
            # Modality groups
            SLICE_NAMES_VISUAL = {'drone_camera_vae','static_camera_vae'}
            SLICE_NAMES_KINEMATIC = {'drone_linear_vel','drone_angular_vel','drone_actions','linear_vel','angular_vel','actions'}
            SLICE_NAMES_SPATIAL = {'drone_position','static_camera_pos','static_camera_orient','drone_orientation'}
            total_val = 0.0
            camera_val = 0.0
            state_val = 0.0
            # Window-specific totals for PURE recent/overall percentages
            total_recent = 0.0
            total_overall = 0.0
            camera_recent = 0.0
            state_recent = 0.0
            camera_overall = 0.0
            state_overall = 0.0
            # Modality windowed totals
            visual_recent = 0.0
            kinematic_recent = 0.0
            spatial_recent = 0.0
            visual_overall = 0.0
            kinematic_overall = 0.0
            spatial_overall = 0.0
            slice_vals = {}
            for name, val in merged.items():
                if not isinstance(name, str) or not name.startswith(('obs_grad/','influence/','grad_attr/')):
                    continue
                parts = name.split('/')
                suffix = parts[-1]
                if suffix.startswith('total_') or suffix == 'backward_passes':
                    continue
                try:
                    scalar = float(val)
                except (ValueError, TypeError):
                    continue
                total_val += scalar
                slice_vals[suffix] = scalar
                base = suffix
                for post in ['_mean_norm_recent','_mean_norm_overall','_recent','_overall','_mean_norm','_mean','_norm']:
                    if base.endswith(post):
                        base = base[: -len(post)]
                        break
                # Classify window
                is_recent = suffix.endswith('_mean_norm_recent') or suffix.endswith('_recent')
                is_overall = suffix.endswith('_mean_norm_overall') or suffix.endswith('_overall')
                if is_recent:
                    total_recent += scalar
                if is_overall:
                    total_overall += scalar
                if (base in SLICE_NAMES_CAMERA) or ('camera_vae' in base):
                    camera_val += scalar
                    if is_recent:
                        camera_recent += scalar
                    if is_overall:
                        camera_overall += scalar
                elif (base in SLICE_NAMES_STATE or
                      base.startswith('drone_position') or base.startswith('drone_orientation') or
                      base.startswith('drone_linear_vel') or base.startswith('drone_angular_vel') or
                      base.startswith('drone_actions') or
                      base.startswith('static_camera_pos') or base.startswith('static_camera_orient')):
                    state_val += scalar
                    if is_recent:
                        state_recent += scalar
                    if is_overall:
                        state_overall += scalar
                # Modality buckets
                if (base in SLICE_NAMES_VISUAL) or ('camera_vae' in base):
                    if is_recent:
                        visual_recent += scalar
                    if is_overall:
                        visual_overall += scalar
                if (base in SLICE_NAMES_KINEMATIC or
                    base.startswith('drone_linear_vel') or base.startswith('drone_angular_vel') or base.startswith('drone_actions')):
                    if is_recent:
                        kinematic_recent += scalar
                    if is_overall:
                        kinematic_overall += scalar
                if (base in SLICE_NAMES_SPATIAL or
                    base.startswith('drone_position') or base.startswith('drone_orientation') or
                    base.startswith('static_camera_pos') or base.startswith('static_camera_orient')):
                    if is_recent:
                        spatial_recent += scalar
                    if is_overall:
                        spatial_overall += scalar
            if total_val > 0.0:
                cam_share = camera_val / total_val
                if state_val <= 0.0 and camera_val > 0.0:
                    state_val = max(total_val - camera_val, 0.0)
                st_share = state_val / total_val
                obs_payload['episode_extra_stats/obs_grad/camera_share'] = float(cam_share)
                obs_payload['episode_extra_stats/obs_grad/state_share'] = float(st_share)
                obs_payload['episode_extra_stats/obs_grad/camera_share_pct'] = float(cam_share * 100.0)
                obs_payload['episode_extra_stats/obs_grad/state_share_pct'] = float(st_share * 100.0)
                # PURE recent/overall camera/state shares
                if total_recent > 0.0:
                    cam_share_r = camera_recent / total_recent
                    st_share_r = state_recent / total_recent
                    obs_payload['episode_extra_stats/obs_grad/camera_share_recent'] = float(cam_share_r)
                    obs_payload['episode_extra_stats/obs_grad/state_share_recent'] = float(st_share_r)
                    obs_payload['episode_extra_stats/obs_grad/camera_share_pct_recent'] = float(cam_share_r * 100.0)
                    obs_payload['episode_extra_stats/obs_grad/state_share_pct_recent'] = float(st_share_r * 100.0)
                if total_overall > 0.0:
                    cam_share_o = camera_overall / total_overall
                    st_share_o = state_overall / total_overall
                    obs_payload['episode_extra_stats/obs_grad/camera_share_overall'] = float(cam_share_o)
                    obs_payload['episode_extra_stats/obs_grad/state_share_overall'] = float(st_share_o)
                    obs_payload['episode_extra_stats/obs_grad/camera_share_pct_overall'] = float(cam_share_o * 100.0)
                    obs_payload['episode_extra_stats/obs_grad/state_share_pct_overall'] = float(st_share_o * 100.0)
                # PURE modality shares (recent/overall)
                if total_recent > 0.0:
                    vis_r = visual_recent / total_recent
                    kin_r = kinematic_recent / total_recent
                    spa_r = spatial_recent / total_recent
                    obs_payload['episode_extra_stats/obs_grad/visual_share_recent'] = float(vis_r)
                    obs_payload['episode_extra_stats/obs_grad/kinematic_share_recent'] = float(kin_r)
                    obs_payload['episode_extra_stats/obs_grad/spatial_share_recent'] = float(spa_r)
                    obs_payload['episode_extra_stats/obs_grad/visual_share_pct_recent'] = float(vis_r * 100.0)
                    obs_payload['episode_extra_stats/obs_grad/kinematic_share_pct_recent'] = float(kin_r * 100.0)
                    obs_payload['episode_extra_stats/obs_grad/spatial_share_pct_recent'] = float(spa_r * 100.0)
                if total_overall > 0.0:
                    vis_o = visual_overall / total_overall
                    kin_o = kinematic_overall / total_overall
                    spa_o = spatial_overall / total_overall
                    obs_payload['episode_extra_stats/obs_grad/visual_share_overall'] = float(vis_o)
                    obs_payload['episode_extra_stats/obs_grad/kinematic_share_overall'] = float(kin_o)
                    obs_payload['episode_extra_stats/obs_grad/spatial_share_overall'] = float(spa_o)
                    obs_payload['episode_extra_stats/obs_grad/visual_share_pct_overall'] = float(vis_o * 100.0)
                    obs_payload['episode_extra_stats/obs_grad/kinematic_share_pct_overall'] = float(kin_o * 100.0)
                    obs_payload['episode_extra_stats/obs_grad/spatial_share_pct_overall'] = float(spa_o * 100.0)
                for sfx, sval in slice_vals.items():
                    obs_payload[f'episode_extra_stats/obs_grad/slice_mag/{sfx}'] = float(sval)
                    # PURE recent/overall percentages by window-specific denominator
                    is_r = sfx.endswith('_mean_norm_recent') or sfx.endswith('_recent')
                    is_o = sfx.endswith('_mean_norm_overall') or sfx.endswith('_overall')
                    denom = total_recent if is_r and total_recent > 0.0 else (total_overall if is_o and total_overall > 0.0 else total_val)
                    obs_payload[f'episode_extra_stats/obs_grad/slice_pct/{sfx}'] = float((sval / denom) * 100.0)
            if len(obs_payload) > 0:
                obs_payload['frames'] = frames
                # Print the two target metrics for debugging
                ss = obs_payload.get('episode_extra_stats/obs_grad/state_share', None)
                ssp = obs_payload.get('episode_extra_stats/obs_grad/state_share_pct', None)
                if (ss is not None) or (ssp is not None):
                    print(f"[OBS_GRAD_DEBUG] frames={frames} state_share={ss if ss is not None else 'None'} state_share_pct={ssp if ssp is not None else 'None'}", flush=True)
                wandb.log(obs_payload, step=frames)
                print(f"[W&B_DEBUG][obs_grad] explicit_log keys={len(obs_payload)}", flush=True)
        except RuntimeError:
            pass
        
        return result

    # Apply monkey patches
    Learner.init = enhanced_learner_init
    Learner.train = enhanced_train

    # Apply wandb monkey patch if available
    if original_wandb_log:
        import wandb
        wandb.log = enhanced_wandb_log

    try:
        print("🚀 Starting enhanced training with complete observation influence tracking...")
        result = run_rl(cfg)
        
        # Final analysis and cleanup
        print("=" * 80)
        print("🎯 COMPLETE OBSERVATION INFLUENCE ANALYSIS - FINAL SUMMARY")
        print("=" * 80)
        
        if influence_tracker:
            print(f"📊 Training completed with {influence_tracker.step_count} analysis steps")
            influence_tracker.print_analysis_summary()
            influence_tracker.cleanup()
        else:
            print("❌ No influence tracker was created - analysis unavailable")
        
        # Gradient attribution final summary
        print("=" * 80)
        print("🧮 GRADIENT-BASED ATTRIBUTION - FINAL SUMMARY")
        print("=" * 80)
        if grad_tracker:
            grad_tracker.print_gradient_summary()
            grad_tracker.cleanup()
        else:
            print("❌ No gradient attribution tracker was created - analysis unavailable")
            
        print("=" * 80)
        
        return result
        
    finally:
        # Restore original functions
        Learner.init = original_learner_init
        Learner.train = original_learner_train
        # Remove grad mirror hooks if present
        if hasattr(self, '_grad_attr_forward_handle'):
            self._grad_attr_forward_handle.remove()
        if hasattr(self, '_grad_attr_backward_handle'):
            self._grad_attr_backward_handle.remove()
        if original_wandb_log:
            import wandb
            wandb.log = original_wandb_log
