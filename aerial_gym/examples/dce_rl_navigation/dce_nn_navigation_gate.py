import isaacgym  # noqa: F401 (ensures gym is loaded)

import torch
import os
from aerial_gym.utils import get_args
from aerial_gym.registry.task_registry import task_registry

from aerial_gym.examples.dce_rl_navigation.dce_navigation_task_gate import (
    DCE_RL_Navigation_Task_Gate,
)
from aerial_gym.examples.dce_rl_navigation.sf_inference_class_gate import (
    NN_Inference_Class_Gate,
)
from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net_gate import (
    parse_aerialgym_cfg,
)


def main():
    args = get_args()
    headless = getattr(args, "headless", False)
    print(f"DCE Gate Inference - Headless mode: {headless}")

    # Build eval cfg and apply static camera overrides to task config before registering
    cfg = parse_aerialgym_cfg(evaluation=True)
    # Instantiate task_config the same way as training does
    task_config_class = task_registry.get_task_config("navigation_task_gate")
    base_task_config = task_config_class()
    try:
        # Env count
        if hasattr(cfg, "env_agents") and cfg.env_agents:
            base_task_config.num_envs = int(cfg.env_agents)
            os.environ["SF_ENV_AGENTS"] = str(int(cfg.env_agents))
        # Apply viewer/headless explicitly (training script exports SF_HEADLESS; here set config directly too)
        if hasattr(base_task_config, "headless"):
            base_task_config.headless = bool(headless)
        # Gate/obstacle overrides (mirror training subprocess setup)
        if hasattr(cfg, "disable_gate_size_randomization"):
            base_task_config.disable_gate_size_randomization = bool(getattr(cfg, "disable_gate_size_randomization", False))
        if hasattr(cfg, "fixed_gate_scale_percent"):
            base_task_config.fixed_gate_scale_percent = int(getattr(cfg, "fixed_gate_scale_percent", 100))
        if hasattr(cfg, "disable_obstacle_randomization"):
            base_task_config.disable_obstacle_randomization = bool(getattr(cfg, "disable_obstacle_randomization", False))
        if hasattr(cfg, "fixed_obstacles_behind_gate"):
            base_task_config.fixed_obstacles_behind_gate = int(getattr(cfg, "fixed_obstacles_behind_gate", 0))
        # Static camera control flags
        if hasattr(base_task_config, "static_camera_yaw_sweep_enabled"):
            base_task_config.static_camera_yaw_sweep_enabled = bool(getattr(cfg, "enable_static_camera_yaw_sweep", False))
        if hasattr(base_task_config, "enable_static_camera_yaw_sweep"):
            base_task_config.enable_static_camera_yaw_sweep = bool(getattr(cfg, "enable_static_camera_yaw_sweep", False))
        if hasattr(base_task_config, "static_camera_yaw_sweep_speed_deg"):
            base_task_config.static_camera_yaw_sweep_speed_deg = float(getattr(cfg, "static_camera_yaw_sweep_speed_deg", 180.0))
        # Base position overrides
        if hasattr(base_task_config, "static_camera_base_y"):
            base_task_config.static_camera_base_y = float(getattr(cfg, "static_camera_base_y", -3.0))
        if hasattr(base_task_config, "static_camera_base_z"):
            base_task_config.static_camera_base_z = getattr(cfg, "static_camera_base_z", "fixed")
        # Orientation randomization toggle
        if hasattr(base_task_config, "disable_static_camera_orientation_randomization"):
            base_task_config.disable_static_camera_orientation_randomization = bool(getattr(cfg, "disable_static_camera_orientation_randomization", False))
        # Dynamic camera following toggles
        try:
            cur = getattr(base_task_config, 'curriculum', None)
            if cur is not None:
                if hasattr(cfg, 'disable_dynamic_camera_following') and bool(getattr(cfg, 'disable_dynamic_camera_following', False)):
                    setattr(cur, 'enable_dynamic_camera_following', False)
                if hasattr(cfg, 'enable_dynamic_camera_following') and getattr(cfg, 'enable_dynamic_camera_following') is not None:
                    setattr(cur, 'enable_dynamic_camera_following', bool(getattr(cfg, 'enable_dynamic_camera_following')))
        except Exception:
            pass
    except Exception:
        pass

    # Register gate task for inference with updated config
    task_registry.register_task(
        task_name="dce_navigation_task_gate",
        task_class=DCE_RL_Navigation_Task_Gate,
        task_config=base_task_config,
    )

    # Build env (respect CLI seed like training)
    seed_val = None
    try:
        if hasattr(cfg, "seed") and cfg.seed is not None:
            seed_val = int(cfg.seed)
    except Exception:
        seed_val = None
    rl_task = task_registry.make_task(
        "dce_navigation_task_gate", seed=seed_val, use_warp=True, headless=headless
    )
    print("Number of environments", rl_task.num_envs)

    # Set compat attribute on the task
    try:
        setattr(rl_task, "sf_cfg", cfg)
    except Exception:
        pass

    # obs/action dims for gate task
    num_actions = rl_task.task_config.action_space_dim
    num_obs = rl_task.task_config.observation_space_dim

    # Inference wrapper uses Sample Factory checkpoints/config
    nn_model = NN_Inference_Class_Gate(rl_task.num_envs, num_actions, num_obs, cfg)
    nn_model.eval()
    nn_model.reset(torch.arange(rl_task.num_envs))

    with torch.no_grad():
        reset_result = rl_task.reset()
        obs_dict = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        # Determine the observation key the model expects
        nn_obs_key = getattr(getattr(nn_model, "cfg", object()), "obs_key", "obs")
        # Apply the same observation ablation used during training if requested via env vars
        def _apply_obs_ablation(obs_tensor: torch.Tensor) -> torch.Tensor:
            import os as _os
            if obs_tensor is None:
                return obs_tensor
            # Work on a private copy to avoid aliasing/view issues
            obs_tensor = obs_tensor.clone()
            debug = _os.environ.get("ABLATE_DEBUG", "false").lower() == "true"
            # Simple switch for drone position (parity with training wrapper)
            if _os.environ.get("ABLATE_DRONE_POS", "false").lower() == "true":
                start, end = 0, 3
                obs_tensor[:, start:end] = 0.0
                if debug:
                    v = obs_tensor[:, start:end]
                    print(f"[ABLATE_DEBUG] applied: {start}:{end}=zero | min={v.min().item():.3e} max={v.max().item():.3e} mean={v.mean().item():.3e} nonzero={int((v!=0).sum().item())}")
            # General ranges
            spec_str = _os.environ.get("ABLATE_OBS_RANGES", "").strip()
            if not spec_str:
                return obs_tensor
            grad_mask = None
            zero_ranges = []
            zerograd_ranges = []
            _dbg_count = 0
            for spec in spec_str.split(","):
                spec = spec.strip()
                if not spec:
                    continue
                if "=" not in spec:
                    continue
                lhs, rhs = spec.split("=", 1)
                lhs = lhs.strip(); rhs = rhs.strip()
                if ":" not in lhs:
                    continue
                try:
                    start_s, end_s = lhs.split(":", 1)
                    start = int(start_s); end = int(end_s)
                except Exception:
                    continue
                op = rhs
                if op == "zero":
                    if grad_mask is None:
                        grad_mask = torch.ones_like(obs_tensor)
                    grad_mask[:, start:end] = 0.0
                    zero_ranges.append((start, end))
                elif op == "zerograd":
                    zerograd_ranges.append((start, end))
                elif op == "shuffle":
                    if obs_tensor.shape[0] > 1:
                        perm = torch.randperm(obs_tensor.shape[0], device=obs_tensor.device)
                        obs_tensor[:, start:end] = obs_tensor[perm, start:end]
                    if debug and _dbg_count < 10:
                        v = obs_tensor[:, start:end]
                        print(f"[ABLATE_DEBUG] applied: {start}:{end}=shuffle | sample_env0={v[0].detach().cpu().numpy()} sample_env1={v[1].detach().cpu().numpy() if v.shape[0]>1 else 'NA'}")
                        _dbg_count += 1
                elif op.startswith("noise:"):
                    try:
                        std = float(op.split(":", 1)[1])
                    except Exception:
                        std = 0.0
                    if std > 0.0:
                        obs_tensor[:, start:end] = obs_tensor[:, start:end] + torch.randn_like(obs_tensor[:, start:end]) * std
                    if debug and _dbg_count < 10:
                        v = obs_tensor[:, start:end]
                        print(f"[ABLATE_DEBUG] applied: {start}:{end}=noise:{std} | std_est={v.std().item():.3e} mean={v.mean().item():.3e}")
                        _dbg_count += 1
            if grad_mask is not None:
                obs_tensor = obs_tensor * grad_mask
                if debug:
                    for (zs, ze) in zero_ranges:
                        if _dbg_count >= 10:
                            break
                        v = obs_tensor[:, zs:ze]
                        print(f"[ABLATE_DEBUG] applied: {zs}:{ze}=zero | min={v.min().item():.3e} max={v.max().item():.3e} mean={v.mean().item():.3e} nonzero={int((v!=0).sum().item())}")
                        _dbg_count += 1
            for (start, end) in zerograd_ranges:
                zero_slice = torch.zeros_like(obs_tensor[:, start:end])
                left = obs_tensor[:, :start]
                right = obs_tensor[:, end:]
                obs_tensor = torch.cat([left, zero_slice, right], dim=-1)
                if debug and _dbg_count < 10:
                    v = obs_tensor[:, start:end]
                    print(f"[ABLATE_DEBUG] applied: {start}:{end}=zerograd | min={v.min().item():.3e} max={v.max().item():.3e} mean={v.mean().item():.3e} nonzero={int((v!=0).sum().item())}")
                    _dbg_count += 1
            return obs_tensor
        while True:
            vec = obs_dict["observations"] if isinstance(obs_dict, dict) and "observations" in obs_dict else obs_dict["obs"]
            # Apply ablation if configured
            vec = _apply_obs_ablation(vec)
            model_obs = {nn_obs_key: vec}
            actions = nn_model.get_action(model_obs)
            step_result = rl_task.step(actions)
            obs_dict = step_result[0] if isinstance(step_result, tuple) else step_result


if __name__ == "__main__":
    main()


