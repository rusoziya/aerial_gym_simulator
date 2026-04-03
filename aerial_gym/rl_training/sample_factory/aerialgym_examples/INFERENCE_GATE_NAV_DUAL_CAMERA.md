## Gate Navigation Inference (Dual Camera)

This guide explains how to run inference for the gate‑navigation policy using the dual‑camera (drone + static/peer) setup.

## Prerequisites
- NVIDIA GPU + CUDA; Isaac Gym installed and importable.
- Aerial Gym + dependencies installed in your Python environment.
- A trained policy checkpoint (either via this repo’s training or provided).

## Script location
The main inference entry for gate navigation is:
```bash
aerial_gym/examples/dce_rl_navigation/dce_nn_navigation_gate.py
```
This script will register the gate task for evaluation, load a Sample‑Factory checkpoint (either from `train_dir/experiment` or from `DCE_MODEL`), and run evaluation episodes with optional W&B logging and GIF saving.

## Quick start (checkpoint provided)
```bash
# Example: 512 episodes, 32 envs, forced Level 23, GIFs ON, W&B disabled (local only)
WANDB_DISABLED=true WANDB_MODE=online \
ABLATE_DEBUG=true ABLATE_OBS_RANGES='0:22=zero' \
DCE_MODEL="/path/to/best_or_checkpoint.pth" \
python -B -m aerial_gym.examples.dce_rl_navigation.dce_nn_navigation_gate \
  --env dce_navigation_task_gate \
  --train_dir=/path/to/train_dir --experiment=selected_network \
  --seed=123 --env_agents=32 --max_num_episodes=512 \
  --eval_deterministic=true --save_gifs=true --headless=false \
  --force_curriculum_level=23 \
  --disable_curriculum_multiplier=true \
  --disable_gate_size_randomization=false --fixed_gate_scale_percent=100 \
  --disable_obstacle_randomization=false --fixed_obstacles_behind_gate=1 \
  --disable_static_camera_orientation_randomization=false \
  --disable_drone_camera_noise_randomization=false --disable_static_camera_noise_randomization=false \
  --disable_drone_camera_frame_dropout=false --disable_static_camera_frame_dropout=false \
  --disable_state_noise_randomization=false \
  --disable_spawn_position_randomization=false --disable_spawn_orientation_randomization=false \
  --enable_dynamic_camera_following=false --enable_static_camera_yaw_sweep=false \
  --static_camera_yaw_sweep_speed_deg=180 --static_camera_base_y=-2.0 --static_camera_base_z=adaptive \
  --enable_static_camera_locked=false --enable_static_camera_arc_follow=false \
  --dynamic_camera_follow_y_offset_m=-2.0
```

## Getting a checkpoint
You need one of the following:
- A `DCE_MODEL` file (e.g., `best_*.pth` or `checkpoint_*.pth`) from `train_dir/<EXPERIMENT_NAME>/policy_*/`.
- Or a `--train_dir` and `--experiment` pair pointing to a folder that contains checkpoints and the saved config (the script will auto‑load the latest/best).

Tip: If you copy a single `.pth` file to a new machine, prefer also copying the run folder containing the saved configuration to ensure normalizer/config parity.

## Using a trained experiment directory instead of `DCE_MODEL`
If `DCE_MODEL` is not provided, the script loads from `--train_dir` + `--experiment` using the last or best checkpoint depending on configuration. Example:
```bash
python -B -m aerial_gym.examples.dce_rl_navigation.dce_nn_navigation_gate \
  --env dce_navigation_task_gate --train_dir=./train_dir --experiment=my_run \
  --env_agents=32 --max_num_episodes=256 --eval_deterministic=true --save_gifs=false --headless=true
```

## Core CLI flags (inference)
- `--env dce_navigation_task_gate`: use gate task.
- `--train_dir`, `--experiment`: where the training run lives (used if `DCE_MODEL` unset).
- `--seed`: evaluation seed for env/task.
- `--env_agents`: number of parallel envs to evaluate.
- `--max_num_episodes`: stop after this many episode resets across envs.
- `--eval_deterministic=true|false`: greedy actions (argmax) vs sampling.
- `--save_gifs=true|false`: save per‑episode GIFs (drone/static variants).
- `--headless=true|false`: show viewer or run headless.
- `--force_curriculum_level=<int|none>`: force difficulty level (e.g., 23 or 33); use `none` to disable forcing.

## Environment variables (common)
- `DCE_MODEL=/path/to/checkpoint.pth`: explicit policy file.
- `WANDB_DISABLED=false|true`, `WANDB_MODE=online|offline`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME`.
- `ABLATE_OBS_RANGES='start:end=op,...'`: apply observation ablation (ops: `zero`, `zerograd`, `shuffle`, `noise:<std>`). Example: `0:22=zero` removes non‑visual state; `22:86=zero` exo‑only; `86:150=zero` ego‑only.
- `ABLATE_DEBUG=true`: print ablation diagnostics.
- `RNN_WARMUP_STEPS=<int>`: warm‑up steps to prime the GRU before logging (alternative to CLI flag).
- `EVAL_STRETCH_ENABLED=1`, `EVAL_STRETCH_END_LEVEL=33`: stretch curriculum schedules during evaluation.
- `CUBLAS_WORKSPACE_CONFIG=:16:8`: deterministic cuBLAS (optional).

Additional toggles (debug/diagnostics):
- `DUMP_OBS_PARITY=true`, `OBS_PARITY_STEPS=<N>`: dump per‑step latent magnitudes for quick train–infer parity checks.
- `VISIBILITY_DEBUG=true`: print visibility/FOV summaries when available.
- `PRINT_ENV0_LATENTS_ONCE=true`, `PRINT_ENV0_LATENTS_ONCE_NORM=true`: one‑episode latent logs for env 0.

## Additional CLI parameters (from script)
The inference script also accepts these overrides via CLI:
- `--run_name`: W&B run name.
- `--wandb_project`, `--wandb_entity`, `--wandb_dir`: W&B configuration.
- `--rnn_warmup_steps`: warm‑up (burn‑in) steps before measurements.
- Static camera pose/motion overrides:
  - `--enable_static_camera_arc_follow=true|false`, `--static_camera_arc_radius_m=<float>`
  - `--dynamic_camera_follow_y_offset_m=<float>` (only if dynamic follow is active)
  - `--disable_dynamic_follow_gate_blending=true|false` (blending toward gate vs pure drone focus)

## RNN warm‑up (optional but recommended)
Because the policy is recurrent, a short burn‑in can stabilize evaluation:
```bash
python -B -m aerial_gym.examples.dce_rl_navigation.dce_nn_navigation_gate ... --rnn_warmup_steps=50
# or: export RNN_WARMUP_STEPS=50
```
During warm‑up, episodes are not counted and RNN states are reset only per‑env when that env finishes.

## Observation layout and ablation indices (150D)
- `0:3` position, `3:6` static‑cam pos (rel), `6:9` static‑cam orient (rel), `9:12` drone orient,
  `12:15` linear vel, `15:18` angular vel, `18:22` recent actions, `22:86` drone VAE (64D), `86:150` static VAE (64D).
- Ego‑only: `86:150=zero`; Exo‑only: `22:86=zero`; pure vision: `0:22=zero`.

## Outputs
- GIFs: saved under `examples/dce_rl_navigation/gif_episodes` when `--save_gifs=true`.
- W&B (optional): logs curriculum snapshots, per‑reset metrics (path efficiency, min gate distance, offsets), throughput, actions, and derived running means keyed by `episodes`.
- Console: summary prints and optional debug taps (normalization, encoder taps, ablation debug, obs‑parity).

Where outputs live:
- Checkpoints are read from `DCE_MODEL` or discovered under `--train_dir/--experiment`.
- GIFs (if enabled) are saved alongside the inference script directory.
- W&B runs (if enabled) appear under your project/entity (online) or are cached locally (offline mode).

## Examples
1) Deterministic evaluation at Level 33 with exo‑only ablation and 16 envs:
```bash
ABLATE_OBS_RANGES='22:86=zero' WANDB_DISABLED=true \
python -B -m aerial_gym.examples.dce_rl_navigation.dce_nn_navigation_gate \
  --env dce_navigation_task_gate --env_agents=16 --max_num_episodes=256 \
  --eval_deterministic=true --save_gifs=false --headless=true --force_curriculum_level=33
```
2) With W&B online logging and run naming:
```bash
WANDB_DISABLED=false WANDB_MODE=online WANDB_PROJECT=final_inference WANDB_RUN_NAME=my_eval \
python -B -m aerial_gym.examples.dce_rl_navigation.dce_nn_navigation_gate \
  --env dce_navigation_task_gate --env_agents=32 --max_num_episodes=512 --eval_deterministic=true
```

## Troubleshooting
- Viewer/Isaac Gym issues: use `--headless=true` for batch inference; ensure only one process opens a viewer.
- No checkpoint found: set `DCE_MODEL` or verify `--train_dir` and `--experiment` point to a run with checkpoints.
- Slow FPS: reduce `--env_agents`, disable `--save_gifs`, set `WANDB_DISABLED=true` for local testing.
- Percentages not summing to 100% in fusion/obs‑grad: a small residual may reside in the 22D non‑visual state and rounding contributes.

Compatibility/pitfalls:
- Architecture mismatch (obs/action dims): ensure the model was trained for gate navigation (150D obs, 4D action, DualFusionEncoder). Mismatches will raise shape errors.
- Normalizer/config coupling: the script attempts to load the saved SF config (normalizers included). If you only provide `DCE_MODEL`, but not the run folder, ensure normalization behavior matches your training; otherwise prefer providing `--train_dir/--experiment` too.
- GPU vs CPU: the wrapper moves the actor‑critic to CUDA if available. If you need CPU‑only, set `CUDA_VISIBLE_DEVICES=""` before running.
- Env count vs VRAM: start with `--env_agents=16` on 8–12 GB GPUs; scale to 32–64 on 24 GB+.

## Notes on metrics
- The script defines W&B metrics for curriculum state, episode‑extra stats, action stats, throughput, visibility/FOV running means, and per‑episode success (binary) aligned by episodes. These match training‑time conventions to simplify comparison.

## Common recipes
- Ego‑only (drop static VAE) at Level 33:
```bash
ABLATE_OBS_RANGES='86:150=zero' \
python -B -m aerial_gym.examples.dce_rl_navigation.dce_nn_navigation_gate \
  --env dce_navigation_task_gate --env_agents=16 --max_num_episodes=256 \
  --eval_deterministic=true --save_gifs=false --headless=true --force_curriculum_level=33
```
- Exo‑only (drop drone VAE) at Level 33:
```bash
ABLATE_OBS_RANGES='22:86=zero' \
python -B -m aerial_gym.examples.dce_rl_navigation.dce_nn_navigation_gate \
  --env dce_navigation_task_gate --env_agents=16 --max_num_episodes=256 \
  --eval_deterministic=true --save_gifs=false --headless=true --force_curriculum_level=33
```
- Asymmetric noise test (Level 33): leave one stream at L33, approximate a clean stream via curriculum flags or task overrides.
  - Example idea: keep defaults, but set `--disable_static_camera_orientation_randomization=true` and lower obstacle count to probe sensitivity.

## Reproducibility
- Fix `--seed`, `--env_agents`, `--max_num_episodes`, and curriculum forcing where relevant.
- Use `WANDB_RUN_NAME` plus `WANDB_PROJECT/ENTITY` for run tracking (or `WANDB_DISABLED=true` to keep local only).
- Keep exact `DCE_MODEL` and, if possible, the matching run folder (config/normalizers) to avoid drift.

## End‑to‑end workflow (summary)
1) Obtain a trained checkpoint (and preferably the run folder with config).
2) Pick the evaluation difficulty: `--force_curriculum_level=23|33|none`.
3) Choose deterministic (`--eval_deterministic=true`) or stochastic actions.
4) Set `--env_agents` based on VRAM; start with 16.
5) (Optional) Enable GIFs and W&B; set `--save_gifs=true`, `WANDB_DISABLED=false`.
6) Run the command and inspect logs/GIFs/W&B charts.

## Deterministic vs stochastic
- Deterministic (`true`): uses greedy actions (argmax); good for reporting, reduces variance.
- Stochastic (`false`): samples from the policy distribution; useful to probe uncertainty or robustness.

## GIF outputs and naming
- When `--save_gifs=true`, the script saves per‑episode GIFs under `examples/dce_rl_navigation/gif_episodes`.
- Typical files include drone D455‑noised depth (`*_drone_depth_D455_NOISED.gif`) and static D455‑noised depth (`*_static_depth_D455_NOISED.gif`). Segmentation GIFs may be saved when available.
- Frequency: saved on env 0 resets; for large batches, consider headless evaluation and lower `--env_agents` to control disk I/O.

## W&B metric groups (overview)
- Curriculum: `curriculum/level`, `curriculum/progress`, totals/rates (`success_rate`, `crash_rate`, `timeout_rate`).
- Episode stats: `episode_extra_stats/path_efficiency`, `.../time_to_gate_steps`, `.../min_gate_distance`, center/height offsets.
- Actions: `action_abs_mean/*`, `action_diff_mean/*`, `action_saturation_rate`.
- Throughput: `throughput/fps_env`, `throughput/episodes_per_min`.
- Visibility/FOV (running): `visibility/*`, `fov/*` (when available).
- Success alignment: `episodes` (primary x‑axis), `episodes/success_binary` (0/1 per reset), `episodes/success_rate_batch`.

## Normalization & compatibility
- The wrapper attempts to load the saved Sample‑Factory config (including normalizer stats) via `load_from_checkpoint`; this preserves training‑time normalization.
- If only `DCE_MODEL` is provided without the saved config, ensure normalization behavior matches training; otherwise prefer supplying `--train_dir/--experiment` too.
- Debug taps: `DISABLE_NORM=true` bypasses normalization (for diagnostics); `NORM_TAP_DEBUG=true` prints pre/post latent magnitudes.

## Camera behavior overrides (at inference)
- Arc follow: `--enable_static_camera_arc_follow=true --static_camera_arc_radius_m=<r>`.
- Dynamic follow offset (active only if dynamic follow is enabled in config): `--dynamic_camera_follow_y_offset_m=<m>`.
- Disable dynamic‑follow gate blending: `--disable_dynamic_follow_gate_blending=true`.
- Static yaw sweep: `--enable_static_camera_yaw_sweep=true --static_camera_yaw_sweep_speed_deg=<deg/s>`.
- Base placement: `--static_camera_base_y=<m> --static_camera_base_z=<float|adaptive>`.

## CPU/headless usage
- To force CPU only: `export CUDA_VISIBLE_DEVICES=""` before running (slower).
- Batch mode (faster): set `--headless=true` and `WANDB_DISABLED=true` during large runs; re‑enable W&B for final reporting.

## Batch evaluation example
Evaluate multiple seeds for the same checkpoint (bash loop):
```bash
for SEED in 0 1 2; do
  WANDB_DISABLED=true \
  python -B -m aerial_gym.examples.dce_rl_navigation.dce_nn_navigation_gate \
    --env dce_navigation_task_gate --env_agents=16 --max_num_episodes=256 \
    --eval_deterministic=true --save_gifs=false --headless=true \
    --force_curriculum_level=33 --seed=${SEED}
done
```

## Common errors and fixes
- Shape mismatch (obs/action): checkpoint not trained for gate (150D/4D) → use a gate‑trained model.
- Isaac Gym import error: ensure Isaac Gym is installed in the active env; test with `python -c 'import isaacgym'`.
- VRAM OOM: reduce `--env_agents`, disable GIFs, or run headless.
- No checkpoints found: verify `DCE_MODEL` path or `--train_dir/--experiment` contents.

## Performance guidance (rule‑of‑thumb)
- 8–12 GB VRAM: `--env_agents=16`, headless, W&B off, GIFs off.
- 24 GB VRAM+: `--env_agents=32–64`, headless; re‑enable W&B for final runs.
- Viewer + GIFs: cut envs by half vs headless baseline.

## Reference for the example command (env vars + flags)
Below maps each element of the example you provided to its effect during inference.

- `DISABLE_WALLS=false`: scenario toggle (if supported by the task); when `true` walls may be removed. If unsupported, it is ignored.
- `VISIBILITY_DEBUG=true`: print visibility/FOV diagnostics when available from the task (helps explain failures vs occlusion/angles).
- `PRINT_ENV0_LATENTS_ONCE_NORM=false`, `PRINT_ENV0_LATENTS_ONCE=false`: one‑episode logging of raw vs normalized latents for env 0; set to `true` to dump once.
- `DISABLE_NORM=false`, `NORM_TAP_DEBUG=false`: bypass/inspect Sample Factory’s observation normalization inside the model; use with care to match training behavior.
- `ENC_TAP_DEBUG=false`: taps encoder inputs to print latent magnitudes periodically (for model sanity checks).
- `EVAL_STRETCH_ENABLED=1`, `EVAL_STRETCH_END_LEVEL=33`: stretch curriculum schedules beyond training cap (e.g., extrapolate noise/dropout from 23→33) for evaluation.
- `WANDB_PROJECT=final_inference`, `WANDB_DISABLED=true`, `WANDB_MODE=online`, `WANDB_RUN_NAME=locked_yaw_L23_SEED123`: W&B configuration; with `WANDB_DISABLED=true` nothing is uploaded (local only), `RUN_NAME` is kept for consistency.
- `DCE_MODEL="/path/to/checkpoint.pth"`: explicit policy checkpoint file to load (bypasses discovery in `train_dir`). Ensure path is correct.
- `ABLATE_DEBUG=true`: enables detailed printouts when applying observation ablation.
- `ABLATE_ZERO_RNN=false`: experimental; when supported, zeros the RNN hidden state periodically or at boundaries. If not wired in your build, it is ignored.
- `ABLATE_OBS_RANGES='0:22=zero'`: sets the 22D non‑visual state to zero (pure‑vision evaluation). Common alternatives: `22:86=zero` (exo‑only), `86:150=zero` (ego‑only).
- `CUBLAS_WORKSPACE_CONFIG=:16:8`: deterministic cuBLAS matmul kernels (optional, minor perf cost).

CLI flags:
- `--env dce_navigation_task_gate`: selects the gate navigation task.
- `--train_dir=... --experiment=...`: run directory to discover checkpoints and the saved config; ignored if `DCE_MODEL` is set.
- `--seed=123`: evaluation seed; keeps randomization repeatable (spawns, noise seeds, etc.).
- `--env_agents=32`: number of parallel environments; increase for throughput, decrease if VRAM limited.
- `--max_num_episodes=512`: stop after this many resets across envs.
- `--eval_deterministic=true`: greedy (argmax) actions; set `false` to sample from the policy distribution.
- `--save_gifs=true`: save depth (and segmentation, if available) GIFs; stored alongside the inference script.
- `--headless=false`: open the Isaac Gym viewer; set `true` for headless batch evaluation (faster).
- `--disable_curriculum_multiplier=true`: disables curriculum reward multiplier (alignment with training config if needed).
- `--disable_gate_size_randomization=false --fixed_gate_scale_percent=100`: keep gate size fixed at 100%.
- `--disable_obstacle_randomization=false --fixed_obstacles_behind_gate=1`: allow obstacle randomization but force at least one object (example setting).
- `--disable_static_camera_orientation_randomization=false`: allow orientation randomization of the static camera.
- `--disable_drone_camera_noise_randomization=false --disable_static_camera_noise_randomization=false`: keep camera noise randomization enabled (no disabling).
- `--disable_drone_camera_frame_dropout=false --disable_static_camera_frame_dropout=false`: keep entire‑frame dropout randomization enabled.
- `--disable_state_noise_randomization=false`: keep state/pose noise enabled.
- `--force_curriculum_level=23`: force all envs to Level 23 (fixed difficulty). Use `none` to disable forcing.
- `--disable_spawn_position_randomization=false --disable_spawn_orientation_randomization=false`: allow spawn randomization.
- `--enable_dynamic_camera_following=false`: keep camera static (no pursuit); toggles dynamic following when `true`.
- `--enable_static_camera_yaw_sweep=false --static_camera_yaw_sweep_speed_deg=180`: disable periodic yaw sweep; when `true`, sets sweep speed.
- `--static_camera_base_y=-2.0 --static_camera_base_z=adaptive`: place static camera behind the gate on Y; adapt Z to gate height.
- `--enable_static_camera_locked=false`: when `true`, lock the camera to center the drone (no sweeping).
- `--enable_static_camera_arc_follow=false`: disabled; when `true`, move static camera on an arc (set radius via `--static_camera_arc_radius_m`).
- `--dynamic_camera_follow_y_offset_m=-2.0`: sets pursuit distance behind the drone when dynamic follow is active (ignored if disabled).

Performance notes for viewer + GIFs:
- Viewer (`--headless=false`) + GIFs (`--save_gifs=true`) significantly increases GPU and I/O load. Reduce `--env_agents` or disable GIFs for speed runs.
- Typical VRAM guidance: 8–12 GB → `--env_agents=16`; 24 GB+ → 32–64 envs with headless.

Sanity checklist before running:
- Verify checkpoint path (`DCE_MODEL`) or that `--train_dir/--experiment` contains saved checkpoints.
- Keep curriculum forcing and randomization flags aligned with your experimental design.
- If using ablations, double‑check slice indices against the 150D layout.
