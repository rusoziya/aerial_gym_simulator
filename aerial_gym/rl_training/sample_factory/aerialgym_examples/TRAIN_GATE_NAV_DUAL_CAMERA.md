## Training Gate Navigation (Dual Camera) with Sample Factory

This guide explains how to train the gate‑navigation policy using the dual‑camera (drone + static) setup and the `train_gate_navigation_dual_camera.sh` runner.

## Prerequisites
- NVIDIA GPU + CUDA; Isaac Gym installed and working.
- Python env with Aerial Gym + Sample Factory (same env you use to run other examples).
- Optional: Weights & Biases account (for online logging).

## Installation (one‑time setup)
1) Create an environment and install PyTorch (CUDA):
```bash
conda create -n aerialgym python=3.10 -y && conda activate aerialgym
# Or use venv if preferred
# Install PyTorch matching your CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
2) Install NVIDIA Isaac Gym (Preview) per NVIDIA’s instructions and verify:
```bash
python -c "import isaacgym; print('Isaac Gym OK')"
```
3) Install project (editable) and dependencies:
```bash
cd <repo_root>/aerial_gym_simulator
pip install -e .
```
4) (Optional) Log into W&B:
```bash
pip install wandb && wandb login
```

## Verify setup
```bash
python - <<'PY'
import torch
import isaacgym
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print('Isaac Gym import: OK')
PY
```

## Script location
```bash
aerial_gym/rl_training/sample_factory/aerialgym_examples/train_gate_navigation_dual_camera.sh
```

## Quick start
```bash
# Example: 2.012M steps, viewer ON, 128 envs, with ablation and W&B enabled
TRAIN_ENV0_LATENTS_NORM=false WANDB_DISABLED=false WANDB_MODE=online \
ABLATE_DEBUG=true ABLATE_OBS_RANGES='0:22=zero' \
./train_gate_navigation_dual_camera.sh drone_and_static_sweep_SEED321 \
  --seed=321 --train_steps=2012416 --headless=false --view --envs=128 \
  --disable_gate_size_randomization=false --fixed_gate_scale_percent=100 \
  --disable_obstacle_randomization=false --fixed_obstacles_behind_gate=1 \
  --disable_static_camera_orientation_randomization=false \
  --disable_drone_camera_noise_randomization=false --disable_static_camera_noise_randomization=false \
  --disable_drone_camera_frame_dropout=false --disable_static_camera_frame_dropout=false \
  --disable_state_noise_randomization=true --disable_spawn_position_randomization=false \
  --disable_spawn_orientation_randomization=false --disable_curriculum_multiplier=true \
  --force_curriculum_level=none --min_curriculum_level=13 --max_curriculum_level=23 \
  --enable_dynamic_camera_following=false --fusion=gated --gate_per_feature=1 \
  --enable_static_camera_yaw_sweep=true --static_camera_yaw_sweep_speed_deg=180 \
  --static_camera_base_y=-2.0 --static_camera_base_z=adaptive --enable_static_camera_locked=false \
  --enable_static_camera_arc_follow=false --dynamic_camera_follow_y_offset_m=-1.0 \
  --disable_dynamic_follow_gate_blending=false
```

## What the runner does
- Configures env count, logging, and GPU monitoring.
- Launches `train_aerialgym_custom_net_gate.py` with PPO/APPO (recurrent) and 4D action space.
- Sets W&B run metadata (if enabled) and logs to `./train_dir/<EXPERIMENT_NAME>`.

## How flags map into training
- The runner passes CLI flags into `train_aerialgym_custom_net_gate.py`, which registers the gate task and sets Sample Factory config overrides.
- Key model defaults for gate training (unless overridden by CLI):
  - Recurrent GRU policy, recurrence 32; encoder MLP `[512, 256, 128]`; nonlinearity ELU.
  - PPO/APPO with clip ratio `0.2`, value clip `1.0` (value function), KL‑adaptive LR schedule (threshold `0.016`).
  - Batch size `8192`, accumulate `2` (effective `16384`), epochs `4`, batches/epoch default `4` (the runner uses `8`).
  - Reward scale `0.1`, γ `0.98`, GAE λ `0.95`, max grad‑norm `1.0`.
  - Observation normalization and return normalization enabled.

## Repository map (key files)
- Training runner (this guide): `rl_training/sample_factory/aerialgym_examples/train_gate_navigation_dual_camera.sh`
- Training entry: `rl_training/sample_factory/aerialgym_examples/train_aerialgym_custom_net_gate.py`
- Inference (gate): `examples/dce_rl_navigation/dce_nn_navigation_gate.py`
- Gate task config: `config/task_config/navigation_task_config_gate.py`
- Gate task implementation: `task/navigation_task_gate/navigation_task_gate.py`
- Ablation suite (inference): `rl_training/sample_factory/aerialgym_examples/run_all_inference_ablation_suite_L33.sh`

## VAE weights (visual encoder)
- The gate task uses depth VAEs for each camera; by default a 64‑D model file is referenced here:
  - `config/task_config/navigation_task_config_gate.py :: task_config.vae_config.model_file`
- Replace with your VAE or set the path to a new weights file; ensure latent size matches 64D.

## Core flags
- **Run identity**
  - `EXPERIMENT_NAME` (positional): folder under `./train_dir/`.
  - `--seed=<int>`: reproducibility seed.
  - `--train_steps=<int>`: env steps (e.g., 2012416).
- **Viewer / headless**
  - `--view` (viewer ON) or omit for headless; you can also pass `--headless=true|false`.
  - Tip: headless is faster; viewer is for visual debugging.
- **Parallelism**
  - `--envs=<int>`: number of parallel envs (e.g., 16, 128).
- **Fusion**
  - `--fusion=gated|concat`: dual‑view fusion strategy (default: `gated`).
  - `--gate_per_feature=1|0`: per‑feature vs scalar gate (recommended: `1`).
- **Static camera behavior** (pick one or combine)
  - `--enable_static_camera_yaw_sweep=true --static_camera_yaw_sweep_speed_deg=<deg/s>`
  - `--enable_static_camera_locked=true` (keeps drone centered)
  - `--enable_static_camera_arc_follow=true --static_camera_arc_radius_m=<m>`
  - `--enable_dynamic_camera_following=true|false`
  - Placement overrides: `--static_camera_base_y=<m>`, `--static_camera_base_z=<float|adaptive>`
  - Dynamic follow offset: `--dynamic_camera_follow_y_offset_m=<m>`
  - Gate blending: `--disable_dynamic_follow_gate_blending=true|false`
- **Curriculum / difficulty**
  - `--min_curriculum_level=<int> --max_curriculum_level=<int>`
  - `--force_curriculum_level=<int|none>`
  - `--disable_curriculum_multiplier=true|false`
- **Randomization toggles** (true/false)
  - Gate size: `--disable_gate_size_randomization`, `--fixed_gate_scale_percent=<40..100>`
  - Obstacles: `--disable_obstacle_randomization`, `--fixed_obstacles_behind_gate=<int>`
  - Static cam orientation: `--disable_static_camera_orientation_randomization`
  - State noise: `--disable_state_noise_randomization`
  - Camera noise: `--disable_camera_noise_randomization` (global) or per‑camera:
    - `--disable_drone_camera_noise_randomization`
    - `--disable_static_camera_noise_randomization`
  - Frame dropout (entire‑frame events): global or per‑camera:
    - `--disable_camera_frame_dropout_randomization`
    - `--disable_drone_camera_frame_dropout`
    - `--disable_static_camera_frame_dropout`
  - Spawns: `--disable_spawn_position_randomization`, `--disable_spawn_orientation_randomization`

## Observation layout and ablation indices
- Gate task observation (150D) is a flat vector consumed under the `obs` key:
  - `0:3`  drone absolute position
  - `3:6`  static camera position (relative)
  - `6:9`  static camera orientation (relative)
  - `9:12` drone orientation (roll/pitch/yaw)
  - `12:15` drone linear velocity (body)
  - `15:18` drone angular velocity (body)
  - `18:22` recent drone actions
  - `22:86` drone (ego) depth VAE latent (64D)
  - `86:150` static (exo) depth VAE latent (64D)
- Common ablations with `ABLATE_OBS_RANGES`:
  - Ego‑only (drop exocentric): `86:150=zero`
  - Exo‑only (drop egocentric): `22:86=zero`
  - Drop non‑visual state for pure vision: `0:22=zero`
  - Shuffle a slice across envs: `22:86=shuffle`
  - Add noise: `22:86=noise:0.1` (std 0.1)

## Curriculum and noise schedules (gate task)
- Levels (training): 3→23. Difficulty rises by increasing obstacle density, camera orientation difficulty, state/camera noise, and frame drop probabilities.
- Camera noise (D455‑style): level‑dependent Gaussian depth noise and pixel dropout; frame‑level freeze/blank events ramp from level 3 to 23.
- You can cap or shift progression with `--min_curriculum_level`, `--max_curriculum_level`, or freeze with `--force_curriculum_level`.

## Camera‑mode presets (suggested)
- Dynamic follow (peer pursuit):
  - `--enable_dynamic_camera_following=true` and optionally `--dynamic_camera_follow_y_offset_m=-1.0`
- Static locked follow (centering):
  - `--enable_static_camera_locked=true`
- Fixed yaw sweep (periodic gaze):
  - `--enable_static_camera_yaw_sweep=true --static_camera_yaw_sweep_speed_deg=180`
- Arc follow (lateral orbit):
  - `--enable_static_camera_arc_follow=true --static_camera_arc_radius_m=2.0`

## Environment variables (optional)
- **W&B**
  - `WANDB_DISABLED=false|true`, `WANDB_MODE=online|offline`, `WANDB_PROJECT`, `WANDB_ENTITY`.
- **Observation ablation**
  - `ABLATE_OBS_RANGES='start:end=op,...'` with ops: `zero`, `zerograd`, `shuffle`, `noise:<std>`.
  - Example: `ABLATE_OBS_RANGES='0:22=zero'` zeros the 22D non‑visual state slice.
  - `ABLATE_DEBUG=true` prints ablation diagnostics (first few steps).
- **Debug taps**
  - `TRAIN_ENV0_LATENTS_NORM=true|false` prints normalized latent magnitudes for env 0 during one episode.

## Outputs
- Checkpoints and configs: `./train_dir/<EXPERIMENT_NAME>/`.
- W&B: if enabled, full metrics, curriculum snapshots, and obs‑grad diagnostics.
- Optional GIFs: add `--gifs` to save per‑episode camera GIFs alongside the script directory.
- GPU log: `./logs/gpu_usage_gate_<EXPERIMENT_NAME>.csv`.

### Checkpointing
- Default save cadence (gate training): regular every 1800 s and best‑model checks every 300–500 s (runner uses 500).
- To resume: set `--restart_behavior=resume` (default) and keep the run folder in `train_dir`.

### W&B tips
- Online logging: set `WANDB_DISABLED=false` and optionally `WANDB_PROJECT`, `WANDB_ENTITY`.
- The script auto‑defines metrics (curriculum/**, episode_extra_stats/**) keyed on `frames`.

## Minimal baseline training (recommended)
```bash
WANDB_DISABLED=true \
./train_gate_navigation_dual_camera.sh baseline_dual \
  --seed=0 --train_steps=1000000 --headless=true --envs=32 \
  --fusion=gated --gate_per_feature=1
```
Expect a run folder at `./train_dir/baseline_dual` with periodic checkpoints.

## Run the ablation suite (inference at Level 33)
```bash
cd aerial_gym/rl_training/sample_factory/aerialgym_examples
bash run_all_inference_ablation_suite_L33.sh
```
This script will iterate through camera‑mode scenarios and evaluation seeds to produce comparable metrics.

## Common recipes
- Baseline dual‑view gated fusion (headless, 128 envs):
```bash
WANDB_DISABLED=false WANDB_MODE=online \
./train_gate_navigation_dual_camera.sh baseline_dual \
  --seed=0 --train_steps=2012416 --headless=true --envs=128 \
  --fusion=gated --gate_per_feature=1
```
- Egocentric‑only ablation (zero static latents 86:150):
```bash
ABLATE_OBS_RANGES='86:150=zero' \
./train_gate_navigation_dual_camera.sh ego_only --envs=128 --headless=true
```
- Exocentric‑only ablation (zero drone latents 22:86):
```bash
ABLATE_OBS_RANGES='22:86=zero' \
./train_gate_navigation_dual_camera.sh exo_only --envs=128 --headless=true
```

## Performance tips
- OOM? Reduce `--envs` (e.g., 128 → 32) or add `--num_batches_per_epoch=4` and lower `--batch_size` in the script if needed.
- Viewer stutter? Use headless (`--headless=true`) during long runs.

## Troubleshooting
- Isaac Gym viewer conflicts: ensure `--headless=true` for multi‑process runs; the worker subprocesses force headless by default.
- Low FPS / GPU overutilization: lower `--envs`, reduce GIF/logging, or disable sweeping camera modes.
- Checkpoint size grows: reduce `--keep_checkpoints` in the Python config or archive older runs.
- W&B offline: set `WANDB_MODE=offline` or `WANDB_DISABLED=true`.

## Runner internals (for power users)
- Batching: the script defines `BATCH_SIZE` (default 2048 in the shell), `NUM_BATCHES_TO_ACCUMULATE`, and computes an effective batch (`BATCH_SIZE * NUM_BATCHES_TO_ACCUMULATE`). It then passes `--batch_size`, `--num_batches_to_accumulate`, and `--num_batches_per_epoch` to Python, targeting an effective ~16K batch by default.
- Python command: the runner assembles a call to `train_aerialgym_custom_net_gate.py` with explicit flags (rollout, learning rate, RNN size/layers, fusion, save cadence, etc.). You can echo the final command from the script to replicate runs verbatim.
- Save cadence: regular `--save_every_sec=1800` and `--save_best_every_sec=500` are passed from the shell (best‑check interval slightly higher than Python default).

## Resource tuning
- Memory budget knobs (approx order): `--envs` → `--num_batches_per_epoch` → `--batch_size`.
- If you see CUDA OOM:
  - Try `--envs=32` (or lower), keep headless, and reduce sweeping cameras.
  - Use `--num_batches_per_epoch=4` and keep `--num_batches_to_accumulate=2` to retain effective batch.
  - Consider disabling GIFs and obs‑grad tracking during long runs.

## GIF saving (optional)
- Add `--gifs` to enable per‑episode GIF dumps (stored next to the script). Saved variants typically include drone and static D455‑noised depth sequences; to reduce I/O, the script saves at episode boundaries.

## Debug taps & diagnostics
- Env‑side one‑episode taps: `PRINT_ENV0_LATENTS_ONCE=true`, `PRINT_ENV0_OBS_ONCE=true` (log only once per worker).
- Encoder taps: `ENC_TAP_DEBUG=true` and/or `TRAIN_ENV0_LATENTS_NORM=true` (normalized latent magnitudes).
- Normalization taps at inference: `NORM_TAP_DEBUG=true`.
- Ablation debug: `ABLATE_DEBUG=true` prints per‑slice application summaries.

## Observation‑influence logging (optional)
- Enable complete observation influence tracking with:
  - `--enable_gradient_monitoring=true --gradient_log_interval=100 --gradient_print_interval=100`.
- Metrics are mirrored to W&B under `episode_extra_stats/obs_grad/*` and include per‑slice shares, camera vs state shares, and recent/overall windows—useful for fusion attribution and ablation sanity checks.

## Resuming and exporting
- Resume: default `--restart_behavior=resume`; keep the `train_dir/<EXPERIMENT_NAME>` folder. You can select checkpoint kind via `--load_checkpoint_kind=latest|best` (Python defaults apply if not specified).
- Export best: after training, copy the `best_*.pth` from `train_dir/<EXPERIMENT_NAME>/policy_*` for inference.

## Evaluation / inference pointers
- Scripted inference for gate navigation is available:
  - `aerial_gym/examples/dce_rl_navigation/dce_nn_navigation_gate.py`
- You can register the trained policy/config and run evaluation with W&B logging enabled, mirroring many of the training‑time episode statistics at reset boundaries.

## FAQ
**Q: Viewer crashes or black window?**
Use `--headless=true`for long runs; ensure only one process opens the viewer.

**Q: How many environments can my GPU handle?**
Start with `--envs=32` on 10–12 GB GPUs; scale up to 128 on 24 GB+ devices.

**Q: My percentages in fusion/obs‑grad don’t sum to 100%.**
A small residual resides in the non‑visual 22D state slice, which trends toward but rarely reaches zero; rounding also contributes.

**Q: How do I train ego‑only/exo‑only policies?**
Set `ABLATE_OBS_RANGES='86:150=zero'` (ego‑only) or `ABLATE_OBS_RANGES='22:86=zero'` (exo‑only).

**Q: Where are metrics logged if W&B is disabled?**
Scalar summaries are printed to stdout; checkpoints/configs live in `train_dir/<EXPERIMENT_NAME>`.

## Reproducibility
- Set `--seed`, fix `--train_steps`, and keep env/flags in your run script.
- W&B captures config + code snapshots; keep `./train_dir/<EXPERIMENT_NAME>` for re‑runs.


