# Gate Navigation Task — End-to-End Guide

This document provides a complete technical walkthrough of the **gate navigation task** in the Aerial Gym Simulator: architecture, training, evaluation, configuration, and the full data flow from raw depth images to learned policies.

---

## Overview

The gate navigation task trains a quadrotor equipped with **dual RGB-D cameras** (drone-mounted + static) to fly through a gate using deep reinforcement learning. The system uses:

- **Deep Collision Encoding (DCE)**: a pre-trained VAE compresses 270×480 depth images into compact 64D latent vectors
- **Gated Dual-Fusion Encoder**: learns to fuse drone camera + static camera latents with proprioceptive state
- **Sample Factory PPO**: high-throughput RL training across hundreds of parallel environments
- **Curriculum Learning**: progressively increases difficulty (noise, obstacles, spawn randomization)

```
┌─────────────────────────────────────────┐
│  Training / Inference Interface         │  aerial_gym/run.py, Makefile
├─────────────────────────────────────────┤
│  RL Algorithm (Sample Factory PPO)      │  train_aerialgym_custom_net_gate.py
├─────────────────────────────────────────┤
│  Environment Wrapper                    │  env_wrapper_gate.py
├─────────────────────────────────────────┤
│  Task + Curriculum + Rewards            │  NavigationTaskGate, CurriculumManager
├─────────────────────────────────────────┤
│  Physics + Rendering (Isaac Gym/Warp)   │  Parallelized simulation + sensors
└─────────────────────────────────────────┘
```

---

## 1. Observation Space (150D)

The agent receives a 150-dimensional observation vector every step, defined in `aerial_gym/task/schemas.py` via `GATE_OBS_LAYOUT`:

| Indices | Dim | Description |
|---------|-----|-------------|
| 0–2 | 3 | Drone position (world frame) |
| 3–5 | 3 | Static camera position (relative to drone) |
| 6–8 | 3 | Static camera orientation (relative to drone) |
| 9–11 | 3 | Drone orientation (roll, pitch, yaw) |
| 12–14 | 3 | Body-frame linear velocity |
| 15–17 | 3 | Body-frame angular velocity |
| 18–21 | 4 | Previous action |
| **22–85** | **64** | **Drone camera depth → VAE latent** |
| **86–149** | **64** | **Static camera depth → VAE latent** |

The 128 dimensions of VAE latents (64 per camera) are the core of Deep Collision Encoding — they compress spatial/depth information into a compact representation the policy can reason over.

## 2. Action Space (4D)

Continuous box `[-1, 1]⁴`:

| Dim | Control |
|-----|---------|
| 0 | X velocity |
| 1 | Y velocity |
| 2 | Z velocity |
| 3 | Yaw rate |

Actions are sanitized (NaN→0, clamped) and passed to an on-GPU velocity controller.

---

## 3. Deep Collision Encoding (VAE)

### Architecture

The VAE (`aerial_gym/utils/vae/VAE.py`) uses a ResNet8-based encoder-decoder:

```
Depth Image (1×270×480)
    → Conv layers with skip connections
    → Flatten → Linear → μ, log σ²   (each 64D)
    → Reparameterize → z (64D latent)
```

### Pre-trained Weights

```
aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth
```

Trained offline on depth images from a D455 camera. The encoder is frozen during RL training — only the fusion encoder and policy are learned.

### Camera Processing Pipeline

`aerial_gym/task/navigation_task_gate/camera_observations.py` handles the full depth→latent pipeline:

```
Raw Depth (270×480)
  → Curriculum-dependent Gaussian noise (σ: 0.000625 → 0.00625)
  → Curriculum-dependent pixel dropout (rate: 0.000625 → 0.00625)
  → Frame dropout (blank/freeze: 0% → 5%)
  → Clamp [0, 1]
  → VAE Encoder
  → 64D latent
```

Both the drone camera and static camera go through this pipeline independently, producing two 64D latent vectors per step.

---

## 4. Dual Fusion Encoder

`aerial_gym/rl_training/sample_factory/aerialgym_examples/dual_fusion_encoder.py`

The encoder fuses the two 64D camera latents with the 22D proprioceptive state before feeding into the policy network.

### Gated Fusion (default)

```
Drone Latents (64D) ──→ LayerNorm → Linear → ELU → z_ego (64D)
                                                          │
Static Latents (64D) ─→ LayerNorm → Linear → ELU → z_static (64D)
                                                          │
                         ┌────────────────────────────────┤
                         │  Concatenate [z_ego, z_static] │
                         │  → Gate Network → sigmoid(g)   │
                         └────────────────────────────────┘
                                      │
                    z_fused = g · z_ego + (1−g) · z_static   (64D)
```

When `gate_per_feature=True` (default), the gate is 64-dimensional (per-feature gating). Otherwise it's a single scalar gate.

### MLP Tail

```
[proprioception (22D), z_fused (64D)] → 86D
  → Linear(512) → ELU
  → Linear(256) → ELU
  → Linear(64) → ELU
  → encoder output (64D)
```

This feeds into a GRU (64 hidden units) then the policy/value heads.

---

## 5. Reward Structure

`aerial_gym/task/navigation_task_gate/reward_functions.py` — all reward functions are `@torch.jit.script` compiled.

### Dense Rewards (per step)

| Component | Description | Scale |
|-----------|-------------|-------|
| **Position** | Exponential decay with distance to gate | 0.5 |
| **Very close** | Bonus within 0.5m of gate | 0.75 |
| **Getting closer** | Shaped reward for decreasing distance; 2× penalty for moving away | 5.0 |
| **Gate approach** | Distance to gate center | 1.25 |
| **Gate alignment** | Body-frame alignment toward gate | 0.5 |
| **Action smoothness** | Penalizes large Δ from previous action (per axis) | varies |
| **Action magnitude** | Penalizes large absolute actions | varies |
| **Time penalty** | Per-step cost scaled by episode progress | negative |
| **Static FOV visibility** | Bonus when drone is in static camera's FOV | curriculum-scaled |

### Sparse Rewards (one-time)

| Component | Value | Trigger |
|-----------|-------|---------|
| **Gate passage** | +100 | Crossing gate plane correctly |
| **Collision** | −100 | Any collision (episode terminates) |
| **Boundary violation** | −5 to −10 | Crossing gate plane outside passage window |
| **Timeout** | −10 | Episode time limit reached |

All rewards are scaled by a curriculum multiplier: `1.0` at level 3, rising to `~1.5` at level 23.

---

## 6. Curriculum Learning

`aerial_gym/task/navigation_task_gate/curriculum_management.py`

The curriculum advances from level 3 → 23 based on success rate. Once a level is reached, it never decreases.

### Progression Rule

- Track success rate over a sliding window
- If success rate > **75%**, advance one level
- Level range: **3 → 23**

### What Changes Per Level

| Parameter | Level 3 | Level 13 | Level 23 |
|-----------|---------|----------|----------|
| Obstacles behind gate | 3 | ~7 | 10 |
| Camera noise σ | 0.000625 | 0.003438 | 0.00625 |
| Pixel dropout | 0.000625 | 0.003438 | 0.00625 |
| Frame dropout (blank/freeze) | 0% | ~2.5% | 5% |
| Spawn Y range | ±1.0m | ±1.25m | ±1.5m |
| Spawn Z range | ±1.0m | ±1.25m | ±1.5m |
| Reward multiplier | 1.0 | ~1.25 | ~1.5 |

---

## 7. Configuration System

### Pydantic Schema

`aerial_gym/config/run_config.py` defines a validated config hierarchy:

```python
RunConfig
├── mode: train | eval | play | inference_suite
├── common: task, num_envs, device, seed, headless
├── training: total_steps, batch_size, learning_rate, gamma
├── sample_factory: fusion, encoder_mlp_layers, rnn_size, ppo_clip_ratio
├── curriculum: min_level, max_level, force_level
├── camera: static_camera_base_y/z, yaw_sweep, arc_follow
├── wandb: enabled, project, tags
├── ablation: observation/reward ablation flags
└── eval: checkpoint, num_episodes
```

### YAML Config Files

Located in `configs/`:

**Training:**
- `train_gate_sf.yaml` — default 256-env gate training
- `train_gate_sf_fixed_orient.yaml` — fixed drone orientation
- `train_gate_sf_arc_follow.yaml` — arc-follow static camera
- `train_gate_sf_dynamic_follow.yaml` — dynamic camera

**Evaluation:**
- `eval_gate_drone_only.yaml` — ablation: no static camera
- `eval_gate_all_modalities.yaml` — all modalities × seeds × levels
- `eval_gate_sweeping.yaml`, `eval_gate_locked_yaw.yaml`, etc.

### CLI Overrides

```bash
python -m aerial_gym.run --config configs/train_gate_sf.yaml \
  --set common.num_envs=512 training.learning_rate=0.0001
```

---

## 8. Training

### Quick Start

```bash
# Validate config
make validate-config CONFIG=configs/train_gate_sf.yaml

# Train (default: 256 envs, ~1.2M steps, gated fusion)
make train-gate

# Or with custom overrides
python -m aerial_gym.run --config configs/train_gate_sf.yaml \
  --set common.num_envs=512 \
       training.total_steps=2_000_000 \
       wandb.enabled=true \
  --log
```

### Makefile Targets

| Command | Description |
|---------|-------------|
| `make train-gate` | Default gate training |
| `make train-gate-fixed` | Fixed orientation variant |
| `make train-gate-arc` | Arc-follow camera variant |
| `make train-gate-dynamic` | Dynamic follow camera variant |
| `make eval` | Generic evaluation |
| `make eval-all-modalities` | Full ablation suite |
| `make validate-config CONFIG=<yaml>` | Validate config only |
| `make dry-run CONFIG=<yaml>` | Show command without running |

### Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| Parallel environments | 256 |
| Batch size | 8192 |
| Learning rate | 3×10⁻⁴ |
| Discount (γ) | 0.98 |
| GAE (λ) | 0.95 |
| PPO clip ratio | 0.2 |
| Rollout horizon | 32 |
| RNN | GRU, 64 hidden units |
| Encoder MLP | [512, 256, 64] |
| Fusion | Gated, per-feature |
| Total steps | ~1.2M |

### Training Outputs

```
train_dir/<experiment_name>/
├── checkpoint_p0/
│   ├── <experiment>_best_<step>_<frames>_reward_<R>.pth
│   └── <experiment>_<step>_<frames>.pth
├── curriculum_<timestamp>.log
└── logs/
```

### Training Data Flow

```
YAML Config → RunConfig (validated) → Environment Variables
    → Sample Factory launches N worker processes
    → Each worker:
        NavigationTaskGate.reset()
          → Spawn drone at curriculum-dependent position
          → Render dual cameras → VAE encode → 64D + 64D
          → Build 150D observation

        NavigationTaskGate.step(action)
          → Velocity controller applies action
          → Isaac Gym physics (10 substeps)
          → Re-render cameras → VAE encode
          → Compute 14 reward components
          → Check termination (collision / passage / timeout)
          → Curriculum update if episode ends

    → PPO collects rollouts → compute GAE advantages → gradient update
    → Repeat until total_steps reached
```

---

## 9. Evaluation & Inference

### Using the Unified Runner

```bash
python -m aerial_gym.run --config configs/eval_gate_all_modalities.yaml \
  --set eval.checkpoint="./train_dir/.../checkpoint_p0/best_*.pth" \
  --log
```

### Using enjoy_aerialgym.py Directly

```bash
python aerial_gym/rl_training/sample_factory/aerialgym_examples/enjoy_aerialgym.py \
  --train_dir=./train_dir/<experiment>/ \
  --experiment=<experiment_name> \
  --env=quad_with_obstacles_gate \
  --max_num_episodes=10 \
  --eval_deterministic=true \
  --save_gifs=true
```

### Programmatic Inference

```python
from aerial_gym.examples.dce_rl_navigation.sf_inference_class_gate import NN_Inference_Class_Gate

model = NN_Inference_Class_Gate(
    num_envs=256, num_actions=4, num_obs=150, cfg=cfg
)
actions, values = model(obs_dict, rnn_states)
```

### GIF Recording

Set `--save_gifs=true` during evaluation. Episodes are saved to `./gif_episodes/` with dual-camera views (drone depth + static depth).

---

## 10. Gate Environment

### Physical Setup

| Parameter | Value |
|-----------|-------|
| Environment bounds | 8m × 8m × 4m |
| Gate position | (0, 0, 1.15) center |
| Gate width | ~2.5m |
| Gate height | ~2.3m |
| Static camera default | (0, −3.0, 1.5) — behind gate |
| Physics substeps | 10 per env step |

### Gate Passage Detection

`aerial_gym/task/navigation_task_gate/gate_geometry.py` handles passage logic:
- Tracks drone position relative to gate plane
- Detects plane crossing via sign change of forward distance
- Verifies passage is within gate opening bounds
- Awards one-time passage reward on success

---

## 11. File Index

### Core Task
| File | Purpose |
|------|---------|
| `aerial_gym/task/navigation_task_gate/navigation_task_gate.py` | Main task class |
| `aerial_gym/task/navigation_task_gate/reward_functions.py` | JIT-compiled reward components |
| `aerial_gym/task/navigation_task_gate/reward_helpers.py` | Reward computation helpers |
| `aerial_gym/task/navigation_task_gate/obs_reward_helpers.py` | Observation + reward processing |
| `aerial_gym/task/navigation_task_gate/curriculum_management.py` | Dynamic difficulty |
| `aerial_gym/task/navigation_task_gate/curriculum_data.py` | Curriculum schedule data |
| `aerial_gym/task/navigation_task_gate/camera_observations.py` | Camera noise + VAE encoding |
| `aerial_gym/task/navigation_task_gate/gate_geometry.py` | Gate detection + passage logic |
| `aerial_gym/task/navigation_task_gate/init_helpers.py` | Task initialization |
| `aerial_gym/task/navigation_task_gate/step_helpers.py` | Per-step logic |
| `aerial_gym/task/schemas.py` | 150D observation layout |

### Training Pipeline
| File | Purpose |
|------|---------|
| `aerial_gym/run.py` | Unified CLI entry point |
| `aerial_gym/config/run_config.py` | Pydantic config schema |
| `aerial_gym/rl_training/sample_factory/aerialgym_examples/train_aerialgym_custom_net_gate.py` | Training script |
| `aerial_gym/rl_training/sample_factory/aerialgym_examples/env_wrapper_gate.py` | Sample Factory wrapper |
| `aerial_gym/rl_training/sample_factory/aerialgym_examples/dual_fusion_encoder.py` | Gated/concat fusion encoder |
| `aerial_gym/rl_training/sample_factory/aerialgym_examples/task_registration.py` | Task registration |
| `aerial_gym/rl_training/sample_factory/aerialgym_examples/cfg_env_bridge.py` | Config→env var bridge |

### VAE
| File | Purpose |
|------|---------|
| `aerial_gym/utils/vae/VAE.py` | ResNet8 encoder/decoder |
| `aerial_gym/utils/vae/vae_image_encoder.py` | Wrapper for RL integration |
| `aerial_gym/utils/vae/weights/ICRA_...epoch_49.pth` | Pre-trained weights (43.8 MB) |

### Inference
| File | Purpose |
|------|---------|
| `aerial_gym/examples/dce_rl_navigation/sf_inference_class_gate.py` | Policy loading + execution |
| `aerial_gym/examples/dce_rl_navigation/episode_metrics.py` | Trajectory tracking |
| `aerial_gym/examples/dce_rl_navigation/gif_recorder.py` | Episode visualization |

### Config
| File | Purpose |
|------|---------|
| `configs/train_gate_sf.yaml` | Default training config |
| `configs/eval_gate_*.yaml` | Evaluation configs |
| `aerial_gym/config/task_config/navigation_task_config_gate.py` | Task parameters |
| `aerial_gym/config/task_config/curriculum_schedules.py` | Curriculum progression |
| `aerial_gym/config/env_config/gate_env.py` | Gate environment assets |

### Shipped Model Weights

Only two weight files are tracked in the repository:

```
aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth   (43.8 MB, VAE)
aerial_gym/examples/dce_rl_navigation/TRAINED/.../HIGH_CONFIG_16ENV_2_best_..._reward_1463.917.pth  (DCE checkpoint)
```

All other training outputs (checkpoints, GIFs, logs) are gitignored and regenerated by running training.

---

## 12. References

- **Aerial Gym Simulator**: [arxiv.org/abs/2305.16510](https://arxiv.org/abs/2305.16510)
- **DCE RL Navigation (ICRA 2024)**: Kulkarni & Alexis, "Reinforcement Learning for Collision-free Flight Exploiting Deep Collision Encoding"
- **Sample Factory**: [github.com/alex-petrenko/sample-factory](https://github.com/alex-petrenko/sample-factory)
- **NVIDIA Isaac Gym**: [developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym)
