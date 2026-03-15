# Isaac Gym → Isaac Lab Migration Guide

## Overview

This guide documents the step-by-step process for migrating aerial_gym from
Isaac Gym Preview 4 (Python 3.8) to Isaac Lab (Python 3.10+).

**Current state:** The abstraction layer is complete. Three Protocol interfaces
define the physics, sensor, and asset backend contracts. Isaac Lab stubs
implement all Protocol methods with TODO placeholders and detailed migration
comments.

## Architecture

```
EnvManager (unchanged)
    │
    ├── PhysicsBackend Protocol
    │   ├── IsaacGymEnv (existing, working)
    │   └── IsaacLabEnv (stub, fill in TODOs)
    │
    ├── CameraBackend Protocol
    │   ├── isaacgym_camera_sensor (existing)
    │   └── IsaacLabCameraBackend (stub)
    │
    └── AssetBackend Protocol
        ├── IsaacGymAsset (existing)
        └── IsaacLabBackend (stub)
```

## Prerequisites

```bash
# Isaac Lab requires:
# - Python 3.10+
# - NVIDIA Isaac Sim (Omniverse)
# - CUDA 11.8+
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics
```

## Step-by-Step Migration

### Step 1: Fill in `isaac_lab_env_manager.py`

**File:** `aerial_gym/env_manager/isaac_lab_env_manager.py`

Each method has a TODO comment showing the Isaac Gym call and its Isaac Lab
equivalent. Key methods to implement:

| Method | Isaac Gym | Isaac Lab |
|--------|-----------|-----------|
| `__init__` | `gymapi.acquire_gym()` + `gym.create_sim()` | `SimulationContext(SimulationCfg(...))` |
| `create_env` | `gym.create_env()` loop | `InteractiveScene(num_envs=N)` (cloned) |
| `add_asset_to_env` | `gym.create_actor()` | `scene.add(ArticulationCfg(...))` |
| `prepare_for_simulation` | `gym.prepare_sim()` + `gym.acquire_*_tensor()` | `scene.reset()` + direct tensor access |
| `physics_step` | `gym.simulate()` | `sim_context.step()` |
| `pre_physics_step` | `gym.apply_rigid_body_force_tensors()` | `robot.set_external_force_and_torque()` |
| `write_to_sim` | `gym.set_actor_root_state_tensor()` | `robot.write_root_state_to_sim()` |

### Step 2: Fill in `isaac_lab_camera_sensor.py`

**File:** `aerial_gym/sensors/isaac_lab_camera_sensor.py`

| Method | Isaac Gym | Isaac Lab |
|--------|-----------|-----------|
| `create_camera` | `gym.create_camera_sensor()` | `CameraCfg(...)` |
| `render_cameras` | `gym.render_all_camera_sensors()` | `sim_context.render()` |
| `get_depth_tensor` | `gym.get_camera_image_gpu_tensor(IMAGE_DEPTH)` | `camera.data.output["distance_to_camera"]` |
| `get_segmentation_tensor` | `gym.get_camera_image_gpu_tensor(IMAGE_SEGMENTATION)` | `camera.data.output["semantic_segmentation"]` |

### Step 3: Fill in `isaac_lab_asset.py`

**File:** `aerial_gym/assets/isaac_lab_asset.py`

| Method | Isaac Gym | Isaac Lab |
|--------|-----------|-----------|
| `load_asset` | `gym.load_asset(sim, folder, file, options)` | `UrdfFileCfg(asset_path=...)` or `UsdFileCfg(...)` |
| `find_body_index` | `gym.find_asset_rigid_body_index()` | Query articulation schema |
| `create_force_sensor` | `gym.create_asset_force_sensor()` | `ContactSensorCfg(...)` |

### Step 4: Test

```bash
# Set backend to Isaac Lab
export AERIAL_GYM_BACKEND=isaaclab

# Or use config:
# common:
#   backend: isaaclab

# Run training
make train-gate-lab

# Or with any config
make train CONFIG=configs/train_gate_sf.yaml --set common.backend=isaaclab
```

### Step 5: Verify

```bash
# Run unit tests
python -m pytest tests/test_physics_backend.py -v

# Run full test suite
python -m pytest tests/test_*.py -v

# Compare Isaac Gym vs Isaac Lab outputs
make train-gate          # Isaac Gym
make train-gate-lab      # Isaac Lab
# Compare reward curves, observation shapes, training metrics
```

## Key Differences to Watch

### Tensor Wrapping
- **Isaac Gym:** `gymtorch.wrap_tensor(ige_tensor)` → PyTorch
- **Isaac Lab:** Tensors are already PyTorch. No wrapping needed.

### Multi-Environment
- **Isaac Gym:** Explicit `for env_id in range(N): gym.create_env()`
- **Isaac Lab:** Single scene with `num_envs` clones. Access via index.

### State Refresh
- **Isaac Gym:** Manual `gym.refresh_rigid_body_state_tensor()` after each step
- **Isaac Lab:** Automatic after `sim_context.step()`. May need `render()` for cameras.

### URDF Loading
- **Isaac Gym:** Direct URDF loading via `gym.load_asset()`
- **Isaac Lab:** Prefer USD format. Use `UrdfConverter` for URDF → USD conversion.

## Files That DON'T Change

These modules are backend-agnostic and require zero modifications:

```
✓ aerial_gym/config/           — All configs, Pydantic schemas, enums
✓ aerial_gym/control/          — Lee controllers (pure torch math)
✓ aerial_gym/task/             — Rewards, curriculum, observations
✓ aerial_gym/registry/         — Component registration
✓ aerial_gym/utils/            — Math, logging, tensor_utils
✓ aerial_gym/rl_training/      — Sample Factory, RL-Games wrappers
✓ aerial_gym/run.py            — Unified runner (backend-aware)
✓ configs/*.yaml               — Training/eval configs
✓ Makefile                     — Build targets (backend-aware)
✓ tests/                       — All unit tests
```

## Estimated Effort

| Phase | Files | Effort |
|-------|-------|--------|
| Fill `isaac_lab_env_manager.py` | 1 | 2 weeks |
| Fill `isaac_lab_camera_sensor.py` | 1 | 1 week |
| Fill `isaac_lab_asset.py` | 1 | 3 days |
| Integration testing | — | 1 week |
| **Total** | **3 files** | **4-5 weeks** |
