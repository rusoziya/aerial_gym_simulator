# Curriculum Learning & Domain Randomization

> Detailed specification from Chapter 4.5 of: Z. Ruso, *"Reinforcement Learning for Cooperative Multi-View Depth-Based Perception in Autonomous UAV Navigation,"* MSc Thesis, UCL, 2025.

## Why Curriculum Learning?

The gate-navigation task is **exploration-hard** and **non-stationary** across viewpoints, geometry, and corruptions:

- **Fixed hard levels** concentrate returns in crashes/timeouts (low signal-to-noise, starving PPO of stable gradients)
- **Fixed easy levels** induce shortcut policies that overfit to large apertures and a static vantage, then fail under yaw sweep, clutter, or depth noise

The curriculum begins at an intermediate difficulty (Level 13) and enforces a **no-decrease policy** — promoting only on success to Level 23. This bypasses trivial early curricula that invite shortcuts and avoids the near-zero-signal regime of the hardest levels from a cold start, yielding more stable advantages and faster learning.

## Curriculum Controller

| Parameter | Value |
|-----------|-------|
| Level range (training) | L in [3, 23] |
| Level range (evaluation stretch) | L in [3, 33] |
| Training start level | 13 |
| Evaluation window | 256 completed episodes |
| Increase threshold | Success rate > 0.60 (step +1) |
| Decrease threshold | Success rate < 0.25 (step -1) |
| Cooldown | 12 windows after any level change |
| Forced-level option | Available (freezes progression for evaluation) |

Levels 23-33 are reserved for evaluation to assess zero-shot generalization beyond the training envelope. Level 33 is a held-out, previously unseen extrapolation test.

## Difficulty Axes

The curriculum increases difficulty along **multiple independent axes** simultaneously. All interpolations between Level 3 and Level 23 are linear.

### Visual Randomization (depth sensing)

Applied to both drone and static camera depth frames post-normalization.

| Aspect | Target | Value @ L=3 | Value @ L=23 |
|--------|--------|-------------|--------------|
| **Gaussian depth noise sigma** | Drone + Static | 0.000625 | 0.0125 |
| **Per-pixel dropout probability** | Drone + Static | 0.000625 | 0.0125 |
| **Whole-frame freeze probability** | Drone | 0.25% | 5.0% |
| **Whole-frame blank probability** | Drone | 0.025% | 0.5% |
| **Whole-frame freeze probability** | Static | 0.25% | 5.0% |
| **Whole-frame blank probability** | Static | 0.025% | 0.5% |

- **Pixel dropout**: Bernoulli masking of individual depth pixels
- **Gaussian noise**: Additive white Gaussian noise on depth values
- **Frame freeze**: Entire frame is held from previous step (simulates sensor stall)
- **Frame blank**: Entire frame is zeroed (simulates complete dropout)

### Camera Viewpoint Randomization

| Aspect | Target | Value @ L=3 | Value @ L=23 |
|--------|--------|-------------|--------------|
| **Exocentric camera azimuth** | About gate centre | +/-3 deg | +/-19 deg |
| **Onboard mount jitter (translation)** | Camera pose (body) | 7-12 cm (xyz) | 7-12 cm (xyz) |
| **Onboard mount jitter (rotation)** | Camera pose (body) | +/-5 deg (r/p/y) | +/-5 deg (r/p/y) |
| **Exocentric jitter (translation)** | Static camera (world) | +/-2 cm (xy), +/-1 cm (z) | +/-2 cm (xy), +/-1 cm (z) |
| **Exocentric jitter (rotation)** | Static camera (world) | +/-1 deg (r), +/-0.5 deg (p) | +/-1 deg (r), +/-0.5 deg (p) |

Camera viewpoint is chosen at reset and held throughout an episode; mount jitters are constant across curriculum levels.

### State/Pose Observation Noise

Applied to observations (not simulator state) to emulate estimation errors:

| Component | Value @ L=3 | Value @ L=23 | Units |
|-----------|-------------|--------------|-------|
| Drone position | 0.001 | 0.02 | m (per-axis std) |
| Drone orientation | 0.025 | 0.5 | deg (per-axis std) |
| Static camera position | 0.0025 | 0.05 | m (per-axis std) |
| Static camera orientation | 0.05 | 1.0 | deg (per-axis std) |

### Scene & Geometry

| Component | Value @ L=3 | Value @ L=23 | Notes |
|-----------|-------------|--------------|-------|
| Obstacles behind gate | 3 | 23 | Clamped by asset capacity (30 total) |
| Gate size unlocking | >= 80% scale | >= 40% scale | Smaller gates unlocked as L increases |

Gate scale is selected at episode reset and fixed intra-episode. This separates "layout hardness" from transient noise.

### Spawn Variation

| Component | Value @ L=3 | Value @ L=23 | Units |
|-----------|-------------|--------------|-------|
| Lateral half-span (X) | +/-0.50 | +/-2.00 | m |
| Longitudinal center (Y) | -1.50 (constant) | -1.50 (constant) | m |
| Longitudinal half-span (Y) | 0.00 (constant) | 0.00 (constant) | m |
| Vertical center (Z) | 1.125 (constant) | 1.125 (constant) | m |
| Vertical half-span (Z) | +/-0.375 | +/-0.625 | m |
| Yaw range at reset | 2 deg | 45 deg | deg |

## Temporal Application Rules

| Perturbation | When applied | Held for |
|-------------|-------------|----------|
| Camera viewpoint (azimuth, position) | At episode reset | Entire episode |
| Whole-frame events (freeze/blank) | Per frame | Single frame |
| Per-pixel noise (Gaussian, dropout) | Per frame | Single frame |
| State/pose noise | Per observation | Single step |
| Gate scale | At episode reset | Entire episode |
| Obstacle count/placement | At episode reset | Entire episode |

## Ablation Toggles

Each difficulty aspect can be independently toggled or fixed at user-specified values via environment variables or CLI flags:

- Gate scale: `SF_FIXED_GATE_SCALE_PERCENT`, `SF_DISABLE_GATE_SIZE_RANDOMIZATION`
- Obstacle count: `SF_FIXED_OBSTACLES_BEHIND_GATE`, `SF_DISABLE_OBSTACLE_RANDOMIZATION`
- Camera orientation: `SF_DISABLE_STATIC_CAMERA_ORIENT_RANDOMIZATION`
- Visual noise: `SF_DISABLE_CAMERA_NOISE_RANDOMIZATION`
- Frame dropout: `SF_DISABLE_CAMERA_FRAME_DROPOUT_RANDOMIZATION`
- State noise: `SF_DISABLE_STATE_NOISE_RANDOMIZATION`
- Spawn variation: `SF_DISABLE_SPAWN_POSITION_RANDOMIZATION`, `SF_DISABLE_SPAWN_ORIENTATION_RANDOMIZATION`
- Curriculum multiplier: `SF_DISABLE_CURRICULUM_MULTIPLIER`
- Forced level: `SF_FORCE_CURRICULUM_LEVEL`

## Implementation

- `aerial_gym/task/navigation_task_gate/curriculum_management.py` — progression controller
- `aerial_gym/task/navigation_task_gate/curriculum_data.py` — schedule data and lookup
- `aerial_gym/config/task_config/curriculum_schedules.py` — curriculum progressions
- `aerial_gym/task/navigation_task_gate/camera_observations.py` — noise/dropout application
