# Reward Shaping

> Detailed specification from Chapter 4.2 of: Z. Ruso, *"Reinforcement Learning for Cooperative Multi-View Depth-Based Perception in Autonomous UAV Navigation,"* MSc Thesis, UCL, 2025.

## Design Principles

The reward function is **deliberately dense in the approach phase and sparse at passage**:

1. **Dense shaping for approach** — position proximity, getting-closer, gate-approach, heading alignment provide step-wise signal aligned with task geometry, shortening credit-assignment paths through the GRU
2. **Sparse bonuses at passage** — a gate-pass bonus and center-pass bonus define success quality at the aperture without relying on repeated plane crossings
3. **Explicit outcome ordering** — crash < timeout < slow success < timely success, enforced by penalty magnitudes
4. **Action regularization** — smoothness and magnitude penalties prevent chatter and saturation while preserving authority for altitude/yaw corrections
5. **Global scaling** — all rewards scaled by S = 0.1 before return normalization to keep PPO clipping and KL schedule numerically stable

## Helper Functions

Two exponential shaping primitives are used throughout:

```
exp_reward(A, k, v)  = A * exp(-k * v^2)        # Positive, decaying with v
exp_penalty(A, k, v) = A * (exp(-k * v^2) - 1)  # Negative, increasing with v
```

## Reward Components

### Dense Per-Step Rewards

| Component | Parameters | Values | Equation |
|-----------|-----------|--------|----------|
| **Position proximity** | magnitude, exponent | 2.5, 1/3.5 | r_pos = exp_reward(2.5, 1/3.5, d) |
| **Very close to goal** | magnitude, exponent | 2.5, 2.0 | r_vcg = exp_reward(2.5, 2.0, d) |
| **Getting closer** | multiplier | 5.0 | r_gc = 5.0 * (d_prev - d) |
| **Gate approach** | magnitude | 1.25 | r_ga ~ 1.25 * max(0, d_plane_prev - d_plane_curr) |
| **Gate alignment** | magnitude | 0.5 | r_align ~ 0.5 * cos(theta_heading_to_gate) |
| **Centering** | magnitude | 1.25 | r_center ~ 1.25 * exp(-k_center * rho^2) |

Where d = Euclidean distance to gate, d_plane = distance to gate plane, rho = lateral offset from centerline, theta = heading angle to gate.

### Sparse One-Time Event Rewards

| Component | Value | Trigger |
|-----------|-------|---------|
| **Gate passage bonus** | +100.0 | Valid traversal (sign change in y, within aperture) |
| **Center passage bonus** | +100.0 | Traversal close to gate centerline (supersedes regular pass) |

### Event Penalties

| Component | Value | Trigger |
|-----------|-------|---------|
| **Collision penalty** | -100.0 | Any collision with environment |
| **Gate collision penalty** | -50.0 | Collision specifically with gate frame |
| **Boundary violation** | -50.0 | Misaligned plane crossing (one-shot) |
| **Timeout penalty** | -65.0 | Episode truncation at horizon (no success/crash) |

**Boundary violation** fires when: y > y_gate + 0.2m AND (lateral offset > 30% of gate width OR vertical offset > 30% of gate height) AND gate has not been passed yet. Applied once per episode.

### Action Smoothness Penalties (per step)

Penalize large changes from previous action (discourages chatter):

| Axis | Magnitude | Exponent |
|------|-----------|----------|
| X velocity | 0.8 | 3.333 |
| Y velocity | 0.8 | 3.333 |
| Z velocity | 0.4 | 2.0 |
| Yaw rate | 0.5 | 2.5 |

```
p_dx = exp_penalty(0.8, 3.333, delta_a_x)
```

Z-axis is gentler (lower magnitude/exponent) to permit altitude adjustments.

### Action Magnitude Penalties (per step)

Penalize large absolute actions (prevents saturation):

| Axis | Magnitude | Exponent |
|------|-----------|----------|
| X velocity | 0.1 | 0.3 |
| Y velocity | 0.1 | 0.3 |
| Z velocity | 0.05 | 0.2 |
| Yaw rate | 1.0 | 1.5 |

Yaw rate has the highest penalty to prevent excessive spinning. Z-axis has the lightest penalty to allow vertical maneuvering.

### Per-Step Time Cost

```
r_time = -lambda_0 * (1 + lambda_1 * s^p)
```

Where s = step_index / horizon, lambda_1 = 1.0, p = 2.0. The constant lambda_0 is chosen so the unscaled total equals 40.0 over a 100-step horizon. After reward scaling S = 0.1, this yields approximately -4.0 if an episode times out.

## Total Reward

### Per-step:

```
r_step = S * [r_pos + r_vcg + r_gc + r_ga + r_align + r_center
              + sum(smoothness_penalties) + sum(magnitude_penalties)
              + r_time + r_boundary_violation]
```

### Event-based (added when triggered):

```
+ r_pass (gate passage)
+ r_pass_center (center passage, supersedes r_pass)
- collision_penalties
- r_timeout (on truncation only)
```

**S = 0.1** (global reward scale applied to all components).

## Scaling & Ordering Rules

1. Timeout penalty is **strictly smaller** than collision penalty (65 < 100)
2. Success bonuses are granted **once per episode** (center-pass supersedes regular pass)
3. Dense terms follow the same global scaler and curriculum multiplier
4. Incentive structure is invariant to rollout length

## Curriculum Reward Multiplier

When enabled, all dense rewards are additionally scaled by a curriculum-dependent multiplier that increases with level, reinforcing shaping signal at higher difficulty where episodes are harder.

## Implementation

All reward functions are JIT-compiled with `@torch.jit.script` for GPU performance:

- `aerial_gym/task/navigation_task_gate/reward_functions.py` — individual reward components
- `aerial_gym/task/navigation_task_gate/reward_helpers.py` — reward computation helpers
- `aerial_gym/task/navigation_task_gate/obs_reward_helpers.py` — combined observation/reward processing

## Limitations

- Position- and alignment-based shaping assumes accurate observation of distance and heading
- Exponential shaping can create local optima at medium distances in rare configurations
- The time penalty may encourage risky shortcuts in highly cluttered scenes
