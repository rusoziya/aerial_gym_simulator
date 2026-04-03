# Exocentric Camera Modes

> Detailed specification from Chapter 4.6 of: Z. Ruso, *"Reinforcement Learning for Cooperative Multi-View Depth-Based Perception in Autonomous UAV Navigation,"* MSc Thesis, UCL, 2025.

## Overview

Six exocentric camera behaviors constitute the **only training-time factor varied** in the experimental study. All other components (architecture, rewards, curricula, budgets, evaluation protocol) remain fixed across configurations, ensuring that observed deltas are attributable to camera behavior alone.

These six modes span the principal axes along which exocentric viewpoint affects the statistics of the static-camera latent:

1. **Origin fixed vs. moving** — does the camera translate?
2. **Orientation-only tracking vs. positional motion** — what drives the view change?
3. **Presence/absence of exocentric information** — is the second stream available?

## Notation

| Symbol | Meaning |
|--------|---------|
| c_t | Camera position at time t |
| t_t | Look-target at time t |
| r_t | Drone position at time t |
| g | Gate center position |
| c_0 | Base camera location (0, y_0, z_0) |
| theta_t | Yaw offset about z-axis |
| R | Arc radius |
| w | Blend weight for look-target |
| F | Horizontal field of view (~87 deg) |
| m | FOV margin (2.5-5 deg) |
| g_z | Current gate center height |
| dt | Time step = 1/60 s |

## Fusion Equation

All modes (except DroneOnly) use gated late fusion with per-feature 64D gating:

```
e' = P_e(e)                          # Ego (drone) projection
s' = P_s(s)                          # Static projection
g_fuse = sigmoid(G([e', s']))        # Gate network
z = g_fuse * s' + (1 - g_fuse) * e'  # Fused latent
```

When a camera stream is ablated, its branch is short-circuited and contributes neither features nor gradients.

---

## Mode 1: FixedYaw (Randomized Fixed Yaw)

**Origin:** Fixed behind gate at c_0 = (0, y_0, z_0), 2.0 m behind gate on centerline

**Orientation:** At each episode reset, the azimuth about the gate centre is sampled uniformly from a curriculum-dependent range: +/-3 deg at L=3 increasing to +/-19 deg at L=23. The sampled yaw is held constant for the entire episode.

**Height:** Adaptive to gate center height (g_z)

**Curriculum-sensitive:** Yes (yaw range grows with level)

**Design rationale:** Provides a consistent gate-centric baseline without temporal view cues. The per-episode randomized yaw reduces overfitting to a single orientation. At higher curriculum levels, the drone may not spawn in view — particularly when its initial heading opposes the fixed yaw.

**Fusion regime:** Complementary, stable context

---

## Mode 2: YawSweep (Randomized Sweeping Yaw)

**Origin:** Fixed at c_0 (no positional motion)

**Orientation:** View direction yaws deterministically with a curriculum-scaled sinusoid. Per-environment phase and direction de-synchronize instances.

**Sweep amplitude:**

```
A(L) = A_min + frac(L) * (A_max - A_min)
A_min = 2 deg,  A_max = 19 deg
```

Linear interpolation across levels 3 to 23.

**Effective sweep speed** (approximately constant peak angular speed as A varies):

```
effective_speed = base_speed * curriculum_scale * (reference_amplitude / current_amplitude)
base_speed = 10.0 deg/s,  curriculum_scale in [1, 2],  reference_amplitude = 50 deg
```

**Yaw evolution:**

```
theta_t = A(L) * sin(d_env * (omega_k) + phi_env)
d_env in {-1, +1},  phi_env in (-pi, pi)
```

**Look-target update:**

```
t_t^(x,y) = c_0^(x,y) + d * [sin(theta_t * pi/180), cos(theta_t * pi/180)]
t_t^z = g_z
```

**Curriculum-sensitive:** Yes (amplitude grows with level)

**Design rationale:** Induces rotational feature drift without origin translation. Stresses fusion under view changes. At difficult levels, the sweep may start off-target so the gate and drone need not be in the same frame at reset, creating partial observability that benefits from recurrent memory (GRU).

**Risks:** Entrainment to periodic motion; HFOV edge clipping at large amplitudes.

**Fusion regime:** View-induced feature drift without parallax

---

## Mode 3: LockedFollow (Target-Locked Yaw)

**Origin:** Fixed at c_0 (no positional motion)

**Orientation:** Optical axis continuously reorients toward the drone. Look-target t_t = r_t.

```
c_t = (0, y_0, z_0),  z_0 in {1.5, g_z}
theta_t chosen so forward axis points along r_t - c_t
```

**Curriculum-sensitive:** No

**Design rationale:** Isolates orientation effects without parallax — tests whether reorientation alone stabilizes perception. Risk: during rapid lateral maneuvers, the gate exits the frame and spatial context is lost.

**Fusion regime:** Stabilized drone-centric framing (orientation only)

---

## Mode 4: DynFollow (Dynamic Drone Follow)

**Origin:** Trails the drone with a fixed world-frame offset.

```
c_t = r_t + Delta,  Delta = (0, -d, 0),  d = 2.0 m
```

**Orientation:** Looks at the drone, with a gate-blending bias when the gate would exit the HFOV:

```
yaw_d = bearing(c_t -> r_t)
yaw_g = bearing(c_t -> g)

if |yaw_g - yaw_d| > (F/2 - m):
    t_t = (1 - w) * r_t + w * g,  w = 0.2
else:
    t_t = r_t  (w = 0)
```

**Curriculum-sensitive:** No

**Design rationale:** Stabilizes a vehicle-centric view while preserving minimal gate awareness through the blending mechanism. Balances stability and context.

**Risks:** An ill-chosen offset d or aggressive drone motion can push the gate out of view and trigger frequent blending oscillations.

**Fusion regime:** Stabilized drone-centric framing (position + orientation)

---

## Mode 5: ArcFollow (Target Arc Follow)

**Origin:** Translates on a circular arc around the gate center.

```
theta(t) = omega * t + phi_env
omega = 2*pi / 600 frames  (~10 s per cycle)
phi_env: small per-environment random phase

c_t = g + R * [sin(theta(t)), -cos(theta(t)), 0]
R = 2.0 m (default)
```

**Vertical alignment:** Adaptive to g_z

**Orientation:** Look-target biased between drone and gate:

```
t_t = (1 - w) * r_t + w * g,  w = 0.3
```

**Curriculum-sensitive:** No

**Design rationale:** Introduces controlled lateral parallax from a moving origin. Probes fusion performance under origin motion with slow, predictable camera translation.

**Risks:** Reduced apparent drone detail at large R; potential entrainment to the oscillation frequency.

**Fusion regime:** Controlled parallax from moving origin

---

## Mode 6: DroneOnly (Drone-Only Ablation)

**Implementation:** The static stream is removed by exact-slice ablation of obs[86:150] (exocentric camera latents) prior to normalization. This short-circuits fusion to the onboard branch:

```
s' = 0
z = e' = P_e(e)
```

The static camera branch contributes neither features nor gradients.

**Design rationale:** Lower-bounds performance without static context. Quantifies reliance on the exocentric stream. Tends to degrade alignment and path efficiency on harder configurations where global spatial context matters.

**Fusion regime:** Single-stream (no fusion)

---

## Summary Table

| Mode | Streams | Origin | Orientation | Curriculum? | Look-target | Height | Distance |
|------|---------|--------|-------------|-------------|-------------|--------|----------|
| FixedYaw | Both | Fixed behind gate | Randomized yaw per episode | Yes | Gate centre | Adaptive | 2.0 m |
| YawSweep | Both | Fixed behind gate | Sinusoidal sweep | Yes | Gate + sweep | Adaptive | 2.0 m |
| LockedFollow | Both | Fixed behind gate | Tracks drone | No | Drone | Adaptive | 2.0 m |
| DynFollow | Both | Trails drone (-2m Y) | Drone + gate blend | No | Drone/gate | Drone height | Variable |
| ArcFollow | Both | Arc around gate (R=2m) | Drone + gate bias | No | Drone/gate | Adaptive | Radius R |
| DroneOnly | Onboard only | N/A | N/A | N/A | N/A | N/A | N/A |

## Induced Fusion Regimes

1. **Complementary, stable context** — FixedYaw
2. **View-induced feature drift without parallax** — YawSweep
3. **Stabilized drone-centric framing** — LockedFollow, DynFollow
4. **Controlled parallax** — ArcFollow
5. **Removal of exocentric stream** — DroneOnly

## Configuration

Camera modes are selected via YAML config or environment variables:

```yaml
camera:
  enable_yaw_sweep: true       # YawSweep
  enable_locked_follow: true   # LockedFollow
  enable_arc_follow: true      # ArcFollow
  enable_dynamic_following: true  # DynFollow
```

DroneOnly is achieved by ablating the static camera observation slice.
