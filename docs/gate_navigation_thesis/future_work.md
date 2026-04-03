# Future Work

> Based on Chapter 7.2 of: Z. Ruso, *"Reinforcement Learning for Cooperative Multi-View Depth-Based Perception in Autonomous UAV Navigation,"* MSc Thesis, UCL, 2025.

## 1. Sim-to-Real Transfer

**Status:** Planned, hardware acquired

All training and evaluation were conducted in simulation. The critical next step is validating the learned policy on real UAV hardware in cluttered, GPS-denied environments.

### Planned Setup

- **Platform:** Holybro X500 V2 + NVIDIA Jetson Orin NX + Intel RealSense D455
- **Software:** ROS 2 Humble, MAVSDK Offboard, PX4 flight stack
- **Inference:** TensorRT (FP16/INT8) targeting sub-50 ms end-to-end latency
- **Communication:** uORB-RTPS bridge for PX4 integration

### Key Challenges

| Challenge | Mitigation |
|-----------|-----------|
| Sensor noise mismatch | Domain randomization during training (Gaussian, dropout, frame freeze) |
| Dynamics mismatch | Force/torque disturbances + system identification on real platform |
| Lighting/appearance | Depth-only pipeline abstracts away appearance; robust to lighting changes |
| Camera calibration | Verify D455 intrinsics/extrinsics match simulation parameters |
| Communication latency | Transmit 64D latents (~256 bytes) instead of full depth frames |
| RNN cold-start | Warm-up period after reset before trusting policy outputs |

### Why Depth Helps

Depth observations abstract away appearance details (textures, lighting, colors) and provide geometric information that translates more directly across sim-to-real domains. This is a deliberate design choice that should improve transfer compared to RGB-trained policies.

See [sim2real.md](sim2real.md) for the full deployment pipeline documentation.

## 2. Architectural & Algorithmic Extensions

### End-to-End Vision Training

The current pipeline freezes the VAE encoder during RL. Future work could:

- **Partially fine-tune** the encoder (unfreeze last N layers) during RL training
- **Train end-to-end** on higher-resolution images with deeper convolutional backbones
- **Use spatiotemporal convolutions** to align features across time and between views

Trade-off: end-to-end training is more expressive but risks representation collapse and requires significantly more compute.

### Advanced Fusion Mechanisms

The current per-feature gating is effective but simple. Alternatives to explore:

| Approach | Advantage | Complexity |
|----------|-----------|-----------|
| **Cross-view transformer** | Spatial attention across views | High |
| **Cross-spatial attention maps** | Learn where to look in each view | High |
| **Confidence-weighted fusion** | Reliability-aware stream weighting | Medium |
| **Multi-head attention** | Multiple fusion strategies in parallel | Medium |

The gating mechanism currently has no notion of confidence or reliability — under distribution shift (miscalibration, latency), routing may be brittle. A confidence-aware fusion gate could improve robustness.

### Alternative RL Algorithms

| Algorithm | Potential Benefit |
|-----------|------------------|
| **Off-policy methods** (SAC, TD3) | Better sample efficiency |
| **Model-based RL** | Better long-term planning and data efficiency |
| **Constrained RL** | Formal safety guarantees during training |
| **Multi-objective RL** | Pareto-optimal trade-off between speed and safety |

The current PPO setup prioritizes stability and traceability over peak sample efficiency. Off-policy methods could reduce the 2M-frame training budget significantly.

### Additional Sensor Modalities

- **Event cameras**: High temporal resolution for fast flight, low bandwidth
- **LiDAR scans**: Complementary range data, especially for sparse environments
- **IMU fusion**: Tighter integration of inertial data into the observation

Any added sensors must be weighed against weight, power, and computational limits on the UAV.

## 3. Multi-Agent / Swarm Extensions

### Vision

Scale from a single-drone + static-camera setup to multiple UAVs that actively cooperate:

- Drones position themselves to cover each other's blind spots
- Ad hoc flying sensor network with overlapping fields of view
- Leapfrogging: drones take turns being the observer and the navigator

### Challenges

| Challenge | Description |
|-----------|-------------|
| **Communication bandwidth** | Sharing visual data or policy information requires reliable, low-latency links |
| **Coordination** | Multi-drone collision avoidance, task allocation, consensus |
| **Scalability** | Extending gated fusion from 2 streams to N streams |
| **Decentralized execution** | Each drone must act on local + shared information |
| **Partial connectivity** | Intermittent links, packet loss, variable latency |

### Communication-Efficient Strategies

- Transmit **compact latent representations** (64D per drone = 256 bytes) rather than raw depth
- **Learned communication protocols**: train drones to decide what/when to share
- **Hierarchical fusion**: local pairs fuse first, then share compressed summaries

## 4. Environment & Task Extensions

### Dynamic Obstacles

Current training uses only static obstacles. Adding moving objects (other drones, people, vehicles) would:
- Test reactive collision avoidance
- Require temporal reasoning about obstacle trajectories
- Better represent real-world deployment scenarios

### Multiple Gates / Waypoints

Extend from single-gate traversal to multi-waypoint navigation:
- Sequential gate traversal (racing)
- Planning under partial observability
- Memory-dependent routing decisions

### Outdoor Environments

Move beyond the 8x8x4m indoor arena:
- Larger environments with longer approach corridors
- Wind disturbances and atmospheric turbulence
- Variable terrain and natural obstacles

## 5. Reward & Training Improvements

### Reward Engineering

- **Automated reward tuning**: Use reward search or intrinsic motivation instead of hand-tuned weights
- **Constrained formulation**: Replace penalty-based safety with hard constraint satisfaction
- **Multi-objective returns**: Separate speed, safety, and accuracy into independent objectives

### Curriculum Improvements

- **Adaptive curriculum**: Data-driven difficulty adjustment instead of fixed success-rate thresholds
- **Reverse curriculum**: Start from success states and expand backward
- **Population-based training**: Evolve curriculum schedules alongside hyperparameters

## 6. Quantization & Deployment Optimization

### Edge Inference

- **TensorRT optimization**: FP16/INT8 quantization for Jetson deployment
- **Knowledge distillation**: Train a smaller student network from the full policy
- **Rate-distortion coding**: Compress latent transmissions with learned quantization

### Latency Optimization

- **Pipeline parallelism**: Overlap VAE encoding with physics stepping
- **Asynchronous inference**: Decouple observation collection from policy execution
- **Predictive models**: Forecast next observation to pre-compute actions

## Priority Ranking

| Priority | Direction | Impact | Effort |
|----------|-----------|--------|--------|
| 1 | **Sim-to-real transfer** | Critical for validation | High |
| 2 | **Dynamic obstacles** | Key for real-world relevance | Medium |
| 3 | **End-to-end encoder training** | Performance ceiling | High |
| 4 | **Multi-agent swarm** | Long-term vision | Very High |
| 5 | **Advanced fusion (transformers)** | Potential gains | Medium |
| 6 | **Off-policy RL** | Training efficiency | Medium |
| 7 | **Edge quantization** | Deployment readiness | Low-Medium |
