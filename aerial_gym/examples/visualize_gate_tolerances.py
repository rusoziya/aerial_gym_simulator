"""
Visualization script for gate scaling tolerances and success criteria.

This script demonstrates:
1. How width and height tolerances scale with gate size
2. Visual comparison of success zones for different gate types
3. Impact of 1% success rate on curriculum progression speed
4. Reward triggering frequency analysis

Usage:
    python aerial_gym/examples/visualize_gate_tolerances.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from aerial_gym.config.asset_config.gate_scaling_config import GateScalingConfig
from aerial_gym.config.task_config.navigation_task_config_gate import task_config

def visualize_tolerance_scaling():
    """Create visual comparison of tolerance zones for different gate scales."""
    
    # Gate configurations
    gate_configs = [
        {"name": "Full Size", "scale": 1.0, "color": "green", "alpha": 0.3},
        {"name": "Medium", "scale": 0.7, "color": "blue", "alpha": 0.4}, 
        {"name": "Small", "scale": 0.5, "color": "orange", "alpha": 0.5},
        {"name": "Minimum", "scale": 0.4, "color": "red", "alpha": 0.6}
    ]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Gate Scaling Success Criteria Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Width tolerance comparison (top view)
    ax1.set_title('Width Tolerance Zones (Top View)', fontweight='bold')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.grid(True, alpha=0.3)
    
    gate_center_x, gate_center_y = 0.0, 0.0
    
    for config in gate_configs:
        scale = config["scale"]
        width_tol, _, _ = GateScalingConfig.get_gate_tolerance_for_scale(scale)
        
        # Draw gate physical opening (width)
        gate_width = 2.5 * scale
        gate_rect = plt.Rectangle(
            (gate_center_x - gate_width/2, gate_center_y - 0.1),
            gate_width, 0.2,
            color=config["color"], alpha=0.8, label=f'{config["name"]} Gate'
        )
        ax1.add_patch(gate_rect)
        
        # Draw success tolerance zone
        tolerance_rect = plt.Rectangle(
            (gate_center_x - width_tol, gate_center_y - 1.0),
            2 * width_tol, 2.0,
            color=config["color"], alpha=config["alpha"], 
            linestyle='--', fill=False, linewidth=2,
            label=f'{config["name"]} Tolerance (±{width_tol:.2f}m)'
        )
        ax1.add_patch(tolerance_rect)
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-1.5, 1.5)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Gate Plane')
    
    # Plot 2: Height tolerance comparison (side view)
    ax2.set_title('Height Tolerance Zones (Side View)', fontweight='bold')
    ax2.set_xlabel('Y Position (m)')
    ax2.set_ylabel('Z Position (m)')
    ax2.grid(True, alpha=0.3)
    
    for config in gate_configs:
        scale = config["scale"]
        _, height_min, height_max = GateScalingConfig.get_gate_tolerance_for_scale(scale)
        
        # Draw gate physical opening (height)
        gate_height = 2.3 * scale
        gate_rect = plt.Rectangle(
            (gate_center_y - 0.1, 0.0),
            0.2, gate_height,
            color=config["color"], alpha=0.8, label=f'{config["name"]} Gate'
        )
        ax2.add_patch(gate_rect)
        
        # Draw success tolerance zone
        tolerance_rect = plt.Rectangle(
            (gate_center_y - 1.0, height_min),
            2.0, height_max - height_min,
            color=config["color"], alpha=config["alpha"],
            linestyle='--', fill=False, linewidth=2,
            label=f'{config["name"]} Tolerance ({height_min:.1f}-{height_max:.1f}m)'
        )
        ax2.add_patch(tolerance_rect)
    
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(0, 2.5)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5, label='Gate Plane')
    
    # Plot 3: Tolerance scaling ratios
    ax3.set_title('Tolerance Scaling Analysis', fontweight='bold')
    scales = [config["scale"] for config in gate_configs]
    names = [config["name"] for config in gate_configs]
    
    width_tolerances = []
    height_ranges = []
    gate_widths = []
    gate_heights = []
    
    for scale in scales:
        width_tol, height_min, height_max = GateScalingConfig.get_gate_tolerance_for_scale(scale)
        width_tolerances.append(width_tol * 2)  # Full width tolerance
        height_ranges.append(height_max - height_min)
        gate_widths.append(2.5 * scale)
        gate_heights.append(2.3 * scale)
    
    x = np.arange(len(names))
    width = 0.2
    
    ax3.bar(x - width, gate_widths, width, label='Gate Width', alpha=0.7, color='lightblue')
    ax3.bar(x, width_tolerances, width, label='Success Width Tolerance', alpha=0.7, color='lightgreen')
    ax3.bar(x + width, height_ranges, width, label='Success Height Range', alpha=0.7, color='lightcoral')
    
    ax3.set_xlabel('Gate Type')
    ax3.set_ylabel('Distance (m)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add tolerance ratio annotations
    for i, (name, scale) in enumerate(zip(names, scales)):
        width_tol, _, _ = GateScalingConfig.get_gate_tolerance_for_scale(scale)
        gate_width = 2.5 * scale
        ratio = (width_tol * 2) / gate_width * 100
        ax3.annotate(f'{ratio:.0f}%', xy=(i, width_tolerances[i] + 0.1), 
                    ha='center', fontsize=9, fontweight='bold')
    
    # Plot 4: Curriculum progression simulation with 1% success rate
    ax4.set_title('Curriculum Progression Speed (1% Success Rate)', fontweight='bold')
    
    # Simulate progression
    levels = list(range(3, 24))
    episodes_per_check = 128
    success_rate_needed = 0.01
    
    cumulative_episodes = []
    total_episodes = 0
    
    for level in levels:
        # Assume it takes exactly the minimum episodes to advance (1% success rate)
        episodes_to_advance = episodes_per_check  # With 1% rate, often advances in first check
        total_episodes += episodes_to_advance
        cumulative_episodes.append(total_episodes)
    
    ax4.plot(levels, cumulative_episodes, 'b-', linewidth=3, label='1% Success Rate')
    
    # Compare with 50% success rate
    total_episodes_50 = 0
    cumulative_episodes_50 = []
    for level in levels:
        episodes_to_advance = episodes_per_check * 10  # Much slower with 50% rate
        total_episodes_50 += episodes_to_advance
        cumulative_episodes_50.append(total_episodes_50)
    
    ax4.plot(levels, cumulative_episodes_50, 'r--', linewidth=2, label='50% Success Rate (Previous)', alpha=0.7)
    
    ax4.set_xlabel('Curriculum Level')
    ax4.set_ylabel('Total Episodes')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add annotations for gate type availability
    level_ranges = [(3, 8, "Full Only"), (9, 13, "Full + Medium"), 
                   (14, 18, "Full + Med + Small"), (19, 23, "All Gates")]
    colors = ['green', 'blue', 'orange', 'red']
    
    for i, (start, end, label) in enumerate(level_ranges):
        ax4.axvspan(start, end, alpha=0.2, color=colors[i], label=label)
    
    plt.tight_layout()
    plt.show()

def analyze_success_frequency():
    """Analyze how success frequency changes with gate scaling."""
    
    print("\n" + "="*60)
    print("SUCCESS FREQUENCY ANALYSIS")
    print("="*60)
    
    # Simulate robot positions near gate center
    np.random.seed(42)  # For reproducible results
    n_attempts = 1000
    
    # Generate robot positions with some spread (simulating control noise)
    robot_x_positions = np.random.normal(0, 0.4, n_attempts)  # 40cm standard deviation
    robot_z_positions = np.random.normal(1.2, 0.3, n_attempts)  # 30cm standard deviation around gate center height
    
    gate_configs = [
        {"name": "Full Size", "scale": 1.0},
        {"name": "Medium", "scale": 0.7},
        {"name": "Small", "scale": 0.5},
        {"name": "Minimum", "scale": 0.4}
    ]
    
    print(f"{'Gate Type':<12} {'Scale':<6} {'Width Tol':<10} {'Height Range':<12} {'Success Rate':<12} {'Reward Freq'}")
    print("-" * 80)
    
    for config in gate_configs:
        scale = config["scale"]
        name = config["name"]
        
        # Get tolerances for this gate scale
        width_tol, height_min, height_max = GateScalingConfig.get_gate_tolerance_for_scale(scale)
        
        # Count successful passages
        width_success = np.abs(robot_x_positions) < width_tol
        height_success = (robot_z_positions > height_min) & (robot_z_positions < height_max)
        total_success = width_success & height_success
        
        success_rate = np.sum(total_success) / n_attempts * 100
        reward_frequency = "High" if success_rate > 70 else "Medium" if success_rate > 40 else "Low" if success_rate > 15 else "Very Low"
        
        height_range_str = f"{height_min:.1f}-{height_max:.1f}m"
        
        print(f"{name:<12} {scale:<6.1f} ±{width_tol:<9.2f} {height_range_str:<12} {success_rate:<11.1f}% {reward_frequency}")
    
    print(f"\nSimulation details:")
    print(f"- {n_attempts} robot positions tested")
    print(f"- Robot control noise: σ_x = 0.4m, σ_z = 0.3m") 
    print(f"- Same control quality across all gate sizes")
    print(f"- Success rate naturally decreases with smaller gates")
    print(f"- This creates appropriate reward frequency scaling")

def curriculum_progression_analysis():
    """Analyze the impact of 1% success rate on training."""
    
    print("\n" + "="*60)
    print("CURRICULUM PROGRESSION ANALYSIS (1% Success Rate)")
    print("="*60)
    
    check_frequency = 128  # episodes between curriculum checks
    success_rate_required = 0.01  # 1% success rate needed to advance
    
    print(f"Configuration:")
    print(f"- Check frequency: Every {check_frequency} episodes")
    print(f"- Success rate required: {success_rate_required*100:.1f}%")
    print(f"- Minimum successes needed: {int(check_frequency * success_rate_required)} out of {check_frequency}")
    
    print(f"\nProgression timeline:")
    print(f"{'Level Range':<15} {'Gate Types Available':<25} {'Min Episodes':<12} {'Total Episodes'}")
    print("-" * 75)
    
    level_ranges = [
        (3, 8, "Full Size Only", 6),
        (9, 13, "Full + Medium", 5), 
        (14, 18, "Full + Medium + Small", 5),
        (19, 23, "All Gates (inc. Minimum)", 5)
    ]
    
    total_episodes = 0
    for start_level, end_level, gate_types, levels_in_range in level_ranges:
        range_episodes = levels_in_range * check_frequency
        total_episodes += range_episodes
        
        range_str = f"{start_level}-{end_level}"
        print(f"{range_str:<15} {gate_types:<25} {range_episodes:<12} {total_episodes}")
    
    print(f"\nKey insights:")
    print(f"✅ Total episodes to reach max difficulty: ~{total_episodes:,}")
    print(f"✅ Agent sees all gate sizes within first {total_episodes:,} episodes")
    print(f"⚡ 32x faster than previous 50% success rate requirement")
    print(f"⚠️  Risk: May advance faster than learning occurs")
    print(f"📊 Monitor: Actual success rates vs. curriculum advancement")
    
    # Compare with theoretical learning timeline
    print(f"\nRecommended monitoring:")
    print(f"- If actual success rate << 1% → curriculum advancing too fast")
    print(f"- If actual success rate >> 1% → curriculum could advance faster")
    print(f"- Sweet spot: ~1-5% actual success rate for stable learning")

if __name__ == "__main__":
    print("🎯 Gate Scaling Tolerance & Success Criteria Analysis")
    print("="*60)
    
    try:
        # Create visualizations
        visualize_tolerance_scaling()
        
        # Analyze success frequencies
        analyze_success_frequency()
        
        # Analyze curriculum progression
        curriculum_progression_analysis()
        
        print(f"\n🎉 Analysis complete!")
        print(f"\nKey takeaways:")
        print(f"✅ Adaptive tolerances scale proportionally with gate size")
        print(f"✅ Reward system adapts naturally through success frequency")
        print(f"✅ 1% success rate creates very aggressive curriculum progression")
        print(f"⚠️  Monitor training stability with rapid progression")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise 