#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize 3D spawn randomization (boxes and samples) with gate and environment bounds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--levels", nargs="*", type=int, default=[3, 23], help="Curriculum levels to visualize (ignored if --level_spans is set)")
    parser.add_argument("--level_spans", type=str, default=None, help="Comma-separated spans like '3-8,8-13,13-18,18-23' (overrides --levels)")
    parser.add_argument("--num_samples", type=int, default=500, help="Samples per level")
    parser.add_argument("--eval_stretch_enabled", type=int, default=None, help="Override EVAL_STRETCH_ENABLED (0/1)")
    parser.add_argument("--eval_stretch_end_level", type=int, default=None, help="Override EVAL_STRETCH_END_LEVEL")
    parser.add_argument("--save", type=str, default=None, help="Save figure to this path instead of showing")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI if saving")
    return parser.parse_args()


def set_eval_env_overrides(args: argparse.Namespace) -> None:
    if args.eval_stretch_enabled is not None:
        os.environ["EVAL_STRETCH_ENABLED"] = str(int(args.eval_stretch_enabled))
    if args.eval_stretch_end_level is not None:
        os.environ["EVAL_STRETCH_END_LEVEL"] = str(int(args.eval_stretch_end_level))


def get_spawn_cfg(level: int):
    """Lightweight import of task_config without importing the full aerial_gym package.
    This avoids initializing Isaac Gym / heavy deps when running visualization.
    """
    import importlib.util
    # Resolve path to navigation_task_config_gate.py relative to this file
    root_dir = os.path.dirname(os.path.dirname(__file__))  # aerial_gym/
    cfg_path = os.path.join(root_dir, "config", "task_config", "navigation_task_config_gate.py")
    spec = importlib.util.spec_from_file_location("nav_task_config", cfg_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    task_config = module.task_config
    return task_config.curriculum.get_spawn_ranges(level)


def sample_spawns(level: int, num_samples: int) -> np.ndarray:
    cfg = get_spawn_cfg(level)
    # X center is at 0 by construction; Y/Z centers returned by schedule
    x_half = float(cfg["x_half_span_m"])  # symmetric around 0
    y_center = float(cfg["y_center_m"])
    y_half = float(cfg["y_half_span_m"])  # can be 0
    z_center = float(cfg["z_center_m"])
    z_half = float(cfg["z_half_span_m"])
    xs = np.random.uniform(-x_half, x_half, size=(num_samples,))
    ys = np.random.uniform(y_center - y_half, y_center + y_half, size=(num_samples,))
    zs = np.random.uniform(z_center - z_half, z_center + z_half, size=(num_samples,))
    return np.stack([xs, ys, zs], axis=1)


def sample_spans(level_start: int, level_end: int, num_samples: int) -> np.ndarray:
    """Uniformly sample integer levels in [level_start, level_end] and draw a spawn per sample."""
    levels = np.random.randint(low=min(level_start, level_end), high=max(level_start, level_end) + 1, size=(num_samples,))
    pts = []
    for lvl in levels:
        pts.append(sample_spawns(int(lvl), 1)[0])
    return np.asarray(pts)


def spawn_box_vertices(level: int) -> Tuple[np.ndarray, np.ndarray]:
    cfg = get_spawn_cfg(level)
    x_half = float(cfg["x_half_span_m"])  # center 0
    y_center = float(cfg["y_center_m"])  # half span possibly 0
    y_half = float(cfg["y_half_span_m"])
    z_center = float(cfg["z_center_m"])  # half span > 0
    z_half = float(cfg["z_half_span_m"])
    x_min, x_max = -x_half, x_half
    y_min, y_max = y_center - y_half, y_center + y_half
    z_min, z_max = z_center - z_half, z_center + z_half
    # 8 corners
    corners = np.array([
        [x_min, y_min, z_min], [x_min, y_min, z_max], [x_min, y_max, z_min], [x_min, y_max, z_max],
        [x_max, y_min, z_min], [x_max, y_min, z_max], [x_max, y_max, z_min], [x_max, y_max, z_max],
    ])
    # 12 edges as pairs of indices into corners
    edges = np.array([
        [0, 1], [0, 2], [0, 4], [7, 5], [7, 3], [7, 6], [2, 3], [2, 6], [4, 5], [4, 6], [1, 3], [1, 5]
    ])
    return corners, edges


def plot_environment(ax):
    # Environment bounds: [-4,4] x [-4,4] x [0,4]
    xmin, xmax = -4.0, 4.0
    ymin, ymax = -4.0, 4.0
    zmin, zmax = 0.0, 4.0
    corners = np.array([
        [xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
        [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax],
    ])
    edges = np.array([
        [0, 1], [0, 2], [0, 4], [7, 5], [7, 3], [7, 6], [2, 3], [2, 6], [4, 5], [4, 6], [1, 3], [1, 5]
    ])
    for e in edges:
        a, b = corners[e[0]], corners[e[1]]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#888888", linewidth=0.8, alpha=0.8)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")


def plot_gate(ax, scale_percent: int = 100):
    # Gate plane at y=0, opening rectangle in X-Z plane
    # Base opening (100%): width in X ≈ 2.5 m (±1.25), height in Z ≈ 2.3 m (0.1..2.4)
    width_x = 2.5 * (scale_percent / 100.0)
    half_x = width_x / 2.0
    z_min, z_max = 0.1 * (scale_percent / 100.0), 2.4 * (scale_percent / 100.0)
    xs = np.array([-half_x, half_x, half_x, -half_x, -half_x])
    ys = np.zeros_like(xs)
    zs = np.array([z_min, z_min, z_max, z_max, z_min])
    ax.plot(xs, ys, zs, color="#d62728", linewidth=2.0, alpha=0.9, label=f"Gate {scale_percent}%")


def color_cycle(n: int) -> List[str]:
    base = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    return [base[i % len(base)] for i in range(n)]


def parse_spans(span_str: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for part in (span_str or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Bad span '{part}', expected a-b")
        a, b = part.split("-", 1)
        spans.append((int(a), int(b)))
    return spans


def main():
    args = parse_args()
    set_eval_env_overrides(args)

    import matplotlib.pyplot as plt  # defer import
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    plot_environment(ax)
    plot_gate(ax, scale_percent=100)

    if args.level_spans:
        spans = parse_spans(args.level_spans)
        colors = color_cycle(len(spans))
        for idx, (a, b) in enumerate(spans):
            # Draw boxes at both endpoints of the span
            for lvl in (a, b):
                corners, edges = spawn_box_vertices(lvl)
                for e in edges:
                    u, v = corners[e[0]], corners[e[1]]
                    ax.plot([u[0], v[0]], [u[1], v[1]], [u[2], v[2]], color=colors[idx], linewidth=1.2, alpha=0.9)
            # Sample points across the entire span
            pts = sample_spans(a, b, args.num_samples)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, alpha=0.35, color=colors[idx], label=f"L{a}–{b}")
    else:
        colors = color_cycle(len(args.levels))
        for idx, lvl in enumerate(args.levels):
            # Box
            corners, edges = spawn_box_vertices(lvl)
            for e in edges:
                a, b = corners[e[0]], corners[e[1]]
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=colors[idx], linewidth=1.2, alpha=0.9)
            # Samples
            pts = sample_spawns(lvl, args.num_samples)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=6, alpha=0.3, color=colors[idx], label=f"L={lvl}")

    ax.legend(loc="upper left")
    ax.set_title("Spawn Randomization vs Gate and Environment")
    ax.view_init(elev=20, azim=-60)

    if args.save:
        fig.tight_layout()
        fig.savefig(args.save, dpi=args.dpi)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()


