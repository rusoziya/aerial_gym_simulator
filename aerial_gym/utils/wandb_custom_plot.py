#!/usr/bin/env python3
from __future__ import annotations

"""
W&B Custom Plot Utility

Examples:
  # Plot policy_stats/avg_successes vs frames for all runs in a project
  python -m aerial_gym.utils.wandb_custom_plot \
    --entity ziya-ruso-ucl \
    --project gate_navigation_dual_camera \
    --y policy_stats/avg_successes \
    --x frames

  # Filter runs by tag and compare multiple metrics
  python -m aerial_gym.utils.wandb_custom_plot \
    --entity ziya-ruso-ucl \
    --project gate_navigation_dual_camera \
    --filter '{"tags":{"$in":["baseline"]}}' \
    --y policy_stats/avg_successes curriculum/level \
    --x frames \
    --title "Baseline: Success & Level vs Frames"

  # Select specific run IDs and save to PNG
  python -m aerial_gym.utils.wandb_custom_plot \
    --entity ziya-ruso-ucl \
    --project gate_navigation_dual_camera \
    --run_ids abc123 def456 \
    --y policy_stats/avg_successes \
    --x frames \
    --output ./wandb_plot.png
"""

import argparse
import json
import sys
from typing import List, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

try:
    from wandb import Api
except Exception as e:
    print("wandb not installed. pip install wandb", file=sys.stderr)
    raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W&B custom plotting utility")
    p.add_argument("--entity", required=True, help="W&B entity (user or team)")
    p.add_argument("--project", required=True, help="W&B project name")
    p.add_argument("--run_ids", nargs="*", default=None, help="Optional list of run IDs to include")
    p.add_argument(
        "--filter",
        type=str,
        default=None,
        help='Optional W&B filter JSON (e.g. \"{\\"tags\\":{\\"$in\\":[\\"baseline\\"]}}\")',
    )
    p.add_argument(
        "--x",
        default="frames",
        help="X-axis key (common choices: frames, global_step)",
    )
    p.add_argument("--y", nargs="+", required=True, help="One or more Y metrics to plot")
    p.add_argument("--smoothing", type=int, default=0, help="EMA window for smoothing (0 disables)")
    p.add_argument("--max_points", type=int, default=2000, help="Downsample to at most N points per series")
    p.add_argument("--output", default=None, help="Path to save the figure (png/pdf). If not set, show()")
    p.add_argument("--title", default=None, help="Figure title")
    p.add_argument("--legend_loc", default="best", help="Legend location")
    p.add_argument(
        "--per_run",
        action="store_true",
        help="Plot each run as a separate line (default). If not set, average across runs per step",
    )
    p.add_argument(
        "--avg_across_runs",
        action="store_true",
        help="Average runs for each metric (mutually exclusive with --per_run)",
    )
    return p.parse_args()


def _ema_smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 0 or y.size == 0:
        return y
    alpha = 2.0 / (window + 1.0)
    out = np.empty_like(y, dtype=np.float64)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> (np.ndarray, np.ndarray):
    if max_points <= 0 or len(x) <= max_points:
        return x, y
    step = max(1, len(x) // max_points)
    return x[::step], y[::step]


def _fetch_runs(api: Api, entity: str, project: str, run_ids: Optional[List[str]], filt: Optional[Dict]) -> List:
    path = f"{entity}/{project}"
    if run_ids:
        runs = []
        for rid in run_ids:
            runs.append(api.run(f"{path}/{rid}"))
        return runs
    return api.runs(path=path, filters=filt or {})


def _get_series(run, x_key: str, y_key: str) -> None:
    # Prefer pandas History for speed and filtering
    try:
        df = run.history(keys=[x_key, y_key], pandas=True)
        if df is None or df.empty:
            return None, None
        x = df[x_key].to_numpy(dtype=np.float64)
        y = df[y_key].to_numpy(dtype=np.float64)
        # Remove NaNs
        good = np.isfinite(x) & np.isfinite(y)
        return x[good], y[good]
    except Exception:
        # Fallback to scan_history
        xs, ys = [], []
        for row in run.scan_history(keys=[x_key, y_key]):
            xv = row.get(x_key, None)
            yv = row.get(y_key, None)
            if xv is None or yv is None:
                continue
            try:
                xs.append(float(xv))
                ys.append(float(yv))
            except Exception:
                continue
        if not xs:
            return None, None
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        good = np.isfinite(x) & np.isfinite(y)
        return x[good], y[good]


def main() -> None:
    args = parse_args()
    if args.per_run and args.avg_across_runs:
        print("Choose either --per_run or --avg_across_runs, not both", file=sys.stderr)
        sys.exit(2)
    if not args.per_run and not args.avg_across_runs:
        args.per_run = True

    try:
        filt = json.loads(args.filter) if args.filter else None
    except Exception as e:
        print(f"Invalid --filter JSON: {e}", file=sys.stderr)
        sys.exit(2)

    api = Api()
    runs = _fetch_runs(api, args.entity, args.project, args.run_ids, filt)
    if not runs:
        print("No runs matched.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel(args.x)
    ax.set_ylabel(", ".join(args.y) if len(args.y) > 1 else args.y[0])
    if args.title:
        ax.set_title(args.title)

    colors = plt.cm.tab10.colors
    color_idx = 0

    for y_key in args.y:
        if args.per_run:
            for run in runs:
                x, y = _get_series(run, args.x, y_key)
                if x is None:
                    continue
                if args.smoothing > 0:
                    y = _ema_smooth(y, args.smoothing)
                x, y = _downsample(x, y, args.max_points)
                label = f"{run.name or run.id}:{y_key}"
                ax.plot(x, y, label=label, alpha=0.9, linewidth=1.5)
        else:
            # Average across runs by interpolating to a common grid
            series = []
            xmin, xmax = float('inf'), -float('inf')
            for run in runs:
                x, y = _get_series(run, args.x, y_key)
                if x is None:
                    continue
                xmin = min(xmin, float(np.min(x)))
                xmax = max(xmax, float(np.max(x)))
                series.append((x, y))
            if not series:
                continue
            grid = np.linspace(xmin, xmax, num=min(args.max_points, 2000))
            vals = []
            for x, y in series:
                try:
                    yi = np.interp(grid, x, y)
                except Exception:
                    # If interpolation fails (unsorted x), sort first
                    order = np.argsort(x)
                    yi = np.interp(grid, x[order], y[order])
                if args.smoothing > 0:
                    yi = _ema_smooth(yi, args.smoothing)
                vals.append(yi)
            mean = np.nanmean(vals, axis=0)
            std = np.nanstd(vals, axis=0)
            color = colors[color_idx % len(colors)]
            color_idx += 1
            ax.plot(grid, mean, color=color, label=f"avg:{y_key}", linewidth=2.0)
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    ax.grid(True, alpha=0.25)
    ax.legend(loc=args.legend_loc, fontsize=9)

    if args.output:
        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        print(f"Saved: {args.output}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()


