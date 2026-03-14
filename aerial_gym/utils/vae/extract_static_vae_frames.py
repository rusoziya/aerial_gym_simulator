from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

from PIL import Image, ImageSequence


DEFAULT_SRC_GLOBS = [
    # Primary: static D455 depth GIFs produced during gate-navigation runs (clean depth only)
    "aerial_gym/rl_training/sample_factory/aerialgym_examples/gif_episodes/episode_*static_depth*.gif",
    # Optional documentation snippets (disabled by default via --include_docs)
    "docs/gifs/d455_camera_depth_frames_*.gif",
]


def find_gifs(src_globs, include_docs=False) -> None:
    candidates = []
    for pattern in src_globs:
        if ("docs/" in pattern) and not include_docs:
            continue
        for p in Path.cwd().glob(pattern):
            candidates.append(p)

    # Filter strictly to clean static depth views (exclude: seg/noised/merged/drone)
    filtered = []
    for p in candidates:
        name = p.name.lower()
        # Must be static depth
        if ("static" not in name) or ("depth" not in name):
            continue
        # Exclude noisy/segmented/merged/other
        if (
            ("drone" in name)
            or ("merged" in name)
            or ("seg" in name)
            or ("noised" in name)
            or ("noise" in name)
        ):
            continue
        filtered.append(p)

    # Deduplicate and sort for reproducibility
    filtered = sorted(set(filtered))
    return filtered


def extract_frames_from_gif(gif_path: Path, out_dir: Path, resize_wh=(480, 270),
                            every_k=1, prefix=None, max_frames=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or gif_path.stem
    saved = []

    with Image.open(gif_path) as im:
        idx = 0
        for frame_idx, frame in enumerate(ImageSequence.Iterator(im)):
            if frame_idx % every_k != 0:
                continue
            if max_frames is not None and len(saved) >= max_frames:
                break

            # Convert to single-channel grayscale and resize to RL resolution (W,H)=(240,135)
            fr = frame.convert("L").resize(resize_wh, Image.BILINEAR)

            out_name = f"{prefix}_f{frame_idx:05d}.png"
            out_path = out_dir / out_name
            fr.save(out_path, format="PNG", optimize=True)
            saved.append(out_path)
            idx += 1
    return saved


def split_by_gif(gif_paths, ratios=(0.8, 0.1, 0.1), seed=17) -> None:
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = random.Random(seed)
    gifs = list(gif_paths)
    rng.shuffle(gifs)
    n = len(gifs)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train = gifs[:n_train]
    val = gifs[n_train:n_train + n_val]
    test = gifs[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def write_index(index_rows, out_csv) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image_path", "width", "height", "near", "far"])
        for r in index_rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract static-camera depth frames from GIFs for VAE training.")
    parser.add_argument("--out_dir", type=str, default="aerial_gym/utils/vae/datasets/static_frames",
                        help="Output directory for extracted PNG frames (train/val/test subdirs will be created)")
    parser.add_argument("--include_docs", action="store_true", help="Also include docs/gifs sources")
    parser.add_argument("--src_glob", action="append", default=None,
                        help="Additional glob(s) for GIFs, e.g., 'aerial_gym/utils/vae/combined/*.gif'. Can be passed multiple times.")
    parser.add_argument("--every_k", type=int, default=1, help="Keep 1 of every K frames from each GIF")
    parser.add_argument("--width", type=int, default=480, help="Resize width for saved frames")
    parser.add_argument("--height", type=int, default=270, help="Resize height for saved frames")
    parser.add_argument("--max_frames_per_gif", type=int, default=None, help="Optional cap per GIF")
    parser.add_argument("--seed", type=int, default=17, help="Split seed")
    parser.add_argument("--near", type=float, default=0.4, help="Near depth used for normalization metadata")
    parser.add_argument("--far", type=float, default=20.0, help="Far depth used for normalization metadata")

    args = parser.parse_args()
    out_root = Path(args.out_dir)

    # Discover candidate GIFs (static depth, D455)
    src_patterns = list(DEFAULT_SRC_GLOBS)
    if args.src_glob:
        src_patterns.extend(args.src_glob)
    gifs = find_gifs(src_patterns, include_docs=args.include_docs)
    if not gifs:
        print("No matching static depth GIFs found. Check source glob patterns.")
        return

    splits = split_by_gif(gifs, seed=args.seed)
    index_rows = []

    resize_wh = (args.width, args.height)

    for split, split_gifs in splits.items():
        split_dir = out_root / split
        for gif_path in split_gifs:
            frames = extract_frames_from_gif(
                gif_path,
                split_dir,
                resize_wh=resize_wh,
                every_k=args.every_k,
                prefix=gif_path.stem,
                max_frames=args.max_frames_per_gif,
            )
            for fp in frames:
                index_rows.append([split, str(fp), resize_wh[0], resize_wh[1], args.near, args.far])

    write_index(index_rows, out_root / "index.csv")
    print(f"Done. Extracted {len(index_rows)} frames from {len(gifs)} GIFs to {out_root}")


if __name__ == "__main__":
    main()


