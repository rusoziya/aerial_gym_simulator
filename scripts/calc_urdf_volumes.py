#!/usr/bin/env python3
"""
Compute approximate volumes for URDF models used as obstacles (non-gate URDFs).

- Scans one or more directories for .urdf files (defaults to resources/models/environment_assets/* except gates/)
- For each URDF, sums volumes of collision geometries (falls back to visual if no collision)
- Supported primitives: box (size="x y z"), cylinder (radius/length), sphere (radius)
- Meshes: uses trimesh if available; otherwise approximates with axis-aligned bounding box (AABB)

Usage:
  python scripts/calc_urdf_volumes.py \
      --dirs resources/models/environment_assets/objects \
             resources/models/environment_assets/trees \
             resources/models/environment_assets/walls \
             resources/models/environment_assets/panels \
             resources/models/environment_assets/thin

Notes:
- Gate URDFs are skipped (any path containing "/gates/" or filenames containing "gate").
- Outputs a CSV-like table to stdout and a total volume summary.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

try:
    import trimesh  # type: ignore
    _HAS_TRIMESH = True
except Exception:
    trimesh = None  # type: ignore
    _HAS_TRIMESH = False


def _to_floats(text: Optional[str]) -> Optional[List[float]]:
    if text is None:
        return None
    try:
        return [float(x) for x in text.strip().split()]
    except Exception:
        return None


def _resolve_mesh_path(urdf_path: str, mesh_filename: str) -> Optional[str]:
    # URDF may use package:// or relative paths
    if not mesh_filename:
        return None
    if mesh_filename.startswith("package://"):
        # Best-effort: strip package:// and treat as relative from repo root
        mesh_rel = mesh_filename.replace("package://", "").lstrip("/")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        cand = os.path.join(repo_root, mesh_rel)
        return cand if os.path.exists(cand) else None
    if mesh_filename.startswith("file://"):
        cand = mesh_filename.replace("file://", "")
        return cand if os.path.exists(cand) else None
    # Relative to the URDF file
    cand = os.path.normpath(os.path.join(os.path.dirname(urdf_path), mesh_filename))
    return cand if os.path.exists(cand) else None


def _mesh_volume(mesh_path: str, scale: Optional[List[float]]) -> Optional[float]:
    try:
        if _HAS_TRIMESH:
            m = trimesh.load(mesh_path, force="mesh")
            if m is None or m.is_empty:
                return None
            vol = float(m.volume)
            if scale is not None:
                if len(scale) == 1:
                    s = scale[0]
                    vol *= (s ** 3)
                elif len(scale) >= 3:
                    vol *= float(scale[0] * scale[1] * scale[2])
            return vol
        # Fallback: AABB approximation
        # Attempt to parse with trimesh anyway to get bounds if possible
        if trimesh is not None:
            m = trimesh.load(mesh_path, force="mesh")
            if m is None or m.is_empty:
                return None
            extents = m.bounding_box.extents
            vol = float(extents[0] * extents[1] * extents[2])
            if scale is not None:
                if len(scale) == 1:
                    s = scale[0]
                    vol *= (s ** 3)
                elif len(scale) >= 3:
                    vol *= float(scale[0] * scale[1] * scale[2])
            return vol
    except Exception:
        return None
    return None


def _geom_volume_from_element(urdf_path: str, geom: ET.Element) -> Optional[float]:
    # Prefer collision geometry; this function expects a <geometry> node
    if geom.find("box") is not None:
        size = _to_floats(geom.find("box").get("size"))  # type: ignore
        if size and len(size) >= 3:
            return float(size[0] * size[1] * size[2])
        return None
    if geom.find("cylinder") is not None:
        cyl = geom.find("cylinder")
        try:
            radius = float(cyl.get("radius"))  # type: ignore
            length = float(cyl.get("length"))  # type: ignore
            return float(math.pi * radius * radius * length)
        except Exception:
            return None
    if geom.find("sphere") is not None:
        sph = geom.find("sphere")
        try:
            radius = float(sph.get("radius"))  # type: ignore
            return float(4.0 / 3.0 * math.pi * (radius ** 3))
        except Exception:
            return None
    if geom.find("mesh") is not None:
        msh = geom.find("mesh")
        filename = msh.get("filename") if msh is not None else None
        scale = _to_floats(msh.get("scale")) if msh is not None else None
        mesh_path = _resolve_mesh_path(urdf_path, filename or "")
        if mesh_path:
            return _mesh_volume(mesh_path, scale)
        return None
    return None


def compute_urdf_volume(urdf_path: str) -> Tuple[float, List[str]]:
    problems: List[str] = []
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except Exception as e:
        return 0.0, [f"PARSE_FAIL:{e}"]

    total = 0.0
    # Prefer collision geometries; if none, fall back to visuals
    colls = root.findall('.//collision')
    use_nodes = colls if len(colls) > 0 else root.findall('.//visual')
    for node in use_nodes:
        geom = node.find('geometry')
        if geom is None:
            continue
        vol = _geom_volume_from_element(urdf_path, geom)
        if vol is None:
            problems.append("UNSUPPORTED_GEOM")
            continue
        total += float(vol)
    return total, problems


def find_urdfs(dirs: List[str]) -> List[str]:
    out: List[str] = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            if os.sep + 'gates' + os.sep in root:
                continue
            for f in files:
                if not f.lower().endswith('.urdf'):
                    continue
                p = os.path.join(root, f)
                # Skip files that look like gate models
                low = p.lower()
                if '/gates/' in low or 'gate' in os.path.basename(low):
                    continue
                out.append(p)
    return sorted(out)


def default_dirs(repo_root: str) -> List[str]:
    base = os.path.join(repo_root, 'resources', 'models', 'environment_assets')
    candidates = [
        os.path.join(base, 'objects'),
        os.path.join(base, 'trees'),
        os.path.join(base, 'walls'),
        os.path.join(base, 'panels'),
        os.path.join(base, 'thin'),
        os.path.join(base, 'objects_gate'),  # keep, but we'll still skip any gate urdfs by name
    ]
    return candidates


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute URDF volumes for obstacles (non-gate)")
    parser.add_argument('--dirs', nargs='*', default=None, help='Directories to scan for URDFs')
    parser.add_argument('--csv', default=None, help='Optional path to write CSV report')
    args = parser.parse_args(argv)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    scan_dirs = args.dirs if args.dirs else default_dirs(repo_root)

    urdfs = find_urdfs(scan_dirs)
    if not urdfs:
        print("No URDFs found in provided directories.")
        return 1

    rows = []
    total_vol = 0.0
    for p in urdfs:
        vol, problems = compute_urdf_volume(p)
        total_vol += vol
        rows.append((p, vol, ";".join(sorted(set(problems))) if problems else ""))

    # Print table
    print("urdf_path,volume_m3,notes")
    for p, v, note in rows:
        print(f"{p},{v:.6f},{note}")
    print("\nTOTAL_VOLUME_M3,%.6f" % total_vol)
    if args.csv:
        try:
            with open(args.csv, 'w', encoding='utf-8') as f:
                f.write("urdf_path,volume_m3,notes\n")
                for p, v, note in rows:
                    f.write(f"{p},{v:.6f},{note}\n")
                f.write("TOTAL_VOLUME_M3,%.6f\n" % total_vol)
            print(f"Report written to {args.csv}")
        except Exception as e:
            print(f"Failed to write CSV: {e}")
    return 0


if __name__ == '__main__':
    sys.exit(main())


