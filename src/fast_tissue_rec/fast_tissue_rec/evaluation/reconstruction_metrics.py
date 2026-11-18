#!/usr/bin/env python
"""
Compute source->target registration accuracy metrics for a single pair of meshes
and print them to the terminal.

The aligned source mesh (in target coordinates), produced by the registration
step, is compared against its reference (ground-truth) target mesh. Both meshes
are surface-sampled into point clouds and nearest-neighbour distances are
computed in a single direction (source->target only):

  rmse = sqrt(mean(d^2))          root-mean-square nearest-neighbour distance
  std  = std(d)                   standard deviation of distances
  p95  = 95th percentile of d     robust Hausdorff

All distances are reported in millimetres (target units are metres).

Usage:
  python -m fast_tissue_rec.evaluation.reconstruction_metrics ALIGNED_SOURCE TARGET
  python -m fast_tissue_rec.evaluation.reconstruction_metrics aligned_source.ply target_m_crop.obj --n_sample 200000 --seed 0
"""
import argparse
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

MM = 1000.0  # metres -> millimetres


def sample_points(mesh_path, n):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0:
        raise ValueError(f"no vertices in {mesh_path}")
    pcd = mesh.sample_points_uniformly(number_of_points=n)
    return np.asarray(pcd.points)


def compute_src_to_tgt(aligned_pts, target_pts):
    """Source->target nearest-neighbour distance metrics."""
    tree_t = cKDTree(target_pts)
    d_s2t, _ = tree_t.query(aligned_pts, k=1)   # aligned source -> target
    return dict(
        rmse=float(np.sqrt(np.mean(d_s2t ** 2)) * MM),
        std=float(np.std(d_s2t) * MM),
        haus_p95=float(np.percentile(d_s2t, 95) * MM),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aligned_source",
                    help="Path to the aligned source mesh (in target coordinates)")
    ap.add_argument("target",
                    help="Path to the reference (ground-truth) target mesh")
    ap.add_argument("--n_sample", type=int, default=200_000,
                    help="Points sampled per mesh (default 200000)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for reproducible sampling (default 0)")
    args = ap.parse_args()

    if args.seed is not None:
        o3d.utility.random.seed(int(args.seed))
        np.random.seed(int(args.seed))

    print(f"Computing source->target metrics "
          f"({args.n_sample} pts/mesh, seed {args.seed})...\n")
    print(f"  aligned source: {args.aligned_source}")
    print(f"  target        : {args.target}\n")

    a_pts = sample_points(args.aligned_source, args.n_sample)
    t_pts = sample_points(args.target, args.n_sample)
    m = compute_src_to_tgt(a_pts, t_pts)

    print(f"  rmse={m['rmse']:.3f}mm  std={m['std']:.3f}mm  "
          f"p95={m['haus_p95']:.3f}mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
