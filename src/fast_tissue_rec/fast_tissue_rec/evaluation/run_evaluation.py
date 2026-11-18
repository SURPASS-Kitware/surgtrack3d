#!/usr/bin/env python
"""
Register every reconstruction to its ground-truth mesh and compute the
source->target surface accuracy (Table 1 of the paper).

For each reconstruction folder <name> in --recon_dir:
  source = <recon_dir>/<name>/output_mesh_high_res.ply
  target = <gt_dir>/<gt_name>/*_m_crop.obj   (gt_name from the match table)
  output = <output_dir>/<name>/              (transform, aligned mesh, report, PNGs)
And a summary table: <output_dir>/reconstruction_metrics.csv

Registration (register_meshes.py): FPFH + RANSAC over a sweep of candidate
scales, rigid point-to-plane ICP, then scale-aware ICP. Similarity transform
source -> target; the ground truth is the target.

Metrics (reconstruction_metrics.py): both meshes are surface-sampled
(--n_sample points, --metric_seed) and source->target nearest-neighbour
distances are reported in mm as RMSE, STD and 95th percentile.

Usage:
  python -m fast_tissue_rec.evaluation.run_evaluation \\
      --recon_dir data_intermediate_outputs/reconstruction \\
      --gt_dir dataset/ground_truth_meshes \\
      --output_dir data_intermediate_outputs/results

  # Override / add reconstruction -> ground-truth folder matches:
  python -m fast_tissue_rec.evaluation.run_evaluation ... --match kidney=kidneys
"""
import argparse
import csv
import glob
import os
import time

import numpy as np
import open3d as o3d

from fast_tissue_rec.evaluation.register_meshes import register
from fast_tissue_rec.evaluation.reconstruction_metrics import (
    compute_src_to_tgt,
    sample_points,
)

SOURCE_MESH_NAME = "output_mesh_high_res.ply"

# Reconstruction folder -> ground-truth folder, for names that differ in the
# released dataset. Any folder not listed is matched to the same name.
DEFAULT_MATCH = {
    "kidney": "kidneys",
    "lungs_left": "lung_left",
    "lungs_right": "lung_right",
}


def find_target_mesh(gt_dir):
    objs = sorted(glob.glob(os.path.join(gt_dir, "*_m_crop.obj")))
    if not objs:
        objs = sorted(glob.glob(os.path.join(gt_dir, "*.obj")))
    return objs[0] if objs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon_dir", required=True,
                    help="Folder with one sub-folder per reconstruction")
    ap.add_argument("--gt_dir", required=True,
                    help="Folder with one sub-folder per ground-truth mesh")
    ap.add_argument("--output_dir", required=True,
                    help="Where registrations and the metrics CSV are written")
    ap.add_argument("--match", nargs="+", default=[], metavar="REC=GT",
                    help="Extra reconstruction=ground_truth folder matches")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Only evaluate these reconstruction folder names")
    ap.add_argument("--seed", type=int, default=47,
                    help="Registration RNG seed (default 47, as in the paper)")
    ap.add_argument("--metric_seed", type=int, default=0,
                    help="Surface-sampling RNG seed for metrics (default 0)")
    ap.add_argument("--n_sample", type=int, default=200_000,
                    help="Points sampled per mesh for metrics (default 200000)")
    ap.add_argument("--n_scale_steps", type=int, default=13,
                    help="Number of scale candidates to sweep (default 13)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="Parallel workers for the scale sweep (default: CPUs-1)")
    ap.add_argument("--no_overlay", action="store_true",
                    help="Do not render overlay/heatmap PNGs")
    args = ap.parse_args()

    match = dict(DEFAULT_MATCH)
    for m in args.match:
        rec, gt = m.split("=", 1)
        match[rec] = gt

    recs = sorted(d for d in os.listdir(args.recon_dir)
                  if os.path.isfile(os.path.join(args.recon_dir, d, SOURCE_MESH_NAME)))
    if args.only:
        recs = [r for r in recs if r in set(args.only)]
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for rec in recs:
        gt = match.get(rec, rec)
        source = os.path.join(args.recon_dir, rec, SOURCE_MESH_NAME)
        target = find_target_mesh(os.path.join(args.gt_dir, gt))
        out_dir = os.path.join(args.output_dir, rec)
        print("=" * 78, f"\n{rec} -> {gt}", flush=True)
        if target is None:
            print(f"  [skip] no ground-truth mesh in {os.path.join(args.gt_dir, gt)}")
            continue

        t0 = time.time()
        reg = register(source, target, out_dir, seed=args.seed,
                       n_scale_steps=args.n_scale_steps, jobs=args.jobs,
                       make_overlay=not args.no_overlay, verbose=True)
        dt = time.time() - t0

        o3d.utility.random.seed(args.metric_seed)
        np.random.seed(args.metric_seed)
        a_pts = sample_points(os.path.join(out_dir, "aligned_source.ply"), args.n_sample)
        t_pts = sample_points(target, args.n_sample)
        m = compute_src_to_tgt(a_pts, t_pts)
        print(f"  scale={reg['scale']:.4f}  rmse={m['rmse']:.3f}mm  "
              f"std={m['std']:.3f}mm  p95={m['haus_p95']:.3f}mm", flush=True)

        rows.append(dict(
            reconstruction=rec, ground_truth=gt,
            scale=round(reg["scale"], 6),
            rmse_mm=round(m["rmse"], 4), std_mm=round(m["std"], 4),
            p95_mm=round(m["haus_p95"], 4),
            final_fitness=round(reg["final_fitness"], 4),
            registration_time_s=round(dt, 1),
        ))

    if not rows:
        print("No reconstructions evaluated.")
        return
    csv_path = os.path.join(args.output_dir, "reconstruction_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
