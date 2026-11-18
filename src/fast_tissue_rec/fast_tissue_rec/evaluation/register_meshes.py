#!/usr/bin/env python
"""
Automatic registration of two meshes (similarity transform: scale + rotation + translation).

Pipeline:
  1. Sample point clouds from both meshes.
  2. Because the two meshes can differ in scale (and are different crops of the
     same anatomy), the scale is treated as an unknown and swept over a small
     range of candidates.
  3. For each candidate scale: FPFH feature extraction + RANSAC global
     registration to recover a coarse pose.
  4. The best coarse result is refined with point-to-plane ICP.
  5. Outputs: a 4x4 transform mapping SOURCE -> TARGET coordinates, the estimated
     scale, the aligned source mesh, and registration quality metrics.

Usage:
  python -m fast_tissue_rec.evaluation.register_meshes \
      --source output_mesh_high_res.ply \
      --target lung_m_crop.obj \
      --out_dir ./registration_out --seed 47

  # Constrain the scale search (meshes already near a known scale) and use more
  # candidates, swept in parallel across CPU cores:
  python -m fast_tissue_rec.evaluation.register_meshes --source ... --target ... \
      --scale_min 0.8 --scale_max 1.2 --n_scale_steps 25 --jobs 8

The scale sweep runs in parallel (one process per candidate, each RANSAC
single-threaded). Every candidate is seeded deterministically (seed + index), so
results are bit-for-bit reproducible regardless of worker scheduling.

Everything is measured in the TARGET's metric scale.
"""

import argparse
import copy
import os

# Pin OpenMP to a single thread BEFORE importing open3d. Open3D's RANSAC is
# multi-threaded and nondeterministic even with a seeded RNG; single-threaded +
# seed makes a run bit-for-bit reproducible. Must be set before the import.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import open3d as o3d


# --------------------------------------------------------------------------- #
# Loading / preprocessing
# --------------------------------------------------------------------------- #
def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) == 0:
        raise ValueError(f"No vertices loaded from {path}")
    mesh.compute_vertex_normals()
    return mesh


def mesh_to_pcd(mesh, n_points):
    """Uniformly sample a point cloud from a mesh surface."""
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    return pcd


def diag(pcd_or_mesh):
    bb = pcd_or_mesh.get_axis_aligned_bounding_box()
    return np.linalg.norm(bb.get_extent())


def preprocess(pcd, voxel):
    """Downsample, estimate normals, and compute FPFH features."""
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
    )
    return down, fpfh


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def global_ransac(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel):
    """RANSAC global registration on FPFH correspondences (rigid)."""
    dist = voxel * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 0.999),
    )
    return result


def refine_icp(src_down, tgt_down, init_T, voxel):
    """Point-to-plane ICP refinement (rigid)."""
    dist = voxel * 1.5
    result = o3d.pipelines.registration.registration_icp(
        src_down,
        tgt_down,
        dist,
        init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    return result


def render_overlay(tgt_pcd, src_pcd_aligned, out_path, n=25_000):
    """Render target and aligned-source point clouds superimposed, multi-view PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def pts(pcd):
        p = np.asarray(pcd.points)
        if len(p) > n:
            p = p[np.random.choice(len(p), n, replace=False)]
        return p

    T = pts(tgt_pcd)
    S = pts(src_pcd_aligned)
    views = [(20, -60), (20, 30), (89, -90)]  # (elev, azim): two obliques + top
    titles = ["oblique 1", "oblique 2", "top"]
    fig = plt.figure(figsize=(15, 5))
    for i, ((elev, azim), title) in enumerate(zip(views, titles)):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(
            T[:, 0],
            T[:, 1],
            T[:, 2],
            s=0.4,
            c="#1f77b4",
            alpha=0.35,
            label="target (reference)",
        )
        ax.scatter(
            S[:, 0],
            S[:, 1],
            S[:, 2],
            s=0.4,
            c="#d62728",
            alpha=0.35,
            label="aligned source",
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_box_aspect((1, 1, 1))
        try:
            ax.set_aspect("equal")
        except Exception:
            pass
        if i == 0:
            ax.legend(markerscale=12, loc="upper left")
    fig.suptitle("Registration overlay — target (blue) vs aligned source (red)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_distance_heatmap(src_aligned, tgt_pcd, out_path, n=120_000):
    """Top view of the aligned source only, each point colored by distance to the
    nearest reference point (turbo colormap)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    S = np.asarray(src_aligned.points)
    if len(S) > n:
        S = S[np.random.choice(len(S), n, replace=False)]
    tree = cKDTree(np.asarray(tgt_pcd.points))
    dist, _ = tree.query(S, k=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    # top view = look down z: plot x vs y
    vmax = np.percentile(dist, 99)  # clip outliers so the scale stays readable
    sc = ax.scatter(
        S[:, 0],
        S[:, 1],
        c=dist * 1000.0,
        s=1.0,
        cmap="turbo",
        vmin=0.0,
        vmax=vmax * 1000.0,
    )
    ax.set_aspect("equal")
    ax.set_title("Aligned source — distance to nearest reference point (top view)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("distance to reference (mm)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return dict(
        mean_mm=float(dist.mean() * 1000),
        median_mm=float(np.median(dist) * 1000),
        p95_mm=float(np.percentile(dist, 95) * 1000),
        max_mm=float(dist.max() * 1000),
    )


def scale_about_centroid(pcd, s, centroid):
    """Return a copy of pcd scaled by s about the given centroid."""
    out = copy.deepcopy(pcd)
    out.scale(s, center=centroid)
    return out


# --------------------------------------------------------------------------- #
# Parallel scale sweep
# --------------------------------------------------------------------------- #
# Worker-process globals, populated once per worker by _sweep_init so the big
# point arrays and the (fixed) target FPFH are not re-pickled/re-computed per
# candidate.
_G = {}


def _sweep_init(src_pts, src_centroid, tgt_pts, voxel):
    _G["src_pts"] = src_pts
    _G["src_centroid"] = src_centroid
    _G["voxel"] = voxel
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_pts)
    _G["tgt_down"], _G["tgt_fpfh"] = preprocess(tgt, voxel)


def _sweep_candidate(task):
    """Prescale the source by one candidate scale, then FPFH + RANSAC.

    Seeds the RNG per candidate so the result is independent of scheduling order
    -> the parallel sweep is bit-for-bit reproducible."""
    scale, seed = task
    o3d.utility.random.seed(int(seed))
    np.random.seed(int(seed))
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(_G["src_pts"])
    src.scale(scale, center=_G["src_centroid"])
    src_down, src_fpfh = preprocess(src, _G["voxel"])
    res = global_ransac(src_down, _G["tgt_down"], src_fpfh, _G["tgt_fpfh"], _G["voxel"])
    return dict(
        scale=float(scale),
        fitness=res.fitness,
        rmse=res.inlier_rmse,
        T=np.asarray(res.transformation).copy(),
    )


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def register(
    source_path,
    target_path,
    out_dir,
    n_sample=80_000,
    voxel_frac=1 / 55,
    scale_span=(0.55, 1.6),
    n_scale_steps=13,
    scale_range=None,
    jobs=None,
    seed=0,
    save_mesh=True,
    make_overlay=True,
    verbose=True,
):
    os.makedirs(out_dir, exist_ok=True)

    # Seed both RNGs (Open3D RANSAC/sampling and NumPy point subsampling) so a
    # run is bit-for-bit reproducible.
    if seed is not None:
        o3d.utility.random.seed(int(seed))
        np.random.seed(int(seed))

    def vp(*a):
        if verbose:
            print(*a)

    vp("Loading meshes...")
    src_mesh = load_mesh(source_path)
    tgt_mesh = load_mesh(target_path)

    src_pcd_full = mesh_to_pcd(src_mesh, n_sample)
    tgt_pcd_full = mesh_to_pcd(tgt_mesh, n_sample)

    d_tgt = diag(tgt_pcd_full)
    d_src = diag(src_pcd_full)
    base_scale = d_tgt / d_src  # rough scale from bbox diagonals
    voxel = d_tgt * voxel_frac  # metric voxel in target scale
    src_centroid = src_pcd_full.get_center()

    vp(f"  target diag = {d_tgt:.4f}, source diag = {d_src:.4f}")
    vp(f"  bbox-ratio scale guess = {base_scale:.4f}")
    vp(f"  voxel size = {voxel:.5f}")

    # Preprocess target once (fixed) for the ICP stages below.
    tgt_down, tgt_fpfh = preprocess(tgt_pcd_full, voxel)

    # ---- candidate scales ----
    # If an absolute scale_range=(lo, hi) is given, sweep it directly (use this
    # when you know the meshes are near a known scale). Otherwise fall back to the
    # bbox-diagonal ratio times a relative span.
    if scale_range is not None:
        lo, hi = scale_range
        candidates = np.linspace(lo, hi, n_scale_steps)
        vp(
            f"\nSweeping {len(candidates)} candidate scales in absolute range "
            f"[{lo:.3f} .. {hi:.3f}] ..."
        )
    else:
        candidates = base_scale * np.linspace(
            scale_span[0], scale_span[1], n_scale_steps
        )
        vp(
            f"\nSweeping {len(candidates)} candidate scales "
            f"[{candidates.min():.3f} .. {candidates.max():.3f}] "
            f"(bbox-ratio x [{scale_span[0]}..{scale_span[1]}]) ..."
        )

    # ---- parallel scale sweep: prescale + FPFH + RANSAC per candidate ----
    # Each candidate is independent and seeded deterministically (seed + index),
    # so the sweep is reproducible regardless of how workers are scheduled.
    src_pts = np.asarray(src_pcd_full.points)
    tgt_pts = np.asarray(tgt_pcd_full.points)
    base_seed = 0 if seed is None else int(seed)
    tasks = [(float(s), base_seed + i) for i, s in enumerate(candidates)]

    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) - 1)
    jobs = min(jobs, len(tasks))

    if jobs > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        vp(f"  running sweep on {jobs} parallel workers")
        # "spawn", not the Linux default "fork": the parent has already run
        # Open3D (thread pools), and forked children can inherit a held lock and
        # deadlock forever.
        with ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_sweep_init,
            initargs=(src_pts, src_centroid, tgt_pts, voxel),
        ) as ex:
            results = list(ex.map(_sweep_candidate, tasks))
    else:
        _sweep_init(src_pts, src_centroid, tgt_pts, voxel)
        results = [_sweep_candidate(t) for t in tasks]

    for r in results:
        vp(
            f"  scale={r['scale']:6.3f}  fitness={r['fitness']:.3f}  rmse={r['rmse']:.5f}"
        )
    best = max(results, key=lambda r: r["fitness"])

    vp(f"\nBest coarse: scale={best['scale']:.4f} fitness={best['fitness']:.3f}")

    # ---- ICP refinement at best scale ----
    src_scaled_best = scale_about_centroid(src_pcd_full, best["scale"], src_centroid)
    src_down_best, _ = preprocess(src_scaled_best, voxel)
    icp = refine_icp(src_down_best, tgt_down, best["T"], voxel)
    vp(f"After rigid ICP: fitness={icp.fitness:.3f} rmse={icp.inlier_rmse:.5f}")

    # ---- compose coarse similarity transform: SOURCE (original) -> TARGET ----
    # Scaling about centroid c by s: p -> s*(p - c) + c = s*p + (1-s)*c
    S = np.eye(4)
    S[:3, :3] = best["scale"] * np.eye(3)
    S[:3, 3] = (1 - best["scale"]) * src_centroid
    T_coarse = icp.transformation @ S

    # ---- final polish: scale-aware ICP (estimates scale + R + t jointly) ----
    # Denser clouds and a tightening correspondence distance so scale settles
    # to the joint least-squares optimum instead of the discrete sweep value.
    fine_voxel = voxel * 0.5
    src_fine = src_pcd_full.voxel_down_sample(fine_voxel)
    tgt_fine = tgt_pcd_full.voxel_down_sample(fine_voxel)
    T_final = T_coarse
    for thr_mult in (3.0, 1.5, 0.75):
        poly = o3d.pipelines.registration.registration_icp(
            src_fine,
            tgt_fine,
            fine_voxel * thr_mult,
            T_final,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                with_scaling=True
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )
        T_final = poly.transformation
    est_scale = float(np.cbrt(np.linalg.det(T_final[:3, :3])))
    vp(
        f"After scale-ICP: fitness={poly.fitness:.3f} rmse={poly.inlier_rmse:.5f} "
        f"scale={est_scale:.4f}"
    )
    best["scale"] = est_scale

    # ---- final evaluation on dense clouds at the transform ----
    src_eval = copy.deepcopy(src_pcd_full).transform(T_final)
    ev = o3d.pipelines.registration.evaluate_registration(
        src_eval, tgt_pcd_full, voxel * 1.5, np.eye(4)
    )
    vp(f"\nFinal (dense) fitness={ev.fitness:.3f} rmse={ev.inlier_rmse:.5f}")

    metrics = dict(
        seed=seed,
        scale=est_scale,
        coarse_fitness=best["fitness"],
        icp_fitness=icp.fitness,
        icp_rmse=icp.inlier_rmse,
        final_fitness=ev.fitness,
        final_rmse=ev.inlier_rmse,
        T=T_final.copy(),
    )

    # ---- save outputs ----
    if save_mesh:
        np.savetxt(
            os.path.join(out_dir, "transform_source_to_target.txt"), T_final, fmt="%.8f"
        )
        # Row-major flattened 16 numbers on one line for pasting into MeshLab's
        # "Set/Freeze Matrix" dialog (Filters -> Normals/Transform).
        flat = " ".join(f"{v:.8f}" for v in T_final.flatten())
        with open(os.path.join(out_dir, "registration_report.txt"), "w") as f:
            f.write(f"source: {source_path}\n")
            f.write(f"target: {target_path}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"estimated scale: {est_scale:.6f}\n")
            f.write(f"coarse fitness: {best['fitness']:.4f}\n")
            f.write(f"icp fitness: {icp.fitness:.4f}\n")
            f.write(f"icp inlier rmse: {icp.inlier_rmse:.6f}\n")
            f.write(f"final dense fitness: {ev.fitness:.4f}\n")
            f.write(f"final dense rmse: {ev.inlier_rmse:.6f}\n")
            f.write("transform (source -> target):\n")
            f.write(np.array2string(T_final, precision=8))
            f.write("\n")
            f.write("flatten transform meshlab:\n")
            f.write(flat + "\n")

        # Aligned full-resolution source mesh in target coordinates.
        aligned = copy.deepcopy(src_mesh)
        aligned.transform(T_final)
        out_mesh = os.path.join(out_dir, "aligned_source.ply")
        o3d.io.write_triangle_mesh(out_mesh, aligned)

    if make_overlay:
        overlay_path = os.path.join(out_dir, "overlay.png")
        render_overlay(tgt_pcd_full, src_eval, overlay_path)
        heat_path = os.path.join(out_dir, "distance_heatmap_top.png")
        heat_stats = render_distance_heatmap(src_eval, tgt_pcd_full, heat_path)
        metrics["heat_stats"] = heat_stats
        vp(f"Saved overlay: {overlay_path}")
        vp(
            f"Saved heatmap: {heat_path}  "
            f"(mean {heat_stats['mean_mm']:.2f} mm, p95 {heat_stats['p95_mm']:.2f} mm)"
        )

    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out_dir", default="./registration_out")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    ap.add_argument(
        "--scale_min",
        type=float,
        default=None,
        help="Lower bound of absolute scale search (use with --scale_max)",
    )
    ap.add_argument(
        "--scale_max",
        type=float,
        default=None,
        help="Upper bound of absolute scale search (use with --scale_min)",
    )
    ap.add_argument(
        "--n_scale_steps",
        type=int,
        default=13,
        help="Number of scale candidates to sweep (default 13)",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel workers for the scale sweep (default: CPUs-1)",
    )
    args = ap.parse_args()

    scale_range = None
    if args.scale_min is not None or args.scale_max is not None:
        if args.scale_min is None or args.scale_max is None:
            ap.error("--scale_min and --scale_max must be given together")
        if args.scale_min > args.scale_max:
            ap.error("--scale_min must be <= --scale_max")
        scale_range = (args.scale_min, args.scale_max)

    register(
        args.source,
        args.target,
        args.out_dir,
        seed=args.seed,
        scale_range=scale_range,
        n_scale_steps=args.n_scale_steps,
        jobs=args.jobs,
    )
