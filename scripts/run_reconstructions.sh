#!/usr/bin/env bash
#
# Run the full tissue reconstruction pipeline (tracking -> bundle adjustment ->
# depth estimation -> volumetric fusion) for one or more datasets.
#
# Expected layout (as released on Zenodo), under $DATA_ROOT:
#   dataset/raw_data/<dataset>/
#     calibration.json  ecm_poses.json  left.<ext>  right.<ext>
#   data_intermediate_outputs/rectified_videos/<dataset>/
#     left_rect.<ext>  right_rect.<ext>        (ext = mkv or avi)
#
# Outputs, under $OUT_DIR/<dataset>/:
#   tracking_output/   ecm_poses.json, tracks/, images/ (stereo_index.json),
#                      slam_poses.json, features.csv
#   depth_maps/        depth_map_index.json + depth_XXX.tiff
#   output_mesh_high_res.ply
#   logs/              one log per step
#
# Requires fast_tissue_rec[all] (and kinematic_ba) installed in the active
# Python environment.
#
# Usage:
#   ./run_reconstructions.sh                    # every dataset in raw_data
#   ./run_reconstructions.sh intestines kidney  # only these datasets
#
# Environment overrides:
#   DATA_ROOT   folder holding dataset/ and data_intermediate_outputs/
#               (default: the parent folder of this repository)
#   RAW_DIR     default: $DATA_ROOT/dataset/raw_data
#   RECT_DIR    default: $DATA_ROOT/data_intermediate_outputs/rectified_videos
#   OUT_DIR     default: $DATA_ROOT/data_intermediate_outputs/reconstruction
#   NUM_VOXELS  voxel count used to derive the TSDF voxel size (default 640000000.0)
#   PYTHON      Python interpreter (default: python)
#
set -euo pipefail

# --- Configuration ----------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$(dirname "$REPO_DIR")}"
RAW_DIR="${RAW_DIR:-$DATA_ROOT/dataset/raw_data}"
RECT_DIR="${RECT_DIR:-$DATA_ROOT/data_intermediate_outputs/rectified_videos}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/data_intermediate_outputs/reconstruction}"
NUM_VOXELS="${NUM_VOXELS:-640000000.0}"
PYTHON="${PYTHON:-python}"

[[ -d "$RAW_DIR" ]]  || { echo "Raw data dir not found: $RAW_DIR" >&2; exit 1; }
[[ -d "$RECT_DIR" ]] || { echo "Rectified videos dir not found: $RECT_DIR" >&2; exit 1; }

# --- Dataset selection ------------------------------------------------------
if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
else
    DATASETS=()
    for d in "$RAW_DIR"/*/; do
        [[ -d "$d" ]] && DATASETS+=("$(basename "$d")")
    done
fi
[[ ${#DATASETS[@]} -gt 0 ]] || { echo "No datasets selected." >&2; exit 1; }

echo "Raw data    : $RAW_DIR"
echo "Rectified   : $RECT_DIR"
echo "Output      : $OUT_DIR"
echo "Voxels      : $NUM_VOXELS"
echo "Datasets    : ${DATASETS[*]}"
echo

# --- Pipeline ---------------------------------------------------------------
# Runs one step, logging to $log; prints a short status line.
run_step() {
    local name="$1" log="$2"; shift 2
    local start=$SECONDS
    if "$@" > "$log" 2>&1; then
        echo "  [ok]   $name ($((SECONDS - start)) s)"
    else
        echo "  [FAIL] $name, see $log" >&2
        return 1
    fi
}

run_reconstruction() {
    local ds="$1"
    local raw="$RAW_DIR/$ds"
    local rect="$RECT_DIR/$ds"
    local out="$OUT_DIR/$ds"
    local tracking="$out/tracking_output"
    local calib="$raw/calibration.json"

    echo "==================================================================="
    echo ">>> $ds"

    # Rectified video container extension (.mkv or .avi).
    local ext=""
    for candidate in mkv avi; do
        if [[ -f "$rect/left_rect.$candidate" && -f "$rect/right_rect.$candidate" ]]; then
            ext="$candidate"
            break
        fi
    done
    if [[ -z "$ext" ]]; then
        echo "  !! no left_rect/right_rect .mkv or .avi in $rect" >&2
        return 1
    fi
    for f in "$calib" "$raw/ecm_poses.json"; do
        [[ -f "$f" ]] || { echo "  !! missing $f" >&2; return 1; }
    done

    mkdir -p "$out/logs"

    run_step "tracking" "$out/logs/1_tracking.log" \
        "$PYTHON" -m fast_tissue_rec.run_tracking \
        -l "$rect/left_rect.$ext" -r "$rect/right_rect.$ext" \
        -p "$raw/ecm_poses.json" -o "$tracking" || return 1

    run_step "bundle adjustment" "$out/logs/2_ba.log" \
        "$PYTHON" -m fast_tissue_rec.run_slam -c "$calib" \
        -p "$tracking/ecm_poses.json" -f "$tracking/tracks/" \
        --poses-output "$tracking/slam_poses.json" \
        --features-output "$tracking/features.csv" || return 1

    run_step "depth estimation" "$out/logs/3_depth.log" \
        "$PYTHON" -m fast_tissue_rec.run_depth_estimation -c "$calib" \
        -s "$tracking/images/stereo_index.json" \
        --output-dir "$out/depth_maps/" || return 1

    run_step "volumetric fusion" "$out/logs/4_fusion.log" \
        "$PYTHON" -m fast_tissue_rec.run_fusion -c "$calib" \
        -s "$tracking/images/stereo_index.json" \
        -d "$out/depth_maps/depth_map_index.json" \
        -p "$tracking/slam_poses.json" -f "$tracking/features.csv" \
        -n "$NUM_VOXELS" -o "$out/output_mesh_high_res.ply" || return 1

    echo "  -> $out/output_mesh_high_res.ply"
}

FAILED=()
for ds in "${DATASETS[@]}"; do
    run_reconstruction "$ds" || FAILED+=("$ds")
done

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed: ${FAILED[*]}" >&2
    exit 1
fi
echo "All reconstructions complete."
