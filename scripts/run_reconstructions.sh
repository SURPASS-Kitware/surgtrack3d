#!/usr/bin/env bash
#
# Run the full tissue reconstruction pipeline (tracking -> SLAM -> depth
# estimation -> fusion) for one or more input videos.
#
# For every video the following outputs are produced under
#   $DATA_ROOT/<scan>/reconstructions/<clip>/
#     tracking_output/   (ecm_poses.json, tracks/, stereo_index.json,
#                         slam_poses.json, features.csv)
#     depth_maps/        (depth_map_index.json + depth maps)
#     output_mesh_high_res.ply
#     output_mesh_med_res.ply
#
# Usage:
#   ./run_reconstructions.sh                          # every clip in all scans
#   ./run_reconstructions.sh scan_26-05-19            # every clip in one scan
#   ./run_reconstructions.sh scan_26-05-19/kidney_7cm # only that clip
#   ./run_reconstructions.sh scan_26-05-19/kidney_7cm scan_26-07-07/intestines_10cm
#
set -euo pipefail

# --- Configuration ----------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-/home/jbarrag3/research/reconstruction_data}"

# Scans processed when no argument is given.
SCANS=(scan_26-05-19 scan_26-06-09 scan_26-07-07)

# Location of the run_*.py scripts.
SCRIPTS="${SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/fast_tissue_rec/fast_tissue_rec}"

# Global stereo calibration used for all videos.
CALIBRATION="${CALIBRATION:-$DATA_ROOT/global_calibration/calibration_storz.json}"

# Number of voxels used to derive the TSDF voxel size in fusion.
HIGH_RES_VOXELS=640000000.0
MED_RES_VOXELS=320000000.0

# --- Clip selection ---------------------------------------------------------
# Each item is a "scan/clip" pair, e.g. "scan_26-05-19/kidney_7cm".
add_scan_clips() {
    local scan="$1"
    local videos_dir="$DATA_ROOT/$scan/videos"
    if [[ ! -d "$videos_dir" ]]; then
        echo "!! videos dir not found, skipping: $videos_dir" >&2
        return 0
    fi
    local d
    for d in "$videos_dir"/*/; do
        [[ -d "$d" ]] && ITEMS+=("$scan/$(basename "$d")")
    done
}

ITEMS=()
if [[ $# -gt 0 ]]; then
    for arg in "$@"; do
        # "scan/clip" -> that single clip; "scan" -> every clip in the scan.
        if [[ "$arg" == */* ]]; then
            ITEMS+=("$arg")
        else
            add_scan_clips "$arg"
        fi
    done
else
    for scan in "${SCANS[@]}"; do
        add_scan_clips "$scan"
    done
fi

if [[ ${#ITEMS[@]} -eq 0 ]]; then
    echo "No clips selected." >&2
    exit 1
fi

echo "Scripts dir : $SCRIPTS"
echo "Data root   : $DATA_ROOT"
echo "Calibration : $CALIBRATION"
echo "Clips       : ${ITEMS[*]}"
echo

# --- Pipeline ---------------------------------------------------------------
run_reconstruction() {
    local item="$1"                     # scan/clip
    local scan="${item%%/*}"
    local clip="${item#*/}"
    local video_dir="$DATA_ROOT/$scan/videos/$clip"
    local out_dir="$DATA_ROOT/$scan/reconstructions/$clip"
    local tracking="$out_dir/tracking_output"

    echo "==================================================================="
    echo ">>> Reconstructing: $scan/$clip"
    echo "==================================================================="

    if [[ ! -d "$video_dir" ]]; then
        echo "!! clip dir not found, skipping: $video_dir" >&2
        return 0
    fi

    set -x
    echo "--- [1/5] Tracking ---"
    ## Cotracker
    # python "$SCRIPTS/run_tracking.py" \
    #     --input-dir "$video_dir" \
    #     --output-dir "$out_dir"

    ## KLT
    # Detect the rectified video container extension (.mkv or .avi).
    local ext=""
    for candidate in mkv avi; do
        if [[ -f "$video_dir/left_rect.$candidate" ]]; then
            ext="$candidate"
            break
        fi
    done
    if [[ -z "$ext" ]]; then
        echo "No left_rect.mkv or left_rect.avi found in $video_dir" >&2
        return 1
    fi

    python $SCRIPTS/run_tracking.py \
    -l "$video_dir/left_rect.$ext" \
    -r "$video_dir/right_rect.$ext"  \
    -p "$video_dir/ecm_poses.json"  \
    -o "$tracking"


    echo "--- [2/5] SLAM ---"
    python "$SCRIPTS/run_slam.py" \
        -c "$CALIBRATION" \
        -p "$tracking/ecm_poses.json" \
        -f "$tracking/tracks/" \
        --poses-output "$tracking/slam_poses.json" \
        --features-output "$tracking/features.csv"

    echo "--- [3/5] Depth estimation ---"
    python "$SCRIPTS/run_depth_estimation.py" \
        -c "$CALIBRATION" \
        -s "$tracking/images/stereo_index.json" \
        --output-dir "$out_dir/depth_maps/"

    echo "--- [4/5] Fusion (high res) ---"
    python "$SCRIPTS/run_fusion.py" \
        -c "$CALIBRATION" \
        -s "$tracking/images/stereo_index.json" \
        -d "$out_dir/depth_maps/depth_map_index.json" \
        -p "$tracking/slam_poses.json" \
        -f "$tracking/features.csv" \
        -n "$HIGH_RES_VOXELS" \
        -o "$out_dir/output_mesh_high_res.ply"

    # echo "--- [5/5] Fusion (med res) ---"
    # python "$SCRIPTS/run_fusion.py" \
    #     -c "$CALIBRATION" \
    #     -s "$tracking/images/stereo_index.json" \
    #     -d "$out_dir/depth_maps/depth_map_index.json" \
    #     -p "$tracking/slam_poses.json" \
    #     -f "$tracking/features.csv" \
    #     -n "$MED_RES_VOXELS" \
    #     -o "$out_dir/output_mesh_med_res.ply"

    set +x
    echo ">>> Done: $scan/$clip"
    echo
}

for item in "${ITEMS[@]}"; do
    run_reconstruction "$item"
done

echo "All reconstructions complete."
