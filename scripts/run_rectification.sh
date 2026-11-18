#!/usr/bin/env bash
#
# Rectify the raw stereo videos of one or more datasets.
#
# Expected layout (as released on Zenodo), under $DATA_ROOT:
#   dataset/raw_data/<dataset>/
#     calibration.json  left.<ext>  right.<ext>      (ext = mkv or avi)
#
# Each pair is rectified with its own calibration.json and written to
#   $RECT_DIR/<dataset>/left_rect.<ext>  right_rect.<ext>
#
# Requires fast_tissue_rec installed in the active Python environment (the core
# install is enough, no CUDA needed).
#
# Usage:
#   ./run_rectification.sh                    # every dataset in raw_data
#   ./run_rectification.sh intestines kidney  # only these datasets
#
# Environment overrides:
#   DATA_ROOT   folder holding dataset/ and data_intermediate_outputs/
#               (default: the parent folder of this repository)
#   RAW_DIR     default: $DATA_ROOT/dataset/raw_data
#   RECT_DIR    default: $DATA_ROOT/data_intermediate_outputs/rectified_videos
#   PYTHON      Python interpreter (default: python)
#
set -euo pipefail

# --- Configuration ----------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$(dirname "$REPO_DIR")}"
RAW_DIR="${RAW_DIR:-$DATA_ROOT/dataset/raw_data}"
RECT_DIR="${RECT_DIR:-$DATA_ROOT/data_intermediate_outputs/rectified_videos}"
PYTHON="${PYTHON:-python}"

[[ -d "$RAW_DIR" ]] || { echo "Raw data dir not found: $RAW_DIR" >&2; exit 1; }

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
echo "Output      : $RECT_DIR"
echo "Datasets    : ${DATASETS[*]}"
echo

# --- Rectification ----------------------------------------------------------
rectify_dataset() {
    local ds="$1"
    local raw="$RAW_DIR/$ds"

    # Raw video container extension (.mkv or .avi).
    local ext=""
    for candidate in mkv avi; do
        if [[ -f "$raw/left.$candidate" && -f "$raw/right.$candidate" ]]; then
            ext="$candidate"
            break
        fi
    done
    if [[ -z "$ext" ]]; then
        echo "  !! skipping $ds (no left/right .mkv or .avi in $raw)" >&2
        return 1
    fi
    [[ -f "$raw/calibration.json" ]] || { echo "  !! missing $raw/calibration.json" >&2; return 1; }

    echo ">>> $ds (.$ext)"
    mkdir -p "$RECT_DIR/$ds"
    "$PYTHON" -m fast_tissue_rec.preprocessing.rectify_stereo_video \
        --calibration "$raw/calibration.json" \
        --left-video "$raw/left.$ext" \
        --right-video "$raw/right.$ext" \
        --output-dir "$RECT_DIR/$ds"
}

FAILED=()
for ds in "${DATASETS[@]}"; do
    rectify_dataset "$ds" || FAILED+=("$ds")
done

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed: ${FAILED[*]}" >&2
    exit 1
fi
echo "All rectifications complete."
