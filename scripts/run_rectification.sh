#!/usr/bin/env bash
#
# Rectify every stereo video pair across the scan datasets using a single
# global calibration.
#
# For each videos/<clip>/ subfolder the script looks for a raw stereo pair
#   left.<ext> / right.<ext>   (ext = mkv or avi)
# and calls rectify_stereo_video.py, which writes the rectified pair next to
# the inputs as:
#   left_rect.<ext> / right_rect.<ext>
#
# Usage:
#   ./run_rectification.sh                 # process every clip in all scans
#   ./run_rectification.sh scan_26-07-07   # process only the given scan dir(s)
#
set -euo pipefail

# --- Configuration ----------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-/home/jbarrag3/research/reconstruction_data}"

# Global stereo calibration used for every video.
CALIBRATION="${CALIBRATION:-$DATA_ROOT/global_calibration/calibration_storz.json}"

# Location of the rectification script.
RECTIFY="${RECTIFY:-/home/jbarrag3/research/surpass_mosaic_video_tracker/rectify_stereo_video.py}"

# Scan directories to process (override by passing them as arguments).
if [[ $# -gt 0 ]]; then
    SCANS=("$@")
else
    SCANS=(scan_26-05-19 scan_26-06-09 scan_26-07-07)
fi

# --- Sanity checks ----------------------------------------------------------
[[ -f "$CALIBRATION" ]] || { echo "Calibration not found: $CALIBRATION" >&2; exit 1; }
[[ -f "$RECTIFY" ]]     || { echo "Rectify script not found: $RECTIFY" >&2; exit 1; }

echo "Data root   : $DATA_ROOT"
echo "Calibration : $CALIBRATION"
echo "Rectify     : $RECTIFY"
echo "Scans       : ${SCANS[*]}"
echo

# --- Rectification ----------------------------------------------------------
rectify_clip() {
    local clip_dir="$1"

    # Detect the raw video container extension (.mkv or .avi).
    local ext=""
    for candidate in mkv avi; do
        if [[ -f "$clip_dir/left.$candidate" && -f "$clip_dir/right.$candidate" ]]; then
            ext="$candidate"
            break
        fi
    done
    if [[ -z "$ext" ]]; then
        echo "  !! skipping (no left/right .mkv or .avi): $clip_dir" >&2
        return 0
    fi

    echo "  -> rectifying $clip_dir (.$ext)"
    python "$RECTIFY" \
        --calibration "$CALIBRATION" \
        --left-video "$clip_dir/left.$ext" \
        --right-video "$clip_dir/right.$ext"
}

for scan in "${SCANS[@]}"; do
    videos_dir="$DATA_ROOT/$scan/videos"
    if [[ ! -d "$videos_dir" ]]; then
        echo "!! videos dir not found, skipping: $videos_dir" >&2
        continue
    fi

    echo "==================================================================="
    echo ">>> Scan: $scan"
    echo "==================================================================="
    for clip in "$videos_dir"/*/; do
        [[ -d "$clip" ]] || continue
        rectify_clip "${clip%/}"
    done
    echo
done

echo "All rectifications complete."
