"""
Rectify a stereo video pair using calibration JSON.

Usage:
  python -m fast_tissue_rec.preprocessing.rectify_stereo_video --calibration input/260303b/calibration.json --left-video path/to/left.mp4 --right-video path/to/right.mp4 [--output-dir out/]

Outputs (written next to inputs, or in --output-dir if given):
  - <left>_rect.<ext>
  - <right>_rect.<ext>
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import cv2
import numpy as np


def get_rect_image_size(lc):
    """Returns (width, height) for the rectified output image.
    Uses rect_image_width/height if present, otherwise falls back to image_width/height.
    """
    w = int(lc.get("rect_image_width", lc["image_width"]))
    h = int(lc.get("rect_image_height", lc["image_height"]))
    return w, h


def compute_rectification(
    cam: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (R1, R2, P1, P2).

    If per-camera 'rectification' and 'projection' are present, returns them directly.
    Otherwise, computes them via cv2.stereoRectify using 'rotation' (R) and 'translation' (T).

    The calibration JSON must contain:
      - left_camera: { intrinsic, distortion, image_width, image_height, ... }
      - right_camera: { intrinsic, distortion, image_width, image_height, ... }

    And either:
      - per-camera rectification + projection matrices, OR
      - global rotation + translation for stereoRectify
    """
    lc = cam["left_camera"]
    rc = cam["right_camera"]

    def _asarr(x, shape=None, dtype=np.float64):
        a = np.asarray(x, dtype=dtype)
        return a.reshape(shape) if shape is not None else a

    # Case 1: precomputed rectification / projection
    has_precomputed = (
        "rectification" in lc
        and "projection" in lc
        and "rectification" in rc
        and "projection" in rc
    )
    if has_precomputed:
        R1 = _asarr(lc["rectification"], (3, 3))
        R2 = _asarr(rc["rectification"], (3, 3))
        P1 = _asarr(lc["projection"], (3, 4))
        P2 = _asarr(rc["projection"], (3, 4))
        return R1, R2, P1, P2

    # Case 2: compute via stereoRectify using R, T
    if "rotation" not in cam or "translation" not in cam:
        raise KeyError(
            "Calibration JSON must contain either per-camera 'rectification' & 'projection' "
            "or global 'rotation' & 'translation'. Neither set was found."
        )

    K1 = _asarr(lc["intrinsic"], (3, 3))
    D1 = _asarr(lc["distortion"])
    K2 = _asarr(rc["intrinsic"], (3, 3))
    D2 = _asarr(rc["distortion"])
    R = _asarr(cam["rotation"], (3, 3))
    T = _asarr(cam["translation"], (3,)).reshape(3, 1)

    # w, h = int(lc["image_width"]), int(lc["image_height"])
    w, h = get_rect_image_size(lc)

    R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
        cameraMatrix1=K1,
        distCoeffs1=D1,
        cameraMatrix2=K2,
        distCoeffs2=D2,
        imageSize=(w, h),
        R=R,
        T=T,
        flags=0,
        alpha=0.0,
    )
    return R1, R2, P1, P2


def _open_writer(path: str, fps: float, size_wh: Tuple[int, int]) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = size_wh
    writer = cv2.VideoWriter(path, fourcc, float(fps), (int(w), int(h)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {path}")
    return writer


def _add_rect_suffix(path: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}_rect{ext}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rectify a stereo video pair using calibration JSON."
    )
    parser.add_argument(
        "--calibration",
        required=True,
        help="Path to stereo calibration JSON (contains intrinsics/distortion and rectification/projection or R/T).",
    )
    parser.add_argument(
        "--left-video", required=True, help="Path to the left (unrectified) video."
    )
    parser.add_argument(
        "--right-video", required=True, help="Path to the right (unrectified) video."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the rectified videos. Defaults to the directory of each input video.",
    )
    parser.add_argument(
        "--show-frame-id",
        action="store_true",
        help="Overlay the 0-based frame index in the top-left corner of each rectified frame.",
    )
    args = parser.parse_args()

    left_video_path = args.left_video
    right_video_path = args.right_video

    with open(args.calibration, "r", encoding="utf-8") as f:
        cam = json.load(f)

    assert "rect_image_width" in cam["left_camera"], (
        "Rectify rect_image_width and rect_image_height needs to be specified"
        + "for new UR ROBOT setup, comment this line if rectifying older data."
    )

    rect_image_width, rect_image_height = get_rect_image_size(cam["left_camera"])

    # Calculate rectification LUTs (identical to pose_optimizer.py)
    R1, R2, P1, P2 = compute_rectification(cam)
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        np.asarray(cam["left_camera"]["intrinsic"], np.float64),
        np.asarray(cam["left_camera"]["distortion"], np.float64),
        R1,
        P1,
        (rect_image_width, rect_image_height),
        cv2.CV_32FC1,
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        np.asarray(cam["right_camera"]["intrinsic"], np.float64),
        np.asarray(cam["right_camera"]["distortion"], np.float64),
        R2,
        P2,
        (rect_image_width, rect_image_height),
        cv2.CV_32FC1,
    )
    if (
        left_map1 is None
        or left_map2 is None
        or right_map1 is None
        or right_map2 is None
    ):
        raise RuntimeError("Could not create rectification LUTs.")

    left_cap = cv2.VideoCapture(left_video_path)
    right_cap = cv2.VideoCapture(right_video_path)
    if not left_cap.isOpened() or not right_cap.isOpened():
        raise RuntimeError(
            f"Could not open one or both video files:\n  left={left_video_path}\n  right={right_video_path}"
        )

    out_w = rect_image_width
    out_h = rect_image_height

    fps_left = float(left_cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_right = float(right_cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = fps_left if fps_left > 0 else fps_right
    if fps <= 0:
        fps = 30.0

    left_out_path = _add_rect_suffix(left_video_path)
    right_out_path = _add_rect_suffix(right_video_path)
    if args.output_dir is not None:
        left_out_path = os.path.join(args.output_dir, os.path.basename(left_out_path))
        right_out_path = os.path.join(args.output_dir, os.path.basename(right_out_path))

    writer_left = _open_writer(left_out_path, fps=fps, size_wh=(out_w, out_h))
    writer_right = _open_writer(right_out_path, fps=fps, size_wh=(out_w, out_h))

    frame_idx = 0
    try:
        while True:
            ok_l, frame_left = left_cap.read()
            ok_r, frame_right = right_cap.read()
            if not ok_l or not ok_r:
                break

            frame_left_rectified = cv2.remap(
                frame_left, left_map1, left_map2, cv2.INTER_LINEAR
            )
            frame_right_rectified = cv2.remap(
                frame_right, right_map1, right_map2, cv2.INTER_LINEAR
            )

            if args.show_frame_id:
                label = str(frame_idx)
                org = (12, 30)
                cv2.putText(
                    frame_left_rectified,
                    label,
                    org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame_left_rectified,
                    label,
                    org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame_right_rectified,
                    label,
                    org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame_right_rectified,
                    label,
                    org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer_left.write(frame_left_rectified)
            writer_right.write(frame_right_rectified)

            frame_idx += 1
            if frame_idx % 200 == 0:
                print(f"Rectified {frame_idx} frames...")
    finally:
        left_cap.release()
        right_cap.release()
        writer_left.release()
        writer_right.release()

    print(f"Done. Wrote:\n  {left_out_path}\n  {right_out_path}\nFrames: {frame_idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
