import argparse
import os
from pathlib import Path

import fast_tissue_rec.data_utils as data_utils
import fast_tissue_rec.tracker as tracker


def main(
    left_video: Path,
    right_video: Path,
    camera_poses_file: Path,
    stereo_output_dir: Path,
):
    os.makedirs(stereo_output_dir, exist_ok=True)
    os.makedirs(stereo_output_dir / "tracks", exist_ok=True)

    # cumulative, so that oversample subsample factor is tracking_subsample * output_subsampler
    tracking_subsample = 2
    output_subsample = 20

    print("Loading stereo videos...")
    left_frames = data_utils.load_video(left_video, tracking_subsample)
    right_frames = data_utils.load_video(right_video, tracking_subsample)

    print("Saving subsampled stereo frames...")
    (stereo_output_dir / "images").mkdir(parents=True, exist_ok=True)
    data_utils.save_stereo_images(
        [
            pair
            for idx, pair in enumerate(zip(left_frames, right_frames))
            if idx % output_subsample == 0
        ],
        stereo_output_dir / "images" / "stereo_index.json",
    )

    print("Loading camera poses...")
    poses = data_utils.load_camera_poses(camera_poses_file)
    print("Saving subsampled camera poses...")
    subsampled_poses = poses[:: (tracking_subsample * output_subsample)]
    data_utils.save_camera_poses(
        stereo_output_dir / camera_poses_file.name, subsampled_poses
    )

    print("Running tracker...")
    features = tracker.track(left_frames, output_subsample)
    data_utils.save_feature_observations(stereo_output_dir / "tracks", features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--left-video",
        required=True,
        help="Left input video file, rectified",
    )
    parser.add_argument(
        "-r",
        "--right-video",
        required=True,
        help="Right input video file, rectified",
    )
    parser.add_argument(
        "-p",
        "--poses",
        required=True,
        help="Input raw camera poses",
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    args = parser.parse_args()

    main(
        Path(args.left_video),
        Path(args.right_video),
        Path(args.poses),
        Path(args.output),
    )

__all__ = ["main"]
