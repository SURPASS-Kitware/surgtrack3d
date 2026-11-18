import argparse
from pathlib import Path

import trimesh

import fast_tissue_rec.data_utils as data_utils
import fast_tissue_rec.fusion.depth_estimation as depth_estimation
import fast_tissue_rec.fusion.fusion as fusion
import fast_tissue_rec.stereo_utils as stereo_utils
from fast_tissue_rec.run_slam import run_slam
from fast_tissue_rec.tracker import track


def run_reconstruction(
    camera_calibration: dict,
    left_video: Path,
    right_video: Path,
    camera_poses_file: Path,
    output_mesh: Path,
):
    # cumulative, so that oversample subsample factor is tracking_subsample * output_subsampler
    tracking_subsample = 2
    output_subsample = 20

    print("Loading stereo videos...")
    left_frames = data_utils.load_video(left_video, tracking_subsample)
    right_frames = data_utils.load_video(right_video, tracking_subsample)

    print("Running feature tracking...")
    features = track(left_frames, output_subsample)

    print("Loading camera poses...")
    poses = data_utils.load_camera_poses(camera_poses_file)
    subsampled_poses = poses[:: (tracking_subsample * output_subsample)]

    print("Running SLAM...")
    camera_matrix = stereo_utils.left_camera_matrix(camera_calibration)
    slam_poses, points = run_slam(camera_matrix, subsampled_poses, features)

    downsample = stereo_utils.ImageDownsampler.square(
        camera_calibration, (1920, 1080), 512
    )

    stereo_pairs = list(zip(left_frames, right_frames))
    stereo_pairs = [
        (downsample(left), downsample(right))
        for left, right in stereo_pairs[::output_subsample]
    ]

    print("Running depth estimation...")
    depth_maps = depth_estimation.stereo_images_to_depth_maps(
        stereo_pairs, downsample.calibration
    )

    print("Running volumetric fusion...")
    # Note: since stereo depth estimation uses cropped + downsampled images, depth fusion
    # also uses the same downsampled images.
    feature_pointcloud = [point for id, point in points.items()]
    mesh = fusion.depth_fusion(
        stereo_utils.left_camera_matrix(downsample.calibration),
        stereo_pairs,
        depth_maps,
        slam_poses,
        feature_pointcloud,
    )

    output_mesh.parent.mkdir(parents=True, exist_ok=True)

    with open(output_mesh, "wb") as f:
        data = trimesh.exchange.ply.export_ply(
            mesh, encoding="binary", include_attributes=True
        )
        f.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--calibration", help="Camera calibration JSON file", required=True
    )
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
    parser.add_argument(
        "-o", "--output-mesh", help="Output mesh PLY file", required=True
    )
    args = parser.parse_args()

    camera_calibration = data_utils.load_camera_calibration(Path(args.calibration))
    run_reconstruction(
        camera_calibration,
        Path(args.left_video),
        Path(args.right_video),
        Path(args.poses),
        Path(args.output_mesh),
    )
