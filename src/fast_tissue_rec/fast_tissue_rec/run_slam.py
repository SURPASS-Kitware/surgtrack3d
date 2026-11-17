import argparse
import numpy as np
import numpy.typing as npt
from pathlib import Path
import sys

import fast_tissue_rec.data_utils as data_utils
import fast_tissue_rec.stereo_utils as stereo_utils
from kinematic_ba import slam


def preprocess_features(
    features_by_frame: list[npt.NDArray[np.float32]], minimum_observations=4
):
    feature_counts = {}

    for frame in features_by_frame:
        ids = frame[:, 0].astype(np.int64)
        for id in ids:
            if id in feature_counts:
                feature_counts[id] += 1
            else:
                feature_counts[id] = 1

    output = []
    for frame in features_by_frame:
        output_frame = []
        ids = frame[:, 0].astype(np.int64)
        observations = frame[:, 1:].astype(np.float32)

        for id, point in zip(ids, observations):
            if feature_counts[id] >= minimum_observations:
                output_frame.append([id, point[0], point[1]])

        output.append(np.array(output_frame))

    return output


def run_slam(
    camera_matrix: npt.NDArray[np.float32],
    camera_poses: list[npt.NDArray[np.float32]],
    features_by_frame: list[npt.NDArray[np.float32]],
):
    """Run SLAM given camera matrix (intrinsics), initial guess at camera poses, and a
    list of features for each corresponding frame. The features for each frame/camera
    pose should be a Nx3 matrix, where each row is a feature observation, the first
    column is an integer to uniquely identify observations of the same feature across
    multiple frames, and the last two columns are x,y pixel coordinates of the
    observation."""
    slam_solver = slam.SLAM()
    camera = slam.CameraModel(
        np.eye(4, dtype=np.float64), camera_matrix.astype(np.float64)
    )

    added_features = {}
    added_frames = []

    def initial_guess(
        camera_pose: npt.NDArray[np.float32], px: float, py: float, assumed_depth: float
    ):
        position = [
            (px - camera_matrix[0, 2]) * assumed_depth / camera_matrix[0, 0],
            (py - camera_matrix[1, 2]) * assumed_depth / camera_matrix[1, 1],
            assumed_depth,
        ]

        world_position = camera_pose[0:3, 0:3] @ position + camera_pose[0:3, 3]
        return world_position

    features_by_frame = preprocess_features(features_by_frame)

    for frame_idx in range(len(camera_poses)):
        estimated_camera_pose = camera_poses[frame_idx]

        feature_ids = features_by_frame[frame_idx][:, 0].astype(np.int64)
        feature_observations = features_by_frame[frame_idx][:, 1:].astype(np.float32)

        frame = slam.Frame(estimated_camera_pose, camera)
        added_frames.append(frame)

        for id, point in zip(feature_ids, feature_observations):
            if id not in added_features.keys():
                assumed_depth = 0.07
                feature_position = initial_guess(
                    estimated_camera_pose, point[0], point[1], assumed_depth
                )
                feature = slam_solver.addFeature(feature_position)
                added_features[id] = feature

            frame.addObservation(added_features[id], point)

        slam_solver.addFrame(frame)

    slam_solver.solve(True)

    poses = [frame.cameraPose() for frame in added_frames]
    points = {
        feature_id: np.array(feature.position)
        for feature_id, feature in added_features.items()
    }

    return poses, points


def main(
    camera_calibration_file: Path,
    camera_poses_file: Path,
    feature_observations_directory: Path,
    output_poses_file: Path,
    output_points_file: Path,
):
    camera_calibration = data_utils.load_camera_calibration(camera_calibration_file)
    camera_matrix = stereo_utils.left_camera_matrix(camera_calibration)
    poses = data_utils.load_camera_poses(camera_poses_file)
    features = data_utils.load_feature_observations(feature_observations_directory)

    if len(poses) != len(features):
        print(
            "Error: number of camera poses differs"
            " from number of frames with feature observations"
        )
        sys.exit(1)

    poses, points = run_slam(camera_matrix, poses, features)

    data_utils.save_camera_poses(output_poses_file, poses)
    data_utils.save_features(output_points_file, points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--camera-calibration", required=True, help="Camera calibration JSON file"
    )
    parser.add_argument(
        "-p",
        "--poses-input",
        required=True,
        help="JSON file containing list of estimated camera poses",
    )
    parser.add_argument(
        "-f",
        "--features-input",
        required=True,
        help="Directory containing CSV files with feature observations for each frame",
    )
    parser.add_argument(
        "--poses-output", required=True, help="JSON file to write camera poses to"
    )
    parser.add_argument(
        "--features-output",
        required=True,
        help="CSV file to write 3D feature positions to",
    )

    args = parser.parse_args()

    main(
        Path(args.camera_calibration),
        Path(args.poses_input),
        Path(args.features_input),
        Path(args.poses_output),
        Path(args.features_output),
    )

__all__ = ["run_slam"]
