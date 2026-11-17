import argparse
import cv2 as cv
import numpy as np
import numpy.typing as npt
from pathlib import Path
from tqdm import tqdm

import fast_tissue_rec.data_utils as data_utils
import fast_tissue_rec.stereo_utils as stereo_utils


def feature_id_to_rgb_color(id: int, dark=True) -> tuple[int, int, int]:
    """Assign color to each feature id, trying to keep each color visually distinct
    without needing to know how many total features/colors will be assigned.

    `dark` parameter can be used to get both a dark and a light color (of the same hue)
    for the same ID, allowing e.g. reprojection vs tracked feature to be marked in
    similar but distinct colors.
    """

    # Really simply 24 bit hash/digest function
    hash_val = (id * 2654435761) & 0xFFFFFF
    hue = hash_val % 180

    # Use hash to control hue, keep color fully saturated so colors stand out
    color_hsv = np.array([[[hue, 255, 150 if dark else 255]]], dtype=np.uint8)
    color_bgr = cv.cvtColor(color_hsv, cv.COLOR_HSV2RGB)

    return (int(color_bgr[0][0][0]), int(color_bgr[0][0][1]), int(color_bgr[0][0][2]))


def save_features_2d_visualization(
    output_file: Path,
    image: npt.NDArray[np.uint8],
    feature_observations: npt.NDArray[np.float32],
    feature_points: dict[int, tuple[float, float]],
    camera_matrix: npt.NDArray[np.float32],
    camera_pose: npt.NDArray[np.float32],
):
    feature_ids = feature_observations[:, 0].astype(np.int64)
    feature_observations = feature_observations[:, 1:].astype(np.int32)

    def project(point3d):
        p = camera_pose[0:3, 0:3].T @ (point3d - camera_pose[0:3, 3])
        q = camera_matrix @ p
        u, v = q[0] / q[2], q[1] / q[2]
        return (int(u), int(v))

    feature_projections = {id: project(point) for id, point in feature_points.items()}
    feature_projections = {
        id: point
        for id, point in feature_projections.items()
        if point[0] >= 0
        and point[1] >= 0
        and point[0] < image.shape[1]
        and point[1] < image.shape[0]
    }

    # Draw tracked 2D features in light colors
    for id, observation in zip(feature_ids, feature_observations):
        color = feature_id_to_rgb_color(id, False)
        cv.circle(image, observation, radius=5, color=color, thickness=-1)

    # Draw projected 3D features in dark colors
    for id, projection in feature_projections.items():
        color = feature_id_to_rgb_color(id)
        cv.circle(image, projection, radius=5, color=color, thickness=-1)

    # Draw lines between projections and corresponding 2D features
    for id, observation in zip(feature_ids, feature_observations):
        if id not in feature_projections:
            continue

        color = feature_id_to_rgb_color(id, False)
        cv.line(image, feature_projections[id], observation, color=color, thickness=2)

    cv.imwrite(str(output_file), image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--camera-calibration", required=True, help="Camera calibration JSON file"
    )
    parser.add_argument(
        "-f",
        "--feature-points",
        required=True,
        help="CSV file containing feature ids and positions",
    )
    parser.add_argument(
        "-p",
        "--poses",
        required=True,
        help="JSON file containing camera poses (from SLAM) per frame",
    )
    parser.add_argument(
        "-s",
        "--stereo-index",
        required=True,
        help="JSON file containing list of stereo pair file paths",
    )
    parser.add_argument(
        "-t",
        "--tracks",
        required=True,
        help="Directory containing CSV files with feature observation tracks for each frame",
    )
    parser.add_argument("-o", "--output-dir", required=True)
    args = parser.parse_args()

    print("Loading data/configs...")

    feature_points = data_utils.load_features(Path(args.feature_points))
    feature_observations = data_utils.load_feature_observations(Path(args.tracks))

    camera_calibration = data_utils.load_camera_calibration(
        Path(args.camera_calibration)
    )
    camera_matrix = stereo_utils.left_camera_matrix(camera_calibration)
    poses = data_utils.load_camera_poses(Path(args.poses))

    stereo_pairs = data_utils.load_stereo_images(Path(args.stereo_index))
    images = [left for left, _ in stereo_pairs]

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    print("Saving annotated images...")
    for idx, (camera_pose, observations, image) in enumerate(
        zip(tqdm(poses), feature_observations, images)
    ):
        output = Path(args.output_dir) / f"features_{idx:04d}.png"
        save_features_2d_visualization(
            output, image, observations, feature_points, camera_matrix, camera_pose
        )

    print("Done!")
