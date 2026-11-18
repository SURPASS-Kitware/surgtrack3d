import csv
import json
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

import cv2 as cv
import numpy as np
import numpy.typing as npt
import scipy.spatial.transform as tr
from natsort import natsorted

# Minimum number of digits in per-frame output file names (points_000.csv, left_000.png,
# depth_000.tiff). Widened automatically if there are more frames than this allows.
FILE_INDEX_MIN_DIGITS = 3


def _index_digits(count: int) -> int:
    return max(FILE_INDEX_MIN_DIGITS, len(str(count)))


def save_array_json(path, array, header):
    data = {
        "header": header,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "data": array.tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_camera_calibration(calibration_file: Path) -> dict:
    """Load stereo camera calibration from JSON file"""
    with open(calibration_file, "r") as f:
        calibration = json.load(f)

    return calibration


def json_to_pose(json):
    pose = np.eye(4)
    position = json["position"]
    pose[0:3, 3] = [position[0], position[1], position[2]]

    q = json["orientation"]
    orientation = tr.Rotation.from_quat([q[0], q[1], q[2], q[3]])
    pose[0:3, 0:3] = orientation.as_matrix()
    return pose.astype(np.float64)


def pose_to_json(pose):
    orientation = tr.Rotation.from_matrix(pose[0:3, 0:3])
    q = orientation.as_quat()

    json = {
        "position": [pose[0, 3], pose[1, 3], pose[2, 3]],
        "orientation": [q[0], q[1], q[2], q[3]],
    }

    return json


def load_camera_poses(camera_poses_file: Path) -> list[npt.NDArray[np.float32]]:
    """Load sequence of camera poses from JSON file"""
    with open(camera_poses_file, "r") as f:
        camera_poses_json = json.load(f)

    camera_poses = [json_to_pose(p) for p in camera_poses_json]

    return camera_poses


def save_camera_poses(
    output_file: Path, camera_poses: list[npt.NDArray[np.float32]]
) -> None:
    """Save sequence of camera poses as JSON file"""
    camera_poses_json = [pose_to_json(p) for p in camera_poses]

    with open(output_file, "w") as f:
        json.dump(camera_poses_json, f)


def save_feature_observations(
    tracks_csv_folder: Path, feature_observations: list[npt.NDArray[np.float32]]
):
    frames_idx_len = _index_digits(len(feature_observations))
    for idx, frame in enumerate(feature_observations):
        with open(
            tracks_csv_folder / f"points_{str(idx).zfill(frames_idx_len)}.csv",
            "w",
        ) as f:
            w = csv.writer(f)
            w.writerow(["point_idx", "x", "y"])
            for id, x, y in frame:
                w.writerow([int(id), str(x), str(y)])


def load_feature_observations(
    tracks_csv_folder: Path,
) -> list[npt.NDArray[np.float32]]:
    """
    Feature observations are stored as separate CSV files, one per frame. Each frame's CSV
    file has three columns: point_idx, x, y, which are the feature id, and observed x/y
    pixel coordinates for each feature observed in that frame.

    Returns a list, where list index corresponds to frame index, and each list is Nx3
    array of the features observed in that frame.
    """
    path_list = natsorted(tracks_csv_folder.glob("*.csv"))
    tracks_per_frame: list[npt.NDArray[np.float32]] = []

    for csv_file in path_list:
        points = []
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            _ = next(reader)  # * Skip the header

            for row in reader:
                feature_id, x, y = int(row[0]), float(row[1]), float(row[2])
                points.append([feature_id, x, y])

        tracks_per_frame.append(np.array(points).astype(np.float32))
    print(f"Loaded {len(tracks_per_frame)} track files")

    return tracks_per_frame


def save_features(
    output_file: Path, feature_points: dict[int, npt.NDArray[np.float32]]
):
    """
    feature_points is a map of feature id to 3D position

    Output format is CSV file with header, where columns are id, x, y, z coordinates,
    and each row is estimated 3D location of a feature, with respect to world frame
    """

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["id", "x", "y", "z"])
        for id, point in feature_points.items():
            writer.writerow([id, *point])


def load_features(features_file: Path) -> dict[int, npt.NDArray[np.float32]]:
    """
    Input format is CSV file with header, where columns are id, x, y, z coordinates,
    and each row is estimated 3D location of a feature, with respect to world frame
    """

    features = {}

    with open(features_file, "r") as f:
        reader = csv.reader(f)
        _ = next(reader)  # * Skip the header

        for row in reader:
            feature_id, x, y, z = (
                int(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
            )
            features[feature_id] = np.array([x, y, z])

    return features


def load_video(video: Path, subsample_factor: int) -> list[npt.NDArray[np.uint8]]:
    """Load frames from video, optionally subsampling (can speed up load a bit)"""
    input = cv.VideoCapture(video)
    if not input.isOpened():
        raise RuntimeError(f"Failed to open input video {video.resolve()}")

    frames = []
    frame_idx = 0

    # read video, only fully decode every nth frame
    while True:
        if frame_idx % subsample_factor != 0:
            ok = input.grab()
            if not ok:
                break
        else:
            ok, frame = input.read()
            if not ok:
                break

            frames.append(frame)

        frame_idx += 1

    return frames


def load_stereo_images(
    index_file: Path,
) -> Sequence[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]]:
    with open(index_file, "r") as f:
        index = json.load(f)

    stereo_pairs = []
    index_dir = Path(index_file).parent

    def load_image(relative_path: Path) -> cv.Mat | None:
        # Resolve image paths relative to parent folder of index file
        image_path = (index_dir / relative_path).resolve()
        if not image_path.is_file():
            print(f'Image path "{image_path}" is not a valid file')
            return None

        image = cv.imread(str(image_path))
        if image is None:
            print(f"Failed to load image {image_path}")
            return None

        return image

    for frame in index:
        left_image = load_image(frame["left_image"])
        right_image = load_image(frame["right_image"])
        if left_image is None or right_image is None:
            return None

        stereo_pairs.append((left_image, right_image))

    return stereo_pairs


def save_stereo_images(
    frames: list[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]], output_file: Path
) -> None:
    # determine how many leading zeros we need to pad to, so that all file names have same
    # total number of digits
    total_count = len(frames)
    count_len = _index_digits(total_count)

    output_dir = output_file.parent

    index = []
    for idx, (left_frame, right_frame) in enumerate(frames):
        left_path = output_dir / f"left_{str(idx).zfill(count_len)}.png"
        right_path = output_dir / f"right_{str(idx).zfill(count_len)}.png"
        cv.imwrite(left_path, left_frame)
        cv.imwrite(right_path, right_frame)

        index.append(
            {
                "left_image": str(left_path.relative_to(output_dir)),
                "right_image": str(right_path.relative_to(output_dir)),
            }
        )

    with open(output_file, "w") as f:
        json.dump(index, f)


def save_depth_maps(
    output_dir: Path, depth_maps: Sequence[npt.NDArray[np.float32]]
) -> list[Path]:
    index = []

    output_dir.mkdir(exist_ok=True, parents=True)

    count_len = _index_digits(len(depth_maps))
    for idx, depth_map in enumerate(depth_maps):
        file_path = output_dir / f"depth_{str(idx).zfill(count_len)}.tiff"
        index.append(file_path)
        cv.imwrite(str(file_path), depth_map)

    return index


def load_depth_maps(depth_index_file: Path) -> Sequence[npt.NDArray[np.float32]]:
    with open(depth_index_file, "r") as f:
        depth_index = json.load(f)

    depth_maps = []
    for relative_file in depth_index:
        file_name = depth_index_file.parent / relative_file
        depth_map = cv.imread(str(file_name), cv.IMREAD_UNCHANGED)
        depth_maps.append(depth_map)

    return depth_maps


__all__ = [
    "load_camera_calibration",
    "load_camera_poses",
    "save_camera_poses",
    "load_feature_observations",
    "load_features",
    "save_features",
    "load_video",
    "load_stereo_images",
    "save_stereo_images",
    "save_depth_maps",
    "load_depth_maps",
]
