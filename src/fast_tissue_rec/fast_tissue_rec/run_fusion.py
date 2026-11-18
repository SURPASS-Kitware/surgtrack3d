import argparse
from pathlib import Path

import trimesh

import fast_tissue_rec.data_utils as data_utils
import fast_tissue_rec.fusion.fusion as fusion
import fast_tissue_rec.stereo_utils as stereo_utils


def main(
    camera_calibration_file: Path,
    stereo_index_file: Path,
    depth_index: Path,
    camera_poses_file: Path,
    features_file: Path,
    output_mesh: Path,
    num_voxels: float = 320000000.0,
):
    camera_calibration = data_utils.load_camera_calibration(camera_calibration_file)
    stereo_pairs = data_utils.load_stereo_images(stereo_index_file)
    downsample = stereo_utils.ImageDownsampler.square(
        camera_calibration, (1920, 1080), 512
    )

    stereo_pairs = [
        (downsample(left), downsample(right)) for left, right in stereo_pairs
    ]
    camera_matrix = stereo_utils.left_camera_matrix(downsample.calibration)

    depth_maps = data_utils.load_depth_maps(depth_index)
    poses = data_utils.load_camera_poses(camera_poses_file)
    features = data_utils.load_features(features_file)

    feature_pointcloud = [feature for id, feature in features.items()]

    mesh = fusion.depth_fusion(
        camera_matrix, stereo_pairs, depth_maps, poses, feature_pointcloud, num_voxels
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
        "-c", "--camera-calibration", help="Camera calibration JSON file", required=True
    )
    parser.add_argument(
        "-s",
        "--stereo-index",
        required=True,
        help="JSON file containing list of stereo pair file paths",
    )
    parser.add_argument(
        "-d",
        "--depth-index",
        required=True,
        help="JSON file containing list of depth map file paths",
    )
    parser.add_argument(
        "-p",
        "--poses",
        required=True,
        help="JSON file containing camera poses (from SLAM) per frame",
    )
    parser.add_argument(
        "-f",
        "--features",
        required=True,
        help="CSV file containing feature ids and positions",
    )
    parser.add_argument(
        "-o", "--output-mesh", required=True, help="Output mesh PLY file"
    )
    parser.add_argument(
        "-n",
        "--num-voxels",
        type=float,
        default=320000000.0,
        help="Number of voxels used to derive the TSDF voxel size",
    )

    args = parser.parse_args()

    main(
        Path(args.camera_calibration),
        Path(args.stereo_index),
        Path(args.depth_index),
        Path(args.poses),
        Path(args.features),
        Path(args.output_mesh),
        args.num_voxels,
    )
