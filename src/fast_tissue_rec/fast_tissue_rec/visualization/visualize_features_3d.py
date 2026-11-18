import argparse
from pathlib import Path

import open3d as o3d

import fast_tissue_rec.data_utils as data_utils


def save_features_3d_visualization(
    output_file: Path, features: dict[int, tuple[float, float]]
):
    feature_points = [point for id, point in features.items()]

    pcd = o3d.geometry.PointCloud()
    print(f"Saving point cloud as {output_file}...")
    pcd.points = o3d.utility.Vector3dVector(feature_points)
    o3d.io.write_point_cloud(output_file, pcd)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--features",
        required=True,
        help="CSV file containing feature ids and positions",
    )
    parser.add_argument("-o", "--output-file", required=True, help="Output PLY file")
    args = parser.parse_args()

    print("Loading features...")
    features = data_utils.load_features(Path(args.features))
    save_features_3d_visualization(Path(args.output_file), features)
