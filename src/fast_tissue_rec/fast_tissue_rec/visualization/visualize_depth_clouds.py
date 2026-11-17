import argparse
from collections.abc import Sequence
import cv2 as cv
import numpy as np
import numpy.typing as npt
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import json

import fast_tissue_rec.data_utils as data_utils
import fast_tissue_rec.stereo_utils as stereo_utils


def save_point_cloud_visualizations(
    output_dir: Path,
    camera_matrix: npt.NDArray[np.float32],
    poses: Sequence[npt.NDArray[np.float32]],
    depth_images: Sequence[npt.NDArray[np.float32]],
    color_images: Sequence[npt.NDArray[np.uint8]],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = o3d.camera.PinholeCameraIntrinsic(512, 512, camera_matrix)

    print("Saving point cloud visualizations...")
    for i, (depth, color, pose) in enumerate(
        zip(tqdm(depth_images), color_images, poses)
    ):
        color = cv.cvtColor(color, cv.COLOR_BGR2RGB)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color.astype(np.uint8)),
            o3d.geometry.Image(depth.astype(np.float32)),
            depth_scale=1.0,  # otherwise it assumes we need to convert from mm to m
            convert_rgb_to_intensity=False,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
        pcd.transform(pose)
        o3d.io.write_point_cloud(output_dir / f"depth_cloud{i:04d}.ply", pcd)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--depth-index", help="Depth map index (JSON)", required=True
    )
    parser.add_argument(
        "-s", "--stereo-index", help="Stereo map index (JSON)", required=True
    )
    parser.add_argument(
        "-c", "--camera-calibration", required=True, help="Camera calibration JSON file"
    )
    parser.add_argument(
        "-p",
        "--poses",
        required=True,
        help="JSON file containing camera poses (from SLAM) per frame",
    )
    parser.add_argument("-o", "--output-dir", required=True)
    args = parser.parse_args()
    
    camera_calibration = data_utils.load_camera_calibration(
        Path(args.camera_calibration)
    )
    downsample = stereo_utils.ImageDownsampler.square(
        camera_calibration, (1920, 1080), 512
    )
    camera_matrix = stereo_utils.left_camera_matrix(downsample.calibration)

    data_utils.save_array_json(Path(args.output_dir) / "downsampled_camera_matrix.json", camera_matrix, "downsampled_left_cam") 

    print("Loading depth maps...")
    depth_maps = data_utils.load_depth_maps(Path(args.depth_index))

    print("Loading stereo images...")
    stereo_pairs = data_utils.load_stereo_images(Path(args.stereo_index))
    color_images = [downsample(left) for left, right in stereo_pairs]

    print("Loading camera poses...")
    poses = data_utils.load_camera_poses(Path(args.poses))

    save_point_cloud_visualizations(
        Path(args.output_dir), camera_matrix, poses, depth_maps, color_images
    )
