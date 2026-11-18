import argparse
import json
from pathlib import Path

import fast_tissue_rec.data_utils as data_utils
import fast_tissue_rec.stereo_utils as stereo_utils
import fast_tissue_rec.fusion.depth_estimation as depth_estimation


def main(camera_calibration_file: Path, stereo_index: Path, output_dir: Path):
    camera_calibration = data_utils.load_camera_calibration(camera_calibration_file)

    downsample = stereo_utils.ImageDownsampler.square(camera_calibration, (1920, 1080), 512)
    
    camera_matrix = stereo_utils.left_camera_matrix(downsample.calibration)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_utils.save_array_json(output_dir / "downsampled_camera_matrix.json", camera_matrix, "downsampled_left_cam")

    stereo_pairs = data_utils.load_stereo_images(stereo_index)
    stereo_pairs = [
        ( downsample(left), downsample(right) ) for left, right in stereo_pairs
    ]

    depth_maps = depth_estimation.stereo_images_to_depth_maps(stereo_pairs, downsample.calibration)

    file_paths = data_utils.save_depth_maps(output_dir, depth_maps)
    index = [ str(path.relative_to(output_dir)) for path in file_paths ]

    with open(output_dir / "depth_map_index.json", "w") as f:
        json.dump(index, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--camera-calibration", required=True, help="Camera calibration JSON file"
    )
    parser.add_argument(
        "-s",
        "--stereo-index",
        required=True,
        help="JSON file containing list of stereo pair file paths",
    )
    parser.add_argument("--output-dir", help="Output directory")

    args = parser.parse_args()
    main(Path(args.camera_calibration), Path(args.stereo_index), Path(args.output_dir))
