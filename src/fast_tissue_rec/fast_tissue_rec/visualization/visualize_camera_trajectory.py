import argparse
import copy
from pathlib import Path

import open3d as o3d
from tqdm import tqdm

import fast_tissue_rec.data_utils as data_utils


def save_camera_trajectory_visualization(
    camera_poses, axlen: float = 0.005, output_file: Path = "trajectory.ply"
):
    trajectory = None
    for i, pose in enumerate(tqdm(camera_poses)):
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axlen)
        mesh_frame.transform(pose)
        if i == 0:
            trajectory = copy.deepcopy(mesh_frame)
        else:
            trajectory += copy.deepcopy(mesh_frame)

    print(f"Saving result to {output_file}")
    o3d.io.write_triangle_mesh(output_file, trajectory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--poses", required=True, help="Camera poses")
    parser.add_argument("-o", "--output", required=True, help="Output mesh file")
    args = parser.parse_args()

    print("Loading camera poses...")
    camera_poses = data_utils.load_camera_poses(Path(args.poses))
    print("    done!")

    save_camera_trajectory_visualization(camera_poses, output_file=Path(args.output))
