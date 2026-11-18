from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import fast_tissue_rec.fusion.tsdf_volume as tsdf_volume
import trimesh
from tqdm import tqdm


class DepthScaling:
    def __init__(self, camera_matrix: npt.NDArray[np.float32], epsilon: float = 1.0e-8):
        self.camera_matrix = camera_matrix
        self.epsilon = epsilon

    def point_depth(
        self,
        camera_pose: npt.NDArray[np.float32],
        feature_point: npt.NDArray[np.float32],
    ):
        in_camera_frame = camera_pose[0:3, 0:3].T @ (
            feature_point - camera_pose[0:3, 3]
        )
        depth = in_camera_frame[2]
        if depth < self.epsilon:
            return None, None

        projective = self.camera_matrix @ in_camera_frame
        projected = projective[0:2] / projective[2]

        return projected, depth

    def average_scale(
        self,
        camera_pose: npt.NDArray[np.float32],
        features: npt.NDArray[np.float32],
        depth_map: npt.NDArray,
    ):
        H, W = depth_map.shape[0:2]

        scales = []

        for point in features:
            point2d, sparse_depth = self.point_depth(camera_pose, point)
            if sparse_depth is None:
                continue
            
            if point2d[0] < 0 or point2d[0] >= W or point2d[1] < 0 or point2d[1] >= H:
                continue

            dense_depth = depth_map[int(point2d[1]), int(point2d[0])]
            if dense_depth == 0:
                continue

            scales.append(sparse_depth / dense_depth)

        if len(scales) == 0:
            return 1.0

        return np.median(scales)


# Get corners of 3D camera view frustum of depth image
# depth_im: 2D image containing depth (Z) of each pixel in camera coordinate system
# camera_intrinsics: 3x3 camera matrix
# camera_pose: 4x4 homogeneous pose of camera with respect to world
# Returns a 5x3 matrix, where each row is one of the five points defining the view frustum
def get_view_frustum(
    depth_image: np.ndarray, camera_intrinsics: np.ndarray, camera_pose: np.ndarray
) -> np.ndarray:
    H, W = depth_image.shape[0:2]
    max_depth = np.nanmax(depth_image)
    min_depth = np.nanmin(depth_image)

    closest = min(0.75 * max_depth, min_depth)

    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]

    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    def unproject(px, py, z):
        x = (px - cx) * z / fx
        y = (py - cy) * z / fy
        return np.array([x, y, z], dtype=np.float32)

    # In camera coordinates
    view_frustum = np.array(
        [
            unproject(0, 0, max_depth),
            unproject(0, H, max_depth),
            unproject(W, 0, max_depth),
            unproject(W, H, max_depth),
            unproject(0, 0, closest),
            unproject(0, H, closest),
            unproject(W, 0, closest),
            unproject(W, H, closest),
        ]
    )

    # In world coordinates
    R = camera_pose[0:3, 0:3]
    t = camera_pose[0:3, 3]
    view_frustum = (R @ view_frustum.T).T + t

    return view_frustum


def depth_fusion(
    camera_matrix: npt.NDArray[np.float32],
    stereo_pairs: Sequence[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]],
    depth_maps: Sequence[npt.NDArray[np.float32]],
    poses: Sequence[npt.NDArray[np.float32]],
    sparse_point_cloud: Sequence[npt.NDArray[np.float32]],
    num_voxels: float = 320000000.0,
    verbose: bool = True,
) -> trimesh.Trimesh:
    assert len(depth_maps) == len(stereo_pairs), "Must have same number of depth maps as stereo pairs"
    assert len(depth_maps) == len(poses), "Must have same number of depth maps as camera poses"

    print("Estimating volume bounds ...")
    volume_bounds = [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]

    scaler = DepthScaling(camera_matrix)

    scaled_depth_maps = []

    for depth_map, pose in zip(tqdm(depth_maps), poses):
        scale = scaler.average_scale(pose, sparse_point_cloud, depth_map)
        if verbose:
            print(f"Scale: {scale}")

        scaled_depth = 1.0 * depth_map
        scaled_depth_maps.append(scaled_depth)

        view_frustum = get_view_frustum(scaled_depth, camera_matrix, pose)

        volume_bounds[0] = np.fmin(volume_bounds[0], np.nanmin(view_frustum, axis=0))
        volume_bounds[1] = np.fmax(volume_bounds[1], np.nanmax(view_frustum, axis=0))

    print("Setting up volume ...")
    print("Volume bounds: ", volume_bounds)
    volume_extent = volume_bounds[1] - volume_bounds[0]
    # Divide volume into largest number of voxels we can still fit on the GPU
    volume = volume_extent[0] * volume_extent[1] * volume_extent[2]
    voxel_size = (volume / num_voxels) ** (1.0 / 3.0)
    print(f"Voxel size: {1000.0 * voxel_size:.04f} mm")

    # Initialize voxel volume
    volume = tsdf_volume.TSDFVolume(
        volume_bounds[0],
        volume_bounds[1] - volume_bounds[0],
        voxel_size,
        truncation_margin=0.02,
    )

    print("Integrating depth images...")

    for depth_map, (left_image, _), pose in zip(tqdm(scaled_depth_maps), stereo_pairs, poses):
        volume.integrate(left_image, depth_map, camera_matrix, pose)

    print("Transferring volume to CPU...")

    vertices, faces, normals, colors = volume.get_mesh()

    print("Isosurface extraction...")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    return mesh
