import importlib

import numpy as np
import numpy.typing as npt
import pycuda.autoinit

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import skimage


class TSDFVolume:
    """Truncated Signed-Distance Field volume

    Builds implicit representation of surface from depth images. Surface is represented
    as a (truncated) signed-distance volume, where each voxel stores distance to closed point on surface.
    """

    def __init__(
        self,
        origin: npt.NDArray[np.float32],
        extent: npt.NDArray[np.float32],
        voxel_size: float,
        truncation_margin: float,
    ):
        # Number of voxels per axis
        self.dimensions = np.ceil(extent / voxel_size).astype(int)
        self.extent = self.dimensions * voxel_size  # Round to whole number of voxels

        self.origin = origin.astype(np.float32)
        self.voxel_size = voxel_size
        self.truncation_margin = truncation_margin

        # Allocate volume on both host and GPU
        self._tsdf_volume_cpu = np.zeros(self.dimensions).astype(np.float32)
        self._weight_volume_cpu = np.zeros(self.dimensions).astype(np.float32)
        self._color_volume_cpu = np.zeros((*self.dimensions, 3)).astype(np.uint8)

        self._tsdf_volume_gpu = cuda.mem_alloc(self._tsdf_volume_cpu.nbytes)
        cuda.memcpy_htod(self._tsdf_volume_gpu, self._tsdf_volume_cpu)

        self._weight_volume_gpu = cuda.mem_alloc(self._weight_volume_cpu.nbytes)
        cuda.memcpy_htod(self._weight_volume_gpu, self._weight_volume_cpu)

        self._color_volume_gpu = cuda.mem_alloc(self._color_volume_cpu.nbytes)
        cuda.memcpy_htod(self._color_volume_gpu, self._color_volume_cpu)

        kernel_path = importlib.resources.files(
            "fast_tissue_rec.resources"
        ).joinpath("tsdf_kernel.cubin")
        self._kernel_module = cuda.module_from_file(str(kernel_path))
        self._integrate = self._kernel_module.get_function("integrate")

        # Determine block/grid size on GPU
        gpu_device = cuda.Device(0)
        self._max_gpu_threads_per_block = gpu_device.MAX_THREADS_PER_BLOCK
        n_blocks = int(
            np.ceil(
                float(np.prod(self.dimensions)) / float(self._max_gpu_threads_per_block)
            )
        )
        grid_dim_x = min(gpu_device.MAX_GRID_DIM_X, int(np.floor(np.cbrt(n_blocks))))
        grid_dim_y = min(
            gpu_device.MAX_GRID_DIM_Y, int(np.floor(np.sqrt(n_blocks / grid_dim_x)))
        )
        grid_dim_z = min(
            gpu_device.MAX_GRID_DIM_Z,
            int(np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))),
        )
        self._max_gpu_grid_dim = np.array([grid_dim_x, grid_dim_y, grid_dim_z]).astype(
            int
        )
        # _n_gpu_loops is number of loops required for the GPU to process the entire volume
        self._n_gpu_loops = int(
            np.ceil(
                float(np.prod(self.dimensions))
                / float(
                    np.prod(self._max_gpu_grid_dim) * self._max_gpu_threads_per_block
                )
            )
        )

    def integrate(
        self,
        color_image: npt.NDArray[np.uint8],
        depth_image: npt.NDArray[np.float32],
        camera_matrix: npt.NDArray,
        camera_pose: npt.NDArray,
        min_depth: float=1.0e-3,
        sample_weight: float=1.0,
    ):
        assert depth_image.shape[0:2] == color_image.shape[0:2], "Depth and color images have have same size for fusion"

        # integrate voxel volume (calls CUDA kernel)
        for gpu_loop_idx in range(self._n_gpu_loops):
            self._integrate(
                self._tsdf_volume_gpu,
                self._weight_volume_gpu,
                self._color_volume_gpu,
                gpuarray.vec.make_float3(*self.origin.astype(np.float32).flatten()),
                gpuarray.vec.make_int3(*self.dimensions.astype(np.int32).flatten()),
                cuda.In(np.ascontiguousarray(camera_matrix, dtype=np.float32)),
                cuda.In(np.ascontiguousarray(camera_pose, dtype=np.float32)),
                cuda.In(np.ascontiguousarray(color_image, dtype=np.uint8)),
                cuda.In(np.ascontiguousarray(depth_image, dtype=np.float32)),
                np.int32(gpu_loop_idx),
                np.float32(self.voxel_size),
                np.float32(min_depth),
                np.int32(depth_image.shape[1]),
                np.int32(depth_image.shape[0]),
                np.float32(self.truncation_margin),
                np.float32(sample_weight),
                block=(self._max_gpu_threads_per_block, 1, 1),
                grid=(
                    int(self._max_gpu_grid_dim[0]),
                    int(self._max_gpu_grid_dim[1]),
                    int(self._max_gpu_grid_dim[2]),
                ),
            )

    def get_volume(self):
        """Copy volume from GPU to CPU"""
        cuda.memcpy_dtoh(self._tsdf_volume_cpu, self._tsdf_volume_gpu)
        cuda.memcpy_dtoh(self._color_volume_cpu, self._color_volume_gpu)
        cuda.memcpy_dtoh(self._weight_volume_cpu, self._weight_volume_gpu)
        return self._tsdf_volume_cpu, self._color_volume_cpu, self._weight_volume_cpu

    def get_mesh(self):
        """Get mesh of voxel volume via marching cubes"""
        tsdf_volume, color_volume, weight_volume = self.get_volume()

        mask = weight_volume > 0.0

        vertices, faces, normals, values = skimage.measure.marching_cubes(
            tsdf_volume, level=0.0, gradient_direction="ascent", allow_degenerate=False, mask=mask
        )

        vertex_indices = np.round(vertices).astype(int)
        # voxel grid coordinates to world coordinates
        vertices = vertices * self.voxel_size + self.origin

        # Get vertex colors
        vertex_colors = color_volume[
            vertex_indices[:, 0], vertex_indices[:, 1], vertex_indices[:, 2], :
        ]
        vertex_colors = vertex_colors.reshape(-1, 3)
        # BGR to RGB
        vertex_colors = np.flip(vertex_colors, axis=1)

        # skimage will include faces even if only one vertex is valid
        valid_vertices = weight_volume[vertex_indices[:, 0], vertex_indices[:, 1], vertex_indices[:, 2]] > 0
        valid_faces_mask = valid_vertices[faces].all(axis=1)
        # skiamge uses left-hand rule, so we also reverse triangle winding order to match right-hand rule
        clean_faces = faces[valid_faces_mask, ::-1]

        return vertices, clean_faces, normals, vertex_colors
