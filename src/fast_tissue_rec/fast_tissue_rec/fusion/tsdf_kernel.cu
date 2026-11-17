extern "C" {

__global__ void integrate(
    float* tsdf_volume,
    float* weight_volume,
    uchar3* color_volume,
    const float3 volume_origin,
    const int3 volume_dimensions,
    const float* camera_matrix,
    const float* camera_pose,
    const uchar3* color_image,
    const float* depth_image,
    const int gpu_loop_idx,
    const float voxel_size,
    const float min_depth,
    const int image_width,
    const int image_height,
    const float truncation_distance,
    const float observation_weight
) {
    // Get voxel index
    // Block is 1D so threadIdx is always (x, 0, 0)
    // Flattened index of current block within overall grid
    const int block_idx = blockIdx.z*gridDim.y*gridDim.x + blockIdx.y*gridDim.x + blockIdx.x;
    const int threads_per_loop = gridDim.x*gridDim.y*gridDim.z*blockDim.x;
    const int voxel_idx = gpu_loop_idx*threads_per_loop + block_idx*blockDim.x + threadIdx.x;

    if (voxel_idx >= volume_dimensions.x*volume_dimensions.y*volume_dimensions.z) {
        return;
    }

    const int voxel_x = voxel_idx / (volume_dimensions.y*volume_dimensions.z);
    const int voxel_y = (voxel_idx - voxel_x*volume_dimensions.y*volume_dimensions.z) / volume_dimensions.z;
    const int voxel_z = voxel_idx - voxel_x*volume_dimensions.y*volume_dimensions.z - voxel_y*volume_dimensions.z;

    // Voxel grid coordinates to world coordinates
    const float3 voxel_wrt_world = make_float3(
        volume_origin.x + voxel_size * voxel_x,
        volume_origin.y + voxel_size * voxel_y,
        volume_origin.z + voxel_size * voxel_z
    );

    // Convert voxel position from world coordinates to camera coordinates
    const float tmp_pt_x = voxel_wrt_world.x - camera_pose[0*4+3];
    const float tmp_pt_y = voxel_wrt_world.y - camera_pose[1*4+3];
    const float tmp_pt_z = voxel_wrt_world.z - camera_pose[2*4+3];
    const float cam_pt_x = camera_pose[0*4+0]*tmp_pt_x + camera_pose[1*4+0]*tmp_pt_y + camera_pose[2*4+0]*tmp_pt_z;
    const float cam_pt_y = camera_pose[0*4+1]*tmp_pt_x + camera_pose[1*4+1]*tmp_pt_y + camera_pose[2*4+1]*tmp_pt_z;
    const float cam_pt_z = camera_pose[0*4+2]*tmp_pt_x + camera_pose[1*4+2]*tmp_pt_y + camera_pose[2*4+2]*tmp_pt_z;

    // Skip voxels too close to camera origin
    if (cam_pt_z < min_depth) {
        return;
    }

    const float3 voxel_wrt_camera = make_float3(cam_pt_x, cam_pt_y, cam_pt_z);
    const float distance_camera_to_voxel = norm3df(voxel_wrt_camera.x, voxel_wrt_camera.y, voxel_wrt_camera.z);

    // Camera coordinates to image pixels
    const float voxel_u = cam_pt_x/cam_pt_z;
    const float voxel_v = cam_pt_y/cam_pt_z;
    const int pixel_x = (int)roundf(camera_matrix[0*3+0]*voxel_u + camera_matrix[0*3+2]);
    const int pixel_y = (int)roundf(camera_matrix[1*3+1]*voxel_v + camera_matrix[1*3+2]);

    // Skip if outside view frustum
    if (pixel_x < 0 || pixel_x >= image_width || pixel_y < 0 || pixel_y >= image_height) {
        return;
    }

    // Skip invalid/masked-out depth
    const float z_camera_to_surface = depth_image[pixel_y*image_width + pixel_x];
    if (z_camera_to_surface <= 0.0) {
        return;
    }

    const float distance_camera_to_surface = z_camera_to_surface * norm3df(voxel_u, voxel_v, 1.0);

    // Distance from camera to voxel + voxel to surface = camera to surface
    const float distance_voxel_to_surface = distance_camera_to_surface - distance_camera_to_voxel;
    if (distance_voxel_to_surface < -truncation_distance || distance_voxel_to_surface > truncation_distance) {
        return;
    }

    // Integrate depth (and weight)
    const float old_weight = weight_volume[voxel_idx];
    const float new_weight = old_weight + observation_weight;
    // Welford's numerically stable (weighted) running average, to avoid some numerical issues we were observing
    tsdf_volume[voxel_idx] += (distance_voxel_to_surface - tsdf_volume[voxel_idx]) * observation_weight / new_weight;
    weight_volume[voxel_idx] = new_weight;

    // Integrate color
    const uchar3 pixel_color = color_image[pixel_y * image_width + pixel_x];
    const uchar3 voxel_color = color_volume[voxel_idx];

    // Welford's weighted running average again
    const float new_b = fmin(roundf(voxel_color.x + ((float)pixel_color.x - (float)voxel_color.x) * observation_weight / new_weight), 255.0f);
    const float new_g = fmin(roundf(voxel_color.y + ((float)pixel_color.y - (float)voxel_color.y) * observation_weight / new_weight), 255.0f);
    const float new_r = fmin(roundf(voxel_color.z + ((float)pixel_color.z - (float)voxel_color.z) * observation_weight / new_weight), 255.0f);

    color_volume[voxel_idx] = make_uchar3((unsigned char)new_b, (unsigned char)new_g, (unsigned char)new_r);
}

}
