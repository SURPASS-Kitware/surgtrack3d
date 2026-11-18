# Installation Notes

## Package installation
1. Bundle adjustment (C++ build through vcpkg; VCPKG_ROOT must be set)
echo $VCPKG_ROOT        # should print /home/jbarrag3/programs/vcpkg
pip install ./src/kinematic_ba -c src/kinematic_ba/requirements.txt

2. Pipeline, all extras (tracking, BA, depth, fusion, visualization)
pip install "./src/fast_tissue_rec[all]" -c src/fast_tissue_rec/requirements.txt \
    -C cmake.define.MODELS_DIR=/home/jbarrag3/research/surpass_models \
    -C cmake.define.TSDF_CUDA_ARCHITECTURES=120-real

3. Check
pip check
cd /tmp && python -c "import kinematic_ba, fast_tissue_rec, fast_tissue_rec.fusion.fusion; print('ok')"

## Notes:
- Step 2 GPU flag: TSDF_CUDA_ARCHITECTURES=120-real is needed because this machine has two different GPU models. It builds the kernel for the RTX 5090, which is CUDA device 0.
- Step 2 model files: MODELS_DIR is where the TensorRT engine and its YAML config are copied from. If you see a warning that they weren't found, depth estimation won't work.
- If something fails to build: the install log is long. Adding -v to the pip command shows the CMake output.
- For rectification only: pip install ./src/fast_tissue_rec -c src/fast_tissue_rec/requirements.txt is enough, with no CUDA and no vcpkg.

## Testing only volumetric fusion.

pip install "./src/fast_tissue_rec[fusion]" \
    -C cmake.define.FTR_BUILD_CUDA=ON \
    -C cmake.define.TSDF_CUDA_ARCHITECTURES=native

- FTR_BUILD_CUDA=ON makes the build fail loudly if nvcc isn't found, instead of silently skipping the kernel.
- TSDF_CUDA_ARCHITECTURES: native is fine with a single GPU. With mixed GPUs, use the one CUDA calls device 0, e.g. 120-real for a 5090, 89-real for a 4090, or 86-real for a 3090.
- The "FoundationStereo engine not found" warning is expected; ignore it.
- Exact versions (optional): add -c src/fast_tissue_rec/requirements.txt.

Check the install:
cd /tmp && python -c "import fast_tissue_rec.fusion.fusion; print('ok')"

Data to copy (example: kidney, about 500 MB)

The index files use relative paths, so you can copy the folders as they are:
dataset/raw_data/kidney/calibration.json
data_intermediate_outputs/reconstruction/kidney/depth_maps/              (tiffs + depth_map_index.json)
data_intermediate_outputs/reconstruction/kidney/tracking_output/images/  (pngs + stereo_index.json)
data_intermediate_outputs/reconstruction/kidney/tracking_output/slam_poses.json
data_intermediate_outputs/reconstruction/kidney/tracking_output/features.csv

Run

O=kidney
python -m fast_tissue_rec.run_fusion -c calibration.json \
    -s $O/tracking_output/images/stereo_index.json \
    -d $O/depth_maps/depth_map_index.json \
    -p $O/tracking_output/slam_poses.json -f $O/tracking_output/features.csv \
    -n 640000000.0 -o $O/output_mesh_high_res.ply

-n 640000000.0 is the voxel count used for the meshes here. On a GPU with less memory, use the default (320M) or lower.
