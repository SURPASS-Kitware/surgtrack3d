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
