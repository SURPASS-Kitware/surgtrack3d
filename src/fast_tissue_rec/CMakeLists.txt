cmake_minimum_required(VERSION 3.23)

# Only builds the GPU resources used by the Python package:
#   - the TSDF integration CUDA kernel (compiled to a .cubin)
#   - the FoundationStereo TensorRT engine + config, copied into the package
project(fast_tissue_rec LANGUAGES CUDA)

# Directory holding the TensorRT engine and its YAML config. The file names are
# fixed because fusion/depth_estimation.py loads them by name.
set(MODELS_DIR "/home/jbarrag3/research/surpass_models" CACHE PATH
    "Directory containing the FoundationStereo TensorRT engine and YAML config")

# GPU architecture for the TSDF kernel. A .cubin holds a single architecture, so on
# machines with mixed GPUs "native" fails; set this to the GPU used at runtime
# (CUDA device 0), e.g. -C cmake.define.TSDF_CUDA_ARCHITECTURES=120-real for an RTX 5090.
set(TSDF_CUDA_ARCHITECTURES "native" CACHE STRING
    "CUDA architecture for tsdf_kernel.cubin (single value)")

add_library(tsdf_kernel OBJECT
    fast_tissue_rec/fusion/tsdf_kernel.cu
)
target_compile_features(tsdf_kernel PUBLIC cxx_std_17)
set_target_properties(tsdf_kernel PROPERTIES
    CUDA_CUBIN_COMPILATION ON
    CUDA_ARCHITECTURES ${TSDF_CUDA_ARCHITECTURES}
)

# Install compiled TSDF kernel into Python module
install(FILES $<TARGET_OBJECTS:tsdf_kernel> DESTINATION fast_tissue_rec/resources)
# For CMake >=4.2 we should instead use:
#   set_target_properties(tsdf_kernel PROPERTIES INSTALL_OBJECT_ONLY_USE_DESTINATION ON)
#   install(TARGETS tsdf_kernel OBJECTS DESTINATION fast_tissue_rec/resources)

# Install Python module resources
install(FILES
            ${MODELS_DIR}/finetuned_foundation_stereo_32_iters.engine
            ${MODELS_DIR}/fast_foundationstereo.yaml
        DESTINATION fast_tissue_rec/resources)
