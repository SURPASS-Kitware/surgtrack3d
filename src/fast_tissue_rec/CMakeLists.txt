cmake_minimum_required(VERSION 3.23)

# Only builds the GPU resources used by the Python package:
#   - the TSDF integration CUDA kernel (compiled to a .cubin), needed by fusion
#   - the FoundationStereo TensorRT engine + config, copied into the package, needed by depth
# Both are optional: without them the package still installs, and the CPU-only parts
# (tracking, rectification, ...) work. No compiler is needed when CUDA is skipped.
project(fast_tissue_rec LANGUAGES NONE)

# AUTO: build the kernel if a CUDA compiler is found. ON: require it. OFF: skip it.
set(FTR_BUILD_CUDA "AUTO" CACHE STRING "Build the TSDF CUDA kernel (AUTO, ON, OFF)")
set_property(CACHE FTR_BUILD_CUDA PROPERTY STRINGS AUTO ON OFF)

# Directory holding the TensorRT engine and its YAML config. The file names are
# fixed because fusion/depth_estimation.py loads them by name.
set(MODELS_DIR "/home/jbarrag3/research/surpass_models" CACHE PATH
    "Directory containing the FoundationStereo TensorRT engine and YAML config")

# GPU architecture for the TSDF kernel. A .cubin holds a single architecture, so on
# machines with mixed GPUs "native" fails; set this to the GPU used at runtime
# (CUDA device 0), e.g. -C cmake.define.TSDF_CUDA_ARCHITECTURES=120-real for an RTX 5090.
set(TSDF_CUDA_ARCHITECTURES "native" CACHE STRING
    "CUDA architecture for tsdf_kernel.cubin (single value)")

# --- TSDF CUDA kernel --------------------------------------------------------
set(_build_cuda OFF)
if(FTR_BUILD_CUDA STREQUAL "ON")
  set(_build_cuda ON)
elseif(FTR_BUILD_CUDA STREQUAL "AUTO")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(_build_cuda ON)
  endif()
endif()

if(_build_cuda)
  enable_language(CUDA)

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
else()
  message(WARNING "fast_tissue_rec: skipping the TSDF CUDA kernel (FTR_BUILD_CUDA=${FTR_BUILD_CUDA}). "
                  "Volumetric fusion will not be available.")
endif()

# --- FoundationStereo model resources -----------------------------------------
set(_stereo_models
    ${MODELS_DIR}/finetuned_foundation_stereo_32_iters.engine
    ${MODELS_DIR}/fast_foundationstereo.yaml)

set(_models_found ON)
foreach(_model IN LISTS _stereo_models)
  if(NOT EXISTS ${_model})
    set(_models_found OFF)
  endif()
endforeach()

if(_models_found)
  # Install Python module resources
  install(FILES ${_stereo_models} DESTINATION fast_tissue_rec/resources)
else()
  message(WARNING "fast_tissue_rec: FoundationStereo engine/config not found in MODELS_DIR=${MODELS_DIR}. "
                  "Depth estimation will not be available.")
endif()
