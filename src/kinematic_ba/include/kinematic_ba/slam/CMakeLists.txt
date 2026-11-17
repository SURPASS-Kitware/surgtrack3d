target_sources(${CMAKE_PROJECT_NAME}_slam_lib
  PUBLIC FILE_SET HEADERS
    BASE_DIRS "../.."
    FILES
      "camera_model.hpp"
      "feature.hpp"
      "frame.hpp"
      "slam.hpp"
)
