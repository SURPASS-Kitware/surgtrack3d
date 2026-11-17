target_sources(${PROJECT_NAME}_slam_lib
  PRIVATE
    "frame.cpp"
    "slam.cpp"
  PUBLIC FILE_SET HEADERS
    BASE_DIRS "."
    FILES
        "camera_reprojection_error.hpp"
        "pose_observation.hpp"
)
