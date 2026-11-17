cmake_minimum_required(VERSION 3.23)

project(kinematic_ba LANGUAGES C CXX)

set (CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
  add_compile_options(-Wpedantic)
endif()

find_package(Ceres CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_library(${PROJECT_NAME}_slam_lib STATIC)
target_include_directories(${PROJECT_NAME}_slam_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
target_compile_features(${PROJECT_NAME}_slam_lib PUBLIC cxx_std_20)
set_property(TARGET ${PROJECT_NAME}_slam_lib PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(${PROJECT_NAME}_slam_lib PUBLIC
  Ceres::ceres
  Eigen3::Eigen
)

add_subdirectory(include)
add_subdirectory(src)

if(SKBUILD STREQUAL "2")
  add_subdirectory(bindings)
endif()

# Skip building C++ unit tests if building Python module
if (NOT DEFINED SKBUILD)
  add_subdirectory(tests)
endif()

