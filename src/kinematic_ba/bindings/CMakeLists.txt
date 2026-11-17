find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_library(${CMAKE_PROJECT_NAME}_py MODULE main_module.cpp slam_module.cpp eigen_geometry.hpp)
target_link_libraries(${CMAKE_PROJECT_NAME}_py PUBLIC ${CMAKE_PROJECT_NAME}_slam_lib)
target_link_libraries(${CMAKE_PROJECT_NAME}_py PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)

pybind11_extension(${CMAKE_PROJECT_NAME}_py)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(${CMAKE_PROJECT_NAME}_py)
endif()

set_target_properties(${CMAKE_PROJECT_NAME}_py
    PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        OUTPUT_NAME _core
)

# Install C++ module/bindings into the Python package directory
install(TARGETS ${CMAKE_PROJECT_NAME}_py LIBRARY DESTINATION kinematic_ba)
