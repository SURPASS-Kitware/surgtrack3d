find_package(Catch2 3 REQUIRED)

set (TEST_FILES
  slam_tests.cpp
)

add_executable(tests ${TEST_FILES})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain ${PROJECT_NAME}_slam_lib)
