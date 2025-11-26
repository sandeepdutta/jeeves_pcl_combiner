cmake_minimum_required(VERSION 3.8)
project(jeeves_pcl_combiner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Option to enable CUDA support
option(USE_CUDA "Enable CUDA support for GPU-accelerated transformations" ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(message_filters REQUIRED)
find_package(Eigen3 REQUIRED)

# Add executable
if(USE_CUDA)
  # Enable CUDA language
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # Add CUDA sources
  add_executable(jeeves_pcl_combiner
    src/jeeves_pcl_combiner.cpp
    src/pointcloud_transform_cuda.cu
  )

  # Set CUDA properties
  set_target_properties(jeeves_pcl_combiner PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
  )

  # Add CUDA compile definition
  target_compile_definitions(jeeves_pcl_combiner PRIVATE USE_CUDA)

  # Include CUDA libraries
  target_link_libraries(jeeves_pcl_combiner
    CUDA::cudart
  )

  # Include directories for CUDA
  target_include_directories(jeeves_pcl_combiner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
  )

  message(STATUS "CUDA support enabled for jeeves_pcl_combiner")
else()
  # CPU-only version
  add_executable(jeeves_pcl_combiner src/jeeves_pcl_combiner.cpp)
  message(STATUS "CUDA support disabled for jeeves_pcl_combiner")
endif()

ament_target_dependencies(jeeves_pcl_combiner
  rclcpp
  sensor_msgs
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  pcl_ros
  pcl_conversions
  message_filters
)

# Install executables
install(TARGETS
  jeeves_pcl_combiner
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files if any
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
  OPTIONAL
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
