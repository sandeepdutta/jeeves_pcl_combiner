/**
 * @file pointcloud_transform_cuda.hpp
 * @brief CUDA-accelerated point cloud transformation
 *
 * @author Sandeep Dutta <sandeep@harmony-robotics.com>
 *
 * MIT License
 *
 * Copyright (c) 2025 Sandeep Dutta
 */

#ifndef POINTCLOUD_TRANSFORM_CUDA_HPP
#define POINTCLOUD_TRANSFORM_CUDA_HPP

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <Eigen/Dense>
#include <vector>

namespace pointcloud_cuda
{

/**
 * @brief Transform point cloud using CUDA GPU acceleration
 *
 * @param input_cloud Input point cloud in sensor_msgs format
 * @param transform 4x4 transformation matrix (Eigen)
 * @param output_cloud Output transformed point cloud
 * @return true if transformation succeeded, false otherwise
 */
bool transformPointCloudCUDA(
    const sensor_msgs::msg::PointCloud2& input_cloud,
    const Eigen::Matrix4f& transform,
    sensor_msgs::msg::PointCloud2& output_cloud);

/**
 * @brief Initialize CUDA resources
 */
void initCUDA();

/**
 * @brief Cleanup CUDA resources
 */
void cleanupCUDA();

} // namespace pointcloud_cuda

#endif // POINTCLOUD_TRANSFORM_CUDA_HPP
