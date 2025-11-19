/**
 * @file pointcloud_transform_cuda.cu
 * @brief CUDA implementation for point cloud transformation
 *
 * @author Sandeep Dutta <sandeep@harmony-robotics.com>
 *
 * MIT License
 *
 * Copyright (c) 2025 Sandeep Dutta
 */

#include "jeeves_pcl_combiner/pointcloud_transform_cuda.hpp"
#include <cuda_runtime.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <iostream>

namespace pointcloud_cuda
{

// CUDA kernel for transforming points
__global__ void transformPointsKernel(
    const float* input_points,
    float* output_points,
    const float* transform_matrix,
    int num_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_points)
    {
        // Input point (x, y, z)
        float x = input_points[idx * 3 + 0];
        float y = input_points[idx * 3 + 1];
        float z = input_points[idx * 3 + 2];

        // Apply transformation: P_out = T * P_in
        // Transform matrix is stored in row-major order
        output_points[idx * 3 + 0] = transform_matrix[0] * x + transform_matrix[1] * y + transform_matrix[2] * z + transform_matrix[3];
        output_points[idx * 3 + 1] = transform_matrix[4] * x + transform_matrix[5] * y + transform_matrix[6] * z + transform_matrix[7];
        output_points[idx * 3 + 2] = transform_matrix[8] * x + transform_matrix[9] * y + transform_matrix[10] * z + transform_matrix[11];
    }
}

bool transformPointCloudCUDA(
    const sensor_msgs::msg::PointCloud2& input_cloud,
    const Eigen::Matrix4f& transform,
    sensor_msgs::msg::PointCloud2& output_cloud)
{
    // Copy header
    output_cloud = input_cloud;

    // Extract XYZ points from input cloud
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(input_cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(input_cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(input_cloud, "z");

    int num_points = input_cloud.width * input_cloud.height;

    if (num_points == 0)
    {
        std::cerr << "Error: Empty point cloud" << std::endl;
        return false;
    }

    // Allocate host memory for input points
    std::vector<float> h_input_points(num_points * 3);

    int point_idx = 0;
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++point_idx)
    {
        h_input_points[point_idx * 3 + 0] = *iter_x;
        h_input_points[point_idx * 3 + 1] = *iter_y;
        h_input_points[point_idx * 3 + 2] = *iter_z;
    }

    // Allocate device memory
    float *d_input_points = nullptr;
    float *d_output_points = nullptr;
    float *d_transform_matrix = nullptr;

    cudaError_t err;

    err = cudaMalloc(&d_input_points, num_points * 3 * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for input points: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc(&d_output_points, num_points * 3 * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for output points: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_points);
        return false;
    }

    err = cudaMalloc(&d_transform_matrix, 16 * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for transform matrix: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_points);
        cudaFree(d_output_points);
        return false;
    }

    // Copy input points to device
    err = cudaMemcpy(d_input_points, h_input_points.data(), num_points * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA memcpy failed for input points: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_points);
        cudaFree(d_output_points);
        cudaFree(d_transform_matrix);
        return false;
    }

    // Copy transform matrix to device (row-major format)
    float h_transform[16];
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            h_transform[i * 4 + j] = transform(i, j);
        }
    }

    err = cudaMemcpy(d_transform_matrix, h_transform, 16 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA memcpy failed for transform matrix: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_points);
        cudaFree(d_output_points);
        cudaFree(d_transform_matrix);
        return false;
    }

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;

    transformPointsKernel<<<num_blocks, threads_per_block>>>(
        d_input_points,
        d_output_points,
        d_transform_matrix,
        num_points);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_points);
        cudaFree(d_output_points);
        cudaFree(d_transform_matrix);
        return false;
    }

    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA device synchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_points);
        cudaFree(d_output_points);
        cudaFree(d_transform_matrix);
        return false;
    }

    // Copy results back to host
    std::vector<float> h_output_points(num_points * 3);
    err = cudaMemcpy(h_output_points.data(), d_output_points, num_points * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA memcpy failed for output points: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_points);
        cudaFree(d_output_points);
        cudaFree(d_transform_matrix);
        return false;
    }

    // Free device memory
    cudaFree(d_input_points);
    cudaFree(d_output_points);
    cudaFree(d_transform_matrix);

    // Write transformed points back to output cloud
    sensor_msgs::PointCloud2Iterator<float> out_iter_x(output_cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> out_iter_y(output_cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> out_iter_z(output_cloud, "z");

    for (int i = 0; i < num_points; ++i, ++out_iter_x, ++out_iter_y, ++out_iter_z)
    {
        *out_iter_x = h_output_points[i * 3 + 0];
        *out_iter_y = h_output_points[i * 3 + 1];
        *out_iter_z = h_output_points[i * 3 + 2];
    }

    return true;
}

void initCUDA()
{
    // Initialize CUDA device
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0)
    {
        std::cerr << "No CUDA devices found!" << std::endl;
        return;
    }

    cudaSetDevice(0);
    std::cout << "CUDA initialized with " << device_count << " device(s)" << std::endl;
}

void cleanupCUDA()
{
    cudaDeviceReset();
}

} // namespace pointcloud_cuda
