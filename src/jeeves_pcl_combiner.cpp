/**
 * @file jeeves_pcl_combiner.cpp
 * @brief Point cloud combiner node for synchronizing and combining multiple point clouds
 *
 * This node subscribes to multiple point cloud topics, transforms them to a common frame,
 * and publishes a combined point cloud.
 *
 * @author Sandeep Dutta <sandeep@harmony-robotics.com>
 *
 * MIT License
 *
 * Copyright (c) 2025 Sandeep Dutta
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <pcl_ros/transforms.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <thread>
#include <future>
#include <mutex>
#include <chrono>
#include <numeric>
#include <tf2_eigen/tf2_eigen.hpp>

#ifdef USE_CUDA
#include "jeeves_pcl_combiner/pointcloud_transform_cuda.hpp"
#endif

using std::placeholders::_1;
using std::placeholders::_2;

class PointCloudCombiner : public rclcpp::Node
{
public:
    PointCloudCombiner() : Node("pointcloud_combiner"),
                           tf_buffer_(this->get_clock()),
                           tf_listener_(tf_buffer_)
    {
        // Declare parameters
        this->declare_parameter<std::string>("target_frame", "base_link");
        this->declare_parameter<bool>("use_cuda", false);

        target_frame_ = this->get_parameter("target_frame").as_string();
        use_cuda_ = this->get_parameter("use_cuda").as_bool();

        // Create subscribers for the two point cloud topics
        pc1_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(this, "/pointcloud1");
        pc2_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(this, "/pointcloud2");

        // Synchronize point clouds using approximate time sync policy
        sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(20), *pc1_sub_, *pc2_sub_));
        sync_->registerCallback(std::bind(&PointCloudCombiner::sync_callback, this, _1, _2));

        // Publisher for the combined point cloud
        pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/combined_pointcloud", 10);

        // Create timer to print statistics every 10 seconds
        stats_timer_ = this->create_wall_timer(
            std::chrono::seconds(10),
            std::bind(&PointCloudCombiner::print_statistics, this));

#ifdef USE_CUDA
        if (use_cuda_)
        {
            pointcloud_cuda::initCUDA();
        }
#endif

        RCLCPP_INFO(this->get_logger(), "Point cloud combiner initialized");
        RCLCPP_INFO(this->get_logger(), "  Target frame: %s", target_frame_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Use CUDA: %s", use_cuda_ ? "true" : "false");
    }

    ~PointCloudCombiner()
    {
#ifdef USE_CUDA
        if (use_cuda_)
        {
            pointcloud_cuda::cleanupCUDA();
        }
#endif
    }

private:
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2> SyncPolicy;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> pc1_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> pc2_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string target_frame_;
    bool use_cuda_;

    // Timing statistics
    std::mutex stats_mutex_;
    std::vector<double> transform_times_;
    std::vector<double> combine_times_;
    std::vector<std::chrono::high_resolution_clock::time_point> publish_timestamps_;
    rclcpp::TimerBase::SharedPtr stats_timer_;

    void sync_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pc1_msg,
                       const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pc2_msg)
    {
        // Get sensor frames from the message headers
        std::string sensor_frame_1 = pc1_msg->header.frame_id;
        std::string sensor_frame_2 = pc2_msg->header.frame_id;

        // Transform the point clouds in parallel using separate threads
        sensor_msgs::msg::PointCloud2 transformed_pc1, transformed_pc2;
        bool success1 = false, success2 = false;

        // Start timing for transform operations
        auto transform_start = std::chrono::high_resolution_clock::now();

        // Launch threads for parallel transformation
        std::thread thread1([this, pc1_msg, sensor_frame_1, &transformed_pc1, &success1]() {
            success1 = this->transform_pointcloud(pc1_msg, sensor_frame_1, transformed_pc1);
        });

        std::thread thread2([this, pc2_msg, sensor_frame_2, &transformed_pc2, &success2]() {
            success2 = this->transform_pointcloud(pc2_msg, sensor_frame_2, transformed_pc2);
        });

        // Wait for both threads to complete
        thread1.join();
        thread2.join();

        auto transform_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> transform_duration = transform_end - transform_start;

        if (!success1 || !success2)
        {
            RCLCPP_WARN(this->get_logger(), "One of the point clouds could not be transformed.");
            return;
        }

        // Record transform time
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            transform_times_.push_back(transform_duration.count());
        }

        // Start timing for combine operation
        auto combine_start = std::chrono::high_resolution_clock::now();

        // Combine the transformed point clouds
        sensor_msgs::msg::PointCloud2 combined_pc = combine_pointclouds(transformed_pc1, transformed_pc2);

        auto combine_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> combine_duration = combine_end - combine_start;

        // Record combine time
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            combine_times_.push_back(combine_duration.count());
        }

        // Publish the combined point cloud
        pc_pub_->publish(combined_pc);

        // Record publish timestamp for frequency calculation
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            publish_timestamps_.push_back(std::chrono::high_resolution_clock::now());
        }
    }

    bool transform_pointcloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &input_cloud,
                              const std::string &source_frame,
                              sensor_msgs::msg::PointCloud2 &output_cloud)
    {
        try
        {
            // Lookup the transform from the sensor frame to the target frame
            geometry_msgs::msg::TransformStamped transform_stamped = tf_buffer_.lookupTransform(target_frame_, source_frame, tf2::TimePointZero);

#ifdef USE_CUDA
            if (use_cuda_)
            {
                // Convert TransformStamped to Eigen Matrix4f for CUDA
                Eigen::Isometry3d transform_eigen = tf2::transformToEigen(transform_stamped.transform);
                Eigen::Matrix4f transform_matrix = transform_eigen.matrix().cast<float>();

                // Use CUDA transformation
                bool success = pointcloud_cuda::transformPointCloudCUDA(*input_cloud, transform_matrix, output_cloud);
                if (success)
                {
                    output_cloud.header.frame_id = target_frame_;
                    output_cloud.header.stamp = input_cloud->header.stamp;
                }
                return success;
            }
            else
#endif
            {
                // Use CPU transformation (pcl_ros)
                pcl_ros::transformPointCloud(target_frame_, transform_stamped, *input_cloud, output_cloud);
                return true;
            }
        }
        catch (tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Could not transform point cloud from %s to %s: %s", source_frame.c_str(), target_frame_.c_str(), ex.what());
            return false;
        }
    }

    sensor_msgs::msg::PointCloud2 combine_pointclouds(const sensor_msgs::msg::PointCloud2 &pc1, const sensor_msgs::msg::PointCloud2 &pc2)
    {
        // Assuming both point clouds have the same fields and we just concatenate points
        sensor_msgs::msg::PointCloud2 combined_pc = pc1;
        combined_pc.width += pc2.width;
        combined_pc.row_step += pc2.row_step;

        // Concatenate point cloud data
        combined_pc.data.insert(combined_pc.data.end(), pc2.data.begin(), pc2.data.end());

        return combined_pc;
    }

    void print_statistics()
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);

        if (transform_times_.empty() && combine_times_.empty())
        {
            return;
        }

        double avg_transform = 0.0;
        double avg_combine = 0.0;
        double publish_hz = 0.0;
        size_t num_samples = transform_times_.size();

        if (!transform_times_.empty())
        {
            avg_transform = std::accumulate(transform_times_.begin(), transform_times_.end(), 0.0) / transform_times_.size();
        }

        if (!combine_times_.empty())
        {
            avg_combine = std::accumulate(combine_times_.begin(), combine_times_.end(), 0.0) / combine_times_.size();
        }

        // Calculate publishing frequency in Hz
        if (publish_timestamps_.size() >= 2)
        {
            auto first_timestamp = publish_timestamps_.front();
            auto last_timestamp = publish_timestamps_.back();
            std::chrono::duration<double> time_span = last_timestamp - first_timestamp;

            if (time_span.count() > 0.0)
            {
                publish_hz = (publish_timestamps_.size() - 1) / time_span.count();
            }
        }

        RCLCPP_INFO(this->get_logger(),
                    "Performance Stats (last %zu samples) (using CUDA %s):", num_samples, use_cuda_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(),
                    "  Avg Transform Time: %.2f ms", avg_transform);
        RCLCPP_INFO(this->get_logger(),
                    "  Avg Combine Time: %.2f ms", avg_combine);
        RCLCPP_INFO(this->get_logger(),
                    "  Avg Total Time: %.2f ms", avg_transform + avg_combine);
        RCLCPP_INFO(this->get_logger(),
                    "  Publishing Frequency: %.2f Hz", publish_hz);

        // Clear the vectors for next interval
        transform_times_.clear();
        combine_times_.clear();
        publish_timestamps_.clear();
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudCombiner>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
