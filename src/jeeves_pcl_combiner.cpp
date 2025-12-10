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
#include <sensor_msgs/msg/laser_scan.hpp>
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
#include <cmath>

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
        this->declare_parameter<bool>("map_frame_sync", false);
        this->declare_parameter<std::string>("map_frame", "map");
        this->declare_parameter<double>("timestamp_tolerance", 0.1);
        this->declare_parameter<bool>("publish_scan", false);
        this->declare_parameter<double>("scan_height_min", -0.1);
        this->declare_parameter<double>("scan_height_max", 0.1);
        this->declare_parameter<double>("scan_range_min", 0.1);
        this->declare_parameter<double>("scan_range_max", 30.0);
        this->declare_parameter<double>("scan_angle_min", -M_PI);
        this->declare_parameter<double>("scan_angle_max", M_PI);
        this->declare_parameter<double>("scan_angle_increment", 0.0087);  // ~0.5 degrees

        target_frame_ = this->get_parameter("target_frame").as_string();
        use_cuda_ = this->get_parameter("use_cuda").as_bool();
        map_frame_sync_ = this->get_parameter("map_frame_sync").as_bool();
        map_frame_ = this->get_parameter("map_frame").as_string();
        timestamp_tolerance_ = this->get_parameter("timestamp_tolerance").as_double();
        publish_scan_ = this->get_parameter("publish_scan").as_bool();
        scan_height_min_ = this->get_parameter("scan_height_min").as_double();
        scan_height_max_ = this->get_parameter("scan_height_max").as_double();
        scan_range_min_ = this->get_parameter("scan_range_min").as_double();
        scan_range_max_ = this->get_parameter("scan_range_max").as_double();
        scan_angle_min_ = this->get_parameter("scan_angle_min").as_double();
        scan_angle_max_ = this->get_parameter("scan_angle_max").as_double();
        scan_angle_increment_ = this->get_parameter("scan_angle_increment").as_double();

        // Create subscribers for the two point cloud topics
        pc1_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(this, "/pointcloud1");
        pc2_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(this, "/pointcloud2");

        // Synchronize point clouds using approximate time sync policy
        sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), *pc1_sub_, *pc2_sub_));
        sync_->registerCallback(std::bind(&PointCloudCombiner::sync_callback, this, _1, _2));

        // Publisher for the combined point cloud
        pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/combined_pointcloud", 10);

        // Publisher for laser scan (if enabled)
        if (publish_scan_)
        {
            scan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/scan", 10);
        }

        // Create timer to print statistics every 10 seconds
        stats_timer_ = this->create_wall_timer(
            std::chrono::seconds(60),
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
        RCLCPP_INFO(this->get_logger(), "  Map frame sync: %s", map_frame_sync_ ? "enabled" : "disabled");
        if (map_frame_sync_) {
            RCLCPP_INFO(this->get_logger(), "  Map frame: %s", map_frame_.c_str());
            RCLCPP_INFO(this->get_logger(), "  Timestamp tolerance: %.3f seconds", timestamp_tolerance_);
        }
        RCLCPP_INFO(this->get_logger(), "  Publish LaserScan: %s", publish_scan_ ? "enabled" : "disabled");
        if (publish_scan_) {
            RCLCPP_INFO(this->get_logger(), "    Scan height range: [%.2f, %.2f]", scan_height_min_, scan_height_max_);
            RCLCPP_INFO(this->get_logger(), "    Scan range: [%.2f, %.2f]", scan_range_min_, scan_range_max_);
            RCLCPP_INFO(this->get_logger(), "    Scan angle range: [%.2f, %.2f] rad", scan_angle_min_, scan_angle_max_);
            RCLCPP_INFO(this->get_logger(), "    Scan angle increment: %.4f rad", scan_angle_increment_);
        }
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
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string target_frame_;
    bool use_cuda_;
    bool map_frame_sync_;
    std::string map_frame_;
    double timestamp_tolerance_;
    bool publish_scan_;
    double scan_height_min_;
    double scan_height_max_;
    double scan_range_min_;
    double scan_range_max_;
    double scan_angle_min_;
    double scan_angle_max_;
    double scan_angle_increment_;

    // Map transform availability tracking
    bool map_transform_received_{false};

    // Timing statistics
    std::mutex stats_mutex_;
    std::vector<double> transform_times_;
    std::vector<double> combine_times_;
    std::vector<std::chrono::high_resolution_clock::time_point> publish_timestamps_;
    rclcpp::TimerBase::SharedPtr stats_timer_;

    bool can_publish_pointcloud(const builtin_interfaces::msg::Time& stamp)
    {
        // If map frame sync is disabled, always allow publishing
        if (!map_frame_sync_) {
            return true;
        }

        // If we haven't received the first map transform yet, silently skip processing
        if (!map_transform_received_)
        {
            // Try to lookup the transform to set the flag
            try
            {
                tf2::TimePoint time_point(std::chrono::seconds(stamp.sec) +
                                         std::chrono::nanoseconds(stamp.nanosec));

                // Try to actually lookup the transform - if it succeeds, we have map frame
                geometry_msgs::msg::TransformStamped transform = tf_buffer_.lookupTransform(
                    map_frame_, target_frame_, time_point,
                    tf2::durationFromSec(timestamp_tolerance_));

                // If we get here, transform is available
                map_transform_received_ = true;
                RCLCPP_INFO(this->get_logger(), "First %s->%s transform received, starting point cloud processing",
                           map_frame_.c_str(), target_frame_.c_str());
                return true;
            }
            catch (const tf2::TransformException &ex)
            {
                // Silently return false on exception - this is expected before SLAM starts
                return false;
            }
        }

        // Once we've seen the map transform at least once, check for availability with normal logging
        try
        {
            // Lookup the transform to verify it's available
            tf2::TimePoint time_point(std::chrono::seconds(stamp.sec) +
                                       std::chrono::nanoseconds(stamp.nanosec));

            geometry_msgs::msg::TransformStamped transform = tf_buffer_.lookupTransform(
                map_frame_, target_frame_, time_point,
                tf2::durationFromSec(timestamp_tolerance_));

            // Transform successfully looked up
            return true;
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_DEBUG(this->get_logger(),
                        "Dropping point cloud - map->base_link transform not available: %s", ex.what());
            return false;
        }
    }

    void sync_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pc1_msg,
                       const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pc2_msg)
    {
        // Check if either point cloud is empty
        bool pc1_empty = (pc1_msg->width == 0 || pc1_msg->height == 0 || pc1_msg->data.empty());
        bool pc2_empty = (pc2_msg->width == 0 || pc2_msg->height == 0 || pc2_msg->data.empty());

        // If one is empty, send the other one directly
        if (pc1_empty && !pc2_empty)
        {
            sensor_msgs::msg::PointCloud2 transformed_pc2;
            if (transform_pointcloud(pc2_msg, pc2_msg->header.frame_id, transformed_pc2))
            {
                // Check if map->base_link transform is available before publishing (if enabled)
                if (can_publish_pointcloud(transformed_pc2.header.stamp))
                {
                    pc_pub_->publish(transformed_pc2);

                    // Convert and publish laserscan if enabled
                    if (publish_scan_)
                    {
                        sensor_msgs::msg::LaserScan scan = pointcloud_to_laserscan(transformed_pc2);
                        scan_pub_->publish(scan);
                    }

                    // Record publish timestamp for frequency calculation
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    publish_timestamps_.push_back(std::chrono::high_resolution_clock::now());
                }
            }
            return;
        }
        else if (pc2_empty && !pc1_empty)
        {
            sensor_msgs::msg::PointCloud2 transformed_pc1;
            if (transform_pointcloud(pc1_msg, pc1_msg->header.frame_id, transformed_pc1))
            {
                // Check if map->base_link transform is available before publishing (if enabled)
                if (can_publish_pointcloud(transformed_pc1.header.stamp))
                {
                    pc_pub_->publish(transformed_pc1);

                    // Convert and publish laserscan if enabled
                    if (publish_scan_)
                    {
                        sensor_msgs::msg::LaserScan scan = pointcloud_to_laserscan(transformed_pc1);
                        scan_pub_->publish(scan);
                    }

                    // Record publish timestamp for frequency calculation
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    publish_timestamps_.push_back(std::chrono::high_resolution_clock::now());
                }
            }
            return;
        }
        else if (pc1_empty && pc2_empty)
        {
            // Both empty, nothing to publish
            return;
        }

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

        // Check if map->base_link transform is available before publishing (if enabled)
        if (can_publish_pointcloud(combined_pc.header.stamp))
        {
            // Publish the combined point cloud
            pc_pub_->publish(combined_pc);

            // Convert and publish laserscan if enabled
            if (publish_scan_)
            {
                sensor_msgs::msg::LaserScan scan = pointcloud_to_laserscan(combined_pc);
                scan_pub_->publish(scan);
            }

            // Record publish timestamp for frequency calculation
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

    sensor_msgs::msg::LaserScan pointcloud_to_laserscan(const sensor_msgs::msg::PointCloud2 &cloud)
    {
        sensor_msgs::msg::LaserScan scan;
        scan.header = cloud.header;
        scan.angle_min = scan_angle_min_;
        scan.angle_max = scan_angle_max_;
        scan.angle_increment = scan_angle_increment_;
        scan.time_increment = 0.0;
        scan.scan_time = 0.0;
        scan.range_min = scan_range_min_;
        scan.range_max = scan_range_max_;

        // Calculate number of rays
        int num_rays = static_cast<int>((scan_angle_max_ - scan_angle_min_) / scan_angle_increment_) + 1;
        scan.ranges.assign(num_rays, std::numeric_limits<float>::infinity());
        scan.intensities.assign(num_rays, 0.0);

        // Iterate through point cloud
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud, "z");

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
        {
            float x = *iter_x;
            float y = *iter_y;
            float z = *iter_z;

            // Filter by height
            if (z < scan_height_min_ || z > scan_height_max_)
            {
                continue;
            }

            // Calculate range and angle
            float range = std::sqrt(x * x + y * y);

            // Filter by range
            if (range < scan_range_min_ || range > scan_range_max_)
            {
                continue;
            }

            float angle = std::atan2(y, x);

            // Check if angle is within scan range
            if (angle < scan_angle_min_ || angle > scan_angle_max_)
            {
                continue;
            }

            // Calculate index in scan array
            int index = static_cast<int>((angle - scan_angle_min_) / scan_angle_increment_);

            if (index >= 0 && index < num_rays)
            {
                // Keep the closest point for each ray
                if (range < scan.ranges[index])
                {
                    scan.ranges[index] = range;
                }
            }
        }

        return scan;
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
