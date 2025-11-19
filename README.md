# jeeves_pcl_combiner

A ROS 2 package for synchronizing and combining multiple point clouds into a common reference frame.

## Overview

This package provides a node that subscribes to two point cloud topics, transforms them to a common target frame using TF2, and publishes a combined point cloud. The node uses approximate time synchronization to handle point clouds that arrive at slightly different times.

## Features

- Time-synchronized point cloud combination using message_filters
- TF2-based transformation to a common reference frame
- Configurable target frame via ROS 2 parameters
- Efficient point cloud concatenation

## Parameters

- `target_frame` (string, default: "base_link"): The target frame to transform all point clouds to before combining

## Topics

### Subscribed Topics

- `/pointcloud1` (sensor_msgs/PointCloud2): First input point cloud
- `/pointcloud2` (sensor_msgs/PointCloud2): Second input point cloud

### Published Topics

- `/combined_pointcloud` (sensor_msgs/PointCloud2): Combined point cloud in the target frame

## Usage

### Building

```bash
cd ~/ros2_ws
colcon build --packages-select jeeves_pcl_combiner
```

### Running

```bash
ros2 run jeeves_pcl_combiner jeeves_pcl_combiner
```

With custom parameters:

```bash
ros2 run jeeves_pcl_combiner jeeves_pcl_combiner --ros-args -p target_frame:=map
```

## Dependencies

- rclcpp
- sensor_msgs
- tf2_ros
- tf2_geometry_msgs
- pcl_ros
- pcl_conversions
- message_filters

## License

MIT License - Copyright (c) 2025 Sandeep Dutta
