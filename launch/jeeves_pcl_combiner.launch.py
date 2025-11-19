"""
Launch file for jeeves_pcl_combiner node

This launch file starts the point cloud combiner node with configurable parameters.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for point cloud combiner"""

    # Declare launch arguments
    target_frame_arg = DeclareLaunchArgument(
        'target_frame',
        default_value='base_link',
        description='Target frame to transform point clouds to'
    )

    pointcloud1_topic_arg = DeclareLaunchArgument(
        'pointcloud1_topic',
        default_value='/pointcloud1',
        description='First point cloud topic'
    )

    pointcloud2_topic_arg = DeclareLaunchArgument(
        'pointcloud2_topic',
        default_value='/pointcloud2',
        description='Second point cloud topic'
    )

    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/combined_pointcloud',
        description='Combined point cloud output topic'
    )

    # Create node
    combiner_node = Node(
        package='jeeves_pcl_combiner',
        executable='jeeves_pcl_combiner',
        name='pointcloud_combiner',
        parameters=[{
            'target_frame': LaunchConfiguration('target_frame')
        }],
        remappings=[
            ('/pointcloud1', LaunchConfiguration('pointcloud1_topic')),
            ('/pointcloud2', LaunchConfiguration('pointcloud2_topic')),
            ('/combined_pointcloud', LaunchConfiguration('output_topic'))
        ],
        output='screen'
    )

    return LaunchDescription([
        target_frame_arg,
        pointcloud1_topic_arg,
        pointcloud2_topic_arg,
        output_topic_arg,
        combiner_node
    ])
