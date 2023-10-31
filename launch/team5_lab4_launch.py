from launch import LaunchDescription
from launch_ros.actions import Node

# ros2 launch turtlebot3_bringup camera_robot.launch.py
# ros2 launch team5_navigate_to_goal team5_lab4_launch.py
def generate_launch_description():
    return LaunchDescription([
        # Node(
        #     package='team5_chase_object',
        #     executable='detect_object'
        # ),
        # Node(
        #     package='team5_chase_object',
        #     executable='get_object_range'
        # ),
        # Node(
        #     package='team5_chase_object',
        #     executable='chase_object'
        # )
    ])