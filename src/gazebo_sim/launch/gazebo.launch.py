import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, Command
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    pkg_name = 'gazebo_sim'
    pkg_share = get_package_share_directory(pkg_name)

    urdf_path = os.path.join(pkg_share, 'urdf', 'robot.xacro')

    # FIX: Use Command properly without extra spaces
    robot_desc = Command(['xacro ', urdf_path])

    # 1. LAUNCH IGNITION GAZEBO
    gz_sim = ExecuteProcess(
        cmd=[
            'ign', 'gazebo', '-v4',
            os.path.join(pkg_share, 'worlds', 'world1.sdf')
        ],
        output='screen'
    )

    # 2. SPAWN ROBOT IN IGNITION
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-string', robot_desc,
            '-name', 'robot'
        ],
        output='screen'
    )

    # 3. PUBLISH TF + ROBOT STATE
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': use_sim_time
        }]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        gz_sim,
        robot_state_publisher,
        spawn_entity
    ])