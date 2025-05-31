import os
from launch import LaunchDescription, LaunchContext
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    package_name='start_rl_environment'

    # Declare a launch argument 'map_number' - This currently works
    map_number_arg = DeclareLaunchArgument(
        'map_number',
        default_value='1',
        description='Number of the map to be launched'
    )
    
    robot_number_arg = DeclareLaunchArgument(
        'robot_number',
        default_value='3',
        description='Number of robots to be launched'
    )
    
    # Add headless mode parameter
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo in headless mode (no GUI)'
    )
    
    # Add fast training mode parameter
    fast_training_arg = DeclareLaunchArgument(
        'fast_training',
        default_value='false',
        description='Run simulation with optimized physics for faster training'
    )

    # This Launch file handles the simulation of the environment (Gazebo)
    start_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory(package_name), 'launch', 'start_world.launch.py')),
        launch_arguments={
            'map_number': LaunchConfiguration('map_number'),
            'robot_number': LaunchConfiguration('robot_number'),
            'headless': LaunchConfiguration('headless'),
            'fast_training': LaunchConfiguration('fast_training')
        }.items()
    )

    # This launch file handles the simulation of the robots
    start_robots = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory(package_name), 'launch', 'start_robots.launch.py')),
        launch_arguments={
            'map_number': LaunchConfiguration('map_number')
        }.items()
    )

    # Launch them all!
    return LaunchDescription([
        map_number_arg,
        robot_number_arg,
        headless_arg,
        fast_training_arg,
        start_world,
        start_robots
    ])