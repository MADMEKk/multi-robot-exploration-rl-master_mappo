import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Define launch parameters
    map_number_arg = DeclareLaunchArgument(
        'map_number',
        default_value='1',
        description='The map number for environment configuration'
    )
    robot_number_arg = DeclareLaunchArgument(
        'robot_number',
        default_value='2',
        description='Number of robots to use in the comparison'
    )
    n_episodes_arg = DeclareLaunchArgument(
        'n_episodes',
        default_value='100',
        description='Number of episodes to run for each algorithm'
    )
    
    # Create the node for algorithm comparison
    algorithm_comparison_node = Node(
        package='start_reinforcement_learning',
        executable='run_algorithm_comparison',
        namespace='algorithm_comparison_ns',
        name='algorithm_comparison_node',
        parameters=[
            {'map_number': LaunchConfiguration('map_number')},
            {'robot_number': LaunchConfiguration('robot_number')},
            {'n_episodes': LaunchConfiguration('n_episodes')}
        ],
        output='screen'
    )
    
    return LaunchDescription([
        map_number_arg,
        robot_number_arg,
        n_episodes_arg,
        algorithm_comparison_node
    ])
