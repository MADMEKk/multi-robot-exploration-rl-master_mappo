"""
Launch script for MAPPO training with configurable parameters.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    map_number = LaunchConfiguration('map_number')
    robot_number = LaunchConfiguration('robot_number')
    config_file = LaunchConfiguration('config_file')
    use_tensorboard = LaunchConfiguration('use_tensorboard')
    
    # Declare launch arguments
    map_number_arg = DeclareLaunchArgument(
        'map_number',
        default_value='1',
        description='Map number to use for training'
    )
    
    robot_number_arg = DeclareLaunchArgument(
        'robot_number',
        default_value='3',
        description='Number of robots to use for training'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='',
        description='Path to configuration YAML file (optional)'
    )
    
    use_tensorboard_arg = DeclareLaunchArgument(
        'use_tensorboard',
        default_value='true',
        description='Whether to use TensorBoard for logging'
    )
    
    # Get package directory
    package_dir = get_package_share_directory('start_reinforcement_learning')
    
    # Create default config path if one is not specified
    default_config_path = os.path.join(
        package_dir, 'config', 'default_config.yaml'
    )
    
    # Determine config path to use
    config_path_expr = PythonExpression([
        '"', config_file, '" if "', config_file, '" else "', default_config_path, '"'
    ])
    
    # Determine if --tensorboard flag should be added
    tensorboard_flag_expr = PythonExpression([
        '"--tensorboard" if ', use_tensorboard, ' else ""'
    ])
    
    # MAPPO node
    mappo_node = Node(
        package='start_reinforcement_learning',
        executable='mappo_main',
        name='mappo_node',
        output='screen',
        arguments=[
            '--map_number', map_number,
            '--robot_number', robot_number,
            '--config', config_path_expr,
            tensorboard_flag_expr
        ]
    )
    
    # TensorBoard process (conditionally launched)
    tensorboard_process = ExecuteProcess(
        condition=PythonExpression([use_tensorboard]),
        cmd=[
            'tensorboard',
            '--logdir', os.path.join(package_dir, 'tensorboard_logs'),
            '--port', '6006',
            '--bind_all'
        ],
        shell=True,
        output='screen'
    )
    
    return LaunchDescription([
        map_number_arg,
        robot_number_arg,
        config_file_arg,
        use_tensorboard_arg,
        mappo_node,
        tensorboard_process
    ]) 