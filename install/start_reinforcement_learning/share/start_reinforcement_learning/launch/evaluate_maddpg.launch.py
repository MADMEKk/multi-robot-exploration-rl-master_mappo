import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, EnvironmentVariable

def generate_launch_description():
    # Get the launch directory
    pkg_dir = get_package_share_directory('start_reinforcement_learning')
    
    # Get environment variables
    map_number = EnvironmentVariable('map_number', default_value='1')
    robot_number = EnvironmentVariable('robot_number', default_value='2')
    model_episode = EnvironmentVariable('model_episode', default_value='0')
    headless = EnvironmentVariable('headless', default_value='true')
    
    # Create the launch configuration variables
    map_number_arg = LaunchConfiguration('map_number')
    robot_number_arg = LaunchConfiguration('robot_number')
    model_episode_arg = LaunchConfiguration('model_episode')
    headless_arg = LaunchConfiguration('headless')
    
    # Declare the launch arguments
    declare_map_number = DeclareLaunchArgument(
        'map_number',
        default_value=map_number,
        description='Map number to use for evaluation'
    )
    
    declare_robot_number = DeclareLaunchArgument(
        'robot_number',
        default_value=robot_number,
        description='Number of robots to use for evaluation'
    )
    
    declare_model_episode = DeclareLaunchArgument(
        'model_episode',
        default_value=model_episode,
        description='Episode number of the model to load (0 for best model, otherwise periodic save)'
    )
    
    declare_headless = DeclareLaunchArgument(
        'headless',
        default_value=headless,
        description='Whether to run in headless mode'
    )
    
    # Include the environment launch file
    environment_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('start_rl_environment'),
            'launch',
            'main.launch.py'
        )),
        launch_arguments={
            'map_number': map_number_arg,
            'robot_number': robot_number_arg,
            'headless': headless_arg,
            'fast_training': 'false'
        }.items()
    )
    
    # Launch the MADDPG evaluation node
    maddpg_node = Node(
        package='start_reinforcement_learning',
        executable='run_maddpg_evaluate',
        name='maddpg_evaluate_node',
        namespace='maddpg_ns',
        parameters=[{
            'map_number': map_number_arg,
            'robot_number': robot_number_arg,
            'model_episode': model_episode_arg
        }],
        output='screen'
    )
    
    return LaunchDescription([
        declare_map_number,
        declare_robot_number,
        declare_model_episode,
        declare_headless,
        environment_launch,
        maddpg_node
    ])
