import os
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, ExecuteProcess, GroupAction
from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PythonExpression, LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition, UnlessCondition

def generate_launch_description():
    
    map_name = PythonExpression(["'map", LaunchConfiguration('map_number'), ".world'"])
    
    # Declare launch arguments
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo in headless mode (no GUI)'
    )
    
    fast_training_arg = DeclareLaunchArgument(
        'fast_training',
        default_value='false',
        description='Run simulation with optimized physics for faster training'
    )
    
    # World file argument
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=PathJoinSubstitution([
            get_package_share_directory('start_rl_environment'), 'worlds', map_name]),
        description='SDF world file'
    )
    
    # Gazebo server parameters
    gazebo_server_params = {
        'verbose': 'false',
    }
    
    # Add fast training parameters if enabled
    gazebo_fast_training_params = {
        'physics': 'ode',
        'max_step_size': '0.002',  # Balanced for speed and accuracy
        'real_time_factor': '5.0',  # 5x speed - good balance for training quality
        'real_time_update_rate': '1500',  # Balanced update rate
    }
    
    # Launch Gazebo server (headless mode)
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gzserver.launch.py')]),
        launch_arguments=gazebo_server_params.items(),
        condition=IfCondition(LaunchConfiguration('headless'))
    )
    
    # Launch Gazebo with GUI (non-headless mode)
    gazebo_with_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
        condition=UnlessCondition(LaunchConfiguration('headless'))
    )
    
    # Launch with fast training parameters
    gazebo_fast_training = ExecuteProcess(
        cmd=['ros2', 'param', 'set', '/gazebo', 'physics.max_step_size', '0.002'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('fast_training'))
    )
    
    gazebo_fast_training_rtf = ExecuteProcess(
        cmd=['ros2', 'param', 'set', '/gazebo', 'physics.real_time_factor', '5.0'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('fast_training'))
    )
    
    gazebo_fast_training_update_rate = ExecuteProcess(
        cmd=['ros2', 'param', 'set', '/gazebo', 'physics.real_time_update_rate', '1500'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('fast_training'))
    )
    
    # Add ODE solver parameters for better stability
    gazebo_solver_iters = ExecuteProcess(
        cmd=['ros2', 'param', 'set', '/gazebo', 'physics.ode.solver.iters', '50'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('fast_training'))
    )
    
    gazebo_solver_type = ExecuteProcess(
        cmd=['ros2', 'param', 'set', '/gazebo', 'physics.ode.solver.type', 'quick'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('fast_training'))
    )
    
    return LaunchDescription([
        headless_arg,
        fast_training_arg,
        world_arg,
        gazebo_server,
        gazebo_with_gui,
        gazebo_fast_training,
        gazebo_fast_training_rtf,
        gazebo_fast_training_update_rate,
        gazebo_solver_iters,
        gazebo_solver_type
    ])