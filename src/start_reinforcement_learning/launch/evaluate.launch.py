import os
from launch import LaunchDescription, actions
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution, PythonExpression
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition

def generate_launch_description():
    # Define launch parameters
    map_number_arg = DeclareLaunchArgument(
        'map_number',
        default_value='1',
        description='The map number show env logic can configure goal locations and respawn locations'
    )
    robot_number_arg = DeclareLaunchArgument(
        'robot_number',
        default_value='3',
        description='Number of robots the model will evaluate'
    )
    algorithm_arg = DeclareLaunchArgument(
        'algorithm',
        default_value='mappo',
        description='RL algorithm to evaluate: mappo or maddpg'
    )

    # Create the MAPPO evaluation node (only runs when algorithm=mappo)
    mappo_evaluate_node = Node(
        package='start_reinforcement_learning',
        executable='run_mappo_evaluate',
        namespace='mappo_evaluate_ns',
        name='mappo_evaluate_node',
        parameters=[
            {'map_number': LaunchConfiguration('map_number')},
            {'robot_number': LaunchConfiguration('robot_number')}
        ],
        condition=IfCondition(PythonExpression(["'mappo' == '", LaunchConfiguration('algorithm'), "'"]))
    )
    
    # Create the MADDPG evaluation node (only runs when algorithm=maddpg)
    # Note: You'll need to create a maddpg_evaluate executable if it doesn't exist
    maddpg_evaluate_node = Node(
        package='start_reinforcement_learning',
        executable='run_maddpg',  # Using regular maddpg with evaluation mode
        namespace='maddpg_evaluate_ns',
        name='maddpg_evaluate_node',
        parameters=[
            {'map_number': LaunchConfiguration('map_number')},
            {'robot_number': LaunchConfiguration('robot_number')},
            {'evaluate': True}  # Set evaluation mode
        ],
        condition=IfCondition(PythonExpression(["'maddpg' == '", LaunchConfiguration('algorithm'), "'"]))
    )
    
    return LaunchDescription([
        map_number_arg,
        robot_number_arg,
        algorithm_arg,
        mappo_evaluate_node,
        maddpg_evaluate_node
    ])
