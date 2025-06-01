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
        description='Number of robots the model will train'
    )
    algorithm_arg = DeclareLaunchArgument(
        'algorithm',
        default_value='mappo',
        description='RL algorithm to use: mappo or maddpg'
    )

    # Create the MAPPO node (only runs when algorithm=mappo)
    mappo_node = Node(
        package='start_reinforcement_learning',
        executable='run_mappo',
        namespace='mappo_ns',
        name='mappo_node',
        parameters=[
            {'map_number': LaunchConfiguration('map_number')},
            {'robot_number': LaunchConfiguration('robot_number')}
        ],
        condition=IfCondition(PythonExpression(["'mappo' == '", LaunchConfiguration('algorithm'), "'"]))
    )
    
    # Create the MADDPG node (only runs when algorithm=maddpg)
    maddpg_node = Node(
        package='start_reinforcement_learning',
        executable='run_maddpg',
        namespace='maddpg_ns',
        name='maddpg_node',
        parameters=[
            {'map_number': LaunchConfiguration('map_number')},
            {'robot_number': LaunchConfiguration('robot_number')}
        ],
        condition=IfCondition(PythonExpression(["'maddpg' == '", LaunchConfiguration('algorithm'), "'"]))
    )
    
    return LaunchDescription([
        map_number_arg,
        robot_number_arg,
        algorithm_arg,
        mappo_node,
        maddpg_node
    ])

