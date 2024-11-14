import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import launch_ros.descriptions

def generate_launch_description():

  # Set the path to this package.
  pkg_share = FindPackageShare(package='ros_interface_mpc').find('ros_interface_mpc')

  # Set the path to the URDF file
  default_urdf_model_path = os.path.join(pkg_share, 'urdf/go2_description.urdf')

  # Launch configuration variables specific to simulation
  use_robot_state_pub = LaunchConfiguration('use_robot_state_pub')
  use_sim_time = LaunchConfiguration('use_sim_time')
  mpc_type = LaunchConfiguration('mpc_type')
  motion_type = LaunchConfiguration('motion_type')

  # Declare the launch arguments  
  declare_robot_name_cmd = DeclareLaunchArgument(
    name='robot_name',
    default_value='go2_description',
    description='The name for the robot')

  declare_urdf_model_path_cmd = DeclareLaunchArgument(
    name='urdf_model', 
    default_value=default_urdf_model_path, 
    description='Absolute path to robot urdf file')

  
  declare_use_robot_state_pub_cmd = DeclareLaunchArgument(
    name='use_robot_state_pub',
    default_value='True',
    description='Whether to start the robot state publisher')
    
  declare_use_sim_time_cmd = DeclareLaunchArgument(
    name='use_sim_time',
    default_value='True',
    description='Use simulation (Gazebo) clock if true')
  
  declare_mpc_type = DeclareLaunchArgument(
    name='mpc_type',
    default_value='fulldynamics',
    description='Dynamic model used by MPC')

  declare_motion_type = DeclareLaunchArgument(
    name='motion_type',
    default_value='walk',
    description='Motion type to execute')
  
  # Specify the actions

  # Subscribe to the joint states of the robot, and publish the 3D pose of each link.
  start_robot_state_publisher_cmd = Node(
    condition=IfCondition(use_robot_state_pub),
    package='robot_state_publisher',
    executable='robot_state_publisher',
    parameters=[{'use_sim_time': use_sim_time, 
    'robot_description': launch_ros.descriptions.ParameterValue( 
    Command(['xacro ',default_urdf_model_path]), value_type=str)}],
    arguments=[default_urdf_model_path])

  start_state_publisher = Node(
    package='ros_interface_mpc',
    executable='subscriber_bullet.py',
    name='subscriber',
    output='screen',
    parameters=[{"mpc_type": mpc_type}])
  
  start_control_node = Node(
    package='ros_interface_mpc',
    executable='publisher_go2.py',
    name='publisher',
    output='screen',
    parameters=[{"mpc_type": mpc_type, "motion_type" : motion_type}])
  

  start_input_publisher = Node(
    package='ros_interface_mpc',
    executable='input.py',
    name='input',
    output='screen')
  
  # Create the launch description and populate
  ld = LaunchDescription()

  # Declare the launch options
  ld.add_action(declare_robot_name_cmd)
  ld.add_action(declare_urdf_model_path_cmd)
  ld.add_action(declare_use_robot_state_pub_cmd)  
  ld.add_action(declare_use_sim_time_cmd)
  ld.add_action(declare_mpc_type)
  ld.add_action(declare_motion_type)

  # Add any actions
  ld.add_action(start_control_node)
  ld.add_action(start_state_publisher)
  ld.add_action(start_robot_state_publisher_cmd)
  ld.add_action(start_input_publisher)

  return ld
