cmake_minimum_required(VERSION 3.8)
#cmake_policy(SET CMP0148 NEW)
#cmake_policy(SET CMP0167 NEW)
project(ros_interface_mpc)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(FetchContent)

# find dependencies

find_package (Python COMPONENTS Interpreter Development)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(tf2 REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED) 
find_package(builtin_interfaces REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(simple-mpc REQUIRED)
find_package(example-robot-data REQUIRED)
pkg_search_module(EIGEN REQUIRED eigen3)
include_directories(${EIGEN_INCLUDE_DIRS})
find_package(rosidl_default_generators REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME}_utils)

set(msg_files
  "msg/State.msg"
  "msg/Torque.msg"
  "msg/RobotState.msg"
  "msg/InputMessage.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs tf2
)

#add_executable(talker src/publisher_member_function.cpp)
#target_link_libraries(talker example-robot-data::example-robot-data simple-mpc::simple-mpc)
#ament_target_dependencies(talker rclcpp std_msgs simple-mpc)
#add_executable(listener src/subscriber_member_function.cpp)
#ament_target_dependencies(listener rclcpp std_msgs)

#rosidl_target_interfaces(talker
#  ${PROJECT_NAME} "rosidl_typesupport_cpp")

# Install Python executables
install(PROGRAMS
  scripts/publisher_talos.py
  scripts/publisher_go2.py
  scripts/subscriber_talos.py
  scripts/subscriber_bullet.py
  scripts/bullet_robot.py
  scripts/subscriber_go2.py
  scripts/subscriber_test.py
  scripts/mpc.py
  scripts/robot_utils.py
  scripts/simulation_args.py
  scripts/simulation_utils.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch urdf meshes rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

#install(TARGETS
  #talker
  #listener
  #DESTINATION lib/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)

ament_package()
