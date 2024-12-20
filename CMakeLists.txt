cmake_minimum_required(VERSION 3.8)
project(ros_interface_mpc)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pinocchio REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(example-robot-data REQUIRED)
find_package(ndcurves REQUIRED)
find_package(simple-mpc REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Build msgs
set(msg_files
  "msg/InitialState.msg"
  "msg/Trajectory.msg"
  "msg/RobotState.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs sensor_msgs tf2_ros pinocchio
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME}_utils)

# Instal C++ modules
add_library(mpc_block src/mpc.cpp)
target_link_libraries(mpc_block  example-robot-data::example-robot-data pinocchio::pinocchio ndcurves::ndcurves simple-mpc::simple-mpc)
add_executable(talker src/publisher_member_function.cpp)
target_link_libraries(talker mpc_block)
ament_target_dependencies(talker rclcpp std_msgs tf2_ros nav_msgs)
#add_executable(listener src/subscriber_member_function.cpp)
#ament_target_dependencies(listener rclcpp std_msgs)

rosidl_target_interfaces(talker
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

# Install Python executables
install(PROGRAMS
  scripts/ocp_solver.py
  scripts/interpolator_controller.py
  scripts/mpc.py
  DESTINATION lib/${PROJECT_NAME}
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

ament_export_dependencies(rosidl_default_runtime)

ament_package()
