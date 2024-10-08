# Ros_interface_mpc

**Ros_interface_mpc** implements a ROS interface to simulate whole-body motions with the Simple simulator and the Aligator library.

##Features

The **Ros_interface_mpc** library provides:

* simulation nodes that subscribe to control topics and display the robot state on rviz
* MPC nodes that subscribe to the robot state topics and compute the next optimal trajectory

## Installation

### Build from source

Create a ROS workspace and clone the repository in the sources:
```
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone git@github.com:edantec/ros_interface_mpc.git --recursive
```

Then build the package:
```
colcon build
```

#### Dependencies

* [simple-mpc](https://github.com/edantec/simple-mpc.git)
* [Simple](https://github.com/Simple-Robotics/Simple.git) 
* ROS Humble installed from Conda
