#!/usr/bin/env python3

# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
import numpy as np
import rclpy.time

from ros_interface_mpc.msg import Torque, State
from rclpy.qos import QoSProfile
from rclpy.time import Time
from ros_interface_mpc_utils.conversions import multiarray_to_numpy_float64


from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

from robot_utils import loadGo2
from go2_control_interface.robot_interface import Go2RobotInterface
from proxsuite_nlp import manifolds
import threading

class MpcSubscriber(Node):

    def __init__(self):
        # Initialization of node
        super().__init__('mpc_subscriber')
        self.robotIf = Go2RobotInterface(self)
        self.start_mpc = False

        # Define state publisher
        qos_profile = QoSProfile(depth=10)
        self.robot_pub = self.create_publisher(State, 'robot_states', qos_profile)

        # Define command subscriber
        self.subscription = self.create_subscription(
            Torque,
            'command',
            self.listener_callback,
            qos_profile)
        self.subscription  # prevent unused variable warning

        # Define odometry subscriber
        self.subscription_odom = self.create_subscription(
            Odometry,
            'odometry/filtered',
            self.listener_callback_odom,
            1)
        self.subscription_odom  # prevent unused variable warning

        # Define at which rate the simulation state is sent to rviz
        timer_period = 0.001  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Message declarations for torque
        self.torque_simu = np.zeros(18)
        self.current_torque = np.array([-3.71, -1.81,  5.25,
            3.14, -1.37, 5.54,
            -1.39, -1.09,  3.36,
            1.95, -0.61,  3.61
        ])

        # Load the robot model
        self.rmodel, geom_model = loadGo2()

        self.space = manifolds.MultibodyPhaseSpace(self.rmodel)

        self.q_current = np.array([0., 0., 0.335, 0., 0., 0., 1.,
            0.0899, 0.8130, -1.596, 
            -0.0405, 0.824, -1.595, 
            0.1695, 0.824, -1.606, 
            -0.1354, 0.820, -1.593
        ])
        """ self.default_standing = np.array([
            0.068, 0.785, -1.440,
            -0.068, 0.785, -1.440,
            0.068, 0.785, -1.440,
            -0.068, 0.785, -1.440,
        ]) """
        self.default_standing = np.array([
            0.0899, 0.8130, -1.596, 
            -0.0405, 0.824, -1.595, 
            0.1695, 0.824, -1.606, 
            -0.1354, 0.820, -1.593
        ])
        self.v_current = np.zeros(18)

        self.ndx = 36
        self.nu = 12
        self.nq = 19
        self.x0 = np.concatenate((self.q_current, self.v_current))
        self.u0 = np.array([-3.71, -1.81,  5.25,
                            3.14, -1.37, 5.54,
                            -1.39, -1.09,  3.36,
                            1.95, -0.61,  3.61])
        self.K0 = np.zeros((12, 36))

        # Define default PD controller that runs before MPC launch
        gain = 100
        self.Kp = [150.]*12
        self.Kd = [10.]*12

        self.timeStamp = self.get_clock().now()
        self.jointCommand = np.array([
            0.0899, 0.8130, -1.596, 
            -0.0405, 0.824, -1.595, 
            0.1695, 0.824, -1.606, 
            -0.1354, 0.820, -1.593
        ])
        self.velocityCommand = np.zeros(12)
        self.torqueCommand = np.array([-3.71, -1.81,  5.25,
            3.14, -1.37, 5.54,
            -1.39, -1.09,  3.36,
            1.95, -0.61,  3.61
        ])
        self.MPC_timestep = 0.01

        # For filtering base position and velocity
        self.base_filter_fq = self.declare_parameter("base_filter_fq", -1.0).value # By default no filter
        self.t_pose_update = None
        self.debug_filter_pub = self.create_publisher(Float64MultiArray, "debug_filter", 10)


    def listener_callback(self, msg):
        self.us = multiarray_to_numpy_float64(msg.us)
        self.xs = multiarray_to_numpy_float64(msg.xs)
        self.K0 = multiarray_to_numpy_float64(msg.k0)
        self.timeStamp = Time.from_msg(msg.stamp)

        self.start_mpc = True

    def listener_callback_odom(self, msg):
        t_meas = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

        pos_meas = msg.pose.pose.position.x, \
                   msg.pose.pose.position.y, \
                   msg.pose.pose.position.z
        quat_meas = msg.pose.pose.orientation.x, \
                    msg.pose.pose.orientation.y, \
                    msg.pose.pose.orientation.z, \
                    msg.pose.pose.orientation.w

        v_meas = msg.twist.twist.linear.x, \
                 msg.twist.twist.linear.y, \
                 msg.twist.twist.linear.z, \
                 msg.twist.twist.angular.x, \
                 msg.twist.twist.angular.y, \
                 msg.twist.twist.angular.z

        # Filter base position and orientation
        b = 0.
        if self.t_pose_update is not None and self.base_filter_fq > 0.:
            b = 1. / (1 + 2 * 3.14 * (t_meas - self.t_pose_update) * self.base_filter_fq)

        self.q_current[:3] = [(1-b) * pos_meas[i] + b * self.q_current[i] for i in range(3)]
        self.q_current[3:7] = [quat_meas[i] for i in range(4)]
        self.v_current[:6] = [(1-b) * v_meas[i]   + b * self.v_current[i] for i in range(6)]
        self.t_pose_update = t_meas

    def interpolate(self, v1, v2, delay):
        return  v1 * (self.MPC_timestep - delay) / self.MPC_timestep + v2 * (delay / self.MPC_timestep)
    
    def timer_callback(self):
        current_tqva = self.robotIf.get_joint_state()
        if current_tqva is None:
            return # No state received yet

        currentTime = current_tqva[0]
        self.q_current[7:] = current_tqva[1]
        self.v_current[6:] = current_tqva[2]


        delay = currentTime - self.timeStamp.nanoseconds * 1e-9
        #self.get_logger().info('Delay : "%s"' % delay)
        #if not(self.start_mpc):
            #self.get_logger().info('Default control')
            #self.current_torque = self.u0 #- self.Kp @ (self.q_current[7:] - self.default_standing) - self.Kd @ self.v_current[6:]
        if self.start_mpc:
            self.Kp = [10.]*12
            self.Kd = [1.]*12
            if delay < self.MPC_timestep:
                x_interpolated = self.interpolate(self.x0, self.x1, delay)
                u_interpolated = self.interpolate(self.u0, self.u1, delay)
            elif delay < 2 * self.MPC_timestep:
                x_interpolated = self.interpolate(self.x1, self.x2, delay)
                u_interpolated = self.interpolate(self.u1, self.u2, delay)
            else:
                x_interpolated = self.x2
                u_interpolated = self.u2

            x_measured = np.concatenate((self.q_current, self.v_current))

            self.jointCommand = x_interpolated[7:self.nq]
            self.velocityCommand = x_interpolated[self.nq + 6:]
            self.torqueCommand = u_interpolated - self.K0 @ self.space.difference(x_measured, x_interpolated)

        x_measured = np.concatenate((self.q_current, self.v_current))
        debug_msg = Float64MultiArray()
        debug_msg.data = x_measured.ravel().tolist()
        self.debug_filter_pub.publish(debug_msg)

        if (self.robotIf.is_init):
            self.robotIf.send_command(self.jointCommand.tolist(),
                                    self.velocityCommand.tolist(),
                                    self.torqueCommand.tolist(),
                                    self.Kp, #self.Kp.tolist(),
                                    self.Kd #self.Kd.tolist()
            )

            state = State()
            state.stamp = rclpy.time.Time(seconds=currentTime).to_msg()
            state.qc = self.q_current.tolist()
            state.vc = self.v_current.tolist()
            self.robot_pub.publish(state)


def main(args=None):
    rclpy.init(args=args)

    mpc_subscriber = MpcSubscriber()

    thread = threading.Thread(target=rclpy.spin, args=(mpc_subscriber, ), daemon=True)
    thread.start()
    mpc_subscriber.robotIf.start(mpc_subscriber.default_standing.tolist())
    input("Ready to start...")

    thread.join()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    mpc_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
