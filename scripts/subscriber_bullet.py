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

import pinocchio as pin
from ros_interface_mpc.msg import Torque, State
from rclpy.qos import QoSProfile

from nav_msgs.msg import Odometry

from robot_utils import loadGo2
from go2_control_interface.robot_interface import Go2RobotInterface
from proxsuite_nlp import manifolds
import example_robot_data
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
        self.current_torque = np.zeros(12)

        # Load the robot model
        self.rmodel, geom_model = loadGo2()

        self.space = manifolds.MultibodyPhaseSpace(self.rmodel)

        self.q_current = np.array([0., 0., 0.335, 0., 0., 0., 1.,
            0.068, 0.785, -1.440,
            -0.068, 0.785, -1.440,
            0.068, 0.785, -1.440,
            -0.068, 0.785, -1.440,
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
        self.Kp = np.ones(12) * gain
        self.Kd = np.ones(12) * 1


    def listener_callback(self, msg):
        self.u0 = np.array(msg.u0.tolist())
        self.x0 = np.array(msg.x0.tolist())
        self.K0 = np.array(msg.riccati.tolist()).reshape((self.nu, self.ndx))

        self.start_mpc = True

    def listener_callback_odom(self, msg):
        self.q_current[0] = msg.pose.pose.position.x
        self.q_current[1] = msg.pose.pose.position.y
        self.q_current[2] = msg.pose.pose.position.z
        self.q_current[3] = msg.pose.pose.orientation.x
        self.q_current[4] = msg.pose.pose.orientation.y
        self.q_current[5] = msg.pose.pose.orientation.z
        self.q_current[6] = msg.pose.pose.orientation.w

        self.v_current[0] = msg.twist.twist.linear.x
        self.v_current[1] = msg.twist.twist.linear.y
        self.v_current[2] = msg.twist.twist.linear.z
        self.v_current[3] = msg.twist.twist.angular.x
        self.v_current[4] = msg.twist.twist.angular.y
        self.v_current[5] = msg.twist.twist.angular.z

    def timer_callback(self):
        current_tqva = self.robotIf.get_joint_state()
        if current_tqva is None:
            return # No state received yet

        self.q_current[7:] = current_tqva[1]
        self.v_current[6:] = current_tqva[2]

        if not(self.start_mpc):
            self.current_torque = self.u0 - self.Kp @ (self.q_current[7:] - self.x0[7:self.nq]) - self.Kd @ self.v_current[6:]
        else:
            x_measured = np.concatenate((self.q_current, self.v_current))
            self.current_torque = self.u0 - self.K0 @ self.space.difference(x_measured, self.x0)

        if (self.robotIf.is_init):
            self.robotIf.send_command(self.q_current[7:].tolist(),
                                    self.v_current[6:].tolist(),
                                    self.current_torque.tolist(),
                                    self.Kp.tolist(),
                                    self.Kd.tolist())

            state = State()
            state.qc = self.q_current.tolist()
            state.vc = self.v_current.tolist()
            self.robot_pub.publish(state)

def main(args=None):
    rclpy.init(args=args)

    mpc_subscriber = MpcSubscriber()

    thread = threading.Thread(target=rclpy.spin, args=(mpc_subscriber, ), daemon=True)
    thread.start()

    mpc_subscriber.robotIf.start(mpc_subscriber.q_current[7:].tolist())
    input("Ready to start...")

    thread.join()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    mpc_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
