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

from std_msgs.msg import String
from ros_interface_mpc.msg import Torque, RobotState
from rclpy.qos import QoSProfile

import numpy as np
import pinocchio as pin

from mpc import ControlBlock


class MpcPublisher(Node):

    def __init__(self):
        super().__init__('mpc_publisher')
        self.declare_parameter('mpc_type', 'fulldynamics')
        qos_profile = QoSProfile(depth=10)
        self.publisher_ = self.create_publisher(Torque, 'command', qos_profile)
        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        parameter = self.get_parameter('mpc_type')

        self.mpc_block = ControlBlock(parameter.value)
        self.mpc_block.create_gait()

        self.ndx = self.mpc_block.param.handler.get_rmodel().nv * 2
        self.nu = self.mpc_block.param.handler.get_rmodel().nv - 6
        self.x0 = self.mpc_block.param.handler.get_x0()
        self.position = np.zeros(22)
        self.velocity = np.zeros(22)
        self.base_pos = np.zeros(7)
        self.base_vel = np.zeros(6)

        self.controlled_joint_ids = [20, 21, 22, 23, 24, 25, # Leg left
                                     26, 27, 28, 29, 30, 31, # Leg right
                                     0, 1, # Torso
                                     4, 5, 6, 7, # Arm left
                                     11, 12, 13, 14 # Arm right
        ]

        self.subscription = self.create_subscription(
            RobotState,
            'robot_states',
            self.listener_callback,
            1)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.position = np.array([msg.position[i] for i in self.controlled_joint_ids])
        self.velocity = np.array([msg.velocity[i] for i in self.controlled_joint_ids])
        self.base_pos[0] = msg.transform.translation.x
        self.base_pos[1] = msg.transform.translation.y
        self.base_pos[2] = msg.transform.translation.z
        self.base_pos[3] = msg.transform.rotation.x
        self.base_pos[4] = msg.transform.rotation.y
        self.base_pos[5] = msg.transform.rotation.z
        self.base_pos[6] = msg.transform.rotation.w

        self.base_vel[0] = msg.twist.linear.x
        self.base_vel[1] = msg.twist.linear.y
        self.base_vel[2] = msg.twist.linear.z
        self.base_vel[3] = msg.twist.angular.x
        self.base_vel[4] = msg.twist.angular.y
        self.base_vel[5] = msg.twist.angular.z
        self.x0 = np.concatenate((self.base_pos, self.position, self.base_vel, self.velocity))
        self.get_logger().info('I heard: "%s"' % msg.position[0])

    def timer_callback(self):
        self.mpc_block.update_mpc(self.x0)
        msg = Torque()
        msg.x0 = self.x0.tolist()
        msg.u0 = self.mpc_block.mpc.us[0].tolist()
        K0 = self.mpc_block.mpc.K0.tolist()
        riccati = []
        for k in K0:
            riccati += k
        msg.riccati = riccati
        msg.ndx = self.ndx
        msg.nu = self.nu
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.x0[0])

        


def main(args=None):
    rclpy.init(args=args)

    mpc_publisher = MpcPublisher()

    rclpy.spin(mpc_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    mpc_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
