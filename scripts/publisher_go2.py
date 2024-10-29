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

from ros_interface_mpc.msg import Torque, RobotState
from rclpy.qos import QoSProfile

import numpy as np
import pinocchio as pin

from mpc import ControlBlockGo2


class MpcPublisher(Node):

    def __init__(self):
        super().__init__('mpc_publisher')
        self.declare_parameter('mpc_type')
        self.declare_parameter('motion_type')
        self.parameter = self.get_parameter('mpc_type')
        self.motion = self.get_parameter('motion_type')
        qos_profile = QoSProfile(depth=10)
        
        self.publisher_ = self.create_publisher(Torque, 'command', qos_profile)
        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.mpc_block = ControlBlockGo2(self.parameter.value, self.motion.value)
        self.mpc_block.create_gait()
        self.mpc_block.mpc.switchToStand()
        self.timeToWalk = 0
        
        self.force_dim = 12
        self.ndx = self.mpc_block.param.handler.getModel().nv * 2
        self.nu = self.mpc_block.param.handler.getModel().nv - 6
        self.nv = self.mpc_block.param.handler.getModel().nv
        self.x0 = self.mpc_block.param.handler.getState()
        self.position = np.zeros(self.nu)
        self.velocity = np.zeros(self.nu)
        self.base_pos = np.zeros(7)
        self.base_vel = np.zeros(6)

        self.controlled_joint_ids = [0, 1, 2,
                                     3, 4, 5,
                                     6, 7, 8,
                                     9, 10, 11
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
        #self.get_logger().info('I heard: "%s"' % msg.position[0])

    def timer_callback(self):
        self.timeToWalk +=1
        if (self.timeToWalk == 200):
            v = pin.Motion.Zero()
            v.linear[0] = 0.1
            self.mpc_block.mpc.switchToWalk(v)
        
        if (self.timeToWalk == 1000):
            self.mpc_block.mpc.switchToStand()
        
        if (self.timeToWalk == 1500):
            v = pin.Motion.Zero()
            v.linear[0] = 0.1
            self.mpc_block.mpc.switchToWalk(v)
        self.mpc_block.update_mpc(self.x0)
        msg = Torque()
        if self.parameter.value == "fulldynamics":
            msg.x0 = self.x0.tolist()
            msg.u0 = self.mpc_block.mpc.us[0].tolist()
            K0 = self.mpc_block.mpc.K0.tolist()
            riccati = []
            for k in K0:
                riccati += k
            msg.riccati = riccati
            msg.ndx = self.ndx
            msg.nu = self.nu
        elif self.parameter.value == "kinodynamics":
            a0 = self.mpc_block.mpc.getSolver().workspace.problem_data.stage_data[0]\
                .dynamics_data.continuous_data.xdot[self.nv:]
            a0[6:] = self.mpc_block.mpc.us[0][self.force_dim:]
            msg.a0 = a0.tolist()
            msg.forces = self.mpc_block.mpc.us[0][:self.force_dim].tolist()
            msg.contact_states = self.mpc_block.mpc.getTrajOptProblem().stages[0].dynamics.differential_dynamics.contact_states.tolist()
        self.publisher_.publish(msg)
        #self.get_logger().info('Publishing: "%s"' % msg.x0[0])


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
