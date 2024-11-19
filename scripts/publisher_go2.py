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

from ros_interface_mpc.msg import Torque, State
from sensor_msgs.msg import Joy
from rclpy.qos import QoSProfile
from rclpy.time import Time

import numpy as np
from ros_interface_mpc_utils.conversions import numpy_to_multiarray_float64, listof_numpy_to_multiarray_float64

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
        
        self.force_dim = 12
        self.ndx = self.mpc_block.param.handler.getModel().nv * 2
        self.nu = self.mpc_block.param.handler.getModel().nv - 6
        self.nv = self.mpc_block.param.handler.getModel().nv
        self.x0 = self.mpc_block.param.handler.getState()
        self.position = np.zeros(self.nu)
        self.velocity = np.zeros(self.nu)
        self.base_pos = np.zeros(7)
        self.base_vel = np.zeros(6)

        self.commanded_vel = np.zeros(6)
        self.walking = False
        self.stamp = Time.to_msg(self.get_clock().now())
        self.timeToWalk = 0

        self.subscription = self.create_subscription(
            State,
            'robot_states',
            self.listener_callback,
            1)
        self.subscription  # prevent unused variable warning

        self.subinput = self.create_subscription(
            Joy,
            'input',
            self.listener_callback_input,
            1)
        self.subinput  # prevent unused variable warning

    def listener_callback(self, msg):
        self.stamp = msg.stamp
        self.x0 = np.array(msg.qc + msg.vc)
        #self.get_logger().info('I heard: "%s"' % msg.position[0])
    
    def listener_callback_input(self, msg):
        self.commanded_vel[0] = msg.axes[1] * 0.25#m/s
        self.commanded_vel[1] = msg.axes[0] * 0.25 #m/s
        self.commanded_vel[5] = msg.axes[2] * 0.15

        if msg.buttons[1]: # Toggle only at first press
            self.get_logger().info('Walk mode')
            self.walking = True
        if msg.buttons[2]: # Toggle only at first press
            self.get_logger().info('Stand mode')
            self.walking = False

    def timer_callback(self):
        self.timeToWalk += 1
        if self.walking:
            self.mpc_block.mpc.switchToWalk(self.commanded_vel)
        else:
            self.mpc_block.mpc.switchToStand()
        
        """ if (self.timeToWalk >= 300):
            self.mpc_block.mpc.switchToWalk(self.commanded_vel) """

        msg = Torque()
        msg.stamp = self.stamp
        self.mpc_block.update_mpc(self.x0)
        msg.xs = listof_numpy_to_multiarray_float64(self.mpc_block.mpc.xs[:3])
        msg.us = listof_numpy_to_multiarray_float64(self.mpc_block.mpc.us[:3])
        if self.parameter.value == "fulldynamics":
            msg.k0 = numpy_to_multiarray_float64(self.mpc_block.mpc.K0)
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
