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

from ros_interface_mpc.msg import Trajectory, InitialState
from sensor_msgs.msg import Joy
from rclpy.qos import QoSProfile

import numpy as np
from ros_interface_mpc_utils.conversions import (
    numpy_to_multiarray_float64, 
    listof_numpy_to_multiarray_float64, 
    listof_numpy_to_multiarray_int8
)

from mpc import ControlBlockGo2


class MpcPublisher(Node):

    def __init__(self):
        super().__init__('mpc_publisher')
        self.mpc_type = self.declare_parameter('mpc_type').value
        motion_type = self.declare_parameter('motion_type').value
        n_threads = self.declare_parameter('n_threads').value

        qos_profile_keeplast = QoSProfile(history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)

        self.traj_publisher_ = self.create_publisher(Trajectory, 'trajectory', qos_profile_keeplast)
        self.traj_msg = Trajectory()


        self.mpc_block = ControlBlockGo2(self.mpc_type, motion_type, n_threads)
        self.mpc_block.create_gait()
        self.mpc_block.mpc.switchToStand()

        self.force_dim = 12
        self.ndx = self.mpc_block.param.handler.getModel().nv * 2
        self.nu = self.mpc_block.param.handler.getModel().nv - 6
        self.nv = self.mpc_block.param.handler.getModel().nv

        self.commanded_vel = np.zeros(6)
        self.walking = False
        self.timeToWalk = 0

        self.state_subscription = self.create_subscription(
            InitialState,
            'initial_state',
            self.listener_callback,
            qos_profile_keeplast)

        self.joystick_subsciption = self.create_subscription(
            Joy,
            'input',
            self.listener_callback_input,
            qos_profile_keeplast)

    def listener_callback(self, msg):
        self.solve(msg.stamp, np.array(msg.x0))

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

    def solve(self, stamp, x0):
        start_time = self.get_clock().now()

        self.timeToWalk += 1
        if self.walking:
            self.mpc_block.mpc.switchToWalk(self.commanded_vel)
        else:
            self.mpc_block.mpc.switchToStand()

        self.traj_msg.stamp = stamp
        self.mpc_block.update_mpc(x0)
        self.traj_msg.xs = listof_numpy_to_multiarray_float64(self.mpc_block.mpc.xs[:4])
        self.traj_msg.us = listof_numpy_to_multiarray_float64(self.mpc_block.mpc.us[:4])
        if self.mpc_type == "fulldynamics":
            self.traj_msg.k0 = numpy_to_multiarray_float64(self.mpc_block.mpc.K0)
        elif self.mpc_type == "kinodynamics":
            accs = []
            forces = []
            contact_states = []
            for i in range(4):
                a0 = self.mpc_block.mpc.getSolver().workspace.problem_data.stage_data[i]\
                    .dynamics_data.continuous_data.xdot[self.nv:]
                a0[6:] = self.mpc_block.mpc.us[i][self.force_dim:]
                accs.append(a0)
                forces.append(self.mpc_block.mpc.us[i][:self.force_dim])
                contact_states.append(self.mpc_block.mpc.getTrajOptProblem().stages[i].dynamics.differential_dynamics.contact_states)
            
            self.traj_msg.a0 = listof_numpy_to_multiarray_float64(accs)
            self.traj_msg.forces = listof_numpy_to_multiarray_float64(forces)
            self.traj_msg.contact_states = listof_numpy_to_multiarray_int8(contact_states)

        duration = self.get_clock().now() - start_time
        self.traj_msg.process_duration = duration.nanoseconds * 1e-9
        self.traj_publisher_.publish(self.traj_msg)


def main(args=None):
    rclpy.init(args=args)

    mpc_publisher = MpcPublisher()

    rclpy.spin(mpc_publisher)

    mpc_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
