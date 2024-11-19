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
from simple_mpc import IDSolver, RobotHandler
import example_robot_data

class MpcSubscriber(Node):

    def __init__(self):
        # Initialization of node
        super().__init__('mpc_subscriber')
        self.declare_parameter('mpc_type')
        self.parameter = self.get_parameter('mpc_type')
        self.robotIf = Go2RobotInterface(self)
        self.start_mpc = False

        SRDF_SUBPATH = "/go2_description/srdf/go2.srdf"
        URDF_SUBPATH = "/go2_description/urdf/go2.urdf"
        modelPath = example_robot_data.getModelPath(URDF_SUBPATH)
        design_conf = dict(
            urdf_path=modelPath + URDF_SUBPATH,
            srdf_path=modelPath + SRDF_SUBPATH,
            robot_description="",
            root_name="root_joint",
            base_configuration="standing",
            controlled_joints_names=[
                "root_joint",
                "FL_hip_joint",
                "FL_thigh_joint",
                "FL_calf_joint",
                "FR_hip_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
            ],
            end_effector_names=[
                "FL_foot",
                "FR_foot",
                "RL_foot",
                "RR_foot",
            ],
            hip_names=[
                "FL_thigh",
                "FR_thigh",
                "RL_thigh",
                "RR_thigh",
            ],
        )
        self.handler = RobotHandler()
        self.handler.initialize(design_conf)

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
        self.ndx = 36
        self.nu = 12
        self.nq = 19
        self.nv = 18

        # Define a default configuration
        self.default_standing_q = np.array([
            0., 0., 0.335, 0., 0., 0., 1.,
            0.0899, 0.8130, -1.596,
            -0.0405, 0.824, -1.595,
            0.1695, 0.824, -1.606,
            -0.1354, 0.820, -1.593
        ])
        self.default_standing_v = np.zeros(self.nv)
        self.default_standing_u = np.array([
            -3.71, -1.81,  5.25,
            3.14, -1.37, 5.54,
            -1.39, -1.09,  3.36,
            1.95, -0.61,  3.61
        ])

        # Start state
        self.q_current = self.default_standing_q.copy()
        self.v_current = self.default_standing_v.copy()
        self.x0 = np.concatenate((self.q_current, self.v_current))
        self.u0 = self.default_standing_u.copy()
        self.K0 = np.zeros((self.nu, self.ndx))

        # Initial commands to be use before MPC is launched
        gain = 100
        self.Kp = [150.] * self.nu
        self.Kd = [10.] * self.nu

        self.timeStamp = self.get_clock().now()
        self.jointCommand = self.default_standing_q[7:]
        self.velocityCommand = self.default_standing_v[6:]
        self.torqueCommand = self.default_standing_u.copy()
        self.MPC_timestep = 0.01

        # For filtering base position and velocity
        self.base_filter_fq = self.declare_parameter("base_filter_fq", -1.0).value # By default no filter
        self.t_pose_update = None
        self.debug_filter_pub = self.create_publisher(Float64MultiArray, "debug_filter", 10)

        """ Initialize whole-body inverse dynamics QP"""
        id_conf = dict(
            contact_ids=self.handler.getFeetIds(),
            x0=self.x0,
            mu=0.8,
            Lfoot=0.01,
            Wfoot=0.01,
            force_size=3,
            kd=0,
            w_force=100,
            w_acc=1,
            verbose=False,
        )

        self.qp = IDSolver()
        self.qp.initialize(id_conf, self.rmodel)


    def listener_callback(self, msg):
        self.us = multiarray_to_numpy_float64(msg.us)
        self.xs = multiarray_to_numpy_float64(msg.xs)
        self.K0 = multiarray_to_numpy_float64(msg.k0)
        self.timeStamp = Time.from_msg(msg.stamp)
        self.a0 = np.array(msg.a0.tolist())
        self.contact_states = msg.contact_states
        self.forces = np.array(msg.forces.tolist())

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
        if self.start_mpc:
            self.Kp = [10.]*self.nu
            self.Kd = [1.]*self.nu

            step_nb = int(delay // self.MPC_timestep)
            step_progress = (delay % self.MPC_timestep) / self.MPC_timestep
            if(step_nb >= len(self.xs) -1):
                step_nb = len(self.xs) - 1
                step_progress = 0.0
                x_interpolated = self.xs[step_nb]
                u_interpolated = self.us[step_nb]
            else:
                x_interpolated = self.xs[step_nb + 1] * step_progress  + self.xs[step_nb] * (1. - step_progress)
                u_interpolated = self.us[step_nb + 1] * step_progress  + self.us[step_nb] * (1. - step_progress)

            x_measured = np.concatenate((self.q_current, self.v_current))

            self.jointCommand = x_interpolated[7:self.nq]
            self.velocityCommand = x_interpolated[self.nq + 6:]
            if self.parameter.value == "fulldynamics":
                self.torqueCommand = u_interpolated - 1 * self.K0 @ self.space.difference(x_measured, x_interpolated)
            elif self.parameter.value == "kinodynamics":
                self.handler.updateState(self.q_current, self.v_current, True)
                self.qp.solve_qp(
                    self.handler.getData(),
                    self.contact_states,
                    self.v_current,
                    self.a0,
                    self.forces,
                    self.handler.getMassMatrix(),
                )
                self.torqueCommand = self.qp.solved_torque
                #self.get_logger().info(f"torque : {self.torqueCommand}")

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
    mpc_subscriber.robotIf.start(mpc_subscriber.default_standing_q[7:].tolist())
    input("Ready to start...")

    thread.join()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    mpc_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
