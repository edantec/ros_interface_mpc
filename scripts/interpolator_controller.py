#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import rclpy.time

from ros_interface_mpc.msg import Trajectory, InitialState
from rclpy.qos import QoSProfile
from rclpy.time import Time
from ros_interface_mpc_utils.conversions import multiarray_to_numpy


from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, Float64

from go2_control_interface.robot_interface import Go2RobotInterface
from proxsuite_nlp import manifolds
from simple_mpc import IDSolver, RobotHandler
import example_robot_data

class InterpolatorControllerNode(Node):

    def __init__(self):
        # Initialization of node
        super().__init__('interpolator_controller')
        self.declare_parameter('mpc_type')
        self.parameter = self.get_parameter('mpc_type')
        self.robotIf = Go2RobotInterface(self)
        self.start_mpc = False

        # Load robot
        self.rmodel = example_robot_data.load("go2").model
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
            feet_to_base_trans=[
                np.array([0.2, 0.1, 0.]),
                np.array([0.2, -0.1, 0.]),
                np.array([-0.2, 0.1, 0.]),
                np.array([-0.2, -0.1, 0.]),
            ]
        )
        self.handler = RobotHandler()
        self.handler.initialize(design_conf)

        # Define state publisher
        qos_profile_keeplast = QoSProfile(history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)
        self.robot_pub = self.create_publisher(InitialState, 'initial_state', qos_profile_keeplast)

        # Define command subscriber
        self.subscription_joints = self.create_subscription(
            Trajectory,
            'trajectory',
            self.trajectory_callback,
            qos_profile_keeplast)

        # Define odometry subscriber
        self.subscription_odom = self.create_subscription(
            Odometry,
            'odometry/filtered',
            self.odometry_callback,
            qos_profile_keeplast)

        # Message declarations for torque
        self.torque_simu = np.zeros(18)

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
        self.trajectoryStamp = self.get_clock().now()
        self.x0 = np.concatenate((self.default_standing_q, self.default_standing_v))
        self.u0 = self.default_standing_u.copy()
        self.K0 = np.zeros((self.nu, self.ndx))

        self.Kp = [150.]*self.nu
        self.Kd = 0.25 * np.sqrt(self.Kp)
        self.jointCommand = self.default_standing_q[7:]
        self.velocityCommand = self.default_standing_v[6:]
        self.torqueCommand = self.default_standing_u.copy()
        self.MPC_timestep = 0.01

        # For filtering base position and velocity
        self.t_odom_update = None
        self.base_pose = [0.,0.,0., 0.,0.,0.,1.]
        self.base_vel = [0.,0.,0., 0.,0.,0.]
        self.base_filter_fq = self.declare_parameter("base_filter_fq", -1.0).value # By default no filter

        # Debugging purposes
        self.debug_filter_pub = self.create_publisher(Float64MultiArray, "/debug/filtered_state", 10)
        self.debug_loop_time_pub = self.create_publisher(Float64, "/debug/loop_time", 10)
        self.debug_loop_time_msg = Float64()
        self.debug_comm_time_pub = self.create_publisher(Float64, "/debug/comm_time", 10)
        self.debug_comm_time_msg = Float64()
        self.debug_comm_time_new_msg = False

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
            w_tau=0,
            verbose=False,
        )

        self.qp = IDSolver(id_conf, self.rmodel)

        self.robotIf.start_async(self.default_standing_q[7:].tolist())
        self.robotIf.register_callback(self.control_loop)

    def trajectory_callback(self, msg):
        self.us = multiarray_to_numpy(msg.us)
        self.xs = multiarray_to_numpy(msg.xs)
        self.K0 = multiarray_to_numpy(msg.k0)
        self.trajectoryStamp = Time.from_msg(msg.stamp)
        self.ddqs = multiarray_to_numpy(msg.ddqs)
        self.contact_states = multiarray_to_numpy(msg.contact_states)
        self.forces = multiarray_to_numpy(msg.forces)

        self.start_mpc = True
        self.debug_comm_time_new_msg = True
        self.debug_comm_time_process_time = msg.process_duration

    def odometry_callback(self, msg):
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
        if self.t_odom_update is not None and self.base_filter_fq > 0.:
            b = 1. / (1 + 2 * 3.14 * (t_meas - self.t_odom_update) * self.base_filter_fq)

        self.base_pose[:3] = [(1-b) * pos_meas[i] + b * self.base_pose[i] for i in range(3)]
        self.base_pose[3:] = [quat_meas[i] for i in range(4)]
        self.base_vel      = [(1-b) * v_meas[i]   + b * self.base_vel[i] for i in range(6)]
        self.t_odom_update = t_meas

    # Main control loop
    def control_loop(self,t, q, v, a):
        delay = t - self.trajectoryStamp.nanoseconds * 1e-9
        x_measured = np.concatenate((self.base_pose, q, self.base_vel, v))

        if self.start_mpc:
            self.Kp = [10.]*self.nu
            self.Kd = (np.sqrt(self.Kp)).tolist()

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

            self.jointCommand = x_interpolated[7:self.nq]
            self.velocityCommand = x_interpolated[self.nq + 6:]
            if self.parameter.value == "fulldynamics":
                self.torqueCommand = u_interpolated - 1.0 * self.K0 @ self.space.difference(x_measured, x_interpolated)

            elif self.parameter.value == "kinodynamics":
                self.Kp = [50.]*self.nu
                self.Kd = (np.sqrt(self.Kp)).tolist()
                if(step_nb >= len(self.xs) -1):
                    a_interpolated = self.ddqs[step_nb]
                    forces_interpolated = self.forces[step_nb]
                else:
                    a_interpolated = self.ddqs[step_nb + 1] * step_progress  + self.ddqs[step_nb] * (1. - step_progress)
                    forces_interpolated = self.forces[step_nb + 1] * step_progress  + self.forces[step_nb] * (1. - step_progress)
                self.handler.updateState(x_measured[:self.nq], x_measured[self.nq:], True)
                self.qp.solve_qp(
                    self.handler.getData(),
                    [bool(self.contact_states[step_nb][i]) for i in range(4)],
                    x_measured[self.nq:],
                    a_interpolated,
                    forces_interpolated,
                    self.handler.getMassMatrix(),
                )
                self.torqueCommand = self.qp.solved_torque


        if (self.robotIf.is_ready):
            self.robotIf.send_command(self.jointCommand.tolist(),
                                    self.velocityCommand.tolist(),
                                    self.torqueCommand.tolist(),
                                    self.Kp, #self.Kp.tolist(),
                                    self.Kd #self.Kd.tolist()
            )

            state = InitialState()
            state.stamp = rclpy.time.Time(seconds=t).to_msg()
            state.x0 = x_measured.tolist()
            self.robot_pub.publish(state)

        ############################
        # Un-comment for debugging #
        ############################

        # # Log delay
        # self.debug_loop_time_msg.data = delay
        # self.debug_loop_time_pub.publish(self.debug_loop_time_msg)

        # # Log filtered state
        # debug_msg = Float64MultiArray()
        # debug_msg.data = x_measured.ravel().tolist()
        # self.debug_filter_pub.publish(debug_msg)

        # Log communication time
        # if(self.debug_comm_time_new_msg):
        #     self.debug_comm_time_new_msg = False
        #     self.debug_comm_time_msg.data = delay - self.debug_comm_time_process_time
        #     self.debug_comm_time_pub.publish(self.debug_comm_time_msg)

def main(args=None):
    rclpy.init(args=args)

    mpc_subscriber = InterpolatorControllerNode()
    rclpy.spin(mpc_subscriber)

    mpc_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
