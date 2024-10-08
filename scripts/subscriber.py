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
from ros_interface_mpc.msg import Torque, RobotState
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster, TransformStamped
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.qos import QoSProfile

from simulation_args import SimulationArgs
from simulation_utils import (
    addFloor,
    removeBVHModelsIfAny,
    setPhysicsProperties,
    Simulation,
    addSystemCollisionPairs
)

from robot_utils import loadTalos
from QP_utils import (
    IDSolver,
    IKIDSolver_f6,
)
from aligator import manifolds

class SimulationWrapper():

    def __init__(self):
        # Load the robot model 
        rmodelComplete, self.rmodel, qComplete, q0, geom_model = loadTalos()
        v0 = np.zeros(self.rmodel.nv)
        args = SimulationArgs()
        np.random.seed(args.seed)
        pin.seed(args.seed)

        # Add plane in geom_model
        visual_model = geom_model.copy()
        addFloor(geom_model, visual_model)

        # Set simulation properties
        setPhysicsProperties(geom_model, args.material, args.compliance)
        removeBVHModelsIfAny(geom_model)
        addSystemCollisionPairs(self.rmodel, geom_model, q0)
        
        # Remove all pair of collision which does not concern floor collision
        """ i = 0
        while i < len(geom_model.collisionPairs):
            cp = geom_model.collisionPairs[i]
            if geom_model.geometryObjects[cp.first].name != 'floor' and geom_model.geometryObjects[cp.second].name != 'floor':
                geom_model.removeCollisionPair(cp)
            else:
                i = i + 1  """
        
        # Create the simulator object
        self.simulator = Simulation(self.rmodel, geom_model, visual_model, q0, v0, args)


class MpcSubscriber(Node):

    def __init__(self):
        # Initialization of node
        super().__init__('mpc_subscriber')
        self.declare_parameter('mpc_type', 'fulldynamics')
        parameter = self.get_parameter('mpc_type')
        self.start_mpc = False
        
        # Define state publisher
        qos_profile = QoSProfile(depth=10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', qos_profile)
        self.robot_pub = self.create_publisher(RobotState, 'robot_states', 10)
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        
        # Define command subscriber
        self.subscription = self.create_subscription(
            Torque,
            'command',
            self.listener_callback,
            qos_profile)
        self.subscription  # prevent unused variable warning

        # Define at which rate the simulation state is sent to rviz
        timer_period = 0.001  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Message declarations for odometry
        self.odom_trans = TransformStamped()
        self.odom_trans.header.frame_id = 'world'
        self.odom_trans.child_frame_id = 'base_link'
        self.robot_state = RobotState()
        
        # Message declarations for torque
        self.wrapper = SimulationWrapper()
        self.ndx = self.wrapper.rmodel.nv * 2
        self.nq = self.wrapper.rmodel.nq
        self.nu = self.wrapper.rmodel.nv - 6
        self.torque_simu = np.zeros(self.wrapper.rmodel.nv)
        self.current_torque = np.zeros(self.wrapper.rmodel.nv - 6)
        self.space = manifolds.MultibodyPhaseSpace(self.wrapper.rmodel)

        # Message declaration for joint states
        self.measure = JointState()
        self.measure.name = ['torso_1_joint', 'torso_2_joint',
                            'head_1_joint', 'head_2_joint',
                            'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 
                            'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint',
                            'arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint', 
                            'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint', 'arm_right_7_joint',
                            'gripper_left_joint', 'gripper_right_joint',
                            'leg_left_1_joint', 'leg_left_2_joint', 'leg_left_3_joint',
                            'leg_left_4_joint', 'leg_left_5_joint', 'leg_left_6_joint',
                            'leg_right_1_joint', 'leg_right_2_joint', 'leg_right_3_joint',
                            'leg_right_4_joint', 'leg_right_5_joint', 'leg_right_6_joint'
        ]
        self.measure.position = [0., 0., # Torso
                                0., 0., # Head
                                0., 0., 0., 0., 0., 0., 0., # Arm left
                                0., 0., 0., 0., 0., 0., 0., # Arm right
                                0., 0., # Grippers
                                0., 0., 0., 0., 0., 0., # Leg left
                                0., 0., 0., 0., 0., 0. # Leg right
        ]
        self.robot_state.position = self.measure.position
        self.measure.velocity = [0., 0., # Torso
                                0., 0., # Head
                                0., 0., 0., 0., 0., 0., 0., # Arm left
                                0., 0., 0., 0., 0., 0., 0., # Arm right
                                0., 0., # Grippers
                                0., 0., 0., 0., 0., 0., # Leg left
                                0., 0., 0., 0., 0., 0. # Leg right
        ]
        self.robot_state.velocity = self.measure.velocity
        self.controlled_joint_ids = [20, 21, 22, 23, 24, 25, # Leg left
                                     26, 27, 28, 29, 30, 31, # Leg right
                                     0, 1, # Torso
                                     4, 5, 6, 7, # Arm left
                                     11, 12, 13, 14 # Arm right
        ]
        
        # Initial state of the robot with feedforward torque equal to
        # gravity-compensating torque in half-sitting
        q_current, v_current = self.wrapper.simulator.get_state()
        self.x0 = np.concatenate((q_current, v_current))
        self.u0 = np.array([ 7.45208859e-03,  2.32855624e+00, -3.84991098e+00, -5.82410660e+01,
            -2.64373784e+00, -2.36745253e+00,  8.64699317e-03, -3.53947463e+00,
            -3.86813412e+00, -5.82001053e+01, -2.65534337e+00,  7.83385935e-01,
            -1.02704937e-02,  5.47839533e+00, -6.30368462e-01,  3.12071325e+00,
                5.61078119e-01, -4.43512345e+00,  6.11342814e-01, -3.13903610e+00,
            -5.78789165e-01, -4.37097602e+00])
        self.K0 = np.zeros((self.nu, self.ndx))

        # Set state message using latest simulation measure
        self.set_messages(q_current, v_current)
        
        # Define default PD controller that runs before MPC launch
        gain = 10000
        self.Kp = np.identity(self.nu) * gain
        self.Kd = np.identity(self.nu) * 20
        
        # Build whole-body control layer depending on the
        # type of MPC in use
        self.WB_solver = None
        mu = 0.8
        Lfoot=0.1
        Wfoot=0.075
        base_id = self.wrapper.rmodel.getFrameId("base_link")
        torso_id = self.wrapper.rmodel.getFrameId("torso_2_link")
        sole_ids = [self.wrapper.rmodel.getFrameId("left_sole_link"), self.wrapper.rmodel.getFrameId("right_sole_link")]
        if (parameter.value == 'kinodynamics'):
            weights_ID = [1, 10000] # Acceleration, forces
            self.WB_solver = IDSolver(self.wrapper.rmodel, weights_ID, 2, mu, Lfoot, Wfoot, sole_ids, 6, False)
        
        elif (parameter.value == 'centroidal'):
            g_p = 400
            g_h = 10
            g_b = 10
            K_gains = []
            g_q = np.diag(np.array([
                0, 0, 0, 10, 10, 10,
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                1, 1,
                10, 10, 10, 10,
                10, 10, 10, 10
            ]))
            K_gains.append([g_q * 10, 2 * np.sqrt(g_q * 10)])
            K_gains.append([np.eye(6) * g_p, np.eye(6) * 2 * np.sqrt(g_p)])
            K_gains.append([np.eye(6) * g_h, np.eye(6) * 2 * np.sqrt(g_h)])
            K_gains.append([np.eye(3) * g_b, np.eye(3) * 2 * np.sqrt(g_b)])

            weights_IKID = [500, 50000, 10, 1000, 100]  # qref, foot_pose, centroidal, base_rot, force
            self.WB_solver = IKIDSolver_f6(self.wrapper.rmodel, weights_IKID, K_gains, 2, mu, Lfoot, Wfoot, sole_ids, base_id, torso_id, 6, False)
        
        elif (parameter.value != 'fulldynamics'):
            print("Error: MPC type not recognized")
            exit()
            

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.x0[0])

        self.u0 = np.array(msg.u0.tolist())
        self.x0 = np.array(msg.x0.tolist())
        self.K0 = np.array(msg.riccati.tolist()).reshape((self.nu, self.ndx))
    
    def timer_callback(self):
        q_current, v_current = self.wrapper.simulator.get_state()
        
        if not(self.start_mpc):
            self.current_torque = self.u0 - self.Kp @ (q_current[7:] - self.x0[7:self.nq]) - self.Kd @ v_current[6:]
        else:
            x_measured = np.concatenate((q_current, v_current))
            self.current_torque = self.u0 - self.K0 @ self.space.difference(x_measured, self.x0)
        self.torque_simu[6:] = self.current_torque
        self.wrapper.simulator.execute(self.torque_simu)

        self.set_messages(q_current, v_current)
        self.joint_pub.publish(self.measure)
        self.robot_pub.publish(self.robot_state)
        self.broadcaster.sendTransform(self.odom_trans)
        #self.get_logger().info('Publishing: "%s"' % self.measure.position[0])
    
    def set_messages(self, q_current, v_current):
        self.measure.header.stamp = self.get_clock().now().to_msg()
        for meas_id, joint_id in enumerate(self.controlled_joint_ids):
            self.measure.position[joint_id] = q_current[7 + meas_id]
            self.measure.velocity[joint_id] = v_current[6 + meas_id]
            self.robot_state.position[joint_id] = q_current[7 + meas_id]
            self.robot_state.velocity[joint_id] = v_current[6 + meas_id]

        self.odom_trans.header.stamp = self.get_clock().now().to_msg()
        self.odom_trans.transform.translation.x = q_current[0]
        self.odom_trans.transform.translation.y = q_current[1]
        self.odom_trans.transform.translation.z = q_current[2]
        self.odom_trans.transform.rotation = Quaternion(x=q_current[3], 
                                                        y=q_current[4], 
                                                        z=q_current[5], 
                                                        w=q_current[6])
        self.robot_state.transform.translation.x = q_current[0]
        self.robot_state.transform.translation.y = q_current[1]
        self.robot_state.transform.translation.z = q_current[2]
        self.robot_state.transform.rotation = Quaternion(x=q_current[3], 
                                                         y=q_current[4], 
                                                         z=q_current[5], 
                                                         w=q_current[6])
        self.robot_state.twist.linear.x = v_current[0]
        self.robot_state.twist.linear.y = v_current[1]
        self.robot_state.twist.linear.z = v_current[2]
        self.robot_state.twist.angular.x = v_current[3]
        self.robot_state.twist.angular.y = v_current[4]
        self.robot_state.twist.angular.z = v_current[5]

def main(args=None):
    rclpy.init(args=args)

    mpc_subscriber = MpcSubscriber()

    rclpy.spin(mpc_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    mpc_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
