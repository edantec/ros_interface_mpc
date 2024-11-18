"""
This script launches a locomotion MPC scheme which solves repeatedly an 
optimal control problem based on the full dynamics model of the humanoid robot Talos. 
The contacts forces are modeled as 6D wrenches. 
"""

import numpy as np
import example_robot_data
from simple_mpc import RobotHandler, FullDynamicsProblem, KinodynamicsProblem, CentroidalProblem, MPC

class TalosParameters():
    def __init__(self, mpc_type):
        print(mpc_type)
        URDF_FILENAME = "talos_reduced.urdf"
        SRDF_FILENAME = "talos.srdf"
        SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
        URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME

        modelPath = example_robot_data.getModelPath(URDF_SUBPATH)
        
        design_conf = dict(
            urdf_path=modelPath + URDF_SUBPATH,
            srdf_path=modelPath + SRDF_SUBPATH,
            robot_description="",
            root_name="root_joint",
            base_configuration="half_sitting",
            controlled_joints_names=[
                "root_joint",
                "leg_left_1_joint",
                "leg_left_2_joint",
                "leg_left_3_joint",
                "leg_left_4_joint",
                "leg_left_5_joint",
                "leg_left_6_joint",
                "leg_right_1_joint",
                "leg_right_2_joint",
                "leg_right_3_joint",
                "leg_right_4_joint",
                "leg_right_5_joint",
                "leg_right_6_joint",
                "torso_1_joint",
                "torso_2_joint",
                "arm_left_1_joint",
                "arm_left_2_joint",
                "arm_left_3_joint",
                "arm_left_4_joint",
                "arm_right_1_joint",
                "arm_right_2_joint",
                "arm_right_3_joint",
                "arm_right_4_joint",
            ],
            end_effector_names=[
                "left_sole_link",
                "right_sole_link",
            ],
        )
        self.handler = RobotHandler()
        self.handler.initialize(design_conf)
        gravity = np.array([0, 0, -9.81])
        u0_forces = np.zeros(self.handler.getModel().nv + 6)
        for i in range(2):
            u0_forces[6 * i + 2] = -gravity[2] * self.handler.getMass() / 2

        if (mpc_type == "fulldynamics"):
            w_x_vec = np.array(
                [
                    0, 0, 0, 100, 100, 100,  # Base pos/ori
                    0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # Left leg
                    0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # Right leg
                    10, 10,  # Torso
                    1, 1, 1, 1,  # Left arm
                    1, 1, 1, 1,  # Right arm
                    1, 1, 1, 1, 1, 1,  # Base pos/ori vel
                    0.1, 0.1, 0.1, 0.1, 1, 1,  # Left leg vel
                    0.1, 0.1, 0.1, 0.1, 1, 1,  # Right leg vel
                    10, 10,  # Torso vel
                    1, 1, 1, 1,  # Left arm vel
                    1, 1, 1, 1,  # Right arm vel
                ]
            )
            w_cent_lin = np.array([0.0, 0.0, 10.])
            w_cent_ang = np.array([0., 0., 10])
            w_forces_lin = np.ones(3) * 0.0001
            w_forces_ang = np.ones(3) * 0.0001

            nu = self.handler.getModel().nv - 6
        
            self.problem_conf = dict(
                x0=self.handler.getState(),
                u0=np.zeros(nu),
                DT=0.01,
                w_x=np.diag(w_x_vec),
                w_u=np.eye(nu) * 1e-4,
                w_cent=np.diag(np.concatenate((w_cent_lin, w_cent_ang))),
                gravity=gravity,
                force_size=6,
                w_forces=np.diag(np.concatenate((w_forces_lin, w_forces_ang))),
                w_frame=np.eye(6) * 4000,
                umin=-self.handler.getModel().effortLimit[6:],
                umax=self.handler.getModel().effortLimit[6:],
                qmin=self.handler.getModel().lowerPositionLimit[7:],
                qmax=self.handler.getModel().upperPositionLimit[7:],
                mu=0.8,
                Lfoot=0.1,
                Wfoot=0.075,
            )
        elif (mpc_type == "kinodynamics"):
            w_x = np.array([
                0, 0, 1000, 1000, 1000, 1000, # Base pos/ori
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, # Left leg
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, # Right leg
                1, 1000, # Torso
                1, 1, 10, 10, # Left arm
                1, 1, 10, 10, # Right arm
                0.1, 0.1, 0.1, 1000, 1000, 1000, # Base pos/ori vel
                1, 1, 1, 1, 1, 1, # Left leg vel
                1, 1, 1, 1, 1, 1, # Right leg vel
                0.1, 100, # Torso vel
                10, 10, 10, 10, # Left arm vel
                10, 10, 10, 10, # Right arm vel
            ]) 
            w_x = np.diag(w_x) * 10
            w_linforce = np.array([0.001,0.001,0.01])
            w_angforce = np.ones(3) * 0.1
            w_u = np.concatenate((
                w_linforce, 
                w_angforce,
                w_linforce, 
                w_angforce,
                np.ones(self.handler.getModel().nv - 6) * 1e-4
            ))
            w_u = np.diag(w_u) 
            w_LFRF = 100000
            w_cent_lin = np.array([0.0,0.0,1])
            w_cent_ang = np.array([0.1,0.1,10])
            w_cent = np.diag(np.concatenate((w_cent_lin,w_cent_ang)))
            w_centder_lin = np.ones(3) * 0.
            w_centder_ang = np.ones(3) * 0.1
            w_centder = np.diag(np.concatenate((w_centder_lin,w_centder_ang)))

            self.problem_conf = dict(
                x0=self.handler.getState(),
                u0=u0_forces,
                DT=0.01,
                w_x=w_x,
                w_u=w_u,
                w_cent=w_cent,
                w_centder=w_centder,
                gravity=gravity,
                force_size=6,
                w_frame=np.eye(6) * w_LFRF,
                umin=-self.handler.getModel().effortLimit[6:],
                umax=self.handler.getModel().effortLimit[6:],
                qmin=self.handler.getModel().lowerPositionLimit[7:],
                qmax=self.handler.getModel().upperPositionLimit[7:],
            )
        elif (mpc_type == "centroidal"):
            w_control_linear = np.ones(3) * 0.001
            w_control_angular = np.ones(3) * 0.1
            w_control = np.diag(np.concatenate((
                w_control_linear,
                w_control_angular,
                w_control_linear,
                w_control_angular
            )))

            self.problem_conf = dict(
                x0=self.handler.getState(),
                u0=u0_forces,
                DT=0.01,
                w_x=w_x,
                w_u=w_control,
                w_centroidal_com=np.diag(np.array([0,0,0])),
                w_linear_mom=np.diag(np.array([0.01,0.01,100])),
                w_linear_acc=0.01 * np.eye(3),
                w_angular_mom=np.diag(np.array([0.1,0.1,1000])),
                w_angular_acc=0.01 * np.eye(3),
                gravity=gravity,
                force_size=6,
            )
        else:
            print("Error: MPC type not recognized")
            exit()

class Go2Parameters():
    def __init__(self, mpc_type):
        print(mpc_type)
        
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

        gravity = np.array([0, 0, -9.81])
        u0_forces = np.zeros(self.handler.getModel().nv + 6)
        for i in range(2):
            u0_forces[6 * i + 2] = -gravity[2] * self.handler.getMass() / 2

        if (mpc_type == "fulldynamics"):
            w_basepos = [0, 0, 0, 0, 0, 0]
            w_legpos = [0.1, 0.1, 0.1]

            w_basevel = [100, 100, 100, 10, 10, 10]
            w_legvel = [1, 1, 1]
            w_x = np.array(w_basepos + w_legpos * 4 + w_basevel + w_legvel * 4)
            w_cent_lin = np.array([0.1, 0.1, 1])
            w_cent_ang = np.array([0.1, 0.1, 1])
            w_forces_lin = np.array([0.001, 0.001, 0.001])

            nu = self.handler.getModel().nv - 6
        
            self.problem_conf = dict(
                DT=0.01,
                w_x=np.diag(w_x),
                w_u=np.eye(nu) * 1e-4,
                w_cent=np.diag(np.concatenate((w_cent_lin, w_cent_ang))),
                gravity=gravity,
                force_size=3,
                w_forces=np.diag(w_forces_lin),
                w_frame=np.eye(3) * 5000,
                umin=-self.handler.getModel().effortLimit[6:],
                umax=self.handler.getModel().effortLimit[6:],
                qmin=self.handler.getModel().lowerPositionLimit[7:],
                qmax=self.handler.getModel().upperPositionLimit[7:],
                mu=0.8,
                Lfoot=0.01,
                Wfoot=0.01,
                torque_limits=False,
                kinematics_limits=False,
                force_cone=False,
            )
        elif (mpc_type == "kinodynamics"):
            w_basepos = [0, 0, 0, 0, 0, 0]
            w_legpos = [1, 1, 1]

            w_basevel = [10, 10, 10, 10, 10, 10]
            w_legvel = [0.1, 0.1, 0.1]
            w_x = np.array(w_basepos + w_legpos * 4 + w_basevel + w_legvel * 4)
            w_x = np.diag(w_x)
            w_linforce = np.array([0.01, 0.01, 0.01])
            w_u = np.concatenate(
                (
                    w_linforce,
                    w_linforce,
                    w_linforce,
                    w_linforce,
                    np.ones(self.handler.getModel().nv - 6) * 1e-4,
                )
            )
            w_u = np.diag(w_u)
            w_LFRF = 1000
            w_cent_lin = np.array([0.1, 0.1, 1])
            w_cent_ang = np.array([0.1, 0.1, 1])
            w_cent = np.diag(np.concatenate((w_cent_lin, w_cent_ang)))
            w_centder_lin = np.ones(3) * 0.0
            w_centder_ang = np.ones(3) * 0.1
            w_centder = np.diag(np.concatenate((w_centder_lin, w_centder_ang)))

            self.problem_conf = dict(
                DT=0.01,
                w_x=w_x,
                w_u=w_u,
                w_cent=w_cent,
                w_centder=w_centder,
                gravity=gravity,
                force_size=3,
                w_frame=np.eye(3) * w_LFRF,
                umin=-self.handler.getModel().effortLimit[6:],
                umax=self.handler.getModel().effortLimit[6:],
                qmin=self.handler.getModel().lowerPositionLimit[7:],
                qmax=self.handler.getModel().upperPositionLimit[7:],
                mu=0.8,
                Lfoot=0.01,
                Wfoot=0.01,
                kinematics_limits=False,
                force_cone=False,
            )
        else:
            print("Error: MPC type not recognized")
            exit()


class ControlBlockGo2():
    def __init__(self, mpc_type, motion):
        print(mpc_type)
        self.param = Go2Parameters(mpc_type)
        self.motion = motion
        self.T = 50
        
        if mpc_type == "fulldynamics":
            problem = FullDynamicsProblem(self.param.handler)
        elif mpc_type == "kinodynamics":
            problem = KinodynamicsProblem(self.param.handler)
        problem.initialize(self.param.problem_conf)
        problem.createProblem(self.param.handler.getState(), self.T, 3, self.param.problem_conf["gravity"][2])
        
        if self.motion == "walk":
            self.T_fly = 40
            self.T_contact = 10
        elif self.motion == "jump":
            self.T_fly = 20
            self.T_contact = 100
        mpc_conf = dict(
            ddpIteration=1,
            support_force=-self.param.handler.getMass() * self.param.problem_conf["gravity"][2],
            TOL=1e-4,
            mu_init=1e-8,
            max_iters=1,
            num_threads=8,
            swing_apex=0.3,
            T_fly=self.T_fly,
            T_contact=self.T_contact,
            T=self.T,
            dt=0.01,
        )

        self.mpc = MPC()
        self.mpc.initialize(mpc_conf, problem)
        self.nq = self.param.handler.getModel().nq
    
    def create_gait(self):
        """ Define contact sequence throughout horizon"""
        contact_phase_quadru = {
            "FL_foot": True,
            "FR_foot": True,
            "RL_foot": True,
            "RR_foot": True,
        }
        contact_phase_lift_FL = {
            "FL_foot": False,
            "FR_foot": True,
            "RL_foot": True,
            "RR_foot": False,
        }
        contact_phase_lift_FR = {
            "FL_foot": True,
            "FR_foot": False,
            "RL_foot": False,
            "RR_foot": True,
        }
        contact_phase_lift_all = {
            "FL_foot": False,
            "FR_foot": False,
            "RL_foot": False,
            "RR_foot": False,
        }
        contact_phases = []
        if self.motion == "walk":
            contact_phases = [contact_phase_quadru] * self.T_contact
            contact_phases += [contact_phase_lift_FL] * self.T_fly
            contact_phases += [contact_phase_quadru] * self.T_contact
            contact_phases += [contact_phase_lift_FR] * self.T_fly
        elif self.motion == "jump":
            contact_phases = [contact_phase_quadru] * self.T_contact
            contact_phases += [contact_phase_lift_all] * self.T_fly
            contact_phases += [contact_phase_quadru] * self.T_contact 

        self.mpc.generateCycleHorizon(contact_phases)

    def update_mpc(self, x_measured):
        self.mpc.iterate(x_measured[:self.nq],x_measured[self.nq:])

class ControlBlock():
    def __init__(self, mpc_type):
        print(mpc_type)
        self.param = TalosParameters(mpc_type)
        self.T = 100

        problem = FullDynamicsProblem(self.param.handler)
        problem.initialize(self.param.problem_conf)
        problem.createProblem(self.param.problem_conf["x0"], self.T, 6, self.param.problem_conf["gravity"][2])

        mpc_conf = dict(
            ddpIteration=1,
            support_force=-self.param.handler.getMass() * self.param.problem_conf["gravity"][2],
            TOL=1e-4,
            mu_init=1e-8,
            max_iters=1,
            num_threads=8,
        )

        self.mpc = MPC()
        self.mpc.initialize(mpc_conf, problem)
        self.nq = self.param.handler.getModel().nq
    
    def create_gait(self):
        """ Define gait and time parameters"""
        T_ds = 20
        T_ss = 80
        dt = 0.01
        self.Nsimu = int(dt / 0.001)

        """ Define contact sequence throughout horizon"""
        contact_phase_double = {
            "left_sole_link" : True,
            "right_sole_link" : True,
        }
        contact_phase_left = {
            "left_sole_link" : True,
            "right_sole_link" : False,
        }
        contact_phase_right = {
            "left_sole_link" : False,
            "right_sole_link" : True,
        }
        contact_phases = [contact_phase_double] * T_ds
        contact_phases += [contact_phase_left] * T_ss
        contact_phases += [contact_phase_double] * T_ds
        contact_phases += [contact_phase_right] * T_ss 

        self.mpc.generateCycleHorizon(contact_phases)

    def update_mpc(self, x_measured):

        self.mpc.iterate(x_measured[:self.nq],x_measured[self.nq:])