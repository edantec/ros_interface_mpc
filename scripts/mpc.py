"""
This script launches a locomotion MPC scheme which solves repeatedly an 
optimal control problem based on the full dynamics model of the humanoid robot Talos. 
The contacts forces are modeled as 6D wrenches. 
"""

import numpy as np
import example_robot_data
from simple_mpc import RobotHandler, FullDynamicsProblem, KinodynamicsProblem, MPC

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
            # Weight for base position and orientation
            w_basepos = [0, 0, 1, 0, 0, 0]

            # Weight for leg position (hip, thigh, ankle)
            w_legpos = [0.1, 0.1, 0.1]
           
            # Weight for base linear and angular velocity
            w_basevel = [10, 10, 10, 10, 10, 10]

            # Weight for leg velocity (hip, thigh, ankle)
            w_legvel = [1, 1, 1]

            # Concatenated state weight
            w_x = np.array(w_basepos + w_legpos * 4 + w_basevel + w_legvel * 4)

            # Weight for linear momentum regularization
            w_cent_lin = np.array([1, 1, 1])
            # Weight for angular momentum regularization
            w_cent_ang = np.array([0.1, 0.1, 1])

            # Weight for force regularization (reference is robot weight divided by nb of contacts)
            w_forces_lin = np.array([0.0002, 0.0002, 0.0002])
            
            # Weight for feet position tracking
            w_foot_tracking = 5000

            nu = self.handler.getModel().nv - 6
        
            self.problem_conf = dict(
                DT=0.01,
                w_x=np.diag(w_x),
                w_u=np.eye(nu) * 1e-4, # Weight for torque regularization
                w_cent=np.diag(np.concatenate((w_cent_lin, w_cent_ang))),
                gravity=gravity,
                force_size=3,
                w_forces=np.diag(w_forces_lin),
                w_frame=np.eye(3) * w_foot_tracking,
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
            # Weight for base position and orientation
            w_basepos = [0, 0, 0, 0, 0, 0]

             # Weight for leg position (hip, thigh, ankle)
            w_legpos = [1, 1, 1]
            
            # Weight for base linear and angular velocity
            w_basevel = [10, 10, 10, 10, 10, 10]

            # Weight for leg velocity (hip, thigh, ankle)
            w_legvel = [0.1, 0.1, 0.1]
            
            # Concatenated weight for state regularization
            w_x = np.diag(np.array(w_basepos + w_legpos * 4 + w_basevel + w_legvel * 4))

            # Weight for force regularization (reference is robot weight divided by nb of contacts)
            w_forces_lin = np.array([0.001, 0.001, 0.001])

            # Concatenated weight for force and acceleration regularization
            w_u = np.diag(np.concatenate(
                (
                    w_forces_lin,
                    w_forces_lin,
                    w_forces_lin,
                    w_forces_lin,
                    np.ones(self.handler.getModel().nv - 6) * 1e-4,
                )
            ))
            
            # Weight for feet position tracking
            w_foot_tracking = 3000

            # Weight for linear momentum regularization
            w_cent_lin = np.array([0.1, 0.1, 1])

            # Weight for angular momentum regularization
            w_cent_ang = np.array([0.1, 0.1, 1])

            # Concatenated weight for momentum reg
            w_cent = np.diag(np.concatenate((w_cent_lin, w_cent_ang)))

            # Weight for derivative of linear momentum regularization
            w_centder_lin = np.ones(3) * 0.0

            # Weight for derivative of angular momentum regularization
            w_centder_ang = np.ones(3) * 0.1

            # Concatenated weight for derivative of momentum reg
            w_centder = np.diag(np.concatenate((w_centder_lin, w_centder_ang)))

            self.problem_conf = dict(
                DT=0.01,
                w_x=w_x,
                w_u=w_u,
                w_cent=w_cent,
                w_centder=w_centder,
                gravity=gravity,
                force_size=3,
                w_frame=np.eye(3) * w_foot_tracking,
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
    def __init__(self, mpc_type, motion, num_threads):
        print(mpc_type)
        self.param = Go2Parameters(mpc_type)
        self.motion = motion
        self.T = 40
        
        if mpc_type == "fulldynamics":
            problem = FullDynamicsProblem(self.param.handler)
        elif mpc_type == "kinodynamics":
            problem = KinodynamicsProblem(self.param.handler)
        problem.initialize(self.param.problem_conf)
        problem.createProblem(self.param.handler.getState(), self.T, 3, self.param.problem_conf["gravity"][2])
        
        if self.motion == "walk":
            self.T_fly = 30
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
            num_threads=num_threads,
            swing_apex=0.2,
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
