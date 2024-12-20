#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/multibody/model.hpp>
#include <simple-mpc/kinodynamics.hpp>
#include <simple-mpc/fulldynamics.hpp>
#include <simple-mpc/mpc.hpp>
#include <simple-mpc/robot-handler.hpp>

class ControlBlock
{
public:
  ControlBlock(const std::string & mpc_type, const std::string motion_type, const int n_threads) {
    pinocchio::Model model;
    std::string urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/urdf/go2.urdf";
    std::string srdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/srdf/go2.srdf";
    std::string base_joint_name ="root_joint";

    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model);
    pinocchio::srdf::loadReferenceConfigurations(model, srdf_path, false);
    pinocchio::srdf::loadRotorParameters(model, srdf_path, false);

    simple_mpc::RobotModelHandler model_handler = 
      simple_mpc::RobotModelHandler(model, "standing", base_joint_name);
    
    pinocchio::SE3 ref_FL_foot = pinocchio::SE3::Identity();
    pinocchio::SE3 ref_FR_foot = pinocchio::SE3::Identity();
    pinocchio::SE3 ref_RL_foot = pinocchio::SE3::Identity();
    pinocchio::SE3 ref_RR_foot = pinocchio::SE3::Identity();
    ref_FL_foot.translation() = Eigen::Vector3d(0.17, 0.15, 0.0);
    ref_FR_foot.translation() = Eigen::Vector3d(0.17, -0.15, 0.0);
    ref_RL_foot.translation() = Eigen::Vector3d(-0.24, 0.15, 0.0); 
    ref_RR_foot.translation() = Eigen::Vector3d(-0.24, -0.15, 0.0);
    model_handler.addFoot("FL_foot", base_joint_name, ref_FL_foot);
    model_handler.addFoot("FR_foot", base_joint_name, ref_FR_foot);
    model_handler.addFoot("RL_foot", base_joint_name, ref_RL_foot);
    model_handler.addFoot("RR_foot", base_joint_name, ref_RR_foot);

    simple_mpc::RobotDataHandler data_handler = simple_mpc::RobotDataHandler(model_handler);

    int nu = model_handler.getModel().nv - 6;
    int nv = model_handler.getModel().nv;
    int force_dim = 3 * model_handler.getFeetNames().size();
    int nu_kino = nu + force_dim;
    int ndx = nv * 2;
    Eigen::Vector3d gravity;
    gravity << 0., 0., -9.81;

    std::shared_ptr<simple_mpc::OCPHandler> ocpPtr;

    if (mpc_type == "fulldynamics") {
      simple_mpc::FullDynamicsSettings problem_settings;

      Eigen::VectorXd w_x_vec(ndx);
      w_x_vec << 0, 0, 0, 10., 10., 0, // Base pos/ori
        1., 1., 1., 1., 1., 1.,    // FL FR leg
        1., 1., 1., 1., 1., 1.,    // RL RR leg
        10., 10., 10., 1., 1., 10.,  // Base vel
        .1, .1, .1, .1, .1, .1,     // FL FR vel
        .1, .1, .1, .1, .1, .1;    // RL RR vel 
      Eigen::VectorXd w_cent(6);
      w_cent << .1, .1, 1., 0.1, 0.1, 1;
      Eigen::VectorXd w_forces(3);
      w_forces << 0.0002, 0.0002, 0.0002;

      Eigen::VectorXd u0 = Eigen::VectorXd::Zero(nu);

      problem_settings.timestep = 0.01;
      problem_settings.w_x = Eigen::MatrixXd::Zero(ndx, ndx);
      problem_settings.w_x.diagonal() = w_x_vec;
      problem_settings.w_u = Eigen::MatrixXd::Identity(nu, nu) * 1e-4;
      problem_settings.w_cent = Eigen::MatrixXd::Zero(6, 6);
      problem_settings.w_cent.diagonal() = w_cent;
      problem_settings.gravity = gravity;
      problem_settings.force_size = 3;
      problem_settings.Kp_correction = Eigen::VectorXd::Zero(3);
      problem_settings.Kd_correction = Eigen::VectorXd::Zero(3);
      problem_settings.w_forces = Eigen::MatrixXd::Zero(3, 3);
      problem_settings.w_forces.diagonal() = w_forces;
      problem_settings.w_frame = Eigen::MatrixXd::Identity(3, 3) * 5000;
      problem_settings.umin = -model_handler.getModel().effortLimit.tail(nu);
      problem_settings.umax = model_handler.getModel().effortLimit.tail(nu);
      problem_settings.qmin = model_handler.getModel().lowerPositionLimit.tail(nu);
      problem_settings.qmax = model_handler.getModel().upperPositionLimit.tail(nu);
      problem_settings.mu = 0.8;
      problem_settings.Lfoot = 0.1;
      problem_settings.Wfoot = 0.075;
      problem_settings.torque_limits = false;
      problem_settings.kinematics_limits = false;
      problem_settings.force_cone = false;

      std::shared_ptr<simple_mpc::OCPHandler> ocpPtr = std::make_shared<simple_mpc::FullDynamicsOCP>(problem_settings, model_handler);
    }
    if (mpc_type == "kinodynamics") {
      simple_mpc::KinodynamicsSettings problem_settings;

      Eigen::VectorXd w_x_vec(ndx);
      w_x_vec << 0, 0, 0, 10., 10., 0, // Base pos/ori
        1., 1., 1., 1., 1., 1.,    // FL FR leg
        1., 1., 1., 1., 1., 1.,    // RL RR leg
        10., 10., 10., 1., 1., 10.,  // Base vel
        .1, .1, .1, .1, .1, .1,     // FL FR vel
        .1, .1, .1, .1, .1, .1;    // RL RR vel 
      
      Eigen::VectorXd w_forces(3);
      w_forces << 0.001, 0.001, 0.001;

      Eigen::VectorXd w_u_vec(nu_kino);
      w_u_vec.setZero();
      w_u_vec.tail(nv - 6) = Eigen::VectorXd::Ones(nv - 6) * 1e-4;
      w_u_vec.head(force_dim) << w_forces, w_forces, w_forces, w_forces;

      Eigen::VectorXd w_cent(6);
      w_cent << 0.1, 0.1, 1., 0.1, 0.1, 1.;

      Eigen::VectorXd w_cent_der(6);
      w_cent_der << 0., 0., 0., 0.1, 0.1, 0.1;

      problem_settings.timestep = 0.01;
      problem_settings.w_x = Eigen::MatrixXd::Zero(ndx, ndx);
      problem_settings.w_x.diagonal() = w_x_vec;
      problem_settings.w_u = Eigen::MatrixXd::Zero(nu_kino, nu_kino);
      problem_settings.w_cent = Eigen::MatrixXd::Zero(6, 6);
      problem_settings.w_cent.diagonal() = w_cent;
      problem_settings.w_centder = Eigen::MatrixXd::Zero(6, 6);
      problem_settings.w_cent.diagonal() = w_cent_der;
      problem_settings.gravity = gravity;
      problem_settings.force_size = 3;
      problem_settings.w_frame = Eigen::MatrixXd::Identity(3, 3) * 5000;
      problem_settings.qmin = model_handler.getModel().lowerPositionLimit.tail(nu);
      problem_settings.qmax = model_handler.getModel().upperPositionLimit.tail(nu);
      problem_settings.mu = 0.8;
      problem_settings.Lfoot = 0.1;
      problem_settings.Wfoot = 0.075;
      problem_settings.kinematics_limits = false;
      problem_settings.force_cone = false;

      std::shared_ptr<simple_mpc::OCPHandler> ocpPtr = std::make_shared<simple_mpc::KinodynamicsOCP>(problem_settings, model_handler);
    }
    size_t T = 100;
    ocpPtr->createProblem(model_handler.getReferenceState(), T, 3, gravity[2], false);

    simple_mpc::MPCSettings mpc_settings;
    std::size_t T_fly;
    std::size_t T_contact;
    if (motion_type == "walk") {
      T_fly = 30;
      T_contact = 5;
    }
    if (motion_type == "jump") {
      T_fly = 20;
      T_contact = 100;
    }
    mpc_settings.support_force = -gravity[2] * model_handler.getMass();
    mpc_settings.TOL = 1e-4;
    mpc_settings.mu_init = 1e-8;
    mpc_settings.max_iters = 1;
    mpc_settings.num_threads = n_threads;
    mpc_settings.T_fly = T_fly;
    mpc_settings.T_contact = T_contact;

    mpc_ = std::make_shared<simple_mpc::MPC>(mpc_settings, ocpPtr);

    std::vector<std::map<std::string, bool>> contact_states;
    std::map<std::string, bool> contact_state_quadru;
    std::map<std::string, bool> contact_phase_lift_FL_RR;
    std::map<std::string, bool> contact_phase_lift_RL_FR;
    std::map<std::string, bool> contact_phase_lift_all;

    contact_state_quadru.insert({"FL_foot", true});
    contact_state_quadru.insert({"FR_foot", true});
    contact_state_quadru.insert({"RL_foot", true});
    contact_state_quadru.insert({"RR_foot", true});

    contact_phase_lift_FL_RR.insert({"FL_foot", false});
    contact_phase_lift_FL_RR.insert({"FR_foot", true});
    contact_phase_lift_FL_RR.insert({"RL_foot", true});
    contact_phase_lift_FL_RR.insert({"RR_foot", false});

    contact_phase_lift_RL_FR.insert({"FL_foot", true});
    contact_phase_lift_RL_FR.insert({"FR_foot", false});
    contact_phase_lift_RL_FR.insert({"RL_foot", false});
    contact_phase_lift_RL_FR.insert({"RR_foot", true});

    contact_phase_lift_all.insert({"FL_foot", false});
    contact_phase_lift_all.insert({"FR_foot", false});
    contact_phase_lift_all.insert({"RL_foot", false});
    contact_phase_lift_all.insert({"RR_foot", false});
    // std::vector<std::vector<bool>> contact_states;

    if (motion_type == "walk") {
      for (std::size_t i = 0; i < T_contact; i++)
        contact_states.push_back(contact_state_quadru);
      for (std::size_t i = 0; i < T_fly; i++)
        contact_states.push_back(contact_phase_lift_FL_RR);
      for (std::size_t i = 0; i < T_contact; i++)
        contact_states.push_back(contact_state_quadru);
      for (std::size_t i = 0; i < T_fly; i++)
        contact_states.push_back(contact_phase_lift_RL_FR);
    }
    if (motion_type == "jump") {
      for (std::size_t i = 0; i < T_contact; i++)
        contact_states.push_back(contact_state_quadru);
      for (std::size_t i = 0; i < T_fly; i++)
        contact_states.push_back(contact_phase_lift_all);
      for (std::size_t i = 0; i < T_contact; i++)
        contact_states.push_back(contact_state_quadru);
    }

    mpc_->generateCycleHorizon(contact_states);
  }
  virtual ~ControlBlock() {};
  
  std::shared_ptr<simple_mpc::MPC> mpc_;
};