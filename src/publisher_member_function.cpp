// Copyright 2016 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define ALIGATOR_PINOCCHIO_V3
//#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include <functional>
#include <memory>
#include <string>

#include <aligator/core/stage-model.hpp>
#include <aligator/core/stage-data.hpp>
#include <simple-mpc/fulldynamics.hpp>
#include <simple-mpc/mpc.hpp>
#include <simple-mpc/robot-handler.hpp>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <ros_interface_mpc/msg/torque.hpp>

//#define EXAMPLE_ROBOT_DATA_MODEL_DIR "home/edantec/miniforge3/envs/ros_env/share/example-robot-data/robots"

using namespace std::chrono_literals;
using simple_mpc::RobotHandler;
using simple_mpc::RobotHandlerSettings;
using PoseVec = std::vector<Eigen::Vector3d>;
using simple_mpc::FullDynamicsProblem;
using simple_mpc::FullDynamicsSettings;
using simple_mpc::MPC;
using simple_mpc::MPCSettings;
using simple_mpc::Problem;

class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher()
  : Node("minimal_publisher"), count_(0)
  { 
    RobotHandlerSettings settings;
    settings.urdf_path =
        EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/urdf/go2.urdf";
    settings.srdf_path =
        EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/srdf/go2.srdf";
    settings.controlled_joints_names = {
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
    };
    settings.end_effector_names = {
      "FL_foot",
      "FR_foot",
      "RL_foot",
      "RR_foot",
    };
    settings.root_name = "root_joint";
    settings.base_configuration = "standing";
    settings.hip_names={
      "FL_thigh",
      "FR_thigh",
      "RL_thigh",
      "RR_thigh",
    };

    simple_mpc::RobotHandler handler = RobotHandler();
    handler.initialize(settings);
    
    int T = 50;
    FullDynamicsSettings problem_settings;
    nu_ = handler.getModel().nv - 6;
    ndx_ = handler.getModel().nv * 2;

    Eigen::VectorXd w_x_vec(ndx);
    w_x_vec << 0, 0, 0, 0, 0, 0,  // Base pos/ori
        1, 1, 1, 1, 1, 1,   // FL / FR
        1, 1, 1, 1, 1, 1,   // RL / RR
        0, 0, 0, 10, 10, 10,  // Base pos/ori vel
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, // FL / FR vel
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1; // RL / RR vel
    Eigen::VectorXd w_cent(6);
    w_cent << 0.1, 0.1,.1, 0.1, 0.1, 1;
    Eigen::VectorXd w_forces(3);
    w_forces << 0.001, 0.001, 0.001;

    
    problem_settings.DT = 0.01;
    problem_settings.w_x = Eigen::MatrixXd::Zero(ndx, ndx);
    problem_settings.w_x.diagonal() = w_x_vec;
    problem_settings.w_u = Eigen::MatrixXd::Identity(nu, nu) * 1e-4;
    problem_settings.w_cent = Eigen::MatrixXd::Zero(6, 6);
    problem_settings.w_cent.diagonal() = w_cent;
    problem_settings.gravity = {0, 0, -9.81};
    problem_settings.force_size = 3,
    problem_settings.w_forces = Eigen::MatrixXd::Zero(3, 3);
    problem_settings.w_forces.diagonal() = w_forces;
    problem_settings.w_frame = Eigen::MatrixXd::Identity(3, 3) * 1000;
    problem_settings.umin = -handler.getModel().effortLimit.tail(nu);
    problem_settings.umax = handler.getModel().effortLimit.tail(nu);
    problem_settings.qmin = handler.getModel().lowerPositionLimit.tail(nu);
    problem_settings.qmax = handler.getModel().upperPositionLimit.tail(nu);
    problem_settings.mu = 0.8;
    problem_settings.Lfoot = 0.01;
    problem_settings.Wfoot = 0.01;
    problem_settings.torque_limits = true;
    problem_settings.kinematics_limits = true;
    problem_settings.force_cone = true;

    FullDynamicsProblem problem(problem_settings, handler);
    problem.createProblem(handler.getState(), T, 3, problem_settings.gravity[2]);

    std::shared_ptr<Problem> problemPtr =
        std::make_shared<FullDynamicsProblem>(problem);
    
    int T_fly = 30;
    int T_contact = 5; 
    MPCSettings mpc_settings;
    mpc_settings.support_force = -handler.getMass() * problem_settings.gravity[2];
    mpc_settings.TOL = 1e-4;
    mpc_settings.mu_init = 1e-8;
    mpc_settings.max_iters = 1;
    mpc_settings.num_threads = 8;
    mpc_settings.swing_apex=0.3;
    mpc_settings.T_fly = T_fly;
    mpc_settings.T_contact = T_contact,
    mpc_settings.T = T;
    mpc_settings.dt=0.01;
    
    mpc_ = MPC(mpc_settings, problemPtr); 
    
    std::vector<std::map<std::string, bool>> contact_states;

    std::map<std::string, bool> contact_states_quadru;
    contact_states_quadru.insert({"FL_foot", true});
    contact_states_quadru.insert({"FR_foot", true});
    contact_states_quadru.insert({"RL_foot", true});
    contact_states_quadru.insert({"RR_foot", true});

    std::map<std::string, bool> contact_phase_lift_FL;
    contact_phase_lift_FL.insert({"FL_foot", false});
    contact_phase_lift_FL.insert({"FR_foot", true});
    contact_phase_lift_FL.insert({"RL_foot", true});
    contact_phase_lift_FL.insert({"RR_foot", false});
    
    std::map<std::string, bool> contact_phase_lift_FR;
    contact_phase_lift_FR.insert({"FL_foot", true});
    contact_phase_lift_FR.insert({"FR_foot", false});
    contact_phase_lift_FR.insert({"RL_foot", false});
    contact_phase_lift_FR.insert({"RR_foot", true});
    
    for (std::size_t i = 0; i < T_contact; i++) {
      contact_states.push_back(contact_states_quadru);
    }
    for (std::size_t i = 0; i < T_fly; i++) {
      contact_states.push_back(contact_phase_lift_FL);
    }
    for (std::size_t i = 0; i < T_contact; i++) {
      contact_states.push_back(contact_states_quadru);
    }
    for (std::size_t i = 0; i < T_fly; i++) {
      contact_states.push_back(contact_phase_lift_FR);
    }

    mpc.generateCycleHorizon(contact_phases);

    position_.resize(nu_);
    position_.setZero();

    velocity_.resize(nu_);
    velocity_.setZero();

    base_pos_.resize(7);
    base_pos_.setZero();

    base_vel_.resize(6);
    base_vel_.setZero();

    x0_ = handler.getState();

    publisher_ = this->create_publisher<ros_interface_mpc::msg::Torque>("command", 10);
    timer_ = this->create_wall_timer(
      10ms, std::bind(&MinimalPublisher::timer_callback, this));
    
    subscriber_ = this->create_subscription<ros_interface_mpc::msg::RobotState>(
      "robot_states", 1, std::bind(&MinimalSubscriber::listener_callback, this, _1));
  }

private:
  void timer_callback()
  {
    count_ ++;
    mpc_.update_mpc(x0_);
    auto message = ros_interface_mpc::msg::Torque();
    msg.x0 = x0_;
    msg.u0 = mpc_.us_[0];
    msg.riccati = mpc_.K0_;
    msg.ndx = ndx_;
    msg.nu = nu_;
    //RCLCPP_INFO_STREAM(this->get_logger(), "Publishing: '" << message.data[0] << "'"); 
    publisher_->publish(message); 
  }
  void listener_callback()
  {
    auto message = ros_interface_mpc::msg::RobotState();
    position_ = message.position;
    velocity_ = message.velocity;
    base_pos_[0] = message.transform.translation.x;
    base_pos_[1] = message.transform.translation.y;
    base_pos_[2] = message.transform.translation.z;
    base_pos_[3] = message.transform.rotation.x;
    base_pos_[4] = message.transform.rotation.y;
    base_pos_[5] = message.transform.rotation.z;
    base_pos_[6] = message.transform.rotation.w;

    base_vel_[0] = message.twist.linear.x;
    base_vel_[1] = message.twist.linear.y;
    base_vel_[2] = message.twist.linear.z;
    base_vel_[3] = message.twist.angular.x;
    base_vel_[4] = message.twist.angular.y;
    base_vel_[5] = message.twist.angular.z;
    x0_ << base_pos_, position_, base_vel_, velocity_;
    //RCLCPP_INFO_STREAM(this->get_logger(), "Publishing: '" << message.data[0] << "'"); 
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<ros_interface_mpc::msg::Torque>::SharedPtr publisher_;
  rclcpp::Subscription<ros_interface_mpc::msg::RobotState>::SharedPtr subscription_;
  size_t count_; 
  MPC mpc_;
  int ndx_;
  int nu_;
  Eigen::VectorXd position_;
  Eigen::VectorXd velocity_;
  Eigen::VectorXd base_pos_;
  Eigen::VectorXd base_vel_;
  Eigen::VectorXd x0_;

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}
