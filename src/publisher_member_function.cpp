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

#define EXAMPLE_ROBOT_DATA_MODEL_DIR "home/edantec/miniforge3/envs/ros_env/share/example-robot-data/robots"

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
        EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_reduced.urdf";
    settings.srdf_path =
        EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf";
    settings.controlled_joints_names = {
        "root_joint",        "leg_left_1_joint",  "leg_left_2_joint",
        "leg_left_3_joint",  "leg_left_4_joint",  "leg_left_5_joint",
        "leg_left_6_joint",  "leg_right_1_joint", "leg_right_2_joint",
        "leg_right_3_joint", "leg_right_4_joint", "leg_right_5_joint",
        "leg_right_6_joint", "torso_1_joint",     "torso_2_joint",
        "arm_left_1_joint",  "arm_left_2_joint",  "arm_left_3_joint",
        "arm_left_4_joint",  "arm_right_1_joint", "arm_right_2_joint",
        "arm_right_3_joint", "arm_right_4_joint",
    };
    settings.end_effector_names = {
        "left_sole_link",
        "right_sole_link",
    };
    settings.root_name = "root_joint";
    settings.base_configuration = "half_sitting";

    RobotHandler handler = RobotHandler();
    handler.initialize(settings);
    
    int T = 100;
    FullDynamicsSettings problem_settings;
    int nu = handler.getModel().nv - 6;
    int ndx = handler.getModel().nv * 2;

    Eigen::VectorXd w_x_vec(ndx);
    w_x_vec << 0, 0, 0, 100, 100, 100,  // Base pos/ori
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,   // Left leg
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,   // Right leg
        10, 10,                         // Torso
        1, 1, 1, 1,                     // Left arm
        1, 1, 1, 1,                     // Right arm
        1, 1, 1, 1, 1, 1,               // Base pos/ori vel
        0.1, 0.1, 0.1, 0.1, 0.01, 0.01, // Left leg vel
        0.1, 0.1, 0.1, 0.1, 0.01, 0.01, // Right leg vel
        10, 10,                         // Torso vel
        1, 1, 1, 1,                     // Left arm vel
        1, 1, 1, 1;                     // Right arm vel
    Eigen::VectorXd w_cent(6);
    w_cent << 0, 0, 0, 0.1, 0., 100;
    Eigen::VectorXd w_forces(6);
    w_forces << 0.0001, 0.0001, 0.0001, 0.01, 0.01, 0.01;

    Eigen::VectorXd u0 = Eigen::VectorXd::Zero(nu);
    
    problem_settings.x0 = handler.getState();
    problem_settings.u0 = u0;
    problem_settings.DT = 0.01;
    problem_settings.w_x = Eigen::MatrixXd::Zero(ndx, ndx);
    problem_settings.w_x.diagonal() = w_x_vec;
    problem_settings.w_u = Eigen::MatrixXd::Identity(nu, nu) * 1e-4;
    problem_settings.w_cent = Eigen::MatrixXd::Zero(6, 6);
    problem_settings.w_cent.diagonal() = w_cent;
    problem_settings.gravity = {0, 0, -9.81};
    problem_settings.force_size = 6,
    problem_settings.w_forces = Eigen::MatrixXd::Zero(6, 6);
    problem_settings.w_forces.diagonal() = w_forces;
    problem_settings.w_frame = Eigen::MatrixXd::Identity(6, 6) * 2000;
    problem_settings.umin = -handler.getModel().effortLimit.tail(nu);
    problem_settings.umax = handler.getModel().effortLimit.tail(nu);
    problem_settings.qmin = handler.getModel().lowerPositionLimit.tail(nu);
    problem_settings.qmax = handler.getModel().upperPositionLimit.tail(nu);
    problem_settings.mu = 0.8;
    problem_settings.Lfoot = 0.1;
    problem_settings.Wfoot = 0.075;

    FullDynamicsProblem problem(problem_settings, handler);
    problem.createProblem(handler.getState(), T, 6, problem_settings.gravity[2]);

    std::shared_ptr<Problem> problemPtr =
        std::make_shared<FullDynamicsProblem>(problem);
    
    MPCSettings mpc_settings;
    mpc_settings.support_force = -handler.getMass() * problem_settings.gravity[2];
    mpc_settings.TOL = 1e-4;
    mpc_settings.mu_init = 1e-8;
    mpc_settings.max_iters = 1;
    mpc_settings.num_threads = 2;
    
    MPC mpc(mpc_settings, problemPtr); 

    publisher_ = this->create_publisher<cpp_pubsub::msg::Torque>("topic", 10);
    timer_ = this->create_wall_timer(
      500ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    auto message = cpp_pubsub::msg::Torque();
    //RCLCPP_INFO_STREAM(this->get_logger(), "Publishing: '" << message.data[0] << "'"); 
    publisher_->publish(message);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<cpp_pubsub::msg::Torque>::SharedPtr publisher_;
  size_t count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}
