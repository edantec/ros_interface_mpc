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

#include <functional>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <rmw/qos_profiles.h>
#include <rclcpp/qos.hpp>
#include <nav_msgs/msg/odometry.hpp>   // for the tf transformation
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>
#include <simple-mpc/robot-handler.hpp>
#include <simple-mpc/lowlevel-control.hpp>
#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/multibody/model.hpp>

#include "ros_interface_mpc/msg/trajectory.hpp"
#include "ros_interface_mpc/msg/initial_state.hpp"

using std::placeholders::_1;

class InterpolatorController : public rclcpp::Node
{
public:
  InterpolatorController()
  : Node("interpolator_controller")
  {
    this->declare_parameter("mpc_type", rclcpp::PARAMETER_STRING);
    this->declare_parameter("base_filter_fq", rclcpp::PARAMETER_DOUBLE);
    mpc_type_ = this->get_parameter("mpc_type").as_string();
    base_filter_fq_ = this->get_parameter("base_filter_fq").as_double();
    start_mpc_ = false;

    // Load robot
    pinocchio::Model model;
    std::string urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/urdf/go2.urdf";
    std::string srdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/srdf/go2.srdf";
    std::string base_joint_name ="root_joint";
    
    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model);
    pinocchio::srdf::loadReferenceConfigurations(model, srdf_path, false);
    //pinocchio::srdf::loadRotorParameters(model, srdf_path, false);

    simple_mpc::RobotModelHandler model_handler = 
      simple_mpc::RobotModelHandler(model, "standing", base_joint_name);
    
    /// Add reference foot for walking
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
    
    // Initialize model related objects
    data_handler_ = std::make_shared<simple_mpc::RobotDataHandler>(model_handler);
    space_ = std::make_shared<proxsuite::nlp::MultibodyPhaseSpace<double>>(model_handler.getModel());
    nq_ = model_handler.getModel().nq;
    nv_ = model_handler.getModel().nv;
    nu_ = nv_ - 6;

    // Initialize vectors and matrices
    t_odom_update_ = 0;
    base_pose_.resize(7);
    base_pose_ << 0, 0, 0, 0, 0, 0, 1; 
    base_vel_.resize(6);
    base_vel_ << 0, 0, 0, 0, 0, 0; 
    
    // Default standing base and joints position for Go2
    Eigen::VectorXd default_standing_q(nq_);
    default_standing_q << 0., 0., 0.335, 0., 0., 0., 1.,
      0.0899, 0.8130, -1.596,
      -0.0405, 0.824, -1.595,
      0.1695, 0.824, -1.606,
      -0.1354, 0.820, -1.593;

    joint_command_.resize(nu_);
    joint_command_ = default_standing_q.tail(nu_);
    velocity_command_.resize(nu_);
    velocity_command_.setZero();
    torque_command_.resize(nu_);
    torque_command_ << -3.71, -1.81, 5.25,
      3.14, -1.37, 5.54,
      -1.39, -1.09,  3.36,
      1.95, -0.61,  3.61;
    x_measured_.resize(nq_ + nv_);
    x_measured_ << default_standing_q, Eigen::VectorXd::Zero(nv_);
    x_interpolated_.resize(nq_ + nv_);
    u_interpolated_.resize(nu_);
    a_interpolated_.resize(nv_);
    forces_interpolated_.resize(model_handler.getFeetNames().size() * 3);
    
    Kp_ = Eigen::VectorXd::Constant(nu_, 150);
    Kd_ = Eigen::VectorXd::Constant(nu_, 0.5 * sqrt(Kp_[0])); 
    MPC_timestep_ = 0.01;

    // Define state publisher
    trajectory_pub_ = this->create_publisher<ros_interface_mpc::msg::InitialState>(
      "initial_state", rclcpp::QoS(rclcpp::KeepLast(1), rmw_qos_profile_sensor_data));

    // Define subscriber
    subscription_joints_ = this->create_subscription<ros_interface_mpc::msg::Trajectory>(
      "trajectory", 
      rclcpp::QoS(rclcpp::KeepLast(1), rmw_qos_profile_sensor_data), 
      std::bind(&InterpolatorController::trajectory_callback, this, std::placeholders::_1)
    );

    subscription_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "odometry/filtered", 
      rclcpp::QoS(rclcpp::KeepLast(1), rmw_qos_profile_sensor_data), 
      std::bind(&InterpolatorController::odometry_callback, this, std::placeholders::_1)
    );

    // Define inverse dynamics QP
    simple_mpc::IDSettings settings;
    settings.contact_ids=model_handler.getFeetIds();
    settings.mu=0.8;
    settings.Lfoot=0.01;
    settings.Wfoot=0.01;
    settings.force_size=3;
    settings.kd=0;
    settings.w_force=100;
    settings.w_acc=1;
    settings.w_tau=0;
    settings.verbose=false;

    qp_ = std::make_shared<simple_mpc::IDSolver>(settings, model_handler.getModel());
  }

private:
  std::vector<Eigen::VectorXd> floatMultiArrayToVector(
    const std_msgs::msg::Float64MultiArray& multiArray)
  {
    std::vector<Eigen::VectorXd> data;
    data.resize(multiArray.layout.dim[0].size, Eigen::VectorXd::Zero(multiArray.layout.dim[1].size)); 
    unsigned int vec_id = 0;
    unsigned int eigen_id = 0;
    for (size_t i = 0; i < multiArray.data.size(); ++i) {
      data[vec_id][eigen_id] = multiArray.data[i];
      eigen_id ++;
      if (eigen_id == multiArray.layout.dim[1].size) {
        eigen_id = 0;
        vec_id ++;
      }
    }
    return data;
  }

  std::vector<std::vector<bool>> intMultiArrayToVector(
    const std_msgs::msg::Int8MultiArray& multiArray)
  {
    std::vector<std::vector<bool>> data;
    data.resize(multiArray.layout.dim[0].size, std::vector<bool>(multiArray.layout.dim[1].size)); 
    unsigned int vec_id = 0;
    unsigned int eigen_id = 0;
    for (size_t i = 0; i < multiArray.data.size(); ++i) {
      data[vec_id][eigen_id] = static_cast<bool>(multiArray.data[i]);
      eigen_id ++;
      if (eigen_id == multiArray.layout.dim[1].size) {
        eigen_id = 0;
        vec_id ++;
      }
    }
    return data;
  }

  Eigen::MatrixXd floatMultiArrayToMatrix(
    const std_msgs::msg::Float64MultiArray& multiArray)
  {
    Eigen::MatrixXd data = 
      Eigen::MatrixXd::Zero(multiArray.layout.dim[0].size, multiArray.layout.dim[1].size);
    unsigned int line = 0;
    unsigned int col = 0;
    for (size_t i = 0; i < multiArray.data.size(); ++i) {
      data(line, col) = multiArray.data[i];
      col ++;
      if (col == multiArray.layout.dim[1].size) {
        col = 0;
        line ++;
      }
    }
    return data;
  } 
  
  void trajectory_callback(const ros_interface_mpc::msg::Trajectory::SharedPtr msg)
  {
    start_mpc_ = true;
    us_ = floatMultiArrayToVector(msg->us);
    xs_ = floatMultiArrayToVector(msg->xs);
    K0_ = floatMultiArrayToMatrix(msg->k0);
    trajectoryStamp_ = rclcpp::Time(msg->stamp.sec, msg->stamp.nanosec);
    ddqs_ = floatMultiArrayToVector(msg->ddqs);
    contact_states_ = intMultiArrayToVector(msg->contact_states);
    forces_ = floatMultiArrayToVector(msg->forces); 
  }

  void odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    auto t_meas = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

    // Filter base position and orientation
    double b = 0.;
    if (t_odom_update_ > 0 and base_filter_fq_ > 0.) {
      b = 1. / (1 + 2 * 3.14 * (t_meas - t_odom_update_) * base_filter_fq_);
    }

    base_pose_[0] = (1-b) * msg->pose.pose.position.x + b * base_pose_[0];
    base_pose_[1] = (1-b) * msg->pose.pose.position.y + b * base_pose_[1];
    base_pose_[2] = (1-b) * msg->pose.pose.position.z + b * base_pose_[2];

    base_pose_[3] = msg->pose.pose.orientation.x;
    base_pose_[4] = msg->pose.pose.orientation.y;
    base_pose_[5] = msg->pose.pose.orientation.z;
    base_pose_[6] = msg->pose.pose.orientation.w;

    base_vel_[0] = (1-b) * msg->twist.twist.linear.x + b * base_vel_[0];
    base_vel_[1] = (1-b) * msg->twist.twist.linear.y + b * base_vel_[1];
    base_vel_[2] = (1-b) * msg->twist.twist.linear.z + b * base_vel_[2];
    base_vel_[3] = (1-b) * msg->twist.twist.angular.x + b * base_vel_[3];
    base_vel_[4] = (1-b) * msg->twist.twist.angular.y + b * base_vel_[4];
    base_vel_[5] = (1-b) * msg->twist.twist.angular.z + b * base_vel_[5];

    t_odom_update_ = t_meas; 
  }
  
  void control_loop(double& t, const Eigen::VectorXd& q, const Eigen::VectorXd& v, const Eigen::VectorXd& a) {
    // Compute control delay
    double delay = t - trajectoryStamp_.nanoseconds() * 1e-9;

    // Concatenate odometry and joint measures
    x_measured_ << base_pose_, q, base_vel_, v;
    
    if (start_mpc_) {
      for (size_t i = 0; i < nu_; i++) {
        Kp_[i] = 100.;
        Kd_[i] = 0.5 * sqrt(Kp_[i]);
      }
      
      // Compute the time knot corresponding to the current delay
      int step_nb = static_cast<int>(delay / MPC_timestep_);
      double step_progress = (delay - step_nb * MPC_timestep_) / MPC_timestep_;
      
      // Interpolate state and command trajectories
      if (step_nb >= (int)xs_.size() -1) {
        step_nb = xs_.size() - 1;
        step_progress = 0.0;
        x_interpolated_ = xs_[step_nb];
        u_interpolated_ = us_[step_nb];
      }
      else {
        x_interpolated_ = xs_[step_nb + 1] * step_progress  + xs_[step_nb] * (1. - step_progress);
        u_interpolated_ = us_[step_nb + 1] * step_progress  + us_[step_nb] * (1. - step_progress);
      }

      joint_command_ = x_interpolated_.segment(7, nu_);
      velocity_command_ = x_interpolated_.tail(nu_);

      // Compute torque command depending on MPC type
      if (mpc_type_ == "fulldynamics") 
        torque_command_ = u_interpolated_ - 1.0 * K0_ * space_->difference(x_measured_, x_interpolated_);
      else if (mpc_type_ == "kinodynamics") {
        for (size_t i = 0; i < nu_; i++) {
          Kp_[i] = 50.;
          Kd_[i] = 0.5 * sqrt(Kp_[i]);
        }

        // Interpolate acceleration and force trajectories
        if (step_nb >= (int)xs_.size() -1) {
          a_interpolated_ = ddqs_[step_nb];
          forces_interpolated_ = forces_[step_nb];
        }
        else {
          a_interpolated_ = ddqs_[step_nb + 1] * step_progress  + ddqs_[step_nb] * (1. - step_progress);
          forces_interpolated_ = forces_[step_nb + 1] * step_progress  + forces_[step_nb] * (1. - step_progress);
        }

        data_handler_->updateInternalData(x_measured_, true);
        pinocchio::Data data = data_handler_->getData();

        // Solve inverse dynamics QP
        qp_->solveQP(
          data,
          contact_states_[step_nb],
          x_measured_.tail(nv_),
          a_interpolated_,
          Eigen::VectorXd::Zero(nu_),
          forces_interpolated_,
          data_handler_->getData().M
        );
        torque_command_ = qp_->solved_torque_;
      }
    }
  }
  
  double t_odom_update_;
  double base_filter_fq_;
  double MPC_timestep_;
  size_t nu_;
  size_t nv_;
  size_t nq_;
  std::string mpc_type_;
  bool start_mpc_;
  rclcpp::Time trajectoryStamp_;

  // Trajectory vectors updated by the OCP solver
  std::vector<Eigen::VectorXd> us_;
  std::vector<Eigen::VectorXd> xs_;
  std::vector<Eigen::VectorXd> ddqs_;
  std::vector<Eigen::VectorXd> forces_;
  std::vector<std::vector<bool>> contact_states_;
  Eigen::MatrixXd K0_;

  // Model-related quantities
  std::shared_ptr<proxsuite::nlp::MultibodyPhaseSpace<double>> space_;
  std::shared_ptr<simple_mpc::RobotDataHandler> data_handler_;
  std::shared_ptr<simple_mpc::IDSolver> qp_;
  
  // Intermediate vectors
  Eigen::VectorXd base_pose_;
  Eigen::VectorXd base_vel_;
  Eigen::VectorXd x_measured_;
  Eigen::VectorXd x_interpolated_;
  Eigen::VectorXd u_interpolated_;
  Eigen::VectorXd a_interpolated_;
  Eigen::VectorXd forces_interpolated_;

  // Command vectors
  Eigen::VectorXd joint_command_;
  Eigen::VectorXd velocity_command_;
  Eigen::VectorXd torque_command_;
  Eigen::VectorXd Kp_;
  Eigen::VectorXd Kd_;

  // ROS topics
  rclcpp::Subscription<ros_interface_mpc::msg::Trajectory>::SharedPtr subscription_joints_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_odom_;
  rclcpp::Publisher<ros_interface_mpc::msg::InitialState>::SharedPtr trajectory_pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<InterpolatorController>());
  rclcpp::shutdown();
  return 0;
}
