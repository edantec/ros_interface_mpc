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

#include <rclcpp/rclcpp.hpp>
#include <rmw/qos_profiles.h>
#include <rclcpp/qos.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/twist.hpp> // for the commanded_vel
#include <nav_msgs/msg/odometry.hpp>   // for the tf transformation
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "mpc.cpp"
#include "ros_interface_mpc/msg/trajectory.hpp"
#include "ros_interface_mpc/msg/initial_state.hpp"

class OCPSolverNode : public rclcpp::Node
{
public:
  explicit OCPSolverNode()
  : Node("ocp_solver")
    //qos_profile_(rclcpp::QoSProfile(rclcpp::QoSHistoryPolicy::KEEP_LAST, 1))
  {
    this->declare_parameter("mpc_type", "fulldynamics");
    this->declare_parameter("motion_type", "walk");
    this->declare_parameter("n_threads", 8);
    mpc_type_ = this->get_parameter("mpc_type").as_string();
    motion_type_ = this->get_parameter("motion_type").as_string();
    n_threads_ = this->get_parameter("n_threads").as_int();
    
    joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
      "joy", 
      rclcpp::QoS(rclcpp::KeepLast(1), rmw_qos_profile_sensor_data), 
      std::bind(&OCPSolverNode::joy_callback, this, std::placeholders::_1)
    );
    state_sub_ = this->create_subscription<ros_interface_mpc::msg::InitialState>(
      "initial_state", 
      rclcpp::QoS(rclcpp::KeepLast(1)), 
      std::bind(&OCPSolverNode::state_callback, this, std::placeholders::_1)
    );
    trajectory_pub_ = this->create_publisher<ros_interface_mpc::msg::Trajectory>("trajectory", 1);

    tf_buffer_ =
      std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ =
      std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    commanded_vel_.resize(6);
    commanded_vel_.setZero();

    mpc_block_ = std::make_shared<ControlBlock>(mpc_type_, motion_type_, n_threads_);
    mpc_block_->mpc_->switchToStand();
  }

private:

  std_msgs::msg::Float64MultiArray convertVectorToFloat64MultiArray(
    const std::vector<Eigen::VectorXd>& eigen_vectors) 
  {  
    std_msgs::msg::Float64MultiArray multi_array_msg;
    // Reserve space for data
    size_t total_size = 0;
    for (const auto& vec : eigen_vectors) {
        total_size += vec.size();
    }
    multi_array_msg.data.reserve(total_size);

    // Copy the data from Eigen vectors into the message
    for (const auto& vec : eigen_vectors) {
        for (int i = 0; i < vec.size(); ++i) {
            multi_array_msg.data.push_back(static_cast<float>(vec(i)));
        }
    }

    return multi_array_msg;
  }

  std_msgs::msg::Int8MultiArray convertVectorToInt8MultiArray(
    const std::vector<std::vector<bool>>& eigen_vectors) 
  {  
    std_msgs::msg::Int8MultiArray multi_array_msg;
    // Reserve space for data
    size_t total_size = 0;
    for (const auto& vec : eigen_vectors) {
        total_size += vec.size();
    }
    multi_array_msg.data.reserve(total_size);

    // Copy the data from Eigen vectors into the message
    for (const auto& vec : eigen_vectors) {
        for (std::size_t i = 0; i < vec.size(); ++i) {
            multi_array_msg.data.push_back(static_cast<int>(vec[i]));
        }
    }

    return multi_array_msg;
  }

  std_msgs::msg::Float64MultiArray convertMatrixToFloat64MultiArray(
    const Eigen::MatrixXd& eigen_matrix) 
  {
    std_msgs::msg::Float64MultiArray multi_array_msg;

    // Set layout dimensions (rows and columns)
    multi_array_msg.layout.dim.resize(2);
    multi_array_msg.layout.dim[0].label = "rows";
    multi_array_msg.layout.dim[0].size = eigen_matrix.rows();
    multi_array_msg.layout.dim[0].stride = eigen_matrix.rows() * eigen_matrix.cols();
    multi_array_msg.layout.dim[1].label = "cols";
    multi_array_msg.layout.dim[1].size = eigen_matrix.cols();
    multi_array_msg.layout.dim[1].stride = eigen_matrix.cols();

    // Reserve space for data
    multi_array_msg.data.reserve(eigen_matrix.size());

    // Copy data from Eigen matrix to Float64MultiArray
    for (int i = 0; i < eigen_matrix.rows(); ++i) {
        for (int j = 0; j < eigen_matrix.cols(); ++j) {
            multi_array_msg.data.push_back(static_cast<float>(eigen_matrix(i, j)));
        }
    }

    return multi_array_msg;
  }

  void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
  { 
    geometry_msgs::msg::TransformStamped t;
    try {
      t = tf_buffer_->lookupTransform(
          "base", "odom",
          tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_INFO(
        this->get_logger(), "Could not transform %s to %s: %s",
          "odom", "base", ex.what());
        return;
    }

    Eigen::Quaternion q = Eigen::Quaternion(t.transform.rotation.z,
      t.transform.rotation.x,
      t.transform.rotation.y,
      t.transform.rotation.z);
    q.normalize();
    auto euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
    double theta = euler[2];

    
    commanded_vel_[0] = cos(theta) * msg->axes[1] + sin(theta) * msg->axes[0];
    commanded_vel_[1] = - sin(theta) * msg->axes[1] + cos(theta) * msg->axes[0];
    commanded_vel_[5] = msg->axes[2];

    commanded_vel_[0] *= 0.25; 
    commanded_vel_[1] *= 0.25; 
    commanded_vel_[5] *= 0.75;

    if (msg->buttons[1])
      walking_ = true;
    if (msg->buttons[2])
      walking_ = false;
  }
  
  void state_callback(const ros_interface_mpc::msg::InitialState::SharedPtr msg)
  { 
    auto trajectory = ros_interface_mpc::msg::Trajectory();
    rclcpp::Time now = this->get_clock()->now();
    
    if (walking_) mpc_block_->mpc_->switchToWalk(commanded_vel_);
    else mpc_block_->mpc_->switchToStand();

    trajectory.stamp = msg->stamp;
    Eigen::VectorXd eigen_x0(msg->x0.size());
    for (std::size_t i = 0; i < msg->x0.size(); ++i)
      eigen_x0[i] = msg->x0[i];
    
    mpc_block_->mpc_->iterate(eigen_x0);
    
    std::vector<Eigen::VectorXd> sub_xs = {mpc_block_->mpc_->xs_.begin(), mpc_block_->mpc_->xs_.begin() + 3}; 
    std::vector<Eigen::VectorXd> sub_us = {mpc_block_->mpc_->us_.begin(), mpc_block_->mpc_->us_.begin() + 3}; 
    trajectory.xs = convertVectorToFloat64MultiArray(sub_xs);
    trajectory.us = convertVectorToFloat64MultiArray(sub_us);
    
    if (mpc_type_ == "fulldynamics") {
      trajectory.k0 = convertMatrixToFloat64MultiArray(mpc_block_->mpc_->Ks_[0]);
    }
    else if (mpc_type_ == "kinodynamics") {
        std::vector<Eigen::VectorXd> accs;
        std::vector<Eigen::VectorXd> forces;
        std::vector<std::vector<bool>> contact_states;
        for (std::size_t i = 0; i < 4; i++) {
            Eigen::VectorXd a = mpc_block_->mpc_->getStateDerivative(i).tail(nv_);
            a.tail(nu_) = mpc_block_->mpc_->us_[i].tail(nu_);
            accs.push_back(a);
            forces.push_back(mpc_block_->mpc_->us_[i].head(force_dim_));
            contact_states.push_back(mpc_block_->mpc_->ocp_handler_->getContactState(i));
        }

        trajectory.ddqs = convertVectorToFloat64MultiArray(accs);
        trajectory.forces = convertVectorToFloat64MultiArray(forces);
        trajectory.contact_states = convertVectorToInt8MultiArray(contact_states);
    }

    auto duration = this->get_clock()->now() - now;
    trajectory.process_duration = duration.nanoseconds() * 1e-9;
    trajectory_pub_->publish(trajectory);
  }

  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Subscription<ros_interface_mpc::msg::InitialState>::SharedPtr state_sub_;
  rclcpp::Publisher<ros_interface_mpc::msg::Trajectory>::SharedPtr trajectory_pub_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  
  std::shared_ptr<ControlBlock> mpc_block_;
  std::string mpc_type_;
  std::string motion_type_;
  int n_threads_;
  Eigen::VectorXd commanded_vel_;
  bool walking_;
  int nu_;
  int nv_;
  int force_dim_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OCPSolverNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}