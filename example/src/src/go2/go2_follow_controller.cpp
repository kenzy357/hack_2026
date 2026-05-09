#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>

#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <unitree_api/msg/request.hpp>
#include <unitree_go/msg/sport_mode_state.hpp>

#include "common/ros2_sport_client.h"

using namespace std::chrono_literals;

class Go2FollowController : public rclcpp::Node {
 public:
  Go2FollowController()
      : Node("go2_follow_controller"), sport_client_(this) {
    target_sub_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
        "/follow/target",
        1,
        [this](geometry_msgs::msg::Vector3Stamped::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(mu_);

          bearing_ = msg->vector.x;
          distance_ = msg->vector.y;

          // Time associated with the perception result.
          target_stamp_ = rclcpp::Time(msg->header.stamp, RCL_ROS_TIME);

          // Time when this controller received the target message.
          last_target_arrival_ = now();

          have_target_ = true;
        });

    state_sub_ = create_subscription<unitree_go::msg::SportModeState>(
        "lf/sportmodestate",
        1,
        [this](unitree_go::msg::SportModeState::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(mu_);
          sport_mode_ = msg->mode;
        });

    timer_ = create_wall_timer(100ms, [this] { Tick(); });
  }

 private:
  void Tick() {
    float bearing = 0.0f;
    float distance = 0.0f;
    rclcpp::Time target_stamp(0, 0, RCL_ROS_TIME);
    rclcpp::Time arrival_stamp(0, 0, RCL_ROS_TIME);
    bool have = false;
    uint8_t mode = 0;

    {
      std::lock_guard<std::mutex> lk(mu_);

      bearing = bearing_;
      distance = distance_;

      target_stamp = target_stamp_;
      arrival_stamp = last_target_arrival_;

      have = have_target_;
      mode = sport_mode_;
    }

    if (mode == 5 || mode == 7) {
      // lieDown / damping mode: do not fight the robot.
      sport_client_.StopMove(req_);
      return;
    }

    if (!have) {
      sport_client_.StopMove(req_);
      return;
    }

    const auto t_now = now();

    const double no_message_time = (t_now - arrival_stamp).seconds();
    const double target_age = (t_now - target_stamp).seconds();

    if (no_message_time > kTargetTimeout || target_age > kMaxTargetAge) {
      sport_client_.StopMove(req_);
      return;
    }

    float dist_err = distance - kDesiredDistance;

    float vx = 0.0f;
    if (std::fabs(dist_err) > kDistDeadband) {
      vx = std::clamp(kKpDist * dist_err, kVxMin, kVxMax);
    }

    if (distance < kMinSafeDist && vx > 0.0f) {
      vx = 0.0f;
    }

    float vyaw = 0.0f;
    if (std::fabs(bearing) > kBearingDeadband) {
      vyaw = std::clamp(-kKpYaw * bearing, -kVyawMax, kVyawMax);
    }

    if (std::fabs(bearing) > 0.4f) {
      vx = std::min(vx, 0.1f);
    }

    sport_client_.Move(req_, vx, 0.0f, vyaw);
  }

  static constexpr float kDesiredDistance = 1.5f;
  static constexpr float kDistDeadband = 0.15f;
  static constexpr float kBearingDeadband = 0.05f;

  static constexpr float kKpDist = 0.8f;
  static constexpr float kKpYaw = 1.5f;

  static constexpr float kVxMax = 0.6f;
  static constexpr float kVxMin = -0.3f;
  static constexpr float kVyawMax = 1.2f;

  static constexpr float kMinSafeDist = 0.6f;

  static constexpr double kTargetTimeout = 0.5;
  static constexpr double kMaxTargetAge = 0.35;

  SportClient sport_client_;
  unitree_api::msg::Request req_;

  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr target_sub_;
  rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr state_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::mutex mu_;

  float bearing_{0.0f};
  float distance_{0.0f};

  rclcpp::Time target_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_target_arrival_{0, 0, RCL_ROS_TIME};

  bool have_target_{false};
  uint8_t sport_mode_{0};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Go2FollowController>());
  rclcpp::shutdown();
  return 0;
}
