#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "std_msgs/msg/string.hpp"
#include "unitree_api/msg/request.hpp"

#include "common/ros2_sport_client.h"

using namespace std::chrono_literals;

class Go2PriorityController : public rclcpp::Node
{
public:
  Go2PriorityController()
  : Node("go2_priority_controller"),
    sport_client_(this)
  {
    this->declare_parameter<std::string>("target_topic", "/follow/target");
    this->declare_parameter<std::string>("voice_action_topic", "/go2/voice_action");

    this->declare_parameter<double>("desired_distance", 1.2);
    this->declare_parameter<double>("distance_deadband", 0.15);
    this->declare_parameter<double>("bearing_deadband", 0.08);

    this->declare_parameter<double>("k_forward", 0.45);
    this->declare_parameter<double>("k_yaw", 0.8);

    this->declare_parameter<double>("max_vx", 0.25);
    this->declare_parameter<double>("max_vy", 0.0);
    this->declare_parameter<double>("max_wz", 0.45);

    this->declare_parameter<double>("forward_sign", -1.0);
    this->declare_parameter<double>("yaw_sign", -1.0);

    this->declare_parameter<double>("target_timeout", 0.5);
    this->declare_parameter<double>("control_rate", 10.0);

    // After any voice action, follow is blocked for this many seconds.
    this->declare_parameter<double>("voice_priority_seconds", 4.0);

    // If the voice command is StopMove/Damp, block follow a bit longer.
    this->declare_parameter<double>("stop_priority_seconds", 4.0);

    this->declare_parameter<bool>("balance_stand_on_start", true);

    target_topic_ = this->get_parameter("target_topic").as_string();
    voice_action_topic_ = this->get_parameter("voice_action_topic").as_string();

    desired_distance_ = this->get_parameter("desired_distance").as_double();
    distance_deadband_ = this->get_parameter("distance_deadband").as_double();
    bearing_deadband_ = this->get_parameter("bearing_deadband").as_double();

    k_forward_ = this->get_parameter("k_forward").as_double();
    k_yaw_ = this->get_parameter("k_yaw").as_double();

    max_vx_ = this->get_parameter("max_vx").as_double();
    max_vy_ = this->get_parameter("max_vy").as_double();
    max_wz_ = this->get_parameter("max_wz").as_double();

    forward_sign_ = this->get_parameter("forward_sign").as_double();
    yaw_sign_ = this->get_parameter("yaw_sign").as_double();

    target_timeout_ = this->get_parameter("target_timeout").as_double();
    voice_priority_seconds_ = this->get_parameter("voice_priority_seconds").as_double();
    stop_priority_seconds_ = this->get_parameter("stop_priority_seconds").as_double();

    balance_stand_on_start_ = this->get_parameter("balance_stand_on_start").as_bool();

    const double control_rate = this->get_parameter("control_rate").as_double();

    req_pub_ = this->create_publisher<unitree_api::msg::Request>(
      "/api/sport/request",
      10
    );

    target_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
      target_topic_,
      10,
      std::bind(&Go2PriorityController::on_target, this, std::placeholders::_1)
    );

    voice_sub_ = this->create_subscription<std_msgs::msg::String>(
      voice_action_topic_,
      10,
      std::bind(&Go2PriorityController::on_voice_action, this, std::placeholders::_1)
    );

    const auto control_period = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::duration<double>(1.0 / std::max(control_rate, 1.0))
    );

    timer_ = this->create_wall_timer(
    control_period,
    std::bind(&Go2PriorityController::control_tick, this)
    );

    RCLCPP_INFO(this->get_logger(), "Go2 priority controller started.");
    RCLCPP_INFO(this->get_logger(), "Subscribing follow target: %s", target_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Subscribing voice action: %s", voice_action_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Publishing robot command: /api/sport/request");
    RCLCPP_INFO(this->get_logger(), "Voice has priority over follow.");

    if (balance_stand_on_start_) {
      RCLCPP_INFO(this->get_logger(), "Sending BalanceStand on start...");
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.BalanceStand(req);
      });
    }
  }

private:
  double now_seconds()
  {
    return this->get_clock()->now().seconds();
  }

  double clamp(double v, double lo, double hi) const
  {
    return std::max(lo, std::min(hi, v));
  }

  void publish_request(
    const std::function<void(unitree_api::msg::Request &)> & fill_request)
  {
    unitree_api::msg::Request req;
    fill_request(req);
    req_pub_->publish(req);
  }

  void on_target(const geometry_msgs::msg::Vector3::SharedPtr msg)
  {
    latest_target_ = *msg;
    have_target_ = true;
    last_target_time_ = now_seconds();
  }

  void on_voice_action(const std_msgs::msg::String::SharedPtr msg)
  {
    const std::string action = msg->data;

    if (dangerous_actions_.count(action) > 0) {
      RCLCPP_WARN(this->get_logger(), "Blocked dangerous action: %s", action.c_str());
      return;
    }

    RCLCPP_INFO(this->get_logger(), "VOICE priority action: %s", action.c_str());

    bool executed = execute_voice_action(action);

    if (!executed) {
      RCLCPP_WARN(this->get_logger(), "Unsupported voice action: %s", action.c_str());
      return;
    }

    const double hold_time =
      (action == "StopMove" || action == "Damp")
      ? stop_priority_seconds_
      : voice_priority_seconds_;

    voice_priority_until_ = now_seconds() + hold_time;

    last_sent_stop_ = false;

    RCLCPP_INFO(
      this->get_logger(),
      "Blocking follow for %.2f seconds after voice command.",
      hold_time
    );
  }

  bool execute_voice_action(const std::string & action)
  {
    if (action == "StopMove") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.StopMove(req);
      });
      return true;
    }

    if (action == "Damp") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.Damp(req);
      });
      return true;
    }

    if (action == "BalanceStand") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.BalanceStand(req);
      });
      return true;
    }

    if (action == "StandUp") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.StandUp(req);
      });
      return true;
    }

    if (action == "StandDown") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.StandDown(req);
      });
      return true;
    }

    if (action == "RecoveryStand") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.RecoveryStand(req);
      });
      return true;
    }

    if (action == "Sit") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.Sit(req);
      });
      return true;
    }

    if (action == "Hello") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.Hello(req);
      });
      return true;
    }

    if (action == "Stretch") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.Stretch(req);
      });
      return true;
    }

    if (action == "Dance1") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.Dance1(req);
      });
      return true;
    }

    if (action == "Dance2") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.Dance2(req);
      });
      return true;
    }

    // Add more actions here only if your ros2_sport_client.h supports them.
    return false;
  }

  bool voice_has_priority()
  {
    return now_seconds() < voice_priority_until_;
  }

  void control_tick()
  {
    if (voice_has_priority()) {
      return;
    }

    if (!have_target_) {
      send_stop_once();
      return;
    }

    const double age = now_seconds() - last_target_time_;

    if (age > target_timeout_) {
      send_stop_once();
      return;
    }

    const double bearing = latest_target_.x;
    const double distance = latest_target_.y;

    if (!std::isfinite(bearing) || !std::isfinite(distance)) {
      send_stop_once();
      return;
    }

    const double distance_error = distance - desired_distance_;

    double vx = 0.0;
    double vy = 0.0;
    double wz = 0.0;

    if (std::abs(distance_error) >= distance_deadband_) {
      vx = forward_sign_ * k_forward_ * distance_error;
    }

    if (std::abs(bearing) >= bearing_deadband_) {
      wz = yaw_sign_ * k_yaw_ * bearing;
    }

    vx = clamp(vx, -max_vx_, max_vx_);
    vy = clamp(vy, -max_vy_, max_vy_);
    wz = clamp(wz, -max_wz_, max_wz_);

    if (std::abs(vx) < 1e-3 && std::abs(vy) < 1e-3 && std::abs(wz) < 1e-3) {
      send_stop_once();
      return;
    }

    send_move(vx, vy, wz);
  }

  void send_move(double vx, double vy, double wz)
  {
    last_sent_stop_ = false;

    publish_request([this, vx, vy, wz](unitree_api::msg::Request & req) {
      sport_client_.Move(req, static_cast<float>(vx), static_cast<float>(vy), static_cast<float>(wz));
    });

    RCLCPP_INFO_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      1000,
      "FOLLOW cmd: vx=%.2f vy=%.2f wz=%.2f",
      vx,
      vy,
      wz
    );
  }

  void send_stop_once()
  {
    if (last_sent_stop_) {
      return;
    }

    last_sent_stop_ = true;

    publish_request([this](unitree_api::msg::Request & req) {
      sport_client_.StopMove(req);
    });

    RCLCPP_INFO(this->get_logger(), "StopMove");
  }

  std::string target_topic_;
  std::string voice_action_topic_;

  double desired_distance_;
  double distance_deadband_;
  double bearing_deadband_;

  double k_forward_;
  double k_yaw_;

  double max_vx_;
  double max_vy_;
  double max_wz_;

  double forward_sign_;
  double yaw_sign_;

  double target_timeout_;
  double voice_priority_seconds_;
  double stop_priority_seconds_;

  bool balance_stand_on_start_;

  SportClient sport_client_;

  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr req_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr target_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr voice_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  geometry_msgs::msg::Vector3 latest_target_;
  bool have_target_ = false;
  double last_target_time_ = 0.0;

  double voice_priority_until_ = 0.0;
  bool last_sent_stop_ = false;

  const std::unordered_set<std::string> dangerous_actions_ = {
    "FrontFlip",
    "BackFlip",
    "LeftFlip",
    "RightFlip",
    "FrontJump",
    "FrontPounce",
    "Bound",
    "Handstand",
    "HandStand"
  };
};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<Go2PriorityController>();

  try {
    rclcpp::spin(node);
  }
  catch (const std::exception & e) {
    RCLCPP_ERROR(node->get_logger(), "Exception: %s", e.what());
  }

  rclcpp::shutdown();
  return 0;
}
