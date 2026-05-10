#include <functional>
#include <memory>
#include <string>
#include <unordered_set>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "unitree_api/msg/request.hpp"

#include "common/ros2_sport_client.h"

class Go2VoiceSportClient : public rclcpp::Node
{
public:
  Go2VoiceSportClient()
  : Node("go2_voice_sport_client"),
    sport_client_(this)
  {
    this->declare_parameter<std::string>("action_topic", "/go2/voice_action");
    this->declare_parameter<bool>("balance_stand_on_start", true);

    action_topic_ = this->get_parameter("action_topic").as_string();
    balance_stand_on_start_ = this->get_parameter("balance_stand_on_start").as_bool();

    req_pub_ = this->create_publisher<unitree_api::msg::Request>(
      "/api/sport/request",
      10
    );

    sub_ = this->create_subscription<std_msgs::msg::String>(
      action_topic_,
      10,
      std::bind(&Go2VoiceSportClient::on_action, this, std::placeholders::_1)
    );

    RCLCPP_INFO(this->get_logger(), "Listening for voice actions on %s", action_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Publishing sport requests to /api/sport/request");

    if (balance_stand_on_start_) {
      RCLCPP_INFO(this->get_logger(), "Sending BalanceStand on start...");
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.BalanceStand(req);
      });
    }
  }

private:
  void publish_request(
    const std::function<void(unitree_api::msg::Request &)> & fill_request)
  {
    unitree_api::msg::Request req;
    fill_request(req);
    req_pub_->publish(req);
  }

  void on_action(const std_msgs::msg::String::SharedPtr msg)
  {
    const std::string action = msg->data;

    if (dangerous_actions_.count(action) > 0) {
      RCLCPP_WARN(this->get_logger(), "Blocked dangerous action: %s", action.c_str());
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Executing voice action: %s", action.c_str());

    if (action == "StopMove") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.StopMove(req);
      });
    }
    else if (action == "Damp") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.Damp(req);
      });
    }
    else if (action == "BalanceStand") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.BalanceStand(req);
      });
    }
    else if (action == "StandUp") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.StandUp(req);
      });
    }
    else if (action == "StandDown") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.StandDown(req);
      });
    }
    else if (action == "RecoveryStand") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.RecoveryStand(req);
      });
    }
    else if (action == "Sit") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.Sit(req);
      });
    }
    else if (action == "Hello") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.Hello(req);
      });
    }
    else if (action == "Stretch") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.Stretch(req);
      });
    }
    else if (action == "Dance1") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.Dance1(req);
      });
    }
    else if (action == "Dance2") {
      publish_request([this](unitree_api::msg::Request & req) {
        sport_client_.Dance2(req);
      });
    }
    else {
      RCLCPP_WARN(
        this->get_logger(),
        "Unsupported action in this Ethernet SportClient wrapper: %s",
        action.c_str()
      );
    }
  }

  std::string action_topic_;
  bool balance_stand_on_start_;

  SportClient sport_client_;

  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr req_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;

  const std::unordered_set<std::string> dangerous_actions_ = {
    "FrontFlip",
    "BackFlip",
    "LeftFlip",
    "RightFlip",
    "FrontJump",
    "FrontPounce",
    "Bound"
  };
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<Go2VoiceSportClient>();
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
