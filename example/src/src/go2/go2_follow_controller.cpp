#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>

#include <geometry_msgs/msg/vector3.hpp>
#include <rclcpp/rclcpp.hpp>
#include <unitree_api/msg/request.hpp>
#include <unitree_go/msg/sport_mode_state.hpp>

#include "common/ros2_sport_client.h"

using namespace std::chrono_literals;

class Go2FollowController : public rclcpp::Node {
 public:
  Go2FollowController()
      : Node("go2_follow_controller"), sport_client_(this) {
    // Keep firmware obstacle avoidance OFF. When it's on, the firmware
    // makes its own stop/go decisions under the hood — which makes our
    // controller behavior inconsistent and impossible to reason about.
    // With it off, everything the dog does comes from Move() commands we
    // send explicitly, including avoidance via the lidar bridge below.
    sport_client_.FreeAvoid(req_, false);
    RCLCPP_INFO(get_logger(),
                "firmware obstacle avoidance disabled — we drive everything");

    target_sub_ = create_subscription<geometry_msgs::msg::Vector3>(
        "/follow/target", 10,
        [this](geometry_msgs::msg::Vector3::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(mu_);
          // Reset the controller-side EMA when we just came back from a loss
          // so we don't blend stale smoothed values with a fresh target.
          const bool was_stale =
              !have_target_ || (now() - last_target_).seconds() > kTargetTimeout;
          if (was_stale) {
            filt_bearing_ = msg->x;
            filt_distance_ = msg->y;
          } else {
            filt_bearing_ =
                kTargetEmaAlpha * msg->x + (1.f - kTargetEmaAlpha) * filt_bearing_;
            filt_distance_ =
                kTargetEmaAlpha * msg->y + (1.f - kTargetEmaAlpha) * filt_distance_;
          }
          last_target_ = now();
          have_target_ = true;
        });

    state_sub_ = create_subscription<unitree_go::msg::SportModeState>(
        "lf/sportmodestate", 1,
        [this](unitree_go::msg::SportModeState::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(mu_);
          sport_mode_ = msg->mode;
        });

    // Optional: reactive obstacle-avoidance steering. Published by
    // go2_lidar_avoidance.py. Controller works fine if it's missing —
    // stale/absent → no bias applied, pure visual follow.
    avoidance_sub_ = create_subscription<geometry_msgs::msg::Vector3>(
        "/follow/avoidance", 10,
        [this](geometry_msgs::msg::Vector3::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(mu_);
          fwd_clear_ = msg->x;
          left_clear_ = msg->y;
          right_clear_ = msg->z;
          last_avoidance_ = now();
          have_avoidance_ = true;
        });

    timer_ = create_wall_timer(
        std::chrono::duration<double>(kTickDt), [this] { Tick(); });

    // Send one final StopMove when the context is shutting down (Ctrl-C etc.).
    // rclcpp::on_shutdown fires before the context is torn down, so the
    // publisher inside SportClient is still valid here.
    rclcpp::on_shutdown([this]() { EmitStop(); });
  }

  // Callable from main() / on_shutdown to halt the robot on exit.
  void EmitStop() { sport_client_.StopMove(req_); }

 private:
  // Unsafe sport-mode values where Move() would fight the robot's own state
  // machine: 5 lieDown, 6 jointLock, 7 damping, 8 recoveryStand, 10 sit.
  static bool IsModeUnsafe(uint8_t mode) {
    return mode == 5 || mode == 6 || mode == 7 || mode == 8 || mode == 10;
  }

  static float SlewLimit(float prev, float target, float max_delta) {
    return std::clamp(target, prev - max_delta, prev + max_delta);
  }

  void Tick() {
    float bearing, distance;
    rclcpp::Time stamp;
    bool have;
    uint8_t mode;
    float fwd_clear, left_clear, right_clear;
    rclcpp::Time avoid_stamp;
    bool avoid_have;
    {
      std::lock_guard<std::mutex> lk(mu_);
      bearing = filt_bearing_;
      distance = filt_distance_;
      stamp = last_target_;
      have = have_target_;
      mode = sport_mode_;
      fwd_clear = fwd_clear_;
      left_clear = left_clear_;
      right_clear = right_clear_;
      avoid_stamp = last_avoidance_;
      avoid_have = have_avoidance_;
    }

    if (IsModeUnsafe(mode)) {  // lieDown/jointLock/damping/recoveryStand/sit
      Stop();
      return;
    }

    if (!have || (now() - stamp).seconds() > kTargetTimeout) {
      Stop();
      return;
    }

    float dist_err = distance - kDesiredDistance;
    float vx = 0.f;
    if (std::fabs(dist_err) > kDistDeadband) {
      vx = std::clamp(kKpDist * dist_err, kVxMin, kVxMax);
    }
    if (distance < kMinSafeDist && vx > 0.f) vx = 0.f;

    float vyaw = 0.f;
    if (std::fabs(bearing) > kBearingDeadband) {
      vyaw = std::clamp(-kKpYaw * bearing, -kVyawMax, kVyawMax);
    }

    // When the target is far off-axis, throttle forward motion so we yaw
    // toward them first rather than driving a curve.
    if (std::fabs(bearing) > 0.4f) vx = std::min(vx, 0.1f);

    // Reactive obstacle avoidance: if lidar says the forward corridor is
    // getting tight, (a) bias yaw toward whichever side has more clearance,
    // and (b) gently squash forward speed. Yaw-in-place is never blocked.
    const bool avoid_fresh =
        avoid_have && (now() - avoid_stamp).seconds() < kAvoidTimeout;
    if (!avoid_fresh && avoid_have) {
      // Log once per staleness episode so we know the feed went quiet.
      if (!avoid_warned_stale_) {
        RCLCPP_WARN(get_logger(), "/follow/avoidance stale; driving unbiased");
        avoid_warned_stale_ = true;
      }
    } else if (avoid_fresh) {
      avoid_warned_stale_ = false;
      if (fwd_clear < kAvoidActivate && vx > 0.f) {
        // urgency: 0 at kAvoidActivate, 1 at kAvoidHard (and clamped).
        float urgency = (kAvoidActivate - fwd_clear) /
                        (kAvoidActivate - kAvoidHard);
        urgency = std::clamp(urgency, 0.f, 1.f);

        // Steer toward the side with more clearance, scaled by urgency.
        // Positive side_diff → right more open → yaw right (negative vyaw
        // per the visual convention: negative vyaw = clockwise = right).
        float side_diff = right_clear - left_clear;
        float bias = -kAvoidGain * urgency * side_diff;
        float vyaw_pre = vyaw;
        vyaw = std::clamp(vyaw + bias, -kVyawMax, kVyawMax);

        // Squash forward speed: full at kAvoidActivate, zero at kAvoidHard.
        float speed_scale = 1.f - urgency;
        float vx_pre = vx;
        vx = std::min(vx, kVxMax * speed_scale);

        // Throttled log so we can see when avoidance actually fires and
        // what it commanded vs what visual-follow alone wanted.
        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 500,
            "avoid active: fwd=%.2f lft=%.2f rgt=%.2f | "
            "urgency=%.2f bias=%+.2f | "
            "vx %.2f→%.2f vyaw %.2f→%.2f",
            fwd_clear, left_clear, right_clear,
            urgency, bias, vx_pre, vx, vyaw_pre, vyaw);
      }
    }

    // Slew-limit the commanded velocities so the gait never jerks.
    vx = SlewLimit(prev_vx_, vx, kVxAccel * kTickDt);
    vyaw = SlewLimit(prev_vyaw_, vyaw, kVyawAccel * kTickDt);

    // Snap sub-threshold commands to zero so the dog doesn't "tick" in place.
    if (std::fabs(vx) < kVxDeadband) vx = 0.f;
    if (std::fabs(vyaw) < kVyawDeadband) vyaw = 0.f;

    prev_vx_ = vx;
    prev_vyaw_ = vyaw;

    // Periodic status log at 1 Hz — shows what the controller is seeing
    // and what it's commanding. Easiest way to tell from the A terminal
    // alone whether the dog is following, stopping, or avoiding.
    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "tick: target bear=%+.2f dist=%.2f | "
        "lidar fwd=%.2f lft=%.2f rgt=%.2f%s | "
        "cmd vx=%+.2f vyaw=%+.2f",
        bearing, distance,
        avoid_fresh ? fwd_clear : -1.f,
        avoid_fresh ? left_clear : -1.f,
        avoid_fresh ? right_clear : -1.f,
        avoid_fresh ? "" : " (stale)",
        vx, vyaw);

    sport_client_.Move(req_, vx, 0.f, vyaw);
  }

  // Stop cleanly and reset slew state so a resume starts from zero.
  void Stop() {
    sport_client_.StopMove(req_);
    prev_vx_ = 0.f;
    prev_vyaw_ = 0.f;
  }

  static constexpr double kTickDt = 0.05;  // 20 Hz control loop
  static constexpr float kDesiredDistance = 1.5f;
  static constexpr float kDistDeadband = 0.15f;
  static constexpr float kBearingDeadband = 0.05f;
  static constexpr float kKpDist = 0.8f;
  static constexpr float kKpYaw = 2.5f;
  static constexpr float kVxMax = 0.8f;
  static constexpr float kVxMin = -0.3f;
  static constexpr float kVyawMax = 1.2f;
  static constexpr float kMinSafeDist = 0.6f;
  static constexpr double kTargetTimeout = 0.5;
  // Max acceleration on commanded velocities (m/s² and rad/s²).
  static constexpr float kVxAccel = 1.5f;
  static constexpr float kVyawAccel = 3.0f;
  // Output deadband — suppress tiny commands that the gait can't execute cleanly.
  static constexpr float kVxDeadband = 0.08f;
  static constexpr float kVyawDeadband = 0.10f;
  // EMA on incoming targets (perception already smooths, so keep this light).
  static constexpr float kTargetEmaAlpha = 0.4f;

  // Obstacle avoidance (lidar-based) — bias applied only when forward
  // corridor is tight. Works on top of the visual follow yaw term.
  // Numbers are intentionally aggressive: controller tick is 50 ms, typical
  // walk speed ~0.4 m/s, so at 0.1 m of closure per tick we want to have
  // started reacting well before the dog is at kAvoidHard.
  static constexpr float kAvoidActivate = 1.8f;  // m, start biasing here
  static constexpr float kAvoidHard = 0.8f;      // m, full bias + vx→0 here
  static constexpr float kAvoidGain = 2.0f;      // side-diff rad/s per meter
  static constexpr double kAvoidTimeout = 0.5;   // s, staleness before we ignore

  SportClient sport_client_;
  unitree_api::msg::Request req_;

  rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr target_sub_;
  rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr state_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr avoidance_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::mutex mu_;
  // Smoothed target values fed to the controller.
  float filt_bearing_{0.f};
  float filt_distance_{0.f};
  rclcpp::Time last_target_{0, 0, RCL_ROS_TIME};
  bool have_target_{false};
  uint8_t sport_mode_{0};
  // Lidar clearance (meters, sentinel large when clear).
  float fwd_clear_{100.f};
  float left_clear_{100.f};
  float right_clear_{100.f};
  rclcpp::Time last_avoidance_{0, 0, RCL_ROS_TIME};
  bool have_avoidance_{false};
  bool avoid_warned_stale_{false};
  // Previous commanded velocities (for slew limiting). Accessed only from Tick.
  float prev_vx_{0.f};
  float prev_vyaw_{0.f};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Go2FollowController>();
  rclcpp::spin(node);
  // spin() returned — context is shutting down; on_shutdown already fired
  // the final StopMove. Fall through to clean exit.
  rclcpp::shutdown();
  return 0;
}
