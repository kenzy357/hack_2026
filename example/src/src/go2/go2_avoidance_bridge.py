#!/usr/bin/env python3
"""go2_avoidance_bridge — publish firmware clearance readings as /follow/avoidance.

The Go2's firmware already runs its own lidar-based obstacle avoidance (the
same one the gamepad uses) and publishes four obstacle distances in every
SportModeState message at ~50 Hz. That's strictly better than our custom
point-cloud pipeline: no frame-convention guesswork, no invalid-return
filtering, no tuning — Unitree already did all of it.

This node is a thin bridge:
  • subscribe to lf/sportmodestate (unitree_go/SportModeState)
  • read `range_obstacle[4]`
  • republish as /follow/avoidance (geometry_msgs/Vector3) using the same
    contract the rest of our stack already speaks (x=forward, y=left, z=right)

Field layout (verified empirically — adjust the indices via params if the
Go2 firmware on your robot uses a different order):
  range_obstacle[0] = front     → /follow/avoidance.x
  range_obstacle[1] = rear      (unused by the follow controller)
  range_obstacle[2] = left      → /follow/avoidance.y
  range_obstacle[3] = right     → /follow/avoidance.z

Run:
  ros2 run unitree_ros2_example go2_avoidance_bridge
  # if your firmware uses a different order:
  ros2 run unitree_ros2_example go2_avoidance_bridge --ros-args \
      -p idx_front:=0 -p idx_left:=2 -p idx_right:=3 -p debug:=true
"""
import sys

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from unitree_go.msg import SportModeState


class Go2AvoidanceBridge(Node):
    def __init__(self) -> None:
        super().__init__("go2_avoidance_bridge")

        # range_obstacle field layout. Defaults match Unitree's documented
        # [front, rear, left, right] ordering. Overridable as ROS params so
        # you don't need to rebuild if a future firmware changes the order.
        self.declare_parameter("idx_front", 0)
        self.declare_parameter("idx_left", 2)
        self.declare_parameter("idx_right", 3)
        # Periodic log showing the live clearance values. Handy for verifying
        # the index mapping while walking an obstacle around the dog.
        self.declare_parameter("debug", False)

        self.idx_front = int(self.get_parameter("idx_front").value)
        self.idx_left = int(self.get_parameter("idx_left").value)
        self.idx_right = int(self.get_parameter("idx_right").value)
        self.debug = bool(self.get_parameter("debug").value)

        self.pub = self.create_publisher(Vector3, "/follow/avoidance", 10)
        self.sub = self.create_subscription(
            SportModeState, "lf/sportmodestate", self.on_state, 1
        )
        # Heartbeat just like the cloud node had — confirms we're actually
        # getting state messages rather than silently idling.
        self._frame_count = 0
        self._got_any = False
        self.create_timer(2.0, self._heartbeat)

        self.get_logger().info(
            f"bridge up: range_obstacle[{self.idx_front}]→x, "
            f"[{self.idx_left}]→y, [{self.idx_right}]→z"
        )

    def on_state(self, msg: SportModeState) -> None:
        ro = msg.range_obstacle
        if len(ro) < 4:
            # Shouldn't happen — the msg type declares float32[4] — but
            # fail loudly rather than index out of bounds.
            self.get_logger().warn(
                f"range_obstacle has {len(ro)} elements, expected 4"
            )
            return

        front = float(ro[self.idx_front])
        left = float(ro[self.idx_left])
        right = float(ro[self.idx_right])
        self.pub.publish(Vector3(x=front, y=left, z=right))

        self._got_any = True
        self._frame_count += 1
        if self.debug and self._frame_count % 25 == 0:
            # Roughly every 0.5 s at 50 Hz. Printing all four lets you
            # confirm the unused "rear" index is sane too.
            all_vals = ", ".join(f"{float(v):.2f}" for v in ro)
            self.get_logger().info(
                f"range_obstacle=[{all_vals}] → fwd={front:.2f} "
                f"lft={left:.2f} rgt={right:.2f}"
            )

    def _heartbeat(self) -> None:
        if not self._got_any:
            self.get_logger().warn(
                "no lf/sportmodestate messages yet — is the robot connected "
                "and publishing? Try: ros2 topic hz /lf/sportmodestate"
            )
            return
        self.get_logger().info(
            f"heartbeat: {self._frame_count} states in last 2s"
        )
        self._frame_count = 0


def main() -> int:
    rclpy.init()
    node = Go2AvoidanceBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
