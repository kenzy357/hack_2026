#!/usr/bin/env python3
"""go2_lidar_avoidance — reactive obstacle-clearance publisher.

Reads the Go2's L1 lidar point cloud and publishes nearest-obstacle distance
in three corridors (forward / left / right) so the follow controller can
yaw around static obstacles without losing its visual lock on the target.

Subscribes:
  /utlidar/cloud  (sensor_msgs/msg/PointCloud2, frame utlidar_lidar)
    L1 lidar. Frame convention on Go2 is REP-103-style: x-forward, y-left,
    z-up. Verify once on your robot by visualizing in rviz2.

Publishes:
  /follow/avoidance  (geometry_msgs/msg/Vector3)
    x = forward clearance  (m)   nearest obstacle with |y| <= corridor_half_width
    y = left    clearance  (m)   nearest obstacle with y ∈ [+half, +full]
    z = right   clearance  (m)   nearest obstacle with y ∈ [-full, -half]
    Sentinel = x_max when a corridor is empty.

Run:
  ros2 run unitree_ros2_example go2_lidar_avoidance
  ros2 run unitree_ros2_example go2_lidar_avoidance --ros-args \
      -p corridor_half_width:=0.40 -p x_max:=3.0 -p debug:=true
"""
import sys

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Vector3
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


class Go2LidarAvoidance(Node):
    def __init__(self) -> None:
        super().__init__("go2_lidar_avoidance")

        # ── Corridor geometry ─────────────────────────────────────────────
        # Expecting body-frame cloud now (/utlidar/cloud_base): +x forward,
        # +y left, +z up. Blind-filter catches self-returns — set to
        # Unitree's own value of 0.5 m (from point_lio_unilidar/config).
        self.declare_parameter("blind_radius", 0.5)
        # How far ahead do we care about obstacles.
        self.declare_parameter("x_min", 0.55)  # just past the blind radius
        self.declare_parameter("x_max", 3.0)   # also the "clear" sentinel value
        # Forward corridor half-width. Dog is ~0.3 m wide; 0.35 keeps a margin.
        self.declare_parameter("corridor_half_width", 0.35)
        # Side corridors extend from half_width out to half_width + side_width.
        self.declare_parameter("side_width", 0.70)

        # Height filter — drop the floor and anything above reasonable
        # obstacles. With the body-frame cloud, z = 0 is robot-base height
        # (approximately the ground-plane level). Obstacles a person would
        # care about: feet → head → low ceilings.
        self.declare_parameter("z_min", 0.05)
        self.declare_parameter("z_max", 1.50)

        # Smoothing + noise rejection.
        # Require at least this many points in a corridor before believing it's
        # actually occupied. Kills single-point specks.
        self.declare_parameter("min_points", 3)
        # Exponential moving average on each corridor's clearance.
        self.declare_parameter("ema_alpha", 0.4)

        # Diagnostics: print per-frame point counts and a z histogram once,
        # useful when you first plug in the robot and aren't sure of the frame.
        self.declare_parameter("debug", False)

        # Frame axis mapping. Defaults match /utlidar/cloud_base which is in
        # the robot body frame (standard: +x forward, +y left, +z up). If
        # you swap the subscription to /utlidar/cloud (raw sensor frame),
        # override with -p forward_axis:=+y -p side_axis:=-x.
        self.declare_parameter("forward_axis", "+x")
        self.declare_parameter("side_axis", "+y")
        self.declare_parameter("up_axis", "+z")

        # Input topic. /utlidar/cloud_base is the Go2's body-frame cloud —
        # axes are sane, and self-returns are already pre-filtered by the
        # firmware. /utlidar/cloud is the raw sensor-frame version which
        # requires all the filtering we'd otherwise do by hand.
        self.declare_parameter("cloud_topic", "/utlidar/cloud_base")

        # Top-down bird's-eye view in an OpenCV window. Independent of rviz2.
        self.declare_parameter("show_window", False)
        # Half-size of the visualized area in meters (square, centered on dog).
        self.declare_parameter("view_range", 3.5)
        # Window size in pixels (square).
        self.declare_parameter("view_pixels", 600)

        self.x_min = float(self.get_parameter("x_min").value)
        self.x_max = float(self.get_parameter("x_max").value)
        self.half_w = float(self.get_parameter("corridor_half_width").value)
        self.side_w = float(self.get_parameter("side_width").value)
        self.z_min = float(self.get_parameter("z_min").value)
        self.z_max = float(self.get_parameter("z_max").value)
        self.blind_r2 = float(self.get_parameter("blind_radius").value) ** 2
        self.min_points = int(self.get_parameter("min_points").value)
        self.alpha = float(self.get_parameter("ema_alpha").value)
        self.debug = bool(self.get_parameter("debug").value)
        self.show = bool(self.get_parameter("show_window").value)
        self.view_range = float(self.get_parameter("view_range").value)
        self.view_pixels = int(self.get_parameter("view_pixels").value)

        # Parse and cache the axis mapping once.
        # Each is (axis_index, sign): ("+y" → (1, +1), "-x" → (0, -1)).
        self._fwd_axis = self._parse_axis(str(self.get_parameter("forward_axis").value))
        self._side_axis = self._parse_axis(str(self.get_parameter("side_axis").value))
        self._up_axis = self._parse_axis(str(self.get_parameter("up_axis").value))

        # EMA state. None until first real reading so we can init from raw.
        self._fwd: float | None = None
        self._lft: float | None = None
        self._rgt: float | None = None

        # One-time diagnostics.
        self._diagnostic_logged = False
        self._frame_count = 0
        # Track when clouds actually arrive vs our startup time.
        self._node_start = self.get_clock().now()
        self._last_cloud_time = None
        self._cloud_counter = 0

        # Viz state. `_viz_slot` decouples the lidar callback from the main-
        # thread OpenCV drawing, same pattern as go2_perception's display.
        self._viz_data = None  # set by on_cloud, read by _display_tick
        if self.show:
            self._display_timer = self.create_timer(1.0 / 20.0, self._display_tick)

        self.pub = self.create_publisher(Vector3, "/follow/avoidance", 10)
        # Always publish the filtered cloud for rviz2 viewing. Cheap; subscribe
        # to /follow/avoidance_debug with frame_id = utlidar_lidar in rviz.
        self.debug_pub = self.create_publisher(
            PointCloud2, "/follow/avoidance_debug", 5
        )

        # L1 publishes on the Go2. Historically the reliability advertised
        # has varied across firmware, so use BEST_EFFORT (most tolerant — it
        # will accept data from either reliable or best-effort publishers
        # in rmw_cyclonedds). If clouds still don't arrive, the heartbeat
        # below will inventory the publishers so we can spot the mismatch.
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        cloud_topic = str(self.get_parameter("cloud_topic").value)
        self.cloud_topic = cloud_topic
        self.sub = self.create_subscription(
            PointCloud2, cloud_topic, self.on_cloud, qos
        )

        # Heartbeat: every 2s, confirm or deny that clouds are flowing and
        # dump QoS of all publishers for this topic. Works even when nothing
        # ever arrives.
        self._heartbeat = self.create_timer(2.0, self._heartbeat_tick)

        self.get_logger().info(
            f"corridors: fwd |y|<{self.half_w:.2f}m; "
            f"sides {self.half_w:.2f}–{self.half_w + self.side_w:.2f}m; "
            f"x∈[{self.x_min:.2f},{self.x_max:.2f}]m; "
            f"z∈[{self.z_min:.2f},{self.z_max:.2f}]m; "
            f"blind_r={float(self.get_parameter('blind_radius').value):.2f}m"
        )
        self.get_logger().info(
            f"subscribed to {cloud_topic} "
            f"(axis mapping: forward={self.get_parameter('forward_axis').value} "
            f"side(+left)={self.get_parameter('side_axis').value} "
            f"up={self.get_parameter('up_axis').value})"
        )

    @staticmethod
    def _parse_axis(spec: str) -> "tuple[int, int]":
        """'+y' → (1, +1);  '-x' → (0, -1);  etc."""
        spec = spec.strip().lower()
        if len(spec) != 2 or spec[0] not in "+-" or spec[1] not in "xyz":
            raise ValueError(
                f"bad axis spec {spec!r}; expected one of +x/-x/+y/-y/+z/-z"
            )
        sign = +1 if spec[0] == "+" else -1
        idx = "xyz".index(spec[1])
        return idx, sign

    def _remap(
        self, arr: np.ndarray
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Turn a raw (N,3) cloud array into (forward, side, up) vectors.

        Uses the three cached axis specs. Letting this be configurable keeps
        the rest of the filtering logic frame-agnostic: we can pretend we're
        always in a '+x forward, +y left, +z up' world after this step.
        """
        fi, fs = self._fwd_axis
        si, ss = self._side_axis
        ui, us = self._up_axis
        return (
            fs * arr[:, fi],
            ss * arr[:, si],
            us * arr[:, ui],
        )

    def _heartbeat_tick(self) -> None:
        """Prove whether cloud data is actually reaching this node.

        Runs every 2 s via a timer, independent of the cloud callback. If
        nothing has arrived, inventories all publishers for the subscribed
        topic and prints their QoS — that's how we catch the two classic
        failure modes: no publisher at all, or a QoS mismatch (RELIABLE pub
        vs BEST_EFFORT sub and friends).
        """
        now = self.get_clock().now()
        if self._cloud_counter == 0 and self._last_cloud_time is None:
            elapsed = (now - self._node_start).nanoseconds / 1e9
            try:
                info = self.get_publishers_info_by_topic(self.cloud_topic)
            except Exception as e:
                info = []
                self.get_logger().warn(f"could not inspect publishers: {e}")
            lines = [
                f"no {self.cloud_topic} messages received after {elapsed:.1f}s. "
                f"found {len(info)} publisher(s):"
            ]
            for p in info:
                qos = p.qos_profile
                lines.append(
                    f"  node={p.node_name} ns={p.node_namespace} "
                    f"reliability={qos.reliability.name} "
                    f"durability={qos.durability.name} depth={qos.depth}"
                )
            if not info:
                lines.append(
                    f"  (none) — either {self.cloud_topic} is not being "
                    "published, or the topic name is different. Try: "
                    "ros2 topic list | grep -i cloud"
                )
            self.get_logger().warn("\n".join(lines))
            return
        # Data is flowing — show rate for this window.
        age = (now - self._last_cloud_time).nanoseconds / 1e9 \
            if self._last_cloud_time is not None else float("inf")
        self.get_logger().info(
            f"heartbeat: {self._cloud_counter} cloud(s) in last 2s; "
            f"newest arrived {age:.2f}s ago"
        )
        self._cloud_counter = 0

    def on_cloud(self, msg: PointCloud2) -> None:
        # First job: register that a cloud arrived, for the heartbeat log.
        self._cloud_counter += 1
        self._last_cloud_time = self.get_clock().now()

        # Iterate once, vectorize the rest. Convert structured array → (N,3) floats.
        pts = point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
        # read_points returns a structured numpy array in humble+; materialize as 2D.
        if pts.size == 0:
            if self._frame_count % 10 == 0:
                self.get_logger().warn("point cloud arrived but had 0 usable points")
            return
        arr = np.stack([pts["x"], pts["y"], pts["z"]], axis=-1).astype(np.float32)
        # Apply the configurable axis mapping immediately so the rest of the
        # pipeline can assume "+x forward, +y left, +z up".
        xs, ys, zs = self._remap(arr)

        # ── Strip invalid / blind returns ─────────────────────────────────
        # Unitree's own preprocessing config (point_lio_unilidar) uses
        # `blind: 0.5` — any point within 0.5 m of the sensor is dropped.
        # That's our knob (configurable via `blind_radius` param). Catches:
        #   • degenerate (0, 0, z) sentinels
        #   • concentric rings at r ≈ 0.05, 0.15 m seen on some firmware
        #   • actual self-returns from the robot's own body
        horiz_r2 = xs * xs + ys * ys
        invalid_mask = horiz_r2 < self.blind_r2
        n_invalid = int(invalid_mask.sum())
        if n_invalid > 0:
            valid = ~invalid_mask
            xs, ys, zs = xs[valid], ys[valid], zs[valid]
            arr = arr[valid]

        # If everything got filtered (e.g., lens covered, every beam is a
        # sentinel), publish the clear-sentinel and skip the rest of the
        # pipeline — the diagnostics and mask code below all assume non-empty
        # arrays and np.min will ValueError otherwise.
        if xs.size == 0:
            if self.debug and self._frame_count % 20 == 0:
                self.get_logger().warn(
                    f"cloud had {n_invalid} points, all invalid — nothing "
                    "to process. (Lens covered? Sensor obscured?)"
                )
            self._publish(self.x_max, self.x_max, self.x_max)
            self._frame_count += 1
            return

        # One-time geometry dump: what are the axes actually pointing at?
        # This is the fastest way to spot a frame-convention mismatch.
        if not self._diagnostic_logged:
            self._log_cloud_geometry(xs, ys, zs)
            self._diagnostic_logged = True

        # Height filter first (cheapest to apply, biggest reduction typically).
        z_mask = (zs > self.z_min) & (zs < self.z_max)
        # Range filter along (remapped) forward axis.
        x_mask = (xs > self.x_min) & (xs < self.x_max)
        base_mask = z_mask & x_mask
        if not base_mask.any():
            if self.debug and self._frame_count % 20 == 0:
                self.get_logger().warn(
                    f"no points survive base filter: z_mask={int(z_mask.sum())}/{len(zs)}, "
                    f"x_mask={int(x_mask.sum())}/{len(xs)} — "
                    "check z_min/z_max and axis mapping"
                )
            self._publish(self.x_max, self.x_max, self.x_max)
            self._frame_count += 1
            return
        bx, by = xs[base_mask], ys[base_mask]

        # Three corridors.
        fwd_mask = np.abs(by) <= self.half_w
        lft_mask = (by > self.half_w) & (by <= self.half_w + self.side_w)
        rgt_mask = (by < -self.half_w) & (by >= -(self.half_w + self.side_w))

        fwd = self._corridor_min(bx[fwd_mask])
        lft = self._corridor_min(bx[lft_mask])
        rgt = self._corridor_min(bx[rgt_mask])

        self._publish(fwd, lft, rgt)
        # Publish the filtered points (in raw/sensor frame) for rviz viewing.
        raw_base = arr[base_mask]
        self._publish_debug_cloud(msg.header, raw_base)
        if self.show:
            # Hand the latest snapshot to the display timer (main thread).
            self._viz_data = {
                "all_x": xs, "all_y": ys,
                "kept_x": bx, "kept_y": by,
                "fwd_mask": fwd_mask, "lft_mask": lft_mask, "rgt_mask": rgt_mask,
                "fwd": float(self._fwd), "lft": float(self._lft), "rgt": float(self._rgt),
            }

        # Diagnostics.
        self._frame_count += 1
        if self.debug and self._frame_count % 20 == 0:
            # Also report the absolute-closest points across any direction
            # with z in our height window — this is what you want to see
            # when tuning x_min or diagnosing "why is clearance always X".
            near_by_z = zs[z_mask]
            near_xs = xs[z_mask]
            near_ys = ys[z_mask]
            if near_xs.size:
                dist = np.sqrt(near_xs**2 + near_ys**2)
                k = min(5, dist.size)
                idx = np.argpartition(dist, k - 1)[:k]
                idx = idx[np.argsort(dist[idx])]
                closest = ", ".join(
                    f"({near_xs[i]:+.2f},{near_ys[i]:+.2f},{near_by_z[i]:+.2f},d={dist[i]:.2f})"
                    for i in idx
                )
            else:
                closest = "none"
            self.get_logger().info(
                f"clearance fwd={fwd:.2f} lft={lft:.2f} rgt={rgt:.2f} | "
                f"corridor pts fwd={int(fwd_mask.sum())} "
                f"lft={int(lft_mask.sum())} rgt={int(rgt_mask.sum())} | "
                f"raw={len(arr) + n_invalid} invalid={n_invalid} "
                f"post-zx-filter={int(base_mask.sum())}"
            )
            self.get_logger().info(
                f"  nearest-5 at proper height (fwd,side,up,dist): {closest}"
            )
            # What are the 3 closest points INSIDE the forward corridor? If
            # fwd clearance is pinned at x_min, these are what's pinning it.
            fwd_xs = bx[fwd_mask]
            fwd_ys = by[fwd_mask]
            if fwd_xs.size:
                k = min(3, fwd_xs.size)
                idx = np.argpartition(fwd_xs, k - 1)[:k]
                idx = idx[np.argsort(fwd_xs[idx])]
                fc = ", ".join(
                    f"(fwd={fwd_xs[i]:.2f}, side={fwd_ys[i]:+.2f})"
                    for i in idx
                )
                self.get_logger().info(f"  nearest-3 in forward corridor: {fc}")

    def _corridor_min(self, xs_in_corridor: np.ndarray) -> float:
        """Return the nearest-point x in this corridor, or x_max when empty/sparse."""
        if xs_in_corridor.size < self.min_points:
            return self.x_max
        return float(xs_in_corridor.min())

    def _publish(self, fwd: float, lft: float, rgt: float) -> None:
        # Per-corridor EMA. Re-init from raw on first sample.
        self._fwd = fwd if self._fwd is None else self._ema(self._fwd, fwd)
        self._lft = lft if self._lft is None else self._ema(self._lft, lft)
        self._rgt = rgt if self._rgt is None else self._ema(self._rgt, rgt)
        self.pub.publish(
            Vector3(x=float(self._fwd), y=float(self._lft), z=float(self._rgt))
        )

    def _ema(self, prev: float, new: float) -> float:
        return self.alpha * new + (1.0 - self.alpha) * prev

    def _publish_debug_cloud(
        self, header: Header, raw_xyz: np.ndarray
    ) -> None:
        """Publish the points that survived the z+x filter so rviz2 can show them.

        Published in the original sensor frame (header copied from input) so
        it overlays correctly on /utlidar/cloud in rviz2.
        """
        if self.debug_pub.get_subscription_count() == 0:
            return  # nobody listening, skip the allocation
        pts = [(float(x), float(y), float(z)) for x, y, z in raw_xyz]
        cloud_msg = point_cloud2.create_cloud_xyz32(header, pts)
        self.debug_pub.publish(cloud_msg)

    def _display_tick(self) -> None:
        """Main-thread OpenCV render. Top-down bird's-eye view."""
        snap = self._viz_data
        if snap is None:
            cv2.waitKey(1)
            return

        n = self.view_pixels
        # px_per_m scales world meters → image pixels. Robot is drawn at the
        # bottom-center of the image, so the full self.view_range fits in the
        # vertical span; horizontally we center on the robot and show
        # ±view_range/1.0 wide.
        px_per_m = (n - 20) / self.view_range  # leave a small margin
        img = np.zeros((n, n, 3), dtype=np.uint8)

        # Robot is at (cx, cy_bot) — bottom-center — with +x going up.
        cx = n // 2
        cy_bot = n - 30

        def to_px(x: float, y: float) -> "tuple[int, int] | None":
            # Canonical frame after _remap: x forward, y left. Image:
            # up = forward, left = +y (robot's left).
            u = int(cx - y * px_per_m)
            v = int(cy_bot - x * px_per_m)
            if 0 <= u < n and 0 <= v < n:
                return u, v
            return None

        # Range rings every 0.5 m, a bit brighter so you can read distances.
        for r in np.arange(0.5, self.view_range + 0.01, 0.5):
            r_px = int(r * px_per_m)
            cv2.circle(img, (cx, cy_bot), r_px, (55, 55, 55), 1)
            # Label each ring on the right side.
            cv2.putText(
                img, f"{r:.1f}m", (cx + r_px + 2, cy_bot - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1, cv2.LINE_AA,
            )
        # Axes.
        cv2.line(img, (cx, cy_bot), (cx, 0), (55, 55, 55), 1)
        cv2.line(img, (0, cy_bot), (n, cy_bot), (55, 55, 55), 1)

        # Dead-zone: the part of space < x_min where self-returns live.
        # Draw as a red semi-transparent band so you visually know "anything
        # in here is ignored."
        overlay = img.copy()
        deadzone_corners = [
            to_px(0.0, -self.half_w - self.side_w - 0.5),
            to_px(self.x_min, -self.half_w - self.side_w - 0.5),
            to_px(self.x_min, +self.half_w + self.side_w + 0.5),
            to_px(0.0, +self.half_w + self.side_w + 0.5),
        ]
        if all(p is not None for p in deadzone_corners):
            cv2.fillPoly(overlay, [np.array(deadzone_corners, dtype=np.int32)],
                         (40, 40, 80))
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
        # Dead-zone label.
        dz = to_px(self.x_min / 2, 0.0)
        if dz:
            cv2.putText(
                img, "DEAD ZONE", (dz[0] - 35, dz[1] + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 180), 1, cv2.LINE_AA,
            )

        # Corridors. Outlined, not filled, so they don't swamp the points.
        def corridor_outline(
            x_lo: float, x_hi: float, y_lo: float, y_hi: float, color
        ) -> None:
            pts = []
            for (x, y) in [(x_lo, y_lo), (x_lo, y_hi), (x_hi, y_hi), (x_hi, y_lo)]:
                pt = to_px(x, y)
                if pt is None:
                    return
                pts.append(pt)
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, color, 1)

        corridor_outline(
            self.x_min, self.x_max, -self.half_w, +self.half_w, (0, 200, 0)
        )
        corridor_outline(
            self.x_min, self.x_max, +self.half_w, +self.half_w + self.side_w,
            (0, 160, 200),
        )
        corridor_outline(
            self.x_min, self.x_max, -(self.half_w + self.side_w), -self.half_w,
            (200, 160, 0),
        )

        # All raw points that passed z filter: white dots. Hide those failing
        # z filter entirely — they were never going to matter and just add noise.
        # (Old behavior showed rejected points in grey; now we drop them.)
        for x, y in zip(snap["kept_x"], snap["kept_y"]):
            pt = to_px(float(x), float(y))
            if pt:
                cv2.circle(img, pt, 1, (180, 180, 180), -1)

        # Corridor-specific points on top, colored.
        colors = [
            (snap["fwd_mask"], (0, 255, 0)),    # green = forward
            (snap["lft_mask"], (0, 200, 255)),  # orange = left
            (snap["rgt_mask"], (255, 200, 0)),  # cyan-blue = right
        ]
        for mask, color in colors:
            cx_arr = snap["kept_x"][mask]
            cy_arr = snap["kept_y"][mask]
            for x, y in zip(cx_arr, cy_arr):
                pt = to_px(float(x), float(y))
                if pt:
                    cv2.circle(img, pt, 2, color, -1)

        # Big crosshair on the nearest obstacle in each corridor (if any).
        def mark_min(mask, color, label_text: str) -> None:
            xs = snap["kept_x"][mask]
            ys = snap["kept_y"][mask]
            if xs.size < self.min_points:
                return
            i = int(np.argmin(xs))
            pt = to_px(float(xs[i]), float(ys[i]))
            if pt is None:
                return
            u, v = pt
            cv2.line(img, (u - 10, v), (u + 10, v), color, 2)
            cv2.line(img, (u, v - 10), (u, v + 10), color, 2)
            cv2.circle(img, (u, v), 12, color, 2)
            cv2.putText(
                img, f"{label_text} {float(xs[i]):.2f}m",
                (u + 14, v - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

        mark_min(snap["fwd_mask"], (0, 255, 0), "fwd")
        mark_min(snap["lft_mask"], (0, 200, 255), "lft")
        mark_min(snap["rgt_mask"], (255, 200, 0), "rgt")

        # Robot marker — a triangle pointing forward.
        tri = np.array(
            [
                [cx, cy_bot - 14],
                [cx - 8, cy_bot + 6],
                [cx + 8, cy_bot + 6],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(img, [tri], (255, 255, 255))

        # Text overlay with clearance values.
        def label(line: int, text: str, color) -> None:
            cv2.putText(
                img, text, (10, 25 + line * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )
        label(0, f"fwd = {snap['fwd']:.2f} m", (0, 255, 0))
        label(1, f"lft = {snap['lft']:.2f} m", (0, 200, 255))
        label(2, f"rgt = {snap['rgt']:.2f} m", (255, 200, 0))
        # Also display number of contributing points so you can tell stale
        # sentinel readings from real ones at a glance.
        cv2.putText(
            img,
            f"pts f/l/r = {int(snap['fwd_mask'].sum())}/"
            f"{int(snap['lft_mask'].sum())}/{int(snap['rgt_mask'].sum())}",
            (10, n - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
        )

        cv2.imshow("go2_lidar_avoidance", img)
        cv2.waitKey(1)

    def _log_cloud_geometry(
        self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray
    ) -> None:
        """One-shot dump of the cloud's min/max in each axis and a z histogram.

        Tells us whether the frame is what we think it is:
          • x in [0, +several]     → forward axis, good
          • x symmetric around 0   → x is horizontal (left/right), WRONG axis
          • z mostly negative      → lidar frame has floor below (expected)
          • all very small values  → cloud is in cm not m, or something is off
        """
        def stats(name: str, a: np.ndarray) -> str:
            return (
                f"{name}: n={len(a)} "
                f"min={a.min():+.2f} max={a.max():+.2f} "
                f"mean={a.mean():+.2f} std={a.std():.2f}"
            )

        self.get_logger().info("── first cloud geometry (after axis remap) ──")
        self.get_logger().info(stats("forward", xs))
        self.get_logger().info(stats("side   ", ys))
        self.get_logger().info(stats("up     ", zs))

        # Histogram on z to help pick z_min/z_max.
        edges = np.linspace(-1.5, 2.0, 15)
        counts, _ = np.histogram(zs, bins=edges)
        lines = [
            f"  z∈[{edges[i]:+.2f},{edges[i+1]:+.2f}) : {counts[i]}"
            for i in range(len(counts))
            if counts[i] > 0
        ]
        self.get_logger().info(
            "z distribution (tune z_min/z_max so the 'floor' bin is excluded):\n"
            + "\n".join(lines)
        )

        # Sanity check: how many points land in the forward lane *before* any
        # height filtering? If this is zero, your forward axis isn't x.
        fwd_pre = np.sum(
            (xs > self.x_min) & (xs < self.x_max) & (np.abs(ys) <= self.half_w)
        )
        self.get_logger().info(
            f"points in forward (x∈[{self.x_min},{self.x_max}], "
            f"|y|<{self.half_w}) BEFORE z filter: {int(fwd_pre)}"
        )
        if fwd_pre == 0:
            self.get_logger().warn(
                "0 points in forward corridor pre-z-filter. The axis "
                "mapping is probably wrong. Inspect the forward/side/up "
                "stats above — 'forward' should have a strong positive "
                "extent; 'side' should be roughly symmetric around zero."
            )
        self.get_logger().info("──────────────────────────────────────────────")


def main() -> int:
    rclpy.init()
    node = Go2LidarAvoidance()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.show:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
