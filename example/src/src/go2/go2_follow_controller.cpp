#!/usr/bin/env python3
"""go2_perception — front-camera person detector for the follow controller.

Subscribes:
  Go2FrontVideoData  (unitree_go/msg/Go2FrontVideoData)
  Empirically only `video720p` is populated by the Go2 firmware, and the
  payload is H.264 NAL data with a small (~4 byte) proprietary prefix —
  FFmpeg's H.264 parser skips past the prefix automatically by scanning
  for the next start code (00 00 00 01).

Publishes:
  /follow/target     (geometry_msgs/msg/Vector3Stamped)
    x = normalized bearing in [-1, 1]   (0 = centered, +ve = person to the right)
    y = monocular distance estimate (m) (calibrate `k_distance` for your camera)
    z = unused

Run:
  ros2 run unitree_ros2_example go2_perception
  ros2 run unitree_ros2_example go2_perception --ros-args \
      -p camera_topic:=/frontvideostream -p show_window:=true
"""
import subprocess
import sys
import threading

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from ultralytics import YOLO

from unitree_go.msg import Go2FrontVideoData


_VIDEO_FIELDS = ("video720p", "video360p", "video180p")


def _split_nals(stream: bytes) -> "list[bytes]":
    """Split an H.264 annex-B byte stream into individual NAL units.

    Each returned chunk starts with a start code (00 00 00 01 or 00 00 01)
    and runs up to (but not including) the next start code.
    """
    units = []
    i = 0
    n = len(stream)
    while i < n:
        # Find the next start code after the one we're sitting on.
        nxt4 = stream.find(b"\x00\x00\x00\x01", i + 3)
        nxt3 = stream.find(b"\x00\x00\x01", i + 3)
        if nxt4 < 0 and nxt3 < 0:
            units.append(stream[i:])
            break
        nxt = min(x for x in (nxt4, nxt3) if x >= 0)
        units.append(stream[i:nxt])
        i = nxt
    return units


class Go2Perception(Node):
    def __init__(self) -> None:
        super().__init__("go2_perception")

        self.declare_parameter("camera_topic", "/frontvideostream")
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("device", "cuda")  # override with "cpu" or "cuda:0"
        self.declare_parameter("conf_threshold", 0.4)
        # distance(m) ≈ k_distance / bbox_height_px — calibrate at a known distance.
        self.declare_parameter("k_distance", 736.0)
        self.declare_parameter("show_window", False)
        # If non-empty, perception ignores DDS and pulls frames from this URL.
        # Anything OpenCV/FFmpeg can open: udp://, rtsp://, http://, file path.
        # Recommended: raw H.264 over UDP from the Jetson, e.g. "udp://0.0.0.0:5000".
        self.declare_parameter("stream_url", "")
        self.declare_parameter("stream_fps", 30.0)
        self.declare_parameter("process_every_n", 3)
        self.process_every_n = int(self.get_parameter("process_every_n").value)

        self.conf = float(self.get_parameter("conf_threshold").value)
        self.k_dist = float(self.get_parameter("k_distance").value)
        self.show = bool(self.get_parameter("show_window").value)
        self.device = str(self.get_parameter("device").value)
        stream_url = str(self.get_parameter("stream_url").value)

        model_path = self.get_parameter("model_path").value
        self.get_logger().info(f"loading YOLO model: {model_path} on {self.device}")
        self.model = YOLO(model_path)

        self.pub = self.create_publisher(Vector3Stamped, "/follow/target", 1)
        self.frame_count = 0
        self._diagnostic_logged = False
        self._seen_nal_types: set[int] = set()
        self._decode_warn_count = 0

        if stream_url:
            self._setup_stream(stream_url, float(self.get_parameter("stream_fps").value))
        else:
            self._setup_dds()

    def _setup_stream(self, url: str, fps: float) -> None:
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open stream: {url}")
        # Keep only the latest frame in OpenCV's internal queue — without this
        # we can build up seconds of latency on a low-fps consumer.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        period_s = 1.0 / max(fps, 1.0)
        self.timer = self.create_timer(period_s, self._poll_stream)
        self.get_logger().info(
            f"stream mode: {url} @ {fps:.0f} Hz; publishing /follow/target"
        )

    def _poll_stream(self) -> None:
        # Drain one extra frame: cap.grab() is decode-free, so we skip a stale
        # frame and decode only the freshest one with cap.retrieve().
        self.cap.grab()
        ok, img = self.cap.retrieve()
        if not ok:
            self.get_logger().warn("stream read failed")
            return
        stamp = self.get_clock().now().to_msg()
        self.process_frame(img, stamp)

    def _setup_dds(self) -> None:
        topic = self.get_parameter("camera_topic").value
        # Hardcoded for the Go2 720p front camera — adjust if the firmware
        # ever switches resolution.
        self.frame_w = 1280
        self.frame_h = 720
        self.frame_size = self.frame_w * self.frame_h * 3  # bgr24

        # Spawn an ffmpeg subprocess: H.264 in on stdin, raw BGR24 out on stdout.
        self.ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel", "warning",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-f", "h264",
                "-i", "pipe:0",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-an",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # let ffmpeg log to the parent's stderr so we can see it
            bufsize=0,
        )
        self._decoder_thread = threading.Thread(
            target=self._decoder_loop, daemon=True
        )
        self._decoder_thread.start()

        video_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(
            Go2FrontVideoData, topic, self.on_video, video_qos
        )
        self.get_logger().info(
            f"DDS mode: subscribed to {topic}; publishing /follow/target"
        )

    def on_video(self, msg: Go2FrontVideoData) -> None:
        if not self._diagnostic_logged:
            self._log_field_diagnostics(msg)
            self._diagnostic_logged = True

        # Pick the first non-empty field. Empirically only video720p is
        # populated, but we don't hardcode it in case firmware behavior changes.
        payload = b""
        for field_name in _VIDEO_FIELDS:
            payload = bytes(getattr(msg, field_name))
            if payload:
                break
        if not payload:
            return

        # Strip Go2's proprietary prefix bytes by jumping to the first NAL start code.
        nal_start = payload.find(b"\x00\x00\x00\x01")
        if nal_start < 0:
            nal_start = payload.find(b"\x00\x00\x01")
        if nal_start < 0:
            return
        nal_stream = payload[nal_start:]

        # Log the NAL types we see (first occurrence of each).
        for nal_unit in _split_nals(nal_stream):
            if len(nal_unit) >= 5:
                sc_len = 4 if nal_unit.startswith(b"\x00\x00\x00\x01") else 3
                self._note_nal_type(nal_unit[sc_len] & 0x1F)

        # Hand the whole NAL stream to ffmpeg; it'll emit decoded BGR24 frames
        # on stdout when it has them, picked up by _decoder_loop in another thread.
        try:
            self.ffmpeg.stdin.write(nal_stream)
        except BrokenPipeError:
            self.get_logger().error("ffmpeg subprocess died — restart the node")

    def _decoder_loop(self) -> None:
        """Pull decoded BGR24 frames out of ffmpeg's stdout, one at a time."""
        while True:
            buf = b""
            while len(buf) < self.frame_size:
                chunk = self.ffmpeg.stdout.read(self.frame_size - len(buf))
                if not chunk:
                    return  # ffmpeg exited
                buf += chunk
            img = np.frombuffer(buf, dtype=np.uint8).reshape(
                self.frame_h, self.frame_w, 3
            )
            stamp = self.get_clock().now().to_msg()
            self.process_frame(img, stamp)

    def _note_nal_type(self, nal_type: int) -> None:
        # NAL types: 1=non-IDR slice (P), 5=IDR (I), 7=SPS, 8=PPS, 6=SEI.
        # Log the first occurrence of each so we can confirm SPS/PPS/IDR arrive.
        if nal_type not in self._seen_nal_types:
            self._seen_nal_types.add(nal_type)
            name = {1: "P-slice", 5: "IDR (I-frame)", 6: "SEI",
                    7: "SPS", 8: "PPS", 9: "AUD"}.get(nal_type, "other")
            self.get_logger().info(f"first NAL type {nal_type} ({name})")

    def _log_field_diagnostics(self, msg: Go2FrontVideoData) -> None:
        for field_name in _VIDEO_FIELDS:
            data = bytes(getattr(msg, field_name))
            head = data[:16].hex(" ")
            self.get_logger().info(
                f"first frame {field_name}: len={len(data)} head={head}"
            )

    def process_frame(self, img: np.ndarray, frame_stamp=None) -> None:
        if frame_stamp is None:
            frame_stamp = self.get_clock().now().to_msg()

        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0:
            return

        h, w = img.shape[:2]

        results = self.model.predict(
            img, classes=[0], conf=self.conf, device=self.device, verbose=False
        )

        if not results:
            return

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            self._maybe_show(img, None)
            return

        xywh = boxes.xywh.cpu().numpy()
        idx = int(np.argmax(xywh[:, 2] * xywh[:, 3]))

        cx, _cy, _bw, bh = xywh[idx]

        bearing = float((cx - w / 2.0) / (w / 2.0))
        distance = float(self.k_dist / max(bh, 1.0))

        msg = Vector3Stamped()
        msg.header.stamp = frame_stamp
        msg.header.frame_id = "front_camera"
        msg.vector.x = bearing
        msg.vector.y = distance
        msg.vector.z = 0.0

        self.pub.publish(msg)

        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"target: bearing={bearing:+.2f}  distance={distance:.2f} m  "
                f"({len(boxes)} ppl, bbox_h={bh:.0f}px)"
            )

        self._maybe_show(img, boxes.xyxy[idx].cpu().numpy().astype(int))

    def _maybe_show(self, img: np.ndarray, xyxy) -> None:
        if not self.show:
            return
        if xyxy is not None:
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("go2_perception", img)
        cv2.waitKey(1)


def main() -> int:
    rclpy.init()
    node = Go2Perception()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.show:
            cv2.destroyAllWindows()
        if hasattr(node, "cap"):
            node.cap.release()
        if hasattr(node, "ffmpeg"):
            try:
                node.ffmpeg.stdin.close()
            except Exception:
                pass
            node.ffmpeg.terminate()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
