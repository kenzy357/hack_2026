#!/usr/bin/env python3
"""go2_perception — front-camera person detector for the follow controller.

Subscribes:
  Go2FrontVideoData  (unitree_go/msg/Go2FrontVideoData)
  Empirically only `video720p` is populated by the Go2 firmware, and the
  payload is H.264 NAL data with a small (~4 byte) proprietary prefix —
  FFmpeg's H.264 parser skips past the prefix automatically by scanning
  for the next start code (00 00 00 01).

Publishes:
  /follow/target     (geometry_msgs/msg/Vector3)
    x = normalized bearing in [-1, 1]   (0 = centered, +ve = person to the right)
    y = monocular distance estimate (m) (calibrate `k_distance` for your camera)
    z = unused

Run:
  ros2 run unitree_ros2_example go2_perception
  ros2 run unitree_ros2_example go2_perception --ros-args \
      -p camera_topic:=/frontvideostream -p show_window:=true
"""
import queue
import subprocess
import sys
import threading

import cv2
import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Vector3
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
        # "auto" → CUDA if available, else CPU. Override with "cpu", "cuda",
        # or a specific device like "cuda:0".
        self.declare_parameter("device", "auto")
        self.declare_parameter("conf_threshold", 0.5)
        # distance(m) ≈ k_distance / bbox_height_px — calibrate at a known distance.
        self.declare_parameter("k_distance", 736.0)
        # Clamp published distance into a sane physical range.
        self.declare_parameter("min_distance", 0.3)
        self.declare_parameter("max_distance", 8.0)
        # EMA smoothing on bearing/distance before publishing.
        self.declare_parameter("ema_alpha", 0.3)
        # Max frames we'll coast on the last known target when detection misses.
        self.declare_parameter("coast_frames", 9)  # ~0.3 s at 30 fps
        # When re-acquiring a lost lock, only accept candidates whose center is
        # within this fraction of frame-diagonal from the last known position.
        self.declare_parameter("reacquire_radius", 0.2)
        self.declare_parameter("show_window", False)
        # If non-empty, perception ignores DDS and pulls frames from this URL.
        # Anything OpenCV/FFmpeg can open: udp://, rtsp://, http://, file path.
        # Recommended: raw H.264 over UDP from the Jetson, e.g. "udp://0.0.0.0:5000".
        self.declare_parameter("stream_url", "")
        self.declare_parameter("stream_fps", 30.0)

        self.conf = float(self.get_parameter("conf_threshold").value)
        self.k_dist = float(self.get_parameter("k_distance").value)
        self.min_dist = float(self.get_parameter("min_distance").value)
        self.max_dist = float(self.get_parameter("max_distance").value)
        self.ema_alpha = float(self.get_parameter("ema_alpha").value)
        self.coast_frames_max = int(self.get_parameter("coast_frames").value)
        self.reacquire_radius = float(self.get_parameter("reacquire_radius").value)
        self.show = bool(self.get_parameter("show_window").value)
        self.device = self._resolve_device(str(self.get_parameter("device").value))
        stream_url = str(self.get_parameter("stream_url").value)

        model_path = self.get_parameter("model_path").value
        self.get_logger().info(f"loading YOLO model: {model_path} on {self.device}")
        self.model = YOLO(model_path)

        self.pub = self.create_publisher(Vector3, "/follow/target", 10)
        self.frame_count = 0
        self._diagnostic_logged = False
        self._seen_nal_types: set[int] = set()
        self._decode_warn_count = 0
        # Tracker state: which ByteTrack ID we're currently locked onto.
        # None until we see a first stable ID; re-acquired when the ID is lost.
        self._locked_id: int | None = None
        # Last-known target geometry (used for spatial re-acquire + coast).
        self._last_cx: float | None = None
        self._last_cy: float | None = None
        self._last_bh: float | None = None
        self._last_bearing: float | None = None
        self._last_distance: float | None = None
        # Count of consecutive frames we've coasted without a detection.
        self._coast_frames = 0

        # Display must live on the main thread: cv2.imshow / waitKey from a
        # worker thread is a classic silent-no-op on Linux. Detection threads
        # drop the latest annotated frame into this slot; a ROS timer (which
        # runs on the main rclpy thread) picks it up and renders.
        self._display_slot: "queue.Queue[np.ndarray] | None" = (
            queue.Queue(maxsize=1) if self.show else None
        )
        if self.show:
            self._display_timer = self.create_timer(1.0 / 30.0, self._display_tick)

        if stream_url:
            self._setup_stream(stream_url, float(self.get_parameter("stream_fps").value))
        else:
            self._setup_dds()

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        self.get_logger().warn(
            "CUDA not available — falling back to CPU. "
            "Expect single-digit FPS on the YOLO path."
        )
        return "cpu"

    def _setup_stream(self, url: str, fps: float) -> None:
        # NOTE: cv2.VideoCapture is deliberately avoided here. It silently
        # buffers many frames inside OpenCV's FFmpeg backend (CAP_PROP_BUFFERSIZE
        # is ignored for the FFmpeg backend), which adds 0.5–2 s of latency.
        # Instead we spawn ffmpeg as a subprocess with explicit low-latency
        # flags and pipe raw BGR24 frames out of stdout — same pattern as the
        # DDS path. A reader thread keeps draining the pipe; a worker thread
        # only ever processes the latest frame (a 1-slot queue drops stale ones).

        # Resolution must match what the Jetson is encoding. Edit if you change
        # -video_size in the Jetson's ffmpeg / systemd unit.
        self.stream_w = 640
        self.stream_h = 480
        self._stream_frame_size = self.stream_w * self.stream_h * 3

        self._stream_proc = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel", "warning",
                "-fflags", "nobuffer+discardcorrupt",
                "-flags", "low_delay",
                "-probesize", "32",
                "-analyzeduration", "0",
                "-i", url,
                "-vf", f"scale={self.stream_w}:{self.stream_h}",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-an",
                "pipe:1",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )

        # Single-slot queue: reader puts the freshest frame; if the worker is
        # busy when a new frame arrives, the old slot value is discarded.
        self._frame_slot: queue.Queue = queue.Queue(maxsize=1)

        self._reader_thread = threading.Thread(
            target=self._stream_reader_loop, daemon=True
        )
        self._reader_thread.start()
        self._worker_thread = threading.Thread(
            target=self._stream_worker_loop, daemon=True
        )
        self._worker_thread.start()

        self.get_logger().info(
            f"stream mode (subprocess): {url}; publishing /follow/target"
        )

    def _stream_reader_loop(self) -> None:
        """Read BGR24 frames from ffmpeg's stdout, push freshest into the slot."""
        while True:
            buf = b""
            while len(buf) < self._stream_frame_size:
                chunk = self._stream_proc.stdout.read(
                    self._stream_frame_size - len(buf)
                )
                if not chunk:
                    return  # ffmpeg exited
                buf += chunk
            img = np.frombuffer(buf, dtype=np.uint8).reshape(
                self.stream_h, self.stream_w, 3
            ).copy()
            # Drop any frame that's still in the slot, then put the new one.
            try:
                self._frame_slot.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_slot.put_nowait(img)
            except queue.Full:
                pass  # racing reader; whatever, next frame comes soon

    def _stream_worker_loop(self) -> None:
        """Pull the freshest frame and run perception on it."""
        while True:
            img = self._frame_slot.get()  # blocks until reader provides one
            self.process_frame(img)

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
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
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
            self.process_frame(img)

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

    def process_frame(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        # ByteTrack (via model.track) gives persistent IDs between frames so we
        # can lock onto one person instead of re-picking the largest bbox each
        # frame (which would hop to whoever walks in closest).
        results = self.model.track(
            img,
            classes=[0],
            conf=self.conf,
            device=self.device,
            verbose=False,
            persist=True,
            tracker="bytetrack.yaml",
        )
        if not results:
            self._handle_miss(img)
            return
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            # No detections (or tracker hasn't committed IDs yet). Coast briefly.
            self._handle_miss(img)
            return

        ids = boxes.id.cpu().numpy().astype(int)
        xywh = boxes.xywh.cpu().numpy()  # (N, 4): cx, cy, w, h

        idx = self._select_target(ids, xywh, w, h)
        if idx is None:
            self._handle_miss(img)
            return

        cx, cy, _bw, bh = xywh[idx]
        bearing_raw = float((cx - w / 2.0) / (w / 2.0))
        distance_raw = float(self.k_dist / max(bh, 1.0))
        distance_raw = max(self.min_dist, min(self.max_dist, distance_raw))

        # EMA on the published values. Reinit from raw when we just re-acquired.
        if self._last_bearing is None:
            bearing, distance = bearing_raw, distance_raw
        else:
            a = self.ema_alpha
            bearing = a * bearing_raw + (1.0 - a) * self._last_bearing
            distance = a * distance_raw + (1.0 - a) * self._last_distance

        self._last_bearing = bearing
        self._last_distance = distance
        self._last_cx = float(cx)
        self._last_cy = float(cy)
        self._last_bh = float(bh)
        self._coast_frames = 0

        self.pub.publish(Vector3(x=bearing, y=distance, z=0.0))

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"target id={self._locked_id}: bearing={bearing:+.2f}  "
                f"distance={distance:.2f} m  ({len(boxes)} ppl, bbox_h={bh:.0f}px)"
            )

        self._maybe_show(img, boxes.xyxy[idx].cpu().numpy().astype(int))

    def _select_target(
        self,
        ids: np.ndarray,
        xywh: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> "int | None":
        """Return index into xywh of the target we're tracking, or None.

        Rules:
          1. If our locked ID is still present, pick it.
          2. Otherwise re-acquire: prefer candidates close to the last known
             position (within reacquire_radius * frame diagonal) with similar
             bbox height. Fall back to largest bbox if we have no history.
        """
        # Case 1: locked ID still visible.
        if self._locked_id is not None and self._locked_id in ids:
            return int(np.where(ids == self._locked_id)[0][0])

        # Case 2: re-acquire. Prefer spatial continuity with the last seen pose.
        if self._last_cx is not None and self._last_bh is not None:
            diag = float(np.hypot(frame_w, frame_h))
            radius_px = self.reacquire_radius * diag
            dx = xywh[:, 0] - self._last_cx
            dy = xywh[:, 1] - self._last_cy
            dist_px = np.hypot(dx, dy)
            # Height ratio close to 1 means same-sized person, likely same one.
            h_ratio = np.minimum(xywh[:, 3], self._last_bh) / np.maximum(
                xywh[:, 3], self._last_bh
            )
            mask = (dist_px < radius_px) & (h_ratio > 0.6)
            if mask.any():
                # Among plausible candidates, take the closest in pixel space.
                cand = int(np.argmin(np.where(mask, dist_px, np.inf)))
                self._locked_id = int(ids[cand])
                self.get_logger().info(
                    f"re-acquired target as track id {self._locked_id} "
                    f"({dist_px[cand]:.0f}px from last seen)"
                )
                return cand
            # Nothing plausible — keep coasting rather than jumping to a stranger.
            return None

        # Cold start: pick the largest bbox.
        self._locked_id = int(ids[np.argmax(xywh[:, 2] * xywh[:, 3])])
        self.get_logger().info(f"locked onto track id {self._locked_id}")
        return int(np.where(ids == self._locked_id)[0][0])

    def _handle_miss(self, img: np.ndarray) -> None:
        """No usable detection this frame — coast on the last known target."""
        if self._last_bearing is None or self._coast_frames >= self.coast_frames_max:
            # Either nothing ever locked, or we've coasted too long — let the
            # controller's own timeout trip and StopMove.
            if self._last_bearing is not None:
                self._last_bearing = None
                self._last_distance = None
                self._locked_id = None
                self.get_logger().warn(
                    f"target lost after {self._coast_frames} coast frames"
                )
            self._maybe_show(img, None)
            return
        # Republish the last smoothed estimate so the controller keeps its
        # downstream state machine happy during brief occlusions.
        self.pub.publish(
            Vector3(x=self._last_bearing, y=self._last_distance, z=0.0)
        )
        self._coast_frames += 1
        self._maybe_show(img, None)

    def _maybe_show(self, img: np.ndarray, xyxy) -> None:
        """Prep an annotated frame and hand it to the main thread for display.

        Runs on the detection thread. cv2.imshow / waitKey MUST NOT be called
        here — they only work on the thread that owns the GUI event loop
        (on Linux, that's the main rclpy thread).
        """
        if not self.show or self._display_slot is None:
            return
        # Copy so the detection thread is free to keep mutating the source buffer.
        annotated = img.copy()
        if xyxy is not None:
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if self._locked_id is not None:
            cv2.putText(
                annotated,
                f"id={self._locked_id}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        # Drop any pending frame (we only care about the newest) then enqueue.
        try:
            self._display_slot.get_nowait()
        except queue.Empty:
            pass
        try:
            self._display_slot.put_nowait(annotated)
        except queue.Full:
            pass

    def _display_tick(self) -> None:
        """Main-thread timer: drain the display slot and render."""
        if self._display_slot is None:
            return
        try:
            img = self._display_slot.get_nowait()
        except queue.Empty:
            # Still need to pump the GUI event loop even if there's no new frame.
            cv2.waitKey(1)
            return
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
        if hasattr(node, "_stream_proc"):
            try:
                if node._stream_proc.stdin is not None:
                    node._stream_proc.stdin.close()
            except Exception:
                pass
            node._stream_proc.terminate()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
