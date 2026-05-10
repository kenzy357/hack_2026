#!/usr/bin/env python3

import json
import os
import queue
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import sounddevice as sd
from vosk import Model, KaldiRecognizer


SAMPLE_RATE = 16000
MIC_DEVICE = None

COMMAND_COOLDOWN = 3.0

DEFAULT_MODEL_PATH = os.environ.get(
    "VOSK_MODEL_PATH",
    "/home/parallels/go2_dimos/ws/vosk-model-small-en-us-0.15",
)

VOICE_TO_ACTION = {
    # stop / emergency
    "stop": "StopMove",
    "robot stop": "StopMove",
    "emergency stop": "Damp",

    # posture
    "robot stand": "BalanceStand",
    "stand up": "StandUp",
    "stand down": "StandDown",
    "sit": "Sit",
    "recover": "RecoveryStand",

    # safe demo actions
    "hello": "Hello",
    "robot hello": "Hello",

    "stretch": "Stretch",
    "robot stretch": "Stretch",

    "dance": "Dance1",
    "robot dance": "Dance1",
    "dance one": "Dance1",
    "robot dance one": "Dance1",
    "dance two": "Dance2",
    "robot dance two": "Dance2",

    "wiggle": "WiggleHips",
    "wiggle hips": "WiggleHips",

    "finger heart": "FingerHeart",
    "moon walk": "MoonWalk",
    "cross walk": "CrossWalk",

    # your desired behavior:
    # saying "stand" triggers Handstand
    "stand": "Handstand",
    "hand stand": "Handstand",
    "handstand": "Handstand",
    "robot handstand": "Handstand",
    "do handstand": "Handstand",
}

DANGEROUS_ACTIONS = {
    "FrontFlip",
    "BackFlip",
    "LeftFlip",
    "RightFlip",
    "FrontJump",
    "FrontPounce",
    "Bound",
}

GRAMMAR = [
    "stop",
    "robot stop",
    "emergency stop",

    "stand",
    "robot stand",
    "stand up",
    "stand down",
    "sit",
    "recover",

    "hello",
    "robot hello",

    "stretch",
    "robot stretch",

    "dance",
    "robot dance",
    "dance one",
    "robot dance one",
    "dance two",
    "robot dance two",

    "wiggle",
    "wiggle hips",

    "finger heart",
    "moon walk",
    "cross walk",

    "hand stand",
    "handstand",
    "robot handstand",
    "do handstand",

    "[unk]",
]


audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}", file=sys.stderr)

    audio_queue.put(bytes(indata))


def validate_vosk_model(model_path: str) -> str:
    path = Path(model_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Vosk model path does not exist: {path}")

    required = ["am", "conf", "graph"]
    missing = [name for name in required if not (path / name).exists()]

    if missing:
        raise RuntimeError(
            f"Invalid Vosk model folder: {path}\n"
            f"Missing: {missing}\n"
            "model_path must point to the folder containing am/, conf/, graph/."
        )

    return str(path)


class Go2VoiceCommandPublisher(Node):
    def __init__(self):
        super().__init__("go2_voice_command_publisher")

        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("command_topic", "/go2/voice_action")
        self.declare_parameter("command_cooldown", COMMAND_COOLDOWN)

        self.model_path = str(self.get_parameter("model_path").value)
        self.command_topic = str(self.get_parameter("command_topic").value)
        self.command_cooldown = float(self.get_parameter("command_cooldown").value)

        self.pub = self.create_publisher(String, self.command_topic, 10)

        self.last_phrase = None
        self.last_time = 0.0

        self.get_logger().info(f"Publishing voice actions to {self.command_topic}")

    def publish_action(self, phrase: str, action_name: str):
        if action_name in DANGEROUS_ACTIONS:
            self.get_logger().warn(f"Blocked dangerous action: {action_name}")
            return

        now = time.time()

        if phrase == self.last_phrase and now - self.last_time < self.command_cooldown:
            self.get_logger().info("Ignored repeated command during cooldown.")
            return

        msg = String()
        msg.data = action_name
        self.pub.publish(msg)

        self.get_logger().info(f"Published voice action: '{phrase}' -> {action_name}")

        self.last_phrase = phrase
        self.last_time = now


def main():
    rclpy.init()

    node = Go2VoiceCommandPublisher()

    try:
        print("Available voice commands:")
        for phrase, action in VOICE_TO_ACTION.items():
            print(f"  '{phrase}' -> {action}")

        model_path = validate_vosk_model(node.model_path)
        print(f"\nLoading Vosk model: {model_path}")

        model = Model(model_path)

        recognizer = KaldiRecognizer(
            model,
            SAMPLE_RATE,
            json.dumps(GRAMMAR),
        )

        print("\nVoice command publisher ready.")
        print("This node only publishes ROS actions. It does not directly control the robot.")
        print("Start the Ethernet/DDS sport client node separately.")
        print("\nSay for example:")
        print("  robot dance")
        print("  dance two")
        print("  hello")
        print("  stretch")
        print("  wiggle hips")
        print("  finger heart")
        print("  moon walk")
        print("  sit")
        print("  stand")
        print("  stop")
        print("\nPress Ctrl+C to quit.\n")

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=8000,
            dtype="int16",
            channels=1,
            device=MIC_DEVICE,
            callback=audio_callback,
        ):
            while rclpy.ok():
                try:
                    data = audio_queue.get(timeout=0.05)
                except queue.Empty:
                    rclpy.spin_once(node, timeout_sec=0.0)
                    continue

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip().lower()

                    if not text:
                        continue

                    node.get_logger().info(f"Heard: '{text}'")

                    action_name = VOICE_TO_ACTION.get(text)
                    if action_name is None:
                        node.get_logger().info("No action mapped.")
                        continue

                    node.publish_action(text, action_name)

                rclpy.spin_once(node, timeout_sec=0.0)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
