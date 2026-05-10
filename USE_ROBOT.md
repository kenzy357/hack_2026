# Working pipeline: laptop ↔ Go2

## A. Hardware bring-up (one-time per session)

1. **Power on the Go2.** Wait for it to finish its self-stand routine (it'll be standing on all four legs, looking around).
2. **Plug Ethernet** from the Go2's network port to the USB-Ethernet adapter on your laptop.
3. **Verify the link**:
   ```bash
   ip addr show enx50a03003bae4 | grep "inet "
   ```
   Should show `192.168.123.99/24`. If not, re-run the static IP command:
   ```bash
   sudo nmcli connection up go2-wired
   ```
4. **Ping the robot**:
   ```bash
   ping -c 2 192.168.123.161
   ```
   Replies = good. No replies = check cable / robot power.

## B. Open three terminals, source the env in each

In **every** new terminal, first thing:
```bash
source ~/hack_2026/unitree_ros2/setup_robot.sh
```
And in any terminal that runs the C++ controller binary, also:
```bash
GO2BIN=~/hack_2026/unitree_ros2/example/install/unitree_ros2_example/bin
```

## C. Sanity checks (terminal 1, before starting anything)

```bash
ros2 topic list | head -20             # should show /api/sport/request, /lf/sportmodestate, etc.
ros2 topic hz /lf/sportmodestate       # ~50 Hz incoming = robot reachable & alive
ros2 topic info /follow/target         # Publisher count = 0 → safe to start controller
```
If `Publisher count >= 1`, kill stale publishers before continuing:
```bash
pkill -f go2_perception
pkill -f "ros2 topic pub.*follow"
```

## D. Start the controller (terminal 1)

```bash
$GO2BIN/go2_follow_controller
```
The controller starts ticking at 10 Hz. With nothing publishing `/follow/target` it sends `StopMove` every 100 ms — the robot stays still.

## E. Verify what the controller is sending (terminal 2)

```bash
ros2 topic echo /api/sport/request
```
You should see steady `Request` messages. While idle, the `parameter` field will encode `StopMove`. This window stays open as your read-out throughout the session.

## F. Choose a perception source (terminal 3)

### F.1 — Test mode: fake target by hand
```bash
# centered, 3 m away → robot walks forward
ros2 topic pub -r 10 /follow/target geometry_msgs/Vector3 "{x: 0.0, y: 3.0}"
```
Other useful one-liners:
```bash
ros2 topic pub -r 10 /follow/target geometry_msgs/Vector3 "{x: 0.3, y: 1.5}"   # turn right in place
ros2 topic pub -r 10 /follow/target geometry_msgs/Vector3 "{x: 0.0, y: 1.5}"   # idle (at follow distance)
ros2 topic pub -r 10 /follow/target geometry_msgs/Vector3 "{x: 0.0, y: 0.5}"   # back up (too close)
```
**Ctrl-C the publisher to stop motion.** Watchdog kicks in after 0.5 s and controller goes back to `StopMove`.

### F.2 — Real mode: live perception from the Go2's front camera
```bash
ros2 run unitree_ros2_example go2_perception
```
This subscribes to `Go2FrontVideoData` (default topic `/frontvideostream`), runs YOLO, and publishes `/follow/target` based on whoever the camera sees. **Robot will follow you in real time.** If `/frontvideostream` is the wrong name, find it with `ros2 topic list | grep -i video` and override:
```bash
ros2 run unitree_ros2_example go2_perception --ros-args -p camera_topic:=<actual>
```

## G. Stop everything

In each terminal: **Ctrl-C** (or **Ctrl-\\** if Ctrl-C is being eaten by log spam).
- Stopping the publisher (F) → controller goes idle within 0.5 s.
- Stopping the controller (D) → no more commands, robot does whatever its high-level controller decides (typically holds still).
- Closing terminals does NOT power down the robot — use the remote/app/power button for that.

## Full reference card

| What | Where | Command |
|---|---|---|
| Source env | every terminal | `source ~/hack_2026/unitree_ros2/setup_robot.sh` |
| Verify link | once per session | `ping -c 2 192.168.123.161` |
| Verify ROS bridge | once per session | `ros2 topic hz /lf/sportmodestate` |
| Check no stale publishers | before controller | `ros2 topic info /follow/target` |
| Run controller | T1 | `$GO2BIN/go2_follow_controller` |
| Watch outgoing | T2 | `ros2 topic echo /api/sport/request` |
| Fake target | T3 (test) | `ros2 topic pub -r 10 /follow/target geometry_msgs/Vector3 "{x: 0.0, y: 3.0}"` |
| Live perception | T3 (real) | `ros2 run unitree_ros2_example go2_perception` |


-----------------------------------------------------
# verify you are connected 
  ping -c 1 192.168.123.161                      # confirm link
  source ~/hack_2026/unitree_ros2/setup_robot.sh  # ROS env
  ros2 topic list | head -20                     # see Go2 topics

  if doesn't work:
  pkill -9 -f _ros2_daemon
  Then verify it's gone:
  pgrep -af ros2_daemon
  (should print nothing). Then retry:
  ros2 topic list --no-daemon
  then can run ros2 topic list

  # at every new terminal 
  GO2BIN=~/hack_2026/unitree_ros2/example/install/unitree_ros2_example/bin

  # test controller

  Terminal 1 — controller:
  $GO2BIN/go2_follow_controller

  Terminal 2 — watch outgoing commands:
  ros2 topic echo /api/sport/request
  You'll see StopMove requests every 100 ms while idle. When you
  publish in T3, you should see a burst of Move requests with
  non-zero vyaw, then return to StopMove after ~0.5 s.

  Terminal 3 — single target, repeat as needed:
  ros2 topic pub --once /follow/target geometry_msgs/Vector3 "{x: 
  0.2, y: 1.5}"
  Robot turns right ~8°. Re-run the same line to step again. Flip
  sign for left.

  # 
  Terminal 1 — perception node (subscribes to Go2's front camera, 
  publishes /follow/target):
  ros2 run unitree_ros2_example go2_perception
  Expect:
  loading YOLO model: yolov8n.pt on cuda
  subscribed to /frontvideostream (360p); publishing /follow/target
  target: bearing=+0.00 distance=4.13 m  (1 ppl, bbox_h=448px)
  ...
  If you get "no messages" / nothing happening, the camera topic
  name is wrong. Check with ros2 topic list | grep -i video and
  override:
  ros2 run unitree_ros2_example go2_perception --ros-args -p
  camera_topic:=<actual_name>

  Terminal 2 — see what perception is publishing:
  ros2 topic echo /follow/target
  You should see Vector3 messages: x = bearing in [-1,1] (negative =
   person left of frame center), y = distance estimate (m), z=0.
  Move in front of the Go2 — x should track which side of the frame
  you're on, y should drop as you walk closer.

  # enter the robot 
  ssh unitree@192.168.123.18

  # test perception
    ros2 run unitree_ros2_example go2_perception

# work with camera
ffmpeg -f v4l2 -input_format yuyv422 -framerate 30 -video_size 640x480 \
         -i /dev/video4 \
         -c:v libx264 -preset ultrafast -tune zerolatency \
         -x264-params "keyint=15:min-keyint=15:scenecut=0:repeat-headers=1" \
         -bf 0 -an \
         -f h264 'udp://192.168.123.99:5000?pkt_size=1316'

  The new bit is -x264-params "...:repeat-headers=1" — that prepends fresh SPS+PPS to every keyframe
  (which now arrives every 15 frames = 0.5 s).

  On the laptop — run perception with the visualization window enabled:

# run perception

  source ~/hack_2026/unitree_ros2/setup_robot.sh
  ros2 run unitree_ros2_example go2_perception --ros-args \
      -p stream_url:='udp://0.0.0.0:5000?fifo_size=1000000&overrun_nonfatal=1' \
      -p show_window:=true

# run controller 
source ~/hack_2026/unitree_ros2/setup_robot.sh
$GO2BIN/go2_follow_controller

# run obstacle avoidance
ros2 run unitree_ros2_example go2_lidar_avoidance --ros-args -p debug:=true


# build
source /opt/ros/humble/setup.bash
source ~/hack_2026/unitree_ros2/cyclonedds_ws/install/setup.bash
cd ~/hack_2026/unitree_ros2/example
colcon build --packages-select unitree_ros2_example
cd build/unitree_ros2_example && cmake --install . && cd -

# delete the auto activation of the camera ffmpeg