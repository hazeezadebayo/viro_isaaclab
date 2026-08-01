# ROS2 Integration Architecture & Bridge Guide

A master, first-principles guide explaining the role of **ROS2 (Robot Operating System 2)** in IsaacLab physics simulations across **Humanoid Bipedal Motion Tracking**, **ANYmal Quadruped Locomotion**, **Autonomous Mobile Robots (AMR)**, and **Cobot 6-DOF Manipulator Arms**.

---

## 1. Multi-Robot ROS2 Subscription Mapping (`core/source/`)

The ROS2 bridge configuration (`custom_bridge.yaml`) maps control inputs from the ROS2 network into each robot system in `core/source/`:

```
   ROS2 TOPIC SUBSCRIPTION                  TARGET SYSTEM IN CORE/SOURCE
   ───────────────────────                  ────────────────────────────

   1. /cmd_vel (geometry_msgs/Twist)  ───>  AMR & ANYmal Locomotion Controller
                                            (Linear speed v, turning rate w)

   2. /initialpose (PoseWithCovariance) ──> AMR Nav2 Localization Reset

   3. /cobot/target_pose (PoseStamped) ──>  Cobot 3D End-Effector Reaching Goal

   4. /cobot/joint_trajectory          ──>  Cobot 6-DOF Arm & Gripper Trajectory
      (trajectory_msgs/JointTrajectory)

   5. /humanoid/motion_target          ──>  Humanoid Motion Capture Target Vector
```

---

## 2. Live ROS2 Simulation Video Streaming

When `PeriodicVideoRecorderWrapper` (`core/utils/video_recorder.py`) is initialized in **`mode='ros2'`** or **`mode='both'`**, simulation camera RGB frames are streamed in real time to the ROS2 topic:

$$\text{Topic}: \mathbf{/camera/rgb/image\_raw} \quad (\text{Type: } \text{sensor\_msgs/msg/Image})$$


---

## 2. Dynamic Launch File (`robot_state_publisher.launch.py`)

Launch `robot_state_publisher` for any target robot head to generate full 3D spatial coordinate frame transformations (`/tf` and `/tf_static`):

```bash
# 1. Launch for Humanoid Biped
ros2 launch viro_ros2_ws robot_state_publisher.launch.py robot_type:=humanoid

# 2. Launch for ANYmal Quadruped
ros2 launch viro_ros2_ws robot_state_publisher.launch.py robot_type:=anymal

# 3. Launch for AMR Mobile Robot (TurtleBot3)
ros2 launch viro_ros2_ws robot_state_publisher.launch.py robot_type:=amr

# 4. Launch for Cobot Manipulator Arm (UR5)
ros2 launch viro_ros2_ws robot_state_publisher.launch.py robot_type:=cobot
```

---

## 3. Topic Reference & QoS Profiles

| Topic | ROS2 Message Type | Direction | Target System | Description |
|-------|-------------------|-----------|---------------|-------------|
| `/clock` | `rosgraph_msgs/msg/Clock` | PUBLISH | All Systems | Simulation physics time synchronization |
| `/joint_states` | `sensor_msgs/msg/JointState` | PUBLISH | All Systems | Joint positions, velocities, torques |
| `/tf`, `/tf_static` | `tf2_msgs/msg/TFMessage` | PUBLISH | All Systems | 3D coordinate frame transform tree |
| `/odom` | `nav_msgs/msg/Odometry` | PUBLISH | AMR & ANYmal | Base odometry position & velocity estimate |
| `/imu/data` | `sensor_msgs/msg/Imu` | PUBLISH | Humanoid & ANYmal | IMU orientation & angular acceleration |
| `/scan` | `sensor_msgs/msg/LaserScan` | PUBLISH | AMR | 2D LiDAR range scan data |
| `/camera/rgb/image_raw` | `sensor_msgs/msg/Image` | PUBLISH | All Systems | Headless simulation camera video stream |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | SUBSCRIBE | AMR & ANYmal | Twist velocity command inputs |
| `/initialpose` | `geometry_msgs/msg/PoseWithCovarianceStamped` | SUBSCRIBE | AMR | Nav2 initial pose localization reset |
| `/cobot/target_pose` | `geometry_msgs/msg/PoseStamped` | SUBSCRIBE | Cobot | End-effector 3D Cartesian target |
| `/cobot/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` | SUBSCRIBE | Cobot | Arm joint trajectory position target |
| `/humanoid/motion_target` | `geometry_msgs/msg/PoseStamped` | SUBSCRIBE | Humanoid | Biped walking heading target |
