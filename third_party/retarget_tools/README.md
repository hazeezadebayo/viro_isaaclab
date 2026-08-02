# Humanoid Motion Capture Video Retargeting Suite (`third_party/retarget_tools/`)

An end-to-end, professor-level guide explaining how human walking/movement videos are converted into robot joint angles using **3D Body Pose Estimation**, **Inverse Kinematics (IK)**, and **Motion Retargeting**.

---

## 1. First Principles: Video-to-Robot Kinematic Pipeline

```
  RGB Video File (.mp4)
         │
         ▼
 ┌────────────────────────┐
 │ 1. bodypose3d          │ ───> Extracts 3D Human Keypoints X_human(t) in meters
 └───────────┬────────────┘
             │
             ▼
 ┌────────────────────────┐
 │ 2. joint_angles_calc   │ ───> Computes 3D Vector Directions & Joint Relative Angles
 └───────────┬────────────┘
             │
             ▼
 ┌────────────────────────┐
 │ 3. retarget_motion.py  │ ───> Maps Human Angles to Humanoid Joint Space q_ref(t)
 └───────────┬────────────┘
             │
             ▼
 Output: human_walk_retargeted.json  (/workspace/core/data/motion_capture/)
```

### Mathematical Pipeline Steps

1. **3D Keypoint Extraction**: `bodypose3d` extracts $(x_i, y_i, z_i)$ spatial positions for 25 key joints (hips, knees, ankles, shoulders, elbows, wrists).
2. **Kinematic Vector Conversion**: Relative link vectors $\mathbf{u}_{ij} = \mathbf{x}_j - \mathbf{x}_i$ are converted into Euler / Quaternion rotations.
3. **Robot Joint Mapping**: Inverse Kinematics solves joint targets $\mathbf{q}_{\text{ref}}(t) \in \mathbb{R}^{17}$ matching target foot height and pelvis heading.

---

## 2. 1-Click Execution Example

### Command Syntax
```bash
bash /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/third_party/retarget_tools/run_retarget.sh \
    [INPUT_VIDEO.mp4] \
    [OUTPUT_DATA_DIR] \
    [CAM1_VIDEO.mp4]        # optional; used for stereo
```

The capture mode is **inferred from the number of input videos** — no flag needed:
- **1 video** → monocular (MediaPipe Pose world landmarks, scale-normalized).
- **2 videos** (cam0 + cam1, e.g. `cam0_test.mp4` and `cam1_test.mp4`) → stereo (DLT triangulation from two calibrated cameras).
- Passing only a video whose path contains `cam0` auto-derives the matching `cam1` stream for the bundled sample.

### Step-by-Step Practical Example

```bash
# Example 1: stereo pipeline on the bundled sample pair (cam0 + cam1)
bash /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/third_party/retarget_tools/run_retarget.sh \
    /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/third_party/retarget_tools/bodypose3d/media/cam0_test.mp4 \
    /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/core/data/motion_capture

# Example 2: monocular pipeline from a single camera video
bash /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/third_party/retarget_tools/run_retarget.sh \
    /path/to/my_single_video.mp4 \
    /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/core/data/motion_capture
```

---

## 3. Output File Format (`human_walk_retargeted.json`)

The output JSON file is formatted for instant loading by `HumanoidImitationEnv`:

```json
{
  "num_frames": 300,
  "fps": 30,
  "joint_names": ["pelvis", "left_thigh_0", "left_shin", "left_foot", ...],
  "joint_positions": [
    [0.0, -0.4, 0.8, 0.0, ...],  // Frame 0 joint angles in radians
    [0.02, -0.42, 0.85, 0.01, ...] // Frame 1 joint angles in radians
  ]
}
```
