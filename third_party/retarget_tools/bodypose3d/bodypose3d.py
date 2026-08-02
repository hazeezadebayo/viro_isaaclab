"""
bodypose3d.py — MediaPipe 1.0.0 Tasks API compatible rewrite.

MediaPipe ≥ 0.10 dropped the legacy `mp.solutions` API entirely.
This version uses `mediapipe.tasks.python.vision.PoseLandmarker`
which is the correct API for mediapipe==1.0.0.
"""

import cv2 as cv
import numpy as np
import sys
import os

# MediaPipe 1.0.0 Tasks API
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles

from utils import DLT, get_projection_matrix, write_keypoints_to_disk

frame_shape = [720, 1280]

# Pose landmark indices we care about (same as before)
# MediaPipe Pose landmark IDs:
# 11=left_shoulder, 12=right_shoulder, 13=left_elbow, 14=right_elbow
# 15=left_wrist, 16=right_wrist, 23=left_hip, 24=right_hip
# 25=left_knee, 26=right_knee, 27=left_ankle, 28=right_ankle
pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]

# ─── Locate the PoseLandmarker model ────────────────────────────────────────
# Try bundled model path first, then fall back to downloading
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_CANDIDATES = [
    os.path.join(_SCRIPT_DIR, "pose_landmarker_lite.task"),
    os.path.join(_SCRIPT_DIR, "pose_landmarker_full.task"),
    os.path.join(_SCRIPT_DIR, "pose_landmarker_heavy.task"),
    "/workspace/core/data/pose_landmarker_lite.task",
    "/workspace/core/data/pose_landmarker_full.task",
]

_MODEL_PATH = None
for _candidate in _MODEL_CANDIDATES:
    if os.path.isfile(_candidate):
        _MODEL_PATH = _candidate
        break

if _MODEL_PATH is None:
    # Download lite model automatically to /workspace/core/data/
    import urllib.request
    _DOWNLOAD_DIR = "/workspace/core/data" if os.path.isdir("/workspace/core/data") else _SCRIPT_DIR
    _MODEL_PATH = os.path.join(_DOWNLOAD_DIR, "pose_landmarker_lite.task")
    _MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    )
    print(f"[bodypose3d] Downloading PoseLandmarker model to {_MODEL_PATH} ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("[bodypose3d] Download complete.")

print(f"[bodypose3d] Using model: {_MODEL_PATH}")


def _has_display():
    """Return True when an interactive display is available for cv2.imshow."""
    if os.environ.get("NO_GUI") in ("1", "true", "True"):
        return False
    if sys.platform.startswith("win"):
        return True
    return bool(os.environ.get("DISPLAY"))


def _extract_world_keypoints_from_result(result):
    """Extract normalized 3D world landmarks for our target pose_keypoints (monocular mode)."""
    frame_keypoints = []
    if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
        landmarks = result.pose_world_landmarks[0]
        for i in pose_keypoints:
            if i < len(landmarks):
                lm = landmarks[i]
                frame_keypoints.append([float(lm.x), float(lm.y), float(lm.z)])
            else:
                frame_keypoints.append([0.0, 0.0, 0.0])
    else:
        frame_keypoints = [[0.0, 0.0, 0.0]] * len(pose_keypoints)
    return frame_keypoints


def _extract_keypoints_from_result(result, frame):
    """Extract pixel coordinates for our target pose_keypoints from a PoseLandmarker result."""
    frame_keypoints = []
    h, w = frame.shape[:2]

    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        landmarks = result.pose_landmarks[0]  # first detected person
        for i in pose_keypoints:
            if i < len(landmarks):
                lm = landmarks[i]
                pxl_x = int(round(lm.x * w))
                pxl_y = int(round(lm.y * h))
                cv.circle(frame, (pxl_x, pxl_y), 3, (0, 0, 255), -1)
                frame_keypoints.append([pxl_x, pxl_y])
            else:
                frame_keypoints.append([-1, -1])
    else:
        frame_keypoints = [[-1, -1]] * len(pose_keypoints)

    return frame_keypoints


def run_mp(input_stream1, input_stream2, P0, P1):
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    show = _has_display()

    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    # Build PoseLandmarker options (VIDEO mode for frame-by-frame processing)
    base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker0, \
         mp_vision.PoseLandmarker.create_from_options(options) as landmarker1:

        frame_idx = 0
        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()

            if not ret0 or not ret1:
                break

            # Crop to 720×720
            if frame0.shape[1] != 720:
                frame0 = frame0[:, frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
                frame1 = frame1[:, frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

            # Convert BGR→RGB for MediaPipe
            rgb0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
            rgb1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

            timestamp_ms = int(frame_idx * 1000 / 30)  # assume 30 fps

            mp_image0 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb0)
            mp_image1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb1)

            result0 = landmarker0.detect_for_video(mp_image0, timestamp_ms)
            result1 = landmarker1.detect_for_video(mp_image1, timestamp_ms)

            frame0_keypoints = _extract_keypoints_from_result(result0, frame0)
            frame1_keypoints = _extract_keypoints_from_result(result1, frame1)

            kpts_cam0.append(frame0_keypoints)
            kpts_cam1.append(frame1_keypoints)

            # Triangulate 3D positions
            frame_p3ds = []
            for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
                if uv1[0] == -1 or uv2[0] == -1:
                    _p3d = [-1, -1, -1]
                else:
                    _p3d = DLT(P0, P1, uv1, uv2)
                frame_p3ds.append(_p3d)

            frame_p3ds = np.array(frame_p3ds).reshape((12, 3))
            kpts_3d.append(frame_p3ds)

            if show:
                cv.imshow('cam1', frame1)
                cv.imshow('cam0', frame0)

            k = cv.waitKey(1) if show else -1
            if k & 0xFF == 27:
                break

            frame_idx += 1

    if show:
        cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)


def run_mp_monocular(input_stream):
    """Single-camera mode: emit MediaPipe Pose world landmarks as the 3D keypoints.

    World landmarks are hip-centered and scale-normalized, so the downstream
    joint-angle stage (which normalizes by bone lengths) works unchanged.
    """
    cap0 = cv.VideoCapture(input_stream)
    cap0.set(3, frame_shape[1])
    cap0.set(4, frame_shape[0])

    show = _has_display()

    base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    kpts_3d = []

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret0, frame0 = cap0.read()
            if not ret0:
                break

            # Crop to 720×720
            if frame0.shape[1] != 720:
                frame0 = frame0[:, frame_shape[1] // 2 - frame_shape[0] // 2:frame_shape[1] // 2 + frame_shape[0] // 2]

            rgb0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
            timestamp_ms = int(frame_idx * 1000 / 30)  # assume 30 fps

            mp_image0 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb0)
            result0 = landmarker.detect_for_video(mp_image0, timestamp_ms)

            frame_3d = np.array(_extract_world_keypoints_from_result(result0)).reshape((12, 3))
            kpts_3d.append(frame_3d)

            if show:
                cv.imshow('cam0', frame0)
                k = cv.waitKey(1)
                if k & 0xFF == 27:
                    break

            frame_idx += 1

    if show:
        cv.destroyAllWindows()
    cap0.release()

    return np.array(kpts_3d)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='3D body pose keypoint estimation (stereo or monocular).')
    parser.add_argument(
        'inputs', nargs='+',
        help='1 input = monocular (MediaPipe Pose world landmarks); '
             '2 inputs = stereo (DLT triangulation from two calibrated cameras).',
    )
    parser.add_argument(
        '--out_dir', default=None,
        help='Directory to write keypoint .dat files. Defaults to /workspace/core/data/motion_capture when available.',
    )
    args = parser.parse_args()

    if args.out_dir:
        _out_dir = args.out_dir
    else:
        _out_dir = "/workspace/core/data/motion_capture" if os.path.isdir("/workspace/core/data") else "/tmp"
    os.makedirs(_out_dir, exist_ok=True)

    if len(args.inputs) >= 2:
        print(f"[bodypose3d] Mode: stereo ({len(args.inputs)} inputs)")
        try:
            input_stream1 = int(args.inputs[0])
        except ValueError:
            input_stream1 = args.inputs[0]

        try:
            input_stream2 = int(args.inputs[1])
        except ValueError:
            input_stream2 = args.inputs[1]

        P0 = get_projection_matrix(0)
        P1 = get_projection_matrix(1)

        kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)
        if kpts_3d.shape[0] == 0:
            raise SystemExit(
                f"[bodypose3d] ERROR: no frames captured from '{input_stream1}' / '{input_stream2}'. "
                "Check that the videos exist and OpenCV can decode them."
            )

        write_keypoints_to_disk(os.path.join(_out_dir, 'kpts_cam0.dat'), kpts_cam0)
        write_keypoints_to_disk(os.path.join(_out_dir, 'kpts_cam1.dat'), kpts_cam1)
        write_keypoints_to_disk(os.path.join(_out_dir, 'kpts_3d.dat'), kpts_3d)
        print(f"[bodypose3d] Keypoints saved to {_out_dir}/")
    else:
        print(f"[bodypose3d] Mode: monocular ({len(args.inputs)} input)")
        try:
            input_stream = int(args.inputs[0])
        except ValueError:
            input_stream = args.inputs[0]

        kpts_3d = run_mp_monocular(input_stream)
        if kpts_3d.shape[0] == 0:
            raise SystemExit(
                f"[bodypose3d] ERROR: no frames captured from '{input_stream}'. "
                "Cannot write empty keypoints."
            )

        write_keypoints_to_disk(os.path.join(_out_dir, 'kpts_3d.dat'), kpts_3d)
        print(f"[bodypose3d] Monocular keypoints saved to {_out_dir}/")
