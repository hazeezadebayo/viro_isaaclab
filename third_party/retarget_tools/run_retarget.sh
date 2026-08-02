#!/usr/bin/env bash
# ==============================================================================
# Humanoid Motion Capture & Retargeting End-to-End Pipeline
# ==============================================================================
# Takes input human video stream(s) -> extracts 3D keypoints -> computes joint angles ->
# retargets to humanoid robot kinematics -> outputs dataset to core/data/motion_capture/
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Python binary resolution
if [ -f "/isaac-sim/python.sh" ]; then
    PYTHON_BIN="/isaac-sim/python.sh"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python"
fi

# Default paths
DEFAULT_VIDEO="$SCRIPT_DIR/bodypose3d/media/cam0_test.mp4"
# Single canonical output location: core/data/motion_capture (writable via the
# compose sub-mount at /workspace/core/data/motion_capture). workspace/data must
# NOT accumulate motion-capture output.
if [ -d "/workspace/core/data" ]; then
    DEFAULT_OUTPUT_DIR="/workspace/core/data/motion_capture"
else
    DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/core/data/motion_capture"
fi

# Positional args: [INPUT_VIDEO] [OUTPUT_DIR] [CAM1_VIDEO]
INPUT_VIDEO="${1:-$DEFAULT_VIDEO}"
REQ_OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"
CAM1_VIDEO="${3:-}"

# Resolve paths to absolute BEFORE the Stage 1 subshell cd's into bodypose3d/.
# Relative paths passed by the caller (e.g. "third_party/...") would otherwise be
# resolved against the bodypose3d/ cwd and no longer exist, making cv2 fail to
# open the videos ("no frames captured").
_resolve_abs() {
    local p="$1"
    case "$p" in
        /*) echo "$p" ;;
        *)  echo "$PWD/$p" ;;
    esac
}
INPUT_VIDEO="$(_resolve_abs "$INPUT_VIDEO")"
REQ_OUTPUT_DIR="$(_resolve_abs "$REQ_OUTPUT_DIR")"

# For the bundled stereo sample, derive the cam1 stream from the cam0 path.
# A single video (no cam0/cam1 pair) is passed through as-is -> monocular.
if [ -z "$CAM1_VIDEO" ] && [[ "$INPUT_VIDEO" == *cam0* ]]; then
    CAM1_VIDEO="${INPUT_VIDEO/cam0/cam1}"
fi
if [ -n "$CAM1_VIDEO" ]; then
    CAM1_VIDEO="$(_resolve_abs "$CAM1_VIDEO")"
fi

# Ensure output directory is writable (mkdir -p alone succeeds on existing
# read-only dirs, so also probe the write bit)
if mkdir -p "$REQ_OUTPUT_DIR" 2>/dev/null && [ -w "$REQ_OUTPUT_DIR" ]; then
    OUTPUT_DIR="$REQ_OUTPUT_DIR"
else
    echo "[INFO] Requested output directory '$REQ_OUTPUT_DIR' is read-only. Redirecting output to /workspace/core/data/motion_capture..."
    OUTPUT_DIR="/workspace/core/data/motion_capture"
    mkdir -p "$OUTPUT_DIR"
fi

BODYPOSE3D_DIR="$SCRIPT_DIR/bodypose3d"
JOINT_ANGLES_DIR="$SCRIPT_DIR/joint_angles_calculate"
RETARGET_SCRIPT="$PROJECT_ROOT/core/source/humanoid/retargeting/retarget_motion.py"

echo "======================================================================"
echo " Humanoid Retargeting Pipeline Initiated"
echo "======================================================================"
if [ -n "$CAM1_VIDEO" ]; then
    echo " Input Videos      : $INPUT_VIDEO"
    echo "                     $CAM1_VIDEO  (stereo)"
else
    echo " Input Video       : $INPUT_VIDEO  (monocular)"
fi
echo " Output Directory  : $OUTPUT_DIR"
echo "======================================================================"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Stage 1: Run bodypose3d keypoint extraction (mode inferred from input count)
echo "[Stage 1/3] Running bodypose3d 3D keypoint estimation..."
if [ -n "$CAM1_VIDEO" ]; then
    (cd "$BODYPOSE3D_DIR" && "$PYTHON_BIN" bodypose3d.py "$INPUT_VIDEO" "$CAM1_VIDEO" --out_dir "$OUTPUT_DIR")
else
    (cd "$BODYPOSE3D_DIR" && "$PYTHON_BIN" bodypose3d.py "$INPUT_VIDEO" --out_dir "$OUTPUT_DIR")
fi

# Stage 2: Calculate joint angles from 3D keypoints
echo "[Stage 2/3] Calculating human joint angles from 3D keypoints..."
# bodypose3d.py writes kpts_3d.dat to the writable output dir
KPTS_FILE="$OUTPUT_DIR/kpts_3d.dat"
if [ ! -f "$KPTS_FILE" ] || [ ! -s "$KPTS_FILE" ]; then
    echo "[ERROR] No 3D keypoints found at '$KPTS_FILE' (missing or empty). Stage 1 failed." >&2
    exit 1
fi
if [ -f "$JOINT_ANGLES_DIR/calculate_joint_angles.py" ]; then
    "$PYTHON_BIN" "$JOINT_ANGLES_DIR/calculate_joint_angles.py" "$KPTS_FILE" "$OUTPUT_DIR"
fi
RAW_ANGLES_FILE="$OUTPUT_DIR/raw_angles.json"
if [ ! -s "$RAW_ANGLES_FILE" ]; then
    echo "[ERROR] Joint angle calculation produced no '$RAW_ANGLES_FILE'." >&2
    exit 1
fi

# Stage 3: Retarget human joint angles to humanoid robot kinematics
echo "[Stage 3/3] Retargeting angles to Humanoid robot URDF dimensions..."
OUTPUT_FILE="$OUTPUT_DIR/human_walk_retargeted.json"

"$PYTHON_BIN" "$RETARGET_SCRIPT" \
    --input "$RAW_ANGLES_FILE" \
    --output "$OUTPUT_FILE" \
    --fps 60.0

echo "======================================================================"
echo " Retargeting Pipeline Successfully Completed!"
echo " Kinematic Motion Data Saved To:"
echo "   -> $OUTPUT_FILE"
echo "======================================================================"
