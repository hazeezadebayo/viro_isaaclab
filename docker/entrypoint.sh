#!/usr/bin/env bash
set -e

# ============================================================
# Isaac RL Studio - Container entrypoint
#
# Sources the ROS2 + Isaac environment, and when VIZ_MODE=1
# starts a virtual display (Xvfb) + x11vnc + noVNC so the Isaac
# Sim GUI can be viewed in a browser at:
#     http://localhost:${VIZ_PORT:-6080}/vnc.html
#
# Idempotent: safe to run at container start AND to `source`
# from a `docker exec` shell to pick up the environment.
# ============================================================

# Source ROS2
if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

# Source Isaac Sim conda env
if [ -f "/isaac-sim/setup_conda_env.sh" ]; then
    source /isaac-sim/setup_conda_env.sh
fi

# Source Isaac Lab env
if [ -f "/isaac-lab/setup_conda_env.sh" ]; then
    source /isaac-lab/setup_conda_env.sh
fi

# Add custom paths
export PYTHONPATH="/workspace/core/source:/workspace/head:${PYTHONPATH:-}"
export ISAAC_TASK_DIR="/workspace/head"
export PATH="/isaac-sim/kit/python/bin:${PATH:-}"

# Isaac Sim ships `python3` in the kit but no `python`; alias it so the documented
# `python ...` commands resolve to the Isaac interpreter.
if [ -f /isaac-sim/kit/python/bin/python3 ] && [ ! -e /isaac-sim/kit/python/bin/python ]; then
    ln -s python3 /isaac-sim/kit/python/bin/python
fi

# Source ROS workspace if built
if [ -f "/workspace/ros_ws/install/setup.bash" ]; then
    source /workspace/ros_ws/install/setup.bash
fi

# Set headless if requested
if [ "${NO_GUI:-0}" = "1" ]; then
    export HEADLESS=1
    export QT_QPA_PLATFORM=offscreen
fi

# If visualization is requested, enforce GUI mode inside the viz container.
if [ "${VIZ_MODE:-0}" = "1" ]; then
    export NO_GUI=0
    unset HEADLESS
    unset QT_QPA_PLATFORM
fi

# Browser visualization mode: virtual display + VNC + noVNC
VIZ_MODE="${VIZ_MODE:-0}"
if [ "${VIZ_MODE}" = "1" ]; then
    VIZ_PORT="${VIZ_PORT:-6080}"
    VNC_PORT="${VNC_PORT:-5900}"
    DISPLAY_NUM="${DISPLAY_NUM:-:99}"
    DISPLAY_ID="${DISPLAY_NUM#:}"
    export DISPLAY="${DISPLAY_NUM}"
    DISPLAY_LOCK="/tmp/.X${DISPLAY_ID}-lock"
    DISPLAY_SOCKET="/tmp/.X11-unix/X${DISPLAY_ID}"

    mkdir -p /var/log/isaac-viz
    if ! pgrep -x Xvfb >/dev/null 2>&1; then
        echo "Starting Xvfb on ${DISPLAY_NUM} ..."
        rm -f "${DISPLAY_LOCK}" "${DISPLAY_SOCKET}"
        Xvfb "${DISPLAY_NUM}" -screen 0 1920x1080x24 -ac >/var/log/isaac-viz/xvfb.log 2>&1 &
        for _ in $(seq 1 40); do
            if [ -S "${DISPLAY_SOCKET}" ] || xdpyinfo -display "${DISPLAY_NUM}" >/dev/null 2>&1; then
                break
            fi
            sleep 0.5
        done
    fi
    if ! pgrep -x fluxbox >/dev/null 2>&1; then
        echo "Starting fluxbox window manager ..."
        fluxbox -display "${DISPLAY_NUM}" >/var/log/isaac-viz/fluxbox.log 2>&1 &
    fi
    if ! pgrep -x x11vnc >/dev/null 2>&1; then
        echo "Starting x11vnc on port ${VNC_PORT} ..."
        for _ in $(seq 1 20); do
            if xdpyinfo -display "${DISPLAY_NUM}" >/dev/null 2>&1; then
                break
            fi
            sleep 0.5
        done
        x11vnc -forever -shared -nopw -display "${DISPLAY_NUM}" -rfbport "${VNC_PORT}" \
            >/var/log/isaac-viz/x11vnc.log 2>&1 &
    fi
    if ! pgrep -f "websockify" >/dev/null 2>&1; then
        echo "Starting noVNC on port ${VIZ_PORT} ..."
        websockify --web=/usr/share/novnc "${VIZ_PORT}" localhost:"${VNC_PORT}" \
            >/var/log/isaac-viz/novnc.log 2>&1 &
    fi
    sleep 1

    # Auto-summon: launch the Isaac task as headless training when VIZ_TASK is set.
    # Renders offscreen (no browser viewport on Docker Desktop/WSL2); the browser
    # view is the live task.log, TensorBoard, and the periodic video clips.
    # Skips if a trainer is already running.
    VIZ_TASK="${VIZ_TASK:-}"
    if [ -n "${VIZ_TASK}" ] && [ "${NO_GUI:-0}" != "1" ]; then
        if pgrep -f "reinforcement_learning/rsl_rl/train.py" >/dev/null 2>&1; then
            echo "Trainer already running - skipping auto-start for ${VIZ_TASK}"
        else
            VIZ_NUM_ENVS="${VIZ_NUM_ENVS:-4}"
            VIZ_MAX_ITERATIONS="${VIZ_MAX_ITERATIONS:-100000}"
            VIDEO_ARGS=""
            if [ "${VIZ_VIDEO:-0}" = "1" ]; then
                VIZ_VIDEO_INTERVAL="${VIZ_VIDEO_INTERVAL:-2000}"
                VIZ_VIDEO_LENGTH="${VIZ_VIDEO_LENGTH:-2000}"
                VIDEO_ARGS="'--video', '--video_interval', '${VIZ_VIDEO_INTERVAL}', '--video_length', '${VIZ_VIDEO_LENGTH}',"
            fi
            echo "Auto-starting task ${VIZ_TASK} (headless, num_envs=${VIZ_NUM_ENVS}, max_iterations=${VIZ_MAX_ITERATIONS}, video=${VIZ_VIDEO:-0}) ..."
            echo "  Log: /var/log/isaac-viz/task.log"
            # Run from /workspace (writable; heads/ is mounted read-only and Hydra
            # needs a writable cwd for its 'outputs/' folder).
            (
                cd /workspace
                nohup python -c "import register; import runpy, sys; sys.path.insert(0, '/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl'); sys.argv=['train.py', '--task', '${VIZ_TASK}', '--num_envs', '${VIZ_NUM_ENVS}', '--max_iterations', '${VIZ_MAX_ITERATIONS}', '--headless', ${VIDEO_ARGS}]; runpy.run_path('/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py', run_name='__main__')" > /var/log/isaac-viz/task.log 2>&1 &
            )
        fi
    fi

    # Auto-start TensorBoard so training progress is viewable in the browser.
    if [ -n "${VIZ_TASK}" ] && ! pgrep -f "tensorboard.main" >/dev/null 2>&1; then
        if python -c "import tensorboard" >/dev/null 2>&1; then
            TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
            echo "Starting TensorBoard on port ${TENSORBOARD_PORT} (logdir=/workspace/logs) ..."
            nohup python -m tensorboard.main --logdir /workspace/logs --host 0.0.0.0 --port "${TENSORBOARD_PORT}" \
                > /var/log/isaac-viz/tensorboard.log 2>&1 &
        fi
    fi
fi

echo "=========================================="
echo " Isaac RL Studio Container "
echo " Head: ${HEAD_NAME:-template}"
echo " ROS:  ${ROS_DISTRO:-humble}"
echo " GUI:  $([ "${NO_GUI:-0}" = "1" ] && echo 'OFF' || echo 'ON')"
if [ "${VIZ_MODE}" = "1" ]; then
    echo " Browser: http://localhost:${VIZ_PORT:-6080}/vnc.html"
fi
echo "=========================================="

exec "$@"
