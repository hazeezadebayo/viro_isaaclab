#!/usr/bin/env bash
set -e

# ============================================================
# Isaac RL Studio - Container Entrypoint
# Sources ROS2 + Isaac Lab environment for headless physics simulation.
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

# Alias python3 -> python if not present
if [ -f /isaac-sim/kit/python/bin/python3 ] && [ ! -e /isaac-sim/kit/python/bin/python ]; then
    ln -s python3 /isaac-sim/kit/python/bin/python
fi

# Source ROS workspace if built
if [ -f "/workspace/ros_ws/install/setup.bash" ]; then
    source /workspace/ros_ws/install/setup.bash
fi

# Set headless execution
export HEADLESS=1
export QT_QPA_PLATFORM=offscreen

echo "=========================================="
echo " Isaac RL Studio Headless Container "
echo " Head: ${HEAD_NAME:-humanoid}"
echo " ROS:  ${ROS_DISTRO:-humble}"
echo " Camera & ROS2 Live Stream Active"
echo "=========================================="

exec "$@"
