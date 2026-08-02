#!/usr/bin/env bash
set -e

# ============================================================
# Isaac RL Studio - Container Entrypoint
# Sources ROS2 + Isaac Lab environment for headless physics simulation.
# ============================================================

# Ensure files/directories created in mounted volumes are readable/writable/deletable by all users
umask 0000
mkdir -p /workspace/core/logs /workspace/core/data /workspace/logs 2>/dev/null || true
chmod -R 777 /workspace/core/logs /workspace/core/data /workspace/logs 2>/dev/null || true

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
export PYTHONPATH="/workspace:/workspace/core/source:/workspace/head:${PYTHONPATH:-}"
export ISAAC_TASK_DIR="/workspace/head"
export PATH="/isaac-sim/kit/python/bin:${PATH:-}"

# Ensure /workspace and /workspace/core/source are permanently on PYTHONPATH for all Python invocations
printf "/workspace\n/workspace/core/source\n" > /usr/local/lib/python3.10/dist-packages/isaac-rl-studio.pth
printf "/workspace\n/workspace/core/source\n" > /isaac-sim/python_packages/isaac-rl-studio.pth
printf "/workspace\n/workspace/core/source\n" > /isaac-sim/kit/python/lib/python3.10/site-packages/isaac-rl-studio.pth

# Auto-register Isaac Lab tasks on Python startup
cat > /isaac-sim/kit/python/lib/python3.10/site-packages/sitecustomize.py << 'SITECUSTOMIZE'
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")
if "/workspace/core/source" not in sys.path:
    sys.path.insert(0, "/workspace/core/source")
try:
    import register_tasks
except Exception:
    pass
SITECUSTOMIZE

cat > /usr/local/lib/python3.10/dist-packages/sitecustomize.py << 'SITECUSTOMIZE'
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")
if "/workspace/core/source" not in sys.path:
    sys.path.insert(0, "/workspace/core/source")
try:
    import register_tasks
except Exception:
    pass
SITECUSTOMIZE

# Register custom Isaac Lab tasks (humanoid, anymal, amr, cobot)
python3 -c "import register_tasks" 2>/dev/null || true

# Alias python3 -> python if not present
if [ -f /isaac-sim/kit/python/bin/python3 ]; then
    ln -sf /isaac-sim/kit/python/bin/python3 /isaac-sim/kit/python/bin/python 2>/dev/null || true
    ln -sf /isaac-sim/kit/python/bin/python3 /usr/local/bin/python 2>/dev/null || true
    ln -sf /isaac-sim/kit/python/bin/python3 /usr/local/bin/python3 2>/dev/null || true
fi

# Ensure numpy version compatibility (<2.0.0) for Isaac Sim extensions
python3 -c "import numpy as np; assert int(np.__version__.split('.')[0]) < 2" 2>/dev/null || python3 -m pip install --no-cache-dir "numpy<2.0.0" "numpy==1.26.4" 2>/dev/null || true

# Ensure retargeting dependencies are installed with numpy<2.0.0 pin
python3 -c "import mediapipe" 2>/dev/null || python3 -m pip install --no-cache-dir "numpy<2.0.0" mediapipe opencv-python-headless 2>/dev/null || true

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
