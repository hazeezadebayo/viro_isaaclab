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

# Register ROS2 libraries in system dynamic linker
if [ -d "/opt/ros/${ROS_DISTRO:-humble}/lib" ]; then
    echo "/opt/ros/${ROS_DISTRO:-humble}/lib" > /etc/ld.so.conf.d/ros2.conf 2>/dev/null || true
    ldconfig 2>/dev/null || true
fi

# Source ROS2
if [ -n "${ROS_DISTRO:-humble}" ] && [ -f "/opt/ros/${ROS_DISTRO:-humble}/setup.bash" ]; then
    source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
fi

# Ensure /isaac-sim/python.sh sources ROS 2 environment when invoked
if [ -f "/isaac-sim/python.sh" ] && ! grep -q "source /opt/ros" /isaac-sim/python.sh; then
    sed -i '2i if [ -f "/opt/ros/${ROS_DISTRO:-humble}/setup.bash" ]; then source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"; fi' /isaac-sim/python.sh 2>/dev/null || true
fi

# Source Isaac Sim conda env
if [ -f "/isaac-sim/setup_conda_env.sh" ]; then
    source /isaac-sim/setup_conda_env.sh
fi

# Source Isaac Lab env
if [ -f "/isaac-lab/setup_conda_env.sh" ]; then
    source /isaac-lab/setup_conda_env.sh
fi

ROS_PY_LIB="/opt/ros/${ROS_DISTRO:-humble}/lib/python3.10/site-packages"
ROS_PY_LOCAL="/opt/ros/${ROS_DISTRO:-humble}/local/lib/python3.10/dist-packages"

# Add custom paths and ROS2 Python & C library paths
export PYTHONPATH="/workspace:/workspace/core/source:/workspace/head:${ROS_PY_LIB}:${ROS_PY_LOCAL}:${PYTHONPATH:-}"
export ISAAC_TASK_DIR="/workspace/head"
export PATH="/isaac-sim/kit/python/bin:${PATH:-}"
export LD_LIBRARY_PATH="/opt/ros/${ROS_DISTRO:-humble}/lib:${LD_LIBRARY_PATH:-}"

# Ensure /workspace, /workspace/core/source, and ROS2 paths are permanently on PYTHONPATH for all Python invocations
printf "/workspace\n/workspace/core/source\n${ROS_PY_LIB}\n${ROS_PY_LOCAL}\n" > /usr/local/lib/python3.10/dist-packages/isaac-rl-studio.pth 2>/dev/null || true
printf "/workspace\n/workspace/core/source\n${ROS_PY_LIB}\n${ROS_PY_LOCAL}\n" > /isaac-sim/python_packages/isaac-rl-studio.pth 2>/dev/null || true
printf "/workspace\n/workspace/core/source\n${ROS_PY_LIB}\n${ROS_PY_LOCAL}\n" > /isaac-sim/kit/python/lib/python3.10/site-packages/isaac-rl-studio.pth 2>/dev/null || true

# Auto-register Isaac Lab tasks and ROS2 paths on Python startup
cat > /isaac-sim/kit/python/lib/python3.10/site-packages/sitecustomize.py << 'SITECUSTOMIZE'
import os, sys, ctypes
ros_lib_dir = "/opt/ros/humble/lib"
if os.path.exists(ros_lib_dir):
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if ros_lib_dir not in current_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{ros_lib_dir}:{current_ld}"
    for so_name in ["librcutils.so", "librcpputils.so", "librcl_logging_interface.so", "librmw.so", "librmw_implementation.so", "librcl.so", "librcl_action.so"]:
        so_path = os.path.join(ros_lib_dir, so_name)
        if os.path.exists(so_path):
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
for p in ["/workspace", "/workspace/core/source", "/opt/ros/humble/lib/python3.10/site-packages", "/opt/ros/humble/local/lib/python3.10/dist-packages"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
try:
    import register_tasks
except Exception:
    pass
SITECUSTOMIZE

cat > /usr/local/lib/python3.10/dist-packages/sitecustomize.py << 'SITECUSTOMIZE'
import os, sys, ctypes
ros_lib_dir = "/opt/ros/humble/lib"
if os.path.exists(ros_lib_dir):
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if ros_lib_dir not in current_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{ros_lib_dir}:{current_ld}"
    for so_name in ["librcutils.so", "librcpputils.so", "librcl_logging_interface.so", "librmw.so", "librmw_implementation.so", "librcl.so", "librcl_action.so"]:
        so_path = os.path.join(ros_lib_dir, so_name)
        if os.path.exists(so_path):
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
for p in ["/workspace", "/workspace/core/source", "/opt/ros/humble/lib/python3.10/site-packages", "/opt/ros/humble/local/lib/python3.10/dist-packages"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
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
