import os
import sys

os.environ.setdefault("CARB_APP_PATH", "/isaac-sim/kit")

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "create_new_stage": False})

import isaaclab.sim as sim_utils
from core.source.amr.descriptions.amr import AMR_BURGER_CFG
from isaaclab.assets import Articulation

print("APP BOOTED", flush=True)

sim = sim_utils.SimulationContext()
articulation = Articulation(AMR_BURGER_CFG.replace(prim_path="/World/amr"))

sim.reset()
articulation.reset()
print("RESET OK", flush=True)

joint_names = sorted(articulation.joint_names)
print("JOINTS:", joint_names, flush=True)
assert "wheel_left_joint" in joint_names, "wheel_left_joint missing"
assert "wheel_right_joint" in joint_names, "wheel_right_joint missing"

print("USD LOAD OK", flush=True)
app.close()
print("DONE", flush=True)
