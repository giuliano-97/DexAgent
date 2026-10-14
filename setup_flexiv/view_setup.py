"""Open a setup in the interactive MuJoCo viewer, frozen at its `home` keyframe.

    python setup_flexiv/view_setup.py                  # this folder's scene.xml
    python setup_flexiv/view_setup.py setup/           # any setup folder (or a scene .xml)

Physics is not stepped (the planning scene's arms have no motors and would sag), so the pose stays put while you
look around: left-drag rotates, right-drag pans, scroll zooms, double-click selects a body. Close the window to quit.
"""
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
scene = arg / "scene.xml" if arg.is_dir() else arg
m = mujoco.MjModel.from_xml_path(str(scene))
d = mujoco.MjData(m)
if m.nkey:
    mujoco.mj_resetDataKeyframe(m, d, max(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"), 0))
mujoco.mj_forward(m, d)
with mujoco.viewer.launch_passive(m, d) as v:
    v.cam.lookat[:] = (0.015, 0.0, 0.25)
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.0, -90, -30   # front: from the far side of the table
    while v.is_running():
        mujoco.mj_forward(m, d)          # kinematics only; no mj_step
        v.sync()
        time.sleep(1 / 60)
