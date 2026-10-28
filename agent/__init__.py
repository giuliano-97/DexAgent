"""DexAgent harness: the loop that lets a language model run a stage with the tool library."""
import os

# MuJoCo picks its GL backend once, on first import; tools import it in this process in any order, so a tool's own
# setdefault comes too late once another tool has loaded mujoco. Headless (cluster) runs need EGL, not GLFW.
os.environ.setdefault("MUJOCO_GL", "egl")
