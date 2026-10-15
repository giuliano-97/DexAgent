"""Build the Flexiv Rizon 4 + Sharpa Wave bench from the default bench.

    python setup_flexiv/build_setup.py            # writes robot/rizon4_arm.xml and scene.xml (keyframe `home` solved by IK)

The table, the ego camera, the wrist cameras, both Sharpa hands (with their floating-wrist proposal joints, servos
and contact pairs) and the `home` hand poses are taken unchanged from ../setup/scene.xml; only the two I2RT YAM arms
are replaced. Keeping the hands' home poses and the rig frame (midpoint of the two `robot_<side>_arm_root` bodies,
see tool_library/skills/stage_on_bench.py) unchanged keeps the bench registered to the default ego camera, so the
default demonstrations and their Stage 2 reconstructions stay in the same frame.

The arm numbers are Flexiv's: flexiv_description (github.com/flexivrobotics/flexiv_description, Apache-2.0) at
commit 6d5145ea74bce2a7f61867a7ae3de3823ec6154f, config/Rizon4/{default_kinematics,inertia_parameters,joint_limits}.yaml.
Joint ranges are the controller's soft limits, not the hard stops. The visual meshes are meshes/Rizon4/visual/link*.obj
of that commit; link7.obj is re-exported as one plain mesh with trimesh (same vertices and 4934 faces), because
MuJoCo's OBJ loader keeps only 3232 of the original's faces (two objects, 132 material groups) and drops the link's
cylinder between joint 7 and the flange. The decorative joint rings (ring.obj) are left out, a 5 mm gap per joint.
"""
from __future__ import annotations

import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_BENCH = REPO / "setup" / "scene.xml"
FLEXIV_COMMIT = "6d5145ea74bce2a7f61867a7ae3de3823ec6154f"

# default_kinematics.yaml: each joint's origin in its parent link (x, y, z, roll, pitch, yaw) and axis.
KINEMATICS = [
    ((0.0, 0.0, 0.155), (0.0, 0.0, np.pi), (0, 0, 1)),
    ((0.0, 0.03, 0.21), (0.0, 0.0, 0.0), (0, 1, 0)),
    ((0.0, 0.035, 0.205), (0.0, 0.0, 0.0), (0, 0, 1)),
    ((-0.02, -0.03, 0.19), (0.0, 0.0, np.pi), (0, 1, 0)),
    ((-0.02, 0.025, 0.195), (0.0, 0.0, np.pi), (0, 0, 1)),
    ((0.0, 0.03, 0.19), (0.0, 0.0, 0.0), (0, 1, 0)),
    ((-0.015, 0.073, 0.11), (0.0, -np.pi / 2, 0.0), (0, 0, 1)),
]
LINK7_TO_FLANGE = ((0.0, 0.0, 0.081), (0.0, 0.0, np.pi))
# inertia_parameters.yaml: mass, centre of mass, (ixx, iyy, izz, ixy, ixz, iyz) about the centre of mass.
INERTIA = {
    "base": (3.863, (0.0, -0.0008, 0.0883), (0.0188, 0.019, 0.0103, 0.0, 0.0, 0.0002)),
    "link1": (3.85, (0.0002, 0.004, 0.1451), (0.0308, 0.0307, 0.0077, 0.0, -0.0001, 0.0012)),
    "link2": (2.904, (0.0, 0.0356, 0.0946), (0.0255, 0.0263, 0.0039, 0.0, 0.0001, -0.0001)),
    "link3": (2.381, (-0.0123, -0.0021, 0.1368), (0.0144, 0.0139, 0.0039, 0.0001, -0.0015, -0.0006)),
    "link4": (2.381, (-0.0129, 0.0309, 0.1047), (0.0184, 0.0188, 0.0029, 0.0001, -0.0016, -0.0012)),
    "link5": (2.336, (0.0003, 0.0024, 0.1365), (0.0144, 0.0136, 0.0036, 0.0, 0.0, 0.0006)),
    "link6": (2.251, (0.0193, 0.0595, 0.0726), (0.0085, 0.0089, 0.0041, 0.0004, 0.0012, 0.0018)),
    "link7": (0.774, (0.0003, 0.0008, 0.025), (0.001, 0.001, 0.0009, 0.0, 0.0, 0.0)),
}
# joint_limits.yaml: soft lower / upper (rad), effort (N m), velocity (rad/s).
SOFT_LO = [-2.7925, -2.2689, -2.9671, -1.8675, -2.9671, -1.3963, -2.9671]
SOFT_HI = [2.7925, 2.2689, 2.9671, 2.6878, 2.9671, 4.5379, 2.9671]
EFFORT_NM = [123, 123, 64, 64, 39, 39, 39]
VELOCITY_RAD_S = [2.0944, 2.0944, 2.4435, 2.4435, 4.8869, 4.8869, 4.8869]

# The hand adapter: an aluminium plate between the ISO flange and the Sharpa wrist (the hand's base_yaw origin is its
# mounting face; its fingers point along its +z). `gripper` is the plate's outer face, so the flange -> hand offset
# is zero as on the default bench and the hand's +z continues the flange's +z.
ADAPTER_M = 0.015
ADAPTER_RADIUS_M = 0.0315
ADAPTER_MASS_KG = 0.12

# Station placement, chosen by setup_flexiv/search_placement.py. The roots keep the default bench's midpoint
# (x = 0.015, y = 0) so the rig frame and the ego camera registration are unchanged; each Rizon base sits inside its
# root at `base_offset`, bolted to a mounting plate on a column (visual only).
PLACEMENT = {
    "half_span": 0.25,                   # root x = 0.015 -/+ half_span
    "root_z": 0.0,
    "base_offset": (0.0, 0.0, 0.25),     # (outward along x, y, z) of the base in its root, metres
    "base_tilt_deg": 45.0,               # each base's joint-1 axis leans this far outward from vertical (angled bracket)
    "base_yaw_deg": 135.0,               # right base's turn about its own (tilted) axis; the left one is mirrored: 180 - yaw
    "mount_roll_deg": {"left": -270.0, "right": 270.0},   # the hand's roll on the adapter, about the flange axis
}
# Keyframe `home`: both hands on edge, palms facing each other 0.36 m apart, thumbs up, fingers forward into the work
# (HOME_HAND_STYLE "palm_down" lays them flat instead: palms down, thumbs inward). Sharpa hand frame: fingers +z, palm faces +x, thumb on +y
# (right) / -y (left).
# x / y are where the ego camera (default bench calibration) saw them at 34% / 67% of the image width and 61% of its
# height at z = 0.16; the hands were then spread to 0.36 m apart about their midpoint and moved 0.20 m forward.
HOME_HANDS = {"left": [-0.200, 0.520, 0.16], "right": [0.160, 0.520, 0.16]}
HOME_HAND_STYLE = "on_edge"
HOME_FINGER_PITCH_UP_DEG = 0.0
INCLUDE_TABLE = True                     # the default bench's table (and its hand-table contact pairs)
TABLE_TOP_Z = 0.0                        # the default bench's: 0.16 m under the hands
RIG_MID_X = 0.015
ARM_RGBA = "0.92 0.92 0.93 1"
PEDESTAL_RGBA = "0.25 0.25 0.27 1"


def _v(x):
    return " ".join(f"{float(v):.9g}" for v in x)


def _quat_rpy(rpy):
    """URDF rpy (extrinsic X, Y, Z) -> MuJoCo quaternion [w, x, y, z]."""
    x, y, z, w = Rotation.from_euler("xyz", rpy).as_quat()
    return [w, x, y, z]


def _quat_yaw(yaw):
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def _quat_tilt_yaw(tilt, yaw):
    """Lean the base about world y by `tilt`, then turn it about its own z by `yaw`; [w, x, y, z]."""
    x, y, z, w = (Rotation.from_euler("y", tilt) * Rotation.from_euler("z", yaw)).as_quat()
    return [w, x, y, z]


def arm_tree(prefix="", proposal=False, mesh_prefix=None):
    """root -> base -> link1..link7 -> flange -> gripper as an ElementTree body.

    proposal=True gives the planning arm of scene.xml: the default bench's 0.25 kg placeholder links and joint
    damping 5 (the runtime compiler restores the real inertia from robot/rizon4_arm.xml)."""
    mesh_prefix = prefix if mesh_prefix is None else mesh_prefix
    root = ET.Element("body", name=f"{prefix}root")

    def link(parent, name, pos=None, quat=None):
        b = ET.SubElement(parent, "body", name=f"{prefix}{name}")
        if pos is not None:
            b.set("pos", _v(pos))
        if quat is not None:
            b.set("quat", _v(quat))
        if proposal:
            ET.SubElement(b, "inertial", pos=_v(INERTIA[name][1]), mass="0.25", diaginertia="0.00025 0.00025 0.00025")
        else:
            mass, com, I = INERTIA[name]
            ET.SubElement(b, "inertial", pos=_v(com), mass=f"{mass}", fullinertia=_v(I))
        mesh = "link0" if name == "base" else name
        ET.SubElement(b, "geom", type="mesh", mesh=f"{mesh_prefix}{mesh}", contype="0", conaffinity="0", group="1",
                      density="0", rgba=ARM_RGBA)
        return b

    body = link(root, "base")
    for i, (pos, rpy, axis) in enumerate(KINEMATICS, start=1):
        body = link(body, f"link{i}", pos, _quat_rpy(rpy))
        j = ET.Element("joint", name=f"{prefix}dof_joint{i}", axis=_v(axis), range=_v([SOFT_LO[i - 1], SOFT_HI[i - 1]]))
        if proposal:
            j.set("damping", "5")
        body.insert(1, j)
    flange = ET.SubElement(body, "body", name=f"{prefix}flange", pos=_v(LINK7_TO_FLANGE[0]),
                           quat=_v(_quat_rpy(LINK7_TO_FLANGE[1])))
    if not proposal:
        ET.SubElement(flange, "inertial", pos=_v([0, 0, ADAPTER_M / 2]), mass=f"{ADAPTER_MASS_KG}",
                      diaginertia=_v([4e-5, 4e-5, 6e-5]))
    ET.SubElement(flange, "geom", type="cylinder", size=_v([ADAPTER_RADIUS_M, ADAPTER_M / 2]),
                  pos=_v([0, 0, ADAPTER_M / 2]), contype="0", conaffinity="0", group="1", density="0",
                  rgba="0.55 0.56 0.58 1")
    ET.SubElement(flange, "body", name=f"{prefix}gripper", pos=_v([0, 0, ADAPTER_M]))
    return root


def mesh_assets(prefix, meshdir):
    return [ET.Element("mesh", name=f"{prefix}link{i}", file=f"{meshdir}link{i}.obj") for i in range(8)]


def write_arm_xml(out=HERE / "robot" / "rizon4_arm.xml"):
    """The arm on its own (unprefixed names, real inertia): the runtime compiler's inertia source."""
    root = ET.Element("mujoco", model="flexiv_rizon4")
    ET.SubElement(root, "compiler", angle="radian", meshdir="../assets/flexiv/")
    asset = ET.SubElement(root, "asset")
    asset.extend(mesh_assets("", ""))
    ET.SubElement(root, "worldbody").append(arm_tree())
    ET.indent(root)
    out.write_text(f"<!-- Flexiv Rizon 4, from flexiv_description {FLEXIV_COMMIT}; built by build_setup.py -->\n"
                   + ET.tostring(root, encoding="unicode") + "\n")
    return out


def station_frames(placement):
    """{side: (root_pos, base_pos_in_root, base_quat_in_root)}."""
    a = placement["half_span"]
    ox, oy, oz = placement["base_offset"]
    yaw = np.radians(placement["base_yaw_deg"])
    tilt = np.radians(placement.get("base_tilt_deg", 0.0))
    return {"left": ([RIG_MID_X - a, 0.0, placement["root_z"]], [-ox, oy, oz], _quat_tilt_yaw(-tilt, np.pi - yaw)),
            "right": ([RIG_MID_X + a, 0.0, placement["root_z"]], [ox, oy, oz], _quat_tilt_yaw(tilt, yaw))}


def scene_tree(placement=PLACEMENT, bench=DEFAULT_BENCH, table=INCLUDE_TABLE):
    """The default bench with its two YAM arms replaced by Rizon 4 stations (no keyframe yet); without `table`, the
    table body and every contact pair naming its geom are left out; with it, its top is moved to TABLE_TOP_Z."""
    root = ET.parse(bench).getroot()
    if table:
        body = root.find("worldbody/body[@name='table']")
        half = float(body.find("geom").get("size").split()[2])
        x, y, _ = (float(v) for v in body.get("pos").split())
        body.set("pos", _v([x, y, TABLE_TOP_Z - half]))
    if not table:
        world0 = root.find("worldbody")
        world0.remove(world0.find("body[@name='table']"))
        contact = root.find("contact")
        for pair in list(contact):
            if "g_table" in (pair.get("geom1"), pair.get("geom2")):
                contact.remove(pair)
    for k in root.findall("keyframe"):
        root.remove(k)
    size = root.find("size")
    if size is not None:
        size.attrib.pop("nkey", None)
    asset = root.find("asset")
    for mesh in list(asset):
        if mesh.get("name", "").startswith(("robot_left_arm_", "robot_right_arm_")):
            asset.remove(mesh)
    world = root.find("worldbody")
    frames = station_frames(placement)
    for side in ("left", "right"):
        old = world.find(f"body[@name='robot_{side}_arm_root']")
        index = list(world).index(old)
        world.remove(old)
        prefix = f"robot_{side}_arm_"
        asset.extend(mesh_assets(prefix, "assets/flexiv/"))
        st = arm_tree(prefix, proposal=True)
        root_pos, base_pos, base_quat = frames[side]
        st.set("pos", _v(root_pos))
        base = st.find("body")
        base.set("pos", _v(base_pos)); base.set("quat", _v(base_quat))
        # a mounting plate under the base, bolted to the face of a column that stands behind it (visual only)
        n = Rotation.from_quat(np.roll(base_quat, -1)).apply([0.0, 0.0, 1.0])     # the base's joint-1 axis
        plate = np.asarray(base_pos) - 0.01 * n
        ET.SubElement(st, "geom", name=f"robot_{side}_mount_plate", type="box", size=_v([0.11, 0.11, 0.01]),
                      pos=_v(plate), quat=_v(base_quat), contype="0", conaffinity="0", group="1", density="0",
                      rgba=PEDESTAL_RGBA)
        top = plate[2] + 0.11 * np.sqrt(max(0.0, 1.0 - n[2] ** 2)) - 0.01 * abs(n[2])   # the plate's top edge
        foot = -0.75 - root_pos[2]
        col = plate - 0.06 * np.r_[n[:2], 0.0]
        ET.SubElement(st, "geom", name=f"robot_{side}_pedestal", type="box", size=_v([0.05, 0.05, (top - foot) / 2]),
                      pos=_v([col[0], col[1], (top + foot) / 2]), contype="0", conaffinity="0", group="1",
                      density="0", rgba=PEDESTAL_RGBA)
        world.insert(index, st)
    return root


def mount_rotation(side, placement=PLACEMENT):
    """Hand orientation in the gripper frame: the hand's +z along the flange axis, rolled about it."""
    return Rotation.from_euler("z", placement["mount_roll_deg"][side], degrees=True).as_matrix()


def home_hand_rotation(side, pitch_up_deg=HOME_FINGER_PITCH_UP_DEG, style=None):
    """World orientation of a hand at home: columns are the hand's x (palm normal), y, z (fingers).
    'palm_down': palms on the table side, fingers forward, thumbs inward; 'on_edge': palms facing each other, thumbs up."""
    style = style or HOME_HAND_STYLE
    if style == "palm_down":
        palm, y = [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]                      # hand +y points -x: right thumb in, left thumb in
    else:
        palm = [-1.0, 0.0, 0.0] if side == "right" else [1.0, 0.0, 0.0]  # palms face each other
        y = [0.0, 0.0, 1.0] if side == "right" else [0.0, 0.0, -1.0]     # thumbs up
    R = np.column_stack([palm, y, np.cross(palm, y)])                    # fingers forward (+y world)
    return Rotation.from_euler("x", pitch_up_deg, degrees=True).as_matrix() @ R


def solve_home(model, bench=DEFAULT_BENCH, placement=PLACEMENT, seeds=48, rng=0):
    """qpos of keyframe `home`: the hands at HOME_HANDS / home_hand_rotation with the default bench's finger values, each
    arm solved so its gripper carries
    the hand exactly (position and orientation through the mount). Among the exact solutions with every joint 10 degrees
    inside its limits, the one whose elbow (joint 4) bows furthest outward of its shoulder-wrist line is kept.
    Returns (qpos, ctrl, report)."""
    import mujoco
    sys.path.insert(0, str(REPO))
    from tool_library.common.trajectory import Chain
    src = mujoco.MjModel.from_xml_path(str(bench))
    sd = mujoco.MjData(src); mujoco.mj_resetDataKeyframe(src, sd, 0)
    d = mujoco.MjData(model)
    for j in range(model.njnt):            # every joint the default bench has keeps its home value
        name = model.joint(j).name
        sj = mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_JOINT, name)
        if sj >= 0:
            n = {0: 7, 1: 4}.get(int(model.jnt_type[j]), 1)
            d.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + n] = sd.qpos[src.jnt_qposadr[sj]:src.jnt_qposadr[sj] + n]
    from tool_library.common import plan as P
    rigs = P.scene_rigs(model)
    for side, rig in rigs.items():           # the hands' home pose: HOME_HANDS, on edge
        wrist = [rig.wrist_qpos[k] for k in ("x", "y", "z", "rx", "ry", "rz")]
        d.qpos[wrist] = P.wrist_joints_from_world(model, d, rig, np.asarray(HOME_HANDS[side], float),
                                                  home_hand_rotation(side), d.qpos[wrist[3:]])
    mujoco.mj_kinematics(model, d)
    rng = np.random.default_rng(rng)
    report = {}
    for side in ("left", "right"):
        hand = model.body(f"robot_{side}_{side}_base_yaw").id
        p_t = d.xpos[hand].copy()
        R_t = d.xmat[hand].reshape(3, 3) @ mount_rotation(side, placement).T
        joints = [model.joint(f"robot_{side}_arm_dof_joint{i}").id for i in range(1, 8)]
        ch = Chain(model, model.body(f"robot_{side}_arm_gripper").id, joints)
        mid = (ch.lo + ch.hi) / 2
        out = 1.0 if side == "right" else -1.0
        shoulder, elbow, wrist = (model.body(f"robot_{side}_arm_link{i}").id for i in (2, 4, 6))
        best = None
        for k in range(seeds):
            q0 = mid if k == 0 else rng.uniform(ch.lo, ch.hi)
            d.qpos[ch.qadr] = q0
            pe, re = ch.solve(d, p_t, R_t, rest=mid, iters=300)
            q = d.qpos[ch.qadr].copy()
            margin = float(np.min(np.minimum(q - ch.lo, ch.hi - q)))
            mujoco.mj_kinematics(model, d)
            v = d.xpos[wrist] - d.xpos[shoulder]; v /= np.linalg.norm(v)
            e = d.xpos[elbow] - d.xpos[shoulder]
            outward = out * float((e - (e @ v) * v)[0])     # how far the elbow bows outward of the shoulder-wrist line
            score = (pe > 1e-4 or re > 1e-3, margin < np.radians(10), -outward, -margin)
            if best is None or score < best[0]:
                best = (score, q, pe, re, margin, outward)
        d.qpos[ch.qadr] = best[1]
        report[side] = {"pos_err_mm": round(best[2] * 1e3, 4), "rot_err_deg": round(np.degrees(best[3]), 4),
                        "min_joint_margin_deg": round(np.degrees(best[4]), 1), "elbow_outward_m": round(best[5], 3), "arm_rad": [round(float(v), 6) for v in best[1]]}
    mujoco.mj_kinematics(model, d)
    ctrl = np.zeros(model.nu)
    for a in range(model.nu):              # actuators are the default bench's (wrist + finger servos)
        ctrl[a] = d.qpos[model.jnt_qposadr[model.actuator_trnid[a, 0]]]
    return d.qpos.copy(), ctrl, report


def build(placement=PLACEMENT, out=HERE / "scene.xml"):
    import mujoco
    write_arm_xml()
    root = scene_tree(placement)
    xml = ET.tostring(root, encoding="unicode")
    spec = mujoco.MjSpec.from_string(xml)
    spec.meshdir = str(out.parent) + "/"        # resolve the assets while solving; the file keeps relative paths
    model = spec.compile()
    qpos, ctrl, report = solve_home(model, placement=placement)
    size = root.find("size")
    if size is None:
        size = ET.Element("size"); root.insert(0, size)
    size.set("nkey", "1")
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", name="home", qpos=_v(qpos), ctrl=_v(ctrl))
    ET.indent(root)
    out.write_text(f"<!-- Flexiv Rizon 4 + Sharpa Wave bench; built by build_setup.py from ../setup/scene.xml -->\n"
                   + ET.tostring(root, encoding="unicode") + "\n")
    return out, report


if __name__ == "__main__":
    path, rep = build()
    print(path)
    for s, r in rep.items():
        print(s, r)
