"""Search the Rizon 4 station placement of build_setup.py.

    python setup_flexiv/search_placement.py [--workers 30]     # prints the best placements, best last

A candidate is (half_span, base offset dy / dz, base yaw about its tilted axis, hand roll on the adapter; the outward
tilt is build_setup.PLACEMENT's), scored on the right station
(the left one is its mirror). It must hold the home hand pose of build_setup.py exactly; its score is the share of
probe hand poses over the work cloth (four orientations: the home one, pitched 60 degrees down, yawed +/-40 degrees)
that the arm holds (the home pose must also bow both elbows at least 5 cm outward of their shoulder-wrist lines) within 1 mm / 1 degree, with every joint 10 degrees inside its soft limits, its links 1 cm above
the table and clear of its own hand. The rig-frame midpoint and the root line stay fixed (see build_setup.py).
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MUJOCO_GL", "egl")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import build_setup as B  # noqa: E402

MARGIN = np.radians(10.0)
SIDE = "right"


def probe_poses(model, data):
    hand = model.body(f"robot_{SIDE}_{SIDE}_base_yaw").id
    R0 = data.xmat[hand].reshape(3, 3).copy()
    Rs = [R0, Rotation.from_euler("x", -60, degrees=True).as_matrix() @ R0,
          Rotation.from_euler("z", 40, degrees=True).as_matrix() @ R0,
          Rotation.from_euler("z", -40, degrees=True).as_matrix() @ R0]
    out = []
    for k, R in enumerate(Rs):
        for x, y, z in itertools.product((-0.25, -0.10, 0.05, 0.20, 0.32), (0.15, 0.35, 0.55, 0.75), (0.10, 0.20, 0.32)):
            out.append((k, np.array([x, y, z]), R))
    return out


def ego_share(m, d, geoms, w=160, h=120):
    """Share of the ego view the arms cover at home."""
    import mujoco
    with mujoco.Renderer(m, height=h, width=w) as r:
        r.enable_segmentation_rendering()
        r.update_scene(d, camera="ego")
        seg = r.render()
    ids = seg[..., 0]
    return float(np.isin(ids, list(geoms)).mean())


def evaluate(cand):
    import mujoco
    from tool_library.common.trajectory import Chain
    half_span, dy, dz, yaw, roll = cand
    placement = {"half_span": half_span, "root_z": 0.0, "base_offset": (0.0, dy, dz), "base_yaw_deg": yaw,
                 "base_tilt_deg": B.PLACEMENT["base_tilt_deg"],
                 "mount_roll_deg": {"left": -roll, "right": roll}}
    spec = mujoco.MjSpec.from_string(ET.tostring(B.scene_tree(placement), encoding="unicode"))
    spec.meshdir = str(HERE) + "/"
    m = spec.compile()
    try:
        qpos, _, rep = B.solve_home(m, placement=placement, seeds=12)
    except Exception as exc:  # noqa: BLE001
        return cand, -1.0, str(exc)
    r = rep[SIDE]
    if r["pos_err_mm"] > 1 or r["rot_err_deg"] > 1 or r["min_joint_margin_deg"] < 10:
        return cand, -1.0, r
    d = mujoco.MjData(m); d.qpos[:] = qpos; mujoco.mj_kinematics(m, d)
    if min(rep[s]["min_joint_margin_deg"] for s in rep) < 10 or min(rep[s]["elbow_outward_m"] for s in rep) < 0.05:
        return cand, -1.0, r
    arm = {s: [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(f"robot_{s}_arm_link")]
           for s in ("left", "right")}
    ft = np.zeros(6)
    r["arms_gap_m"] = round(min(mujoco.mj_geomDistance(m, d, a, b, 0.5, ft) for a in arm["left"] for b in arm["right"]), 3)
    r["ego_arm_share"] = round(ego_share(m, d, set(arm["left"] + arm["right"])), 3)
    joints = [m.joint(f"robot_{SIDE}_arm_dof_joint{i}").id for i in range(1, 8)]
    ch = Chain(m, m.body(f"robot_{SIDE}_arm_gripper").id, joints)
    links = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(f"robot_{SIDE}_arm_link")
             and m.body(m.geom_bodyid[g]).name[-1] in "234567"]
    near = [g for g in links if m.body(m.geom_bodyid[g]).name[-1] in "45"]
    table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "g_table")          # -1 without the table
    hand = [g for g in range(m.ngeom) if m.geom(g).name.startswith(f"robot_{SIDE}_collision_hand_")]
    mount = B.mount_rotation(SIDE, placement)
    home_q = d.qpos[ch.qadr].copy()
    mid = (ch.lo + ch.hi) / 2
    ok = 0
    probes = probe_poses(m, d)
    hand_root = m.body(f"robot_{SIDE}_{SIDE}_base_yaw")
    wrist = [m.jnt_qposadr[m.joint(f"robot_{SIDE}_{SIDE}_{k}").id] for k in ("pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z")]
    from tool_library.common import plan as P
    rig = P.scene_rigs(m)[SIDE]
    for k, p, R in probes:
        good = False
        for seed in (home_q, mid):
            d.qpos[ch.qadr] = seed
            pe, re = ch.solve(d, p, R @ mount.T, rest=home_q, iters=200)
            q = d.qpos[ch.qadr]
            if pe > 1e-3 or re > np.radians(1) or np.any(q < ch.lo + MARGIN) or np.any(q > ch.hi - MARGIN):
                continue
            # put the floating hand where the arm holds it, for the clearance checks
            d.qpos[wrist] = P.wrist_joints_from_world(m, d, rig, p, R, d.qpos[wrist[3:]])
            mujoco.mj_kinematics(m, d)
            if table >= 0 and any(mujoco.mj_geomDistance(m, d, g, table, 0.05, ft) < 0.01 for g in links):
                continue
            if any(mujoco.mj_geomDistance(m, d, g, h, 0.02, ft) < 0.0 for g in near for h in hand):
                continue
            good = True
            break
        ok += good
    return cand, ok / len(probes), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--half-spans", type=float, nargs="+", default=[0.25, 0.35, 0.45])
    ap.add_argument("--log", default=str(HERE / "search_results" / "results.jsonl"),
                    help="every evaluated candidate is appended here; a rerun skips the ones already in it")
    ap.add_argument("--max-ego-share", type=float, default=0.30,
                    help="largest share of the ego view the arms may cover at home (forearms enter the frame from below)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate at most this many new candidates, then rank")
    a = ap.parse_args()
    grid = list(itertools.product(a.half_spans, (-0.45, -0.30, -0.15), (-0.25, -0.10, 0.05),
                                  range(0, 360, 45), range(0, 360, 45)))
    import json
    log = Path(a.log); log.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line); done[tuple(r["cand"])] = (tuple(r["cand"]), r["score"], r["report"])
    todo = [c for c in grid if tuple(float(v) for v in c) not in {tuple(float(v) for v in k) for k in done}]
    todo = todo[:a.limit] if a.limit else todo
    print(f"{len(done)} candidates already in {log}, {len(todo)} to evaluate", flush=True)
    with Pool(a.workers) as pool, log.open("a") as out:   # appended as they finish, so a stopped run keeps its work
        for r in pool.imap_unordered(evaluate, todo, chunksize=2):
            out.write(json.dumps({"cand": list(r[0]), "score": r[1], "report": r[2]}) + "\n"); out.flush()
            done[tuple(r[0])] = r
    res = list(done.values())
    held = [r for r in res if r[1] >= 0]
    res = sorted((r for r in held if r[2]["arms_gap_m"] > 0.05 and r[2]["ego_arm_share"] <= a.max_ego_share),
                 key=lambda r: r[1])
    print(f"{len(done)} evaluated (of {len(grid)}), {len(held)} hold the home pose, {len(res)} also keep the arms 5 cm "
          f"apart and cover at most {a.max_ego_share:.0%} of the ego view")
    for cand, score, rep in res[-a.top:]:
        print(f"half_span {cand[0]:.2f}  dy {cand[1]:+.2f}  dz {cand[2]:+.2f}  yaw {cand[3]:3d}  roll {cand[4]:3d}"
              f"  coverage {score:.3f}  home margin {rep['min_joint_margin_deg']} deg  arms gap {rep['arms_gap_m']} m"
              f"  ego arm share {rep['ego_arm_share']}  elbow out {rep['elbow_outward_m']} m")


if __name__ == "__main__":
    main()
