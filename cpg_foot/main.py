#!/usr/bin/env python3
"""Freeze 3 legs and draw a half-elliptic foot path with a CPG on the 4th.

Usage (macOS viewer requires ``mjpython``):

    python cpg_foot/main.py                 # headless + plot
    .venv/bin/mjpython cpg_foot/main.py --viewer
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from robot_studio.model_loader import urdf_to_mjcf, apply_scene_style  # noqa: E402

from kinematics import fk_chain, ik, nominal_targets, LEG_YAW  # noqa: E402
from controller import EllipticCpgController  # noqa: E402


CONTROL_PERIOD = 0.02


def load_model(urdf: str) -> mujoco.MjModel:
    xml = urdf_to_mjcf(urdf, add_ground=True)
    return apply_scene_style(mujoco.MjModel.from_xml_string(xml))


def place_on_floor(model, data):
    """Give the robot its nominal stance, then lift it just clear of floor."""
    m = model
    mujoco.mj_resetData(m, data)
    free = None
    for j in range(m.njnt):
        if int(m.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
            free = j
            break
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        for leg in ("FL", "FR", "RL", "RR"):
            if name == "%s_coxa_joint" % leg:
                data.qpos[m.jnt_qposadr[j]] = LEG_YAW[leg]
            elif name == "%s_femur_joint" % leg:
                data.qpos[m.jnt_qposadr[j]] = -0.95
            elif name == "%s_tibia_joint" % leg:
                data.qpos[m.jnt_qposadr[j]] = 0.82
    mujoco.mj_forward(m, data)
    if free is not None:
        lowest = float("inf")
        for g in range(m.ngeom):
            if int(m.geom_type[g]) in (mujoco.mjtGeom.mjGEOM_PLANE,):
                continue
            s = np.asarray(m.geom_size[g], dtype=float)
            r = float(np.sqrt(float(np.dot(s, s))))
            lowest = min(lowest, float(data.geom_xpos[g, 2]) - r)
        data.qpos[m.jnt_qposadr[free] + 2] = max(0.0, 0.02 - lowest)
        mujoco.mj_forward(m, data)


def foot_tip_chain(model, data, leg):
    """Active foot tip in chain coords (u, v) from the current joints."""
    qf = q_for(model, data, leg, "femur")
    qt = q_for(model, data, leg, "tibia")
    return fk_chain(leg, qf, qt)


def q_for(model, data, leg, part):
    name = "%s_%s_joint" % (leg, part)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qpos[model.jnt_qposadr[jid]])


def base_fdof(model):
    """Return (qpos adr, qvel adr, n) of the floating base free joint."""
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
            return (model.jnt_qposadr[j], model.jnt_dofadr[j], 7, 6)
    return (None, None, 0, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", default=os.path.join(HERE, "pla_hexapod.urdf"))
    ap.add_argument("--leg", default="FL", help="active (CPG) leg")
    ap.add_argument("--freq", type=float, default=0.4, help="CPG frequency Hz")
    ap.add_argument("--amp-u", type=float, default=0.04, help="ellipse radial half-axis")
    ap.add_argument("--amp-v", type=float, default=0.05, help="ellipse height half-axis")
    ap.add_argument("--dur", type=float, default=6.0, help="simulated seconds")
    ap.add_argument("--viewer", action="store_true", help="show the MuJoCo viewer")
    ap.add_argument("--hold-base", action="store_true",
                    help="pin the torso in place (kinematic) so only the leg sweeps")
    args = ap.parse_args()

    model = load_model(args.urdf)
    data = mujoco.MjData(model)
    place_on_floor(model, data)

    ctrl = EllipticCpgController(model, active_leg=args.leg,
                                 freq_hz=args.freq,
                                 amp_u=args.amp_u, amp_v=args.amp_v,)
    ctrl.reset()
    print(ctrl.describe())

    fq, fv, nq, nv = base_fdof(model)
    pin = (args.hold_base and fq is not None)
    base_qpos = data.qpos[fq:fq + 7].copy().astype(np.float64) if pin else None

    def do_substep():
        for i in range(n_sub):
            tau = ctrl.torque(model, data, advance_cpg=(i == 0),
                              dt=CONTROL_PERIOD)
            mujoco.mj_step1(model, data)
            data.ctrl[:] = tau
            mujoco.mj_step2(model, data)
            if pin:
                data.qpos[fq:fq + nq] = base_qpos
                data.qvel[fv:fv + nv] = 0.0
                mujoco.mj_kinematics(model, data)

    # ---- reference ellipse (for the plot) ---------------------------------
    n_ell = 360
    ref_u = ctrl.u0 + np.array([ctrl.cpg.amp_u * math.cos(2 * math.pi * k / n_ell)
                                for k in range(n_ell)])
    ref_v = ctrl.v0 + np.array([max(0.0, ctrl.cpg.amp_v * math.sin(2 * math.pi * k / n_ell))
                                for k in range(n_ell)])

    # ---- run -----------------------------------------------------------------
    u_track, v_track = [], []
    n_sub = max(1, round(CONTROL_PERIOD / model.opt.timestep))
    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as v:
            n = 0
            while v.is_running() and n * CONTROL_PERIOD < args.dur:
                do_substep()
                u, vv = foot_tip_chain(model, data, args.leg)
                u_track.append(u)
                v_track.append(vv)
                n += 1
                v.sync()
    else:
        n_steps = int(args.dur / CONTROL_PERIOD)
        for n in range(n_steps):
            do_substep()
            u, vv = foot_tip_chain(model, data, args.leg)
            u_track.append(u)
            v_track.append(vv)

    # ---- diagnostics + plot ---------------------------------------------------
    u_track, v_track = np.array(u_track), np.array(v_track)
    peak = float(np.max(v_track))
    span = float(np.max(u_track) - np.min(u_track))
    overshoot = float(np.max(np.abs(np.maximum(u_track - (ctrl.u0 + ctrl.cpg.amp_u), 0))))
    print("CPG root:   u0=%.3f v0=%.3f  ell A_u=%.2f A_v=%.2f" % (
        ctrl.u0, ctrl.v0, ctrl.cpg.amp_u, ctrl.cpg.amp_v))
    print("trace:      n=%d  peak_v=%.3f  u_span=%.3f  (ground line v=%.3f)" % (
        len(u_track), peak, span, ctrl.v0))
    print("reach check: u_max=%.3f  (limit ~%.3f)" % (
        float(np.max(u_track)), 0.1607 + 0.2121))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ref_u, ref_v, "--", color="gray", lw=1.2, label="CPG reference")
        ax.plot(u_track, v_track, "-", color="crimson", lw=1.6, label="foot trace")
        ax.axhline(ctrl.v0, color="k", lw=0.8)
        ax.set_xlabel("u  (radial in leg plane, m)")
        ax.set_ylabel("v  (up in leg plane, m)")
        ax.set_title("Half-elliptic foot path over the ground  [%s]" % (
            "Coxa frozen, other legs at stance" if False else "CPG %s leg" % args.leg))
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        out = os.path.join(HERE, "foot_path.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print("saved plot -> %s" % out)
    except Exception as exc:  # keep running even if matplotlib is missing
        print("(plot skipped: %s)" % exc)


if __name__ == "__main__":
    main()