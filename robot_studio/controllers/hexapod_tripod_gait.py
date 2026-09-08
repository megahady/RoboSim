"""Tripod walking gait for 6-legged robots (PLA hexapod and similar).

Every 3-DOF leg gets a periodic pattern; the two leg groups move in
anti-phase so three legs support while three swing. Torque command =
PD around the gait target with gravity/Coriolis bias feed-forward.

The controller maps motors to legs by parsing actuator joint names like
``FL_coxa_joint`` / ``FL_femur_joint`` / ``FL_tibia_joint``.
It works for any number of legs with coxa/femur/tibia joints.
"""

import math

import mujoco
import numpy as np

TITLE = "Hexapod tripod gait"
HELP = ("Anti-phase tripod gait. Raise the stride frequency to move faster, "
        "grow step length for longer strides, or increase step height to "
        "clear the ground. Body pitch tilts the nose (negative pitches down).")

PARAMS = {
    "Stride frequency (Hz)": (0.1, 2.5, 0.6),
    "Step length (m)": (0.05, 0.6, 0.22),
    "Step height (m)": (0.05, 0.35, 0.14),
    "Body pitch ctrl (rad)": (-0.35, 0.35, 0.0),
    "Kp": (5.0, 60.0, 20.0),
    "Kd": (0.2, 8.0, 2.0),
}

_PHASE = {
    "FL": 0.0, "MR": 0.0, "RL": 0.0,
    "FR": math.pi, "ML": math.pi, "RR": math.pi,
}

_MOTORS = {}   # ctrl index -> (leg, task)
_MODEL = None


def _tasks(model, data):
    global _MOTORS, _MODEL
    if _MOTORS or model is None:
        return
    _MODEL = model
    legs = []
    found = {}
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        parts = jname.split("_")
        leg = parts[0].upper() if parts else ""
        task = parts[1] if len(parts) > 1 else "coxa"
        if leg and task in ("coxa", "femur", "tibia"):
            found[i] = (leg, task)
            if leg not in legs:
                legs.append(leg)
    if found:
        _MOTORS = found
        print("[controller] hexapod gait: %d motors across %d legs" % (len(found), len(legs)))


def _phase(leg, freq, t):
    return _PHASE.get(leg, 0.0) + 2.0 * math.pi * freq * t


def _targets(data, params):
    freq = params["Stride frequency (Hz)"]
    amp = params["Step length (m)"]
    lift = params["Step height (m)"]
    pitch = params["Body pitch ctrl (rad)"]
    t = float(data.time)
    out = {}
    for i, (leg, task) in _MOTORS.items():
        ph = _phase(leg, freq, t)
        sw = 0.5 * (1.0 - math.cos(ph))          # 0..1, peaks mid-cycle
        if task == "coxa":
            out[i] = 0.5 * amp * math.sin(ph)    # yaw swing step length
        elif task == "femur":
            out[i] = pitch - lift * sw           # lift during swing
        else:                                    # tibia
            out[i] = -0.55 * lift * sw           # curl during swing
    return out


def init(model, data, params):
    _MOTORS.clear()
    _tasks(model, data)
    if not _MOTORS:
        raise RuntimeError("No coxa/femur/tibia motors found; this controller "
                           "expects a hexapod-style URDF.")


def reference(data, params):
    """Target joint angles aligned with qpos (free-joint entries stay zero)."""
    m = _MODEL
    if m is None or not _MOTORS:
        return np.zeros(len(data.qpos))
    tgt = _targets(data, params)
    ref = np.zeros(len(data.qpos))
    for i, (leg, task) in _MOTORS.items():
        jid = int(m.actuator_trnid[i, 0])
        ref[int(m.jnt_qposadr[jid])] = tgt[i]
    return ref


def step(model, data, params):
    _tasks(model, data)
    if not _MOTORS:
        return
    targets = _targets(data, params)
    kp = params["Kp"]
    kd = params["Kd"]
    for i, (leg, task) in _MOTORS.items():
        jid = int(model.actuator_trnid[i, 0])
        q = float(data.qpos[model.jnt_qposadr[jid]])
        dq = float(data.qvel[model.jnt_dofadr[jid]])
        data.ctrl[i] = (kp * (targets[i] - q) - kd * dq
                        + float(data.qfrc_bias[model.jnt_dofadr[jid]]))