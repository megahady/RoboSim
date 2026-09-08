"""Plain diagonal PD plus gravity/Coriolis bias compensation for the acrobot.

The bias term ``data.qfrc_bias`` is already evaluated by MuJoCo during the
previous step, so no extra inverse-dynamics call is needed.

Motor mapping: ctrl[0] drives joint1, ctrl[1] drives joint2 (URDF joint order).
"""

import math

import numpy as np

TITLE = "Acrobot PD + gravity compensation"
HELP = "Diagonal PD with bias feedforward on the two joints. Simpler and more\nrobust than full inverse dynamics, but tracks only asymptotically."

PARAMS = {
    "Target amplitude (rad)": (0.2, 1.5, 0.8),
    "Target frequency (Hz)": (0.1, 1.0, 0.3),
    "Phase offset joint 2 (rad)": (0.0, 6.283, 2.0),
    "Kp": (1.0, 200.0, 40.0),
    "Kd": (0.1, 40.0, 6.0),
}


def _target(t, params):
    amp = params["Target amplitude (rad)"]
    f = params["Target frequency (Hz)"]
    ph = params["Phase offset joint 2 (rad)"]
    w = 2.0 * math.pi * f
    return (
        np.array([amp * math.sin(w * t), amp * math.sin(w * t + ph)]),
        np.array([amp * w * math.cos(w * t), amp * w * math.cos(w * t + ph)]),
    )


def init(model, data, params):
    if model.nv < 2:
        raise RuntimeError("This controller needs at least 2 dofs.")
    print("[controller] acrobot PD + gravity compensation active")


def reference(data, params):
    return _target(data.time, params)[0]


def step(model, data, params):
    if model.nv < 2:
        raise RuntimeError("This controller needs at least 2 dofs.")
    q = np.asarray(data.qpos[0:2])
    dq = np.asarray(data.qvel[0:2])
    qref, dqref = _target(data.time, params)
    kp = params["Kp"]
    kd = params["Kd"]
    tau = kp * (qref - q) + kd * (dqref - dq) + np.asarray(data.qfrc_bias[0:2])
    data.ctrl[0:2] = tau