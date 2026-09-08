"""Inverse-dynamics (computed-torque) trajectory tracking for the acrobot.

Motor mapping: the loader injects one torque motor per URDF joint in joint
order, so ``ctrl[0]`` drives joint1 and ``ctrl[1]`` drives joint2. The model
has no free joint, so qpos[0:2] are the two hinge angles (radians).

    tau = M(q) @ qdd_des + bias(q, dq)
    qdd_des = Kp*(q_ref - q) + Kd*(dq_ref - dq) + ddq_ref

The simulated joint state is plotted solid; the ``reference()`` target is
plotted dashed in the Joint state graph.
"""

import math

import mujoco
import numpy as np

TITLE = "Acrobot inverse-dynamics tracking"
HELP = (
    "Computed-torque tracking of a sinusoidal reference on both joints.\n"
    "Requires the acrobot (2 hinged motors, ctrl[0:2]). Matrix feedforward "
    "uses the mass matrix + gravity/Coriolis bias."
)

PARAMS = {
    "Target amplitude (rad)": (0.2, 1.5, 1.0),
    "Target frequency (Hz)": (0.1, 1.0, 0.3),
    "Phase offset joint 2 (rad)": (0.0, 6.283, 2.0),
    "Kp": (10.0, 200.0, 60.0),
    "Kd": (1.0, 30.0, 8.0),
}


def _target(t, params):
    amp = params["Target amplitude (rad)"]
    f = params["Target frequency (Hz)"]
    ph = params["Phase offset joint 2 (rad)"]
    w = 2.0 * math.pi * f
    return (
        np.array([amp * math.sin(w * t), amp * math.sin(w * t + ph)]),
        np.array([amp * w * math.cos(w * t), amp * w * math.cos(w * t + ph)]),
        np.array([-amp * w * w * math.sin(w * t), -amp * w * w * math.sin(w * t + ph)]),
    )


def init(model, data, params):
    if model.nv < 2:
        raise RuntimeError("This controller needs at least 2 dofs.")
    print("[controller] acrobot inverse-dynamics tracking active")


def reference(data, params):
    return _target(data.time, params)[0]


def step(model, data, params):
    nv = model.nv
    if nv < 2:
        raise RuntimeError("This controller needs at least 2 dofs.")
    q = np.asarray(data.qpos[0:2])
    dq = np.asarray(data.qvel[0:2])
    qref, dqref, ddqref = _target(data.time, params)
    kp = params["Kp"]
    kd = params["Kd"]
    qdd = kp * (qref - q) + kd * (dqref - dq) + ddqref

    flat = np.zeros((nv, nv))
    mujoco.mj_fullM(model, data, flat)
    M = flat[0:2, 0:2]
    tau = M @ qdd + np.asarray(data.qfrc_bias[0:2])
    data.ctrl[0:2] = tau