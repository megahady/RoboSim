"""Step 3 architecture: a layered controller for floating-base legged robots.

We deliberately separate the *two things a legged controller computes* into
independent layers so each can be tested/replaced on its own:

    Layer 1  gaits.PhaseGait — the *reference generator*.  Turns a phase
        oscillator (tripod gait: two anti-phase leg groups) into a joint-space
        target q_ref(t) for coxa/femur/tibia across all 6 legs.  This is the
        "what the legs should be doing" layer.

    Layer 2  gaits.JointPD   — the *stabilizer*.  Converts joint-space targets
        into motor torques:

            tau = Kp*(q_ref - q) + Kd*(dq_ref - dq) + bias_feedforward(model)

        The bias term cancels gravity/Coriolis (makes the leg behave roughly
        linear), which keeps the gait's performance close to design intent.

``LayeredController`` composes them into the single ``step(model, data, obs)``
signature the env calls — and the same object can be dropped into the batch
experiments, the GUI prototype, or an RL loop unchanged.
"""

from __future__ import annotations

import math

import numpy as np

import mujoco

# Coordinate frame of the legs for the reduced four-leg robot.  The gait still
# alternates phase between two groups, but now the robot has only the corner
# legs: front-left, front-right, rear-left, rear-right.
LEGS = ["FL", "FR", "RL", "RR"]
_GROUP_A = ["FL", "RR"]                 # diagonal pair A
_GROUP_B = ["FR", "RL"]                 # diagonal pair B (anti-phase)

# Default parameters exposed as a dict, mirroring the app's PARAMS convention.
DEFAULT_PARAMS = {
    "stride_freq_hz": 0.6,
    "step_length_m": 0.22,
    "step_height_m": 0.07,
    "body_pitch_rad": 0.0,
    "kp": 30.0,
    "kd": 4.0,
}


def _lookup(model) -> dict:
    """actuator index -> (joint id, qpos index, dof index, joint name)."""
    out = {}
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        out[i] = (jid, int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]),
                  mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "")
    return out


class PhaseGait:
    """Layer 1 — phase-based tripod reference generator (periodic targets)."""

    def __init__(self, legs=LEGS, group_a=_GROUP_A, params=None):
        self.legs = legs
        self.group_a = set(group_a)
        self.params = dict(DEFAULT_PARAMS)
        self.params.update(params or {})

    def reference(self, t: float) -> dict:
        """Return per-joint target angles {joint_name: q_ref} at time t."""
        p = self.params
        w = 2.0 * math.pi * p["stride_freq_hz"]
        out = {}
        for leg in self.legs:
            # Tripod: group B is offset by pi, so while A swings B is in stance.
            phase = 0.0 if leg in self.group_a else math.pi
            ph = phase + w * t
            swing = 0.5 * (1.0 - math.cos(ph))        # 0..1, peaks at mid-cycle
            out["%s_coxa_joint" % leg] = 0.5 * p["step_length_m"] * math.sin(ph)
            out["%s_femur_joint" % leg] = p["body_pitch_rad"] - p["step_height_m"] * swing
            out["%s_tibia_joint" % leg] = -0.55 * p["step_height_m"] * swing
        return out


class JointPD:
    """Layer 2 — PD + bias feed-forward at the actuated joints."""

    def __init__(self, kp=20.0, kd=2.0):
        self.kp = kp
        self.kd = kd

    def torque(self, model, data, targets: dict) -> np.ndarray:
        tau = np.zeros(model.nu)
        lookup = _lookup(model)
        for i, (jid, qadr, dadr, jname) in lookup.items():
            if jname not in targets:
                continue
            q = float(data.qpos[qadr])
            dq = float(data.qvel[dadr])
            tau[i] = (self.kp * (targets[jname] - q) - self.kd * dq
                      + float(data.qfrc_bias[dadr]))
        return tau


class LayeredController:
    """Full stack: PhaseGait (L1) -> JointPD (L2). Env-facing interface."""

    def __init__(self, params=None, legs=LEGS):
        pa = dict(DEFAULT_PARAMS)
        pa.update(params or {})
        self.gait = PhaseGait(legs=legs, params=pa)
        self.pd = JointPD(kp=pa["kp"], kd=pa["kd"])
        self.params = pa

    def reset(self, model, data, obs=None):
        pass

    def step(self, model, data, obs=None) -> np.ndarray:
        """Return torque vector for this control period (step 3 contract)."""
        targets = self.gait.reference(data.time)
        return self.pd.torque(model, data, targets)

    def reference(self, t: float):
        """Expose the joint-space targets (for plotting / diagnostics)."""
        return self.gait.reference(t)

    # ------------------------------------------------------------- accessible
    def describe(self) -> str:
        p = self.params
        return ("LayeredController(tripod | freq=%.2fHz step=%.2fm lift=%.2fm | "
                "kp=%.1f kd=%.1f)" % (p["stride_freq_hz"], p["step_length_m"],
                                      p["step_height_m"], p["kp"], p["kd"]))