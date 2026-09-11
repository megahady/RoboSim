"""Controller: freeze 3 legs at stance, drive the 4th leg's CPG to trace a
half-elliptic foot path in its own plane (radial u, up v)."""

from __future__ import annotations

import math

import numpy as np

import mujoco

from cpg import Cpg
from kinematics import fk_chain, ik, nominal_targets, LEG_YAW

LEGS = ["FL", "FR", "RL", "RR"]


def _lookup(model):
    """actuator index -> (joint id, qpos idx, dof idx, joint name)."""
    out = []
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        out.append((i, jid, int(model.jnt_qposadr[jid]),
                    int(model.jnt_dofadr[jid]),
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""))
    return out


class EllipticCpgController:
    """Joint-space PD + gravity bias holding 3 legs, tracking 1 CPG leg."""

    def __init__(self, model, active_leg: str = "FL",
                 freq_hz: float = 0.4, amp_u: float = 0.04,
                 amp_v: float = 0.05, kp: float = 25.0,
                 kd: float = 3.0, bias_ff: bool = True):
        if active_leg not in LEGS:
            raise ValueError("active_leg must be one of %s" % LEGS)
        self.active_leg = active_leg
        self.frozen_legs = [leg for leg in LEGS if leg != active_leg]
        self.kp, self.kd = kp, kd
        self.bias_ff = bias_ff
        self.lookup = _lookup(model)

        self.cpg = Cpg(freq_hz=freq_hz, amp_u=amp_u, amp_v=amp_v)
        # nominal stance (the ellipse anchors to the resting foot tip)
        self.frozen_targets = nominal_targets(self.frozen_legs)
        self.nominal_q = nominal_targets([active_leg])
        u0, v0 = fk_chain(active_leg,
                          self.nominal_q[active_leg + "_femur_joint"],
                          self.nominal_q[active_leg + "_tibia_joint"])
        self.u0, self.v0 = u0, v0
        self.t = 0.0
        self._prev_ref: dict | None = None
        self._prev_t = 0.0

    def reset(self):
        self.cpg.reset()
        self._prev_ref = None
        self._prev_t = 0.0

    # --------------------------------------------------- CPG -> joint refs
    def leg_reference(self, t: float):
        """Foot target in chain coords, then femur/tibia from IK."""
        u, v = self.cpg.outp()
        v_up = max(0.0, v)                    # upper half ellipse only
        u_t = self.u0 + u
        v_t = self.v0 + v_up
        qf, qt = ik(self.active_leg, u_t, v_t)
        return {
            self.active_leg + "_coxa_joint": LEG_YAW[self.active_leg],
            self.active_leg + "_femur_joint": qf,
            self.active_leg + "_tibia_joint": qt,
        }

    # -------------------------------------------------------------- control
    def torque(self, model, data, obs=None, advance_cpg: bool = True,
               dt: float = 0.02) -> np.ndarray:
        """Joint-space PD + gravity bias.

        ``advance_cpg`` should be True once per CPG period (step the phase
        oscillators), False on the intermediate substeps where only the PD is
        re-evaluated from the same reference.
        """
        tau = np.zeros(model.nu)
        if advance_cpg:
            self.cpg.step(dt)
        t = data.time
        targets = dict(self.frozen_targets)
        targets.update(self.leg_reference(t))
        dt = max(1e-6, t - self._prev_t)
        for (i, jid, qadr, dadr, jname) in self.lookup:
            if jname not in targets:
                continue
            q = float(data.qpos[qadr])
            dq = float(data.qvel[dadr])
            if self._prev_ref is not None and jname in self._prev_ref:
                dq_des = (targets[jname] - self._prev_ref[jname]) / dt
            else:
                dq_des = 0.0
            tau[i] = (self.kp * (targets[jname] - q)
                      + self.kd * (dq_des - dq))
            if self.bias_ff:
                tau[i] += float(data.qfrc_bias[dadr])
        if advance_cpg:
            self._prev_ref = targets
            self._prev_t = t
        return tau

    def step(self, model, data, obs=None) -> np.ndarray:
        return self.torque(model, data, obs)

    def reference(self, t: float):
        targets = dict(self.frozen_targets)
        targets.update(self.leg_reference(t))
        return targets

    def describe(self) -> str:
        p = self.cpg
        return ("EllipticCpgController(active=%s freq=%.2fHz A_u=%.2f A_v=%.2f "
                "| frozen=%s kp=%.0f kd=%.0f)"
                % (self.active_leg, p.w / (2 * math.pi), p.amp_u, p.amp_v,
                   ",".join(self.frozen_legs), self.kp, self.kd))