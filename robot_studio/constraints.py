"""Physical actuator limits: torque, velocity, and position.

MuJoCo enforces *position* limits natively (``jnt_range``, from the URDF's
``<limit>``).  Torque is only bounded by ``actuator_ctrlrange`` — but that
limits the **command input**, not the force the actuator can actually apply;
the real cap is ``actuator_forcerange`` (which defaults to [+0, +0] = unlimited).
And MuJoCo has **no native joint velocity limit** at all.

So to mimic a real servo we add what's missing:
  - torque:   set ``actuator_forcerange = +-ctrlrange`` (hard force saturation)
  - velocity: clamp the commanded torque once ``|qvel|`` approaches ``vmax``
              (a servomotor's torque drops off past its speed limit)
  - position: inherited from the URDF limits + a soft PD-like brake near the
              mechanical stops

``LimitSet`` inspects one compiled model and derives the per-joint limits;
``ClampedController`` wraps any ``step(model, data, obs)->tau`` controller so
the *same* physics gets limited command channels the way a real motor driver
would.
"""

from __future__ import annotations

import numpy as np
import mujoco


def _joint_for_actuator(model, act: int) -> int:
    return int(model.actuator_trnid[act, 0])


def _dof_of_joint(model, jnt: int) -> int:
    return int(model.jnt_dofadr[jnt])


def apply_forcerange(model: mujoco.MjModel) -> None:
    """Set actuator force limits equal to the command (torque) limits.

    Without this the solver can apply unbounded force even though ``ctrl`` is
    bounded — unrealistic for a real motor.
    """
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        model.actuator_forcerange[i] = (lo, hi)
    model.actuator_forcelimited[:] = model.actuator_ctrllimited[:]


def joint_velocity_limits(model: mujoco.MjModel) -> np.ndarray:
    """Return a single velocity cap (rad/s) per actuator = per actuated joint.

    Uses the joint-position range as a rough proxy for the servo's speed class:
    wider-range joints (legs) get a higher cap.  Real hexapod servos (HS-485
    class) run ~ 4-8 rad/s.
    """
    vmax = np.full(model.nu, 6.0)
    for i in range(model.nu):
        jnt = _joint_for_actuator(model, i)
        span = float(model.jnt_range[jnt, 1] - model.jnt_range[jnt, 0])
        if span <= 0.0:                      # continuous joint: default cap
            vmax[i] = 8.0
        elif span <= 3.2:                    # ~ coxa: rotate more slowly
            vmax[i] = 5.0
        else:                                # femur/tibia: lift faster
            vmax[i] = 7.0
    return vmax


class LimitSet:
    """Per-actuator (torque, velocity) limits derived from one model."""

    def __init__(self, model: mujoco.MjModel):
        apply_forcerange(model)
        tmax = np.empty(model.nu)
        for i in range(model.nu):
            hi = float(model.actuator_forcerange[i, 1])
            lo = float(model.actuator_forcerange[i, 0])
            tmax[i] = hi if abs(hi) >= abs(lo) else -lo
        self.tmax = tmax
        self.vmax = joint_velocity_limits(model)
        # dof index (qvel) + qpos index + lower/upper angle per actuated joint
        self.brake = [
            (int(model.jnt_dofadr[_joint_for_actuator(model, i)]),
             int(model.jnt_qposadr[_joint_for_actuator(model, i)]),
             float(model.jnt_range[_joint_for_actuator(model, i), 0]),
             float(model.jnt_range[_joint_for_actuator(model, i), 1]))
            for i in range(model.nu)]

    def describe(self) -> str:
        m = self.tmax.min()
        return ("limits | torque max=%.1f N.m/joint, joint vel cap "
                "%.0f..%.0f rad/s, position from URDF ranges" % (m, self.vmax.min(), self.vmax.max()))


def clamp_torque(tau, qvel, qpos, lims: "LimitSet") -> np.ndarray:
    """Apply the three limits to a raw torque command (in-place into new array).

    torque:   hard saturation to +-tmax
    velocity: servo torque rolls off as |qvel| approaches/ exceeds vmax
              (``tau *= vmax/|qvel|`` when fighting above the speed limit)
    position: soft brake as the joint nears its mechanical stop (like the
              compliant end-stop of a real gearbox)
    """
    out = np.clip(tau, -lims.tmax, lims.tmax)
    for i, (dof, _qadr, _lo, _hi) in enumerate(lims.brake):
        qv = float(qvel[dof])
        av = abs(qv)
        if av > lims.vmax[i]:
            # torque in the direction that would push beyond the speed cap is
            # scaled down (a real motor cannot out-torque past its no-load rpm)
            if out[i] * qv > 0.0:
                out[i] *= lims.vmax[i] / av
    for i, (_dof, qadr, lo, hi) in enumerate(lims.brake):
        q = float(qpos[qadr])
        margin = 0.12                      # rad of soft-stop cushion
        d = max(lo + margin - q, 0.0) if q < lo + margin else (
            min(q - (hi - margin), 0.0) if q > hi - margin else 0.0)
        out[i] += d * -20.0                # spring pushing away from the stop
    return out


class ClampedController:
    """Wrap a ``step(model, data, obs) -> tau`` controller with real limiter."""

    def __init__(self, inner, model: mujoco.MjModel):
        self.inner = inner
        self.lims = LimitSet(model)

    def step(self, model, data, obs=None) -> np.ndarray:
        tau = self.inner.step(model, data, obs)
        return clamp_torque(tau, data.qvel[:model.nv], data.qpos[:model.nq],
                            self.lims)

    def __getattr__(self, name):
        # Never forward dunder lookups: copy.deepcopy probes __deepcopy__/
        # __getstate__/... and forwarding them into the wrapper recurses.
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    @property
    def params(self):
        return getattr(self.inner, "params", {})

    def describe(self) -> str:
        base = self.inner.describe() if hasattr(self.inner, "describe") else "controller"
        return base + " + " + self.lims.describe()