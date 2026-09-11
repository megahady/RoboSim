"""Step 5 — efficacy metrics: turning a recording into judeable numbers.

The whole point of a controlled study is that "this controller is good" is
replaceable by a handful of measured numbers.  Every metric below is defined
from the *recorded observations only*, so any two runs (different designs,
ablation, seeds) can be compared apples-to-apples.

A recording is a list of queried observations in time order, each a dict as
produced by ``LeggedEnv.observe()`` (plus ``ctrl`` appended at step time).
"""

from __future__ import annotations

import math

import numpy as np

import mujoco

FALL_TILT = math.radians(50.0)


class Metrics:
    """Compute the §5 metric set from a run's recording."""

    def __init__(self, rec: list[dict], dt: float | None = None):
        self.rec = rec
        self.t = np.array([r["time"] for r in rec]) if rec else np.zeros(0)
        if dt is None and len(self.t) > 1:
            dt = self.t[1] - self.t[0]
        self.dt = dt or 0.0

    # ---------------------------------------------------------------- helpers
    def _series(self, key):
        return np.array([r[key] for r in self.rec]) if self.rec else np.zeros((0,))

    def _base(self, key):
        arr = self._series(key)
        return arr.reshape(len(arr), -1) if arr.ndim == 1 else arr

    # --------------------------------------------------------------- metrics
    def velocity_tracking(self, cmd_dir=(1.0, 0.0), cmd_speed=0.2):
        """Mean forward speed along cmd_dir vs commanded; RMS lateral drift."""
        pos = self._base("base_pos")
        v = self._series("base_linvel")
        if pos.shape[0] < 2 or v.size == 0:
            return {"v_mean": math.nan, "v_rms_err": math.nan, "cmd": cmd_speed}
        ex, ey = cmd_dir[0], cmd_dir[1]
        n = np.linalg.norm([ex, ey]) or 1.0
        ex, ey = ex / n, ey / n
        v_par = v[:, 0] * ex + v[:, 1] * ey
        v_perp = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 - v_par ** 2)
        return {
            "v_mean": float(np.mean(v_par)),
            "v_rms_err": float(np.sqrt(np.mean((v_par - cmd_speed) ** 2))),
            "v_side_rms": float(np.sqrt(np.mean(v_perp ** 2))),
            "cmd": cmd_speed,
        }

    def stability(self):
        """Body tilt envelope + base height variance (smaller = more stable)."""
        eul = self._base("base_euler")
        pos = self._base("base_pos")
        if eul.shape[0] == 0:
            return {"max_tilt_deg": float("nan"), "height_std": float("nan"), "fallen": bool(self.fallen)}
        tilt = np.rad2deg(np.max(np.abs(eul[:, 0:2]), axis=1))
        return {
            "max_tilt_deg": float(np.max(tilt)),
            "mean_tilt_deg": float(np.mean(tilt)),
            "height_std": float(np.std(pos[:, 2])) if pos.shape[1] >= 3 else float("nan"),
            "fallen": bool(self.fallen),
        }

    @property
    def fallen(self):
        """True if any frame exceeded the tilt fall threshold."""
        eul = self._base("base_euler")
        if eul.shape[0] == 0:
            return False
        return bool(np.any(np.max(np.abs(eul[:, 0:2]), axis=1) > FALL_TILT))

    def energy(self):
        """Mean total mechanical energy excursion (KE+PE) — actuator use proxy."""
        if not self.rec or "ek" not in self.rec[0]:
            return {"mean_energy": float("nan"), "ctrl_rms": float("nan")}
        ek = np.array([r["ek"] for r in self.rec])
        ep = np.array([r["ep"] for r in self.rec])
        ctrl = np.array([r.get("ctrl", np.zeros(1)) for r in self.rec]).reshape(len(self.rec), -1)
        return {
            "mean_energy": float(np.mean(ek + ep)),
            "ctrl_rms": float(np.sqrt(np.mean(ctrl ** 2))) if ctrl.size else float("nan"),
        }

    def cost_of_transport(self, mass: float, g: float = 9.81):
        """CoT = mechanical work / (m*g*distance). ~0.5..2 for legged robots."""
        d = self._series("base_linvel")
        qvel = self._base("joint_qvel")
        if len(self.rec) == 0 or qvel.shape[0] == 0 or d.shape[0] == 0:
            return float("nan")
        work = 0.0
        for i, r in enumerate(self.rec):
            ctrl = np.asarray(r.get("ctrl", np.zeros(qvel.shape[1])), dtype=float)
            # consumed mechanical power proxies as |sum_j tau_j * dq_j| (no negative
            # cancelation), integrated over the control period.
            joints = qvel[i] if qvel.ndim == 2 else qvel
            k = min(len(ctrl), len(joints))
            work += abs(float(np.sum(ctrl[:k] * joints[:k]))) * self.dt
        dist = float(np.sum(np.hypot(d[:, 0], d[:, 1]) * self.dt))
        return work / (mass * g * dist) if dist > 1e-6 else float("nan")

    def summary(self, mass: float) -> dict:
        """The compact 'efficacy card' printed for every run."""
        vt = self.velocity_tracking()
        st = self.stability()
        en = self.energy()
        cot = self.cost_of_transport(mass)
        return {
            **vt, **st, **en,
            "cot": cot,
            "success": not self.fallen,
            "runs": 1, "falls": int(self.fallen),
        }


def fall_percentage(records: list[list[dict]]) -> float:
    """Share of falled runs across a set of recordings (robustness metric)."""
    if not records:
        return float("nan")
    return 100.0 * sum(1 for r in records if Metrics(r).fallen) / len(records)