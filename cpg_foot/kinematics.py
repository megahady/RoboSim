"""Leg kinematics for one PLA hexapod leg.

Kept intentionally free of MuJoCo: the chain from the coxa (yaw) joint
origin to the foot tip is modeled with two planar segments (femur, tibia)
that rotate about the leg's y-axis.  ``fk_chain`` / ``ik`` operate in the
leg's own "chain coordinates":

    u  = coxa-frame x  (radial, out of the body; URDF femur extends +x)
    v  = coxa-frame z  (up; URDF coxa z is the vertical axis)

``foot_in_legframe`` folds in the 5 cm coxa offset so ``u`` runs from the
coxa origin; ``foot_base`` adds the coxa yaw rotation and the joint origin
so results can be checked against MuJoCo.
"""

from __future__ import annotations

import math

L_FEMUR = math.hypot(0.15, 0.06)    # femur pivot -> tibia pivot segment
L_TIBIA = math.hypot(0.15, 0.15)    # tibia pivot -> foot tip segment
THETA_FEMUR = math.atan2(-0.06, 0.15)
THETA_TIBIA = math.atan2(-0.15, 0.15)

# URDF leg origins (coxa joint frame in base_link coordinates) and the
# fixed coxa yaw (rpy z) that swings the leg out of the body plane.
LEG_ORIGIN = {
    "FL": (0.18, 0.11, -0.01),
    "FR": (0.18, -0.11, -0.01),
    "RL": (-0.18, 0.11, -0.01),
    "RR": (-0.18, -0.11, -0.01),
}
LEG_YAW = {"FL": 1.5708, "FR": -1.5708, "RL": -1.5708, "RR": 1.5708}

# Nominal stance (radians) used by the frozen legs and as the ellipse center.
NOMINAL = {"coxa": 0.0, "femur": -0.95, "tibia": 0.82}
NOMINAL_COXA = LEG_YAW  # coxa holds its fixed yaw in stance


def _dir(leg: str) -> int:
    """+1 for front legs, -1 for rear legs (their chains point -x)."""
    return 1 if leg in ("FL", "FR") else -1


def _rot2d(a: float, u: float, v: float) -> tuple[float, float]:
    c, s = math.cos(a), math.sin(a)
    return (c * u - s * v, s * u + c * v)


def fk_chain(leg: str, qf: float, qt: float) -> tuple[float, float]:
    """Foot tip in chain coordinates (u, v) from femur/tibia angles."""
    s = _dir(leg)
    p1 = _rot2d(-s * qf, 0.15, -0.06)
    p2 = _rot2d(-s * (qf + qt), 0.15, -0.15)
    return (p1[0] + p2[0], p1[1] + p2[1])


def _two_link_ik(ux: float, vx: float, s: int) -> tuple[float, float]:
    """Planar 2-link IK in chain coords.  ``s`` picks the elbow branch."""
    d = math.hypot(ux, vx)
    beta = math.atan2(vx, ux)
    c2 = (d * d - L_FEMUR * L_FEMUR - L_TIBIA * L_TIBIA) / (2 * L_FEMUR * L_TIBIA)
    c2 = max(-1.0, min(1.0, c2))
    delta = -s * math.acos(c2)
    t1 = beta - math.atan2(L_TIBIA * math.sin(delta), L_FEMUR + L_TIBIA * math.cos(delta))
    r1 = t1 - THETA_FEMUR
    r2 = delta - (THETA_TIBIA - THETA_FEMUR)
    return -s * r1, -s * r2


def ik(leg: str, ux: float, vx: float) -> tuple[float, float]:
    """Femur/tibia angles that place the foot at chain (ux, vx)."""
    return _two_link_ik(ux, vx, _dir(leg))


def foot_in_legframe(leg: str, qf: float, qt: float) -> tuple[float, float]:
    """Foot tip in the coxa frame: (radial-out, up), including the 5cm coxa."""
    u, v = fk_chain(leg, qf, qt)
    return (_dir(leg) * (0.05 + u), v)


def foot_base(leg: str, qc: float, qf: float, qt: float) -> tuple[float, float, float]:
    """Foot tip in base_link coordinates (MuJoCo-verified)."""
    s = _dir(leg)
    u, v = fk_chain(leg, qf, qt)
    cx = s * (0.05 + u)
    a = LEG_YAW[leg] + qc
    ca, sa = math.cos(a), math.sin(a)
    bx, by, bz = LEG_ORIGIN[leg]
    return (bx + ca * cx, by + sa * cx, bz + v)


def nominal_targets(legs: list[str]) -> dict:
    """Joint-name -> angle dict holding every listed leg at stance."""
    t = {}
    for leg in legs:
        t[f"{leg}_coxa_joint"] = LEG_YAW[leg]
        t[f"{leg}_femur_joint"] = NOMINAL["femur"]
        t[f"{leg}_tibia_joint"] = NOMINAL["tibia"]
    return t