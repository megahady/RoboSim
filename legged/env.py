"""Step 1-4 architecture: the environment wrapper.

``LeggedEnv`` owns the *plant side* of the problem:
    - reset: drop/place the robot at a sensible start pose on the floor,
    - step:  one control period (several physics substeps) using the
             mj_step1 (dynamics) / mj_step2 (contact) split,
    - observe: the exact observation vector the controller is allowed to see
               (this is the "interface contract" with the policy),
    - reward / done / success: RL-style hooks and the fall detector.

Floating-base conventions (MuJoCo):
    - a free joint stores qpos[0:3] = base position, qpos[3:7] = base quat;
      qvel[0:3] = linear vel, qvel[3:6] = angular vel.
    - joint-space entries thereafter.  We locate everything from the model
      (qposadr/dofadr) so the code works for any robot, not just this one.
"""

from __future__ import annotations

import math

import numpy as np

import mujoco



def resolve_freejoint(model) -> int | None:
    """Index of the free joint if the model has a floating base, else None."""
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
            return j
    return None


def euler_from_quat(qw, qx, qy, qz):
    """Roll/pitch/yaw from a scalar-first quaternion (MuJoCo order)."""
    # Aerospace convention (yaw about z, pitch about y, roll about x).
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class LeggedEnv:
    """A single floating-base leg simulation + observation space."""

    # Control period (seconds). Physics still runs at model.opt.timestep:
    # the controller is evaluated once per period (step 4 of the recipe).
    CONTROL_PERIOD = 0.02

    def __init__(self, model: mujoco.MjModel, seed: int = 0,
                 drop_height: float = 0.0, noise_scale: float = 0.0):
        self.model = model
        self.free = resolve_freejoint(model)
        if self.free is None:
            raise ValueError("LeggedEnv requires a floating-base (free-joint) model.")

        self.drop_height = drop_height
        self.ctrl_per = self.CONTROL_PERIOD
        self.n_sub = max(1, round(self.ctrl_per / model.opt.timestep))
        self.rng = np.random.default_rng(seed)
        self.noise_scale = noise_scale

        # Keep mechanical energy in MjData.energy[] so metrics can separate
        # KE/PE (it costs ~nothing and makes the efficacy table meaningful).
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY

        # Precompute geometry indirection so observe() is trivial + general.
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.base_body = bid if bid >= 0 else 0
        self.feet = self._find_feet()
        self.data = mujoco.MjData(model)

        self.reset()

    # ------------------------------------------------------------ model census
    def _find_feet(self):
        """Map free-joint-relative foot bodies -> (bodyid, geomids) by name.

        A "foot" is any body whose name ends with '_foot'.  We later detect a
        touchdown by checking contacts between the ground and these geoms.
        """
        feet = []
        for b in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if name.endswith("_foot"):
                geoms = [g for g in range(self.model.ngeom)
                         if self.model.geom_bodyid[g] == b]
                feet.append((name, b, geoms))
        return feet

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None):
        """Reset and place on the floor.

        Legs start in a slightly *spread* nominal stance (femur out, tibia
        bent) mimicking a real robot at rest - starting with straight legs
        gives the gait no stance leverage to walk on.  The base height is set
        from that *actual* stance pose (not a straight-leg rest pose) so the
        feet begin just above the floor instead of through it - a feet-through-
        floor start is a guaranteed tumble.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        idx = m.jnt_qposadr[self.free]
        d.qpos[idx + 0] = 0.0                  # x
        d.qpos[idx + 1] = 0.0                  # y
        d.qpos[idx + 2] = 0.0                  # z: fixed below
        d.qpos[idx + 3] = 1.0                  # quat w (level)
        self._apply_nominal_stance()
        mujoco.mj_forward(m, d)
        d.qpos[idx + 2] = self._z_lift() + self.drop_height
        mujoco.mj_forward(m, d)
        self.t = 0.0
        self._start_qpos = d.qpos.copy()
        self._start_z = float(d.qpos[idx + 2])
        return self.observe()

    def _z_lift(self) -> float:
        """Lowest foot point (below zero => shift the base up to clear it)."""
        m, d = self.model, self.data
        lowest = float("inf")
        for _name, _bid, geoms in self.feet:
            for g in geoms:
                r = float(np.linalg.norm(np.asarray(m.geom_size[g])))
                lowest = min(lowest, float(d.geom_xpos[g, 2]) - r)
        if not np.isfinite(lowest):
            return 0.0
        return max(0.0, 0.01 - lowest)        # keep ~1 cm hover

    def _apply_nominal_stance(self):
        m, d = self.model, self.data
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if name.endswith("_femur_joint"):
                d.qpos[m.jnt_qposadr[j]] = -0.8
            elif name.endswith("_tibia_joint"):
                d.qpos[m.jnt_qposadr[j]] = 0.64
            elif name.endswith("_coxa_joint"):
                # group A faces +; group B faces - (symmetric fore/aft)
                d.qpos[m.jnt_qposadr[j]] = 0.15 if name.startswith(("FL", "ML", "RL")) else -0.15

    # ------------------------------------------------------------------ observe
    def observe(self) -> dict:
        """The observation vector handed to the controller (step 3 contract)."""
        m, d = self.model, self.data
        idx = m.jnt_qposadr[self.free]
        qw, qx, qy, qz = d.qpos[idx + 3:idx + 7]
        roll, pitch, yaw = euler_from_quat(qw, qx, qy, qz)

        feet_p = []
        for name, bid, _geoms in self.feet:
            p = d.xpos[bid]
            feet_p.append(p[0]); feet_p.append(p[1]); feet_p.append(p[2])

        obs = {
            "base_pos": np.array(d.qpos[idx:idx + 3], dtype=float),       # xyz
            "base_euler": np.array([roll, pitch, yaw], dtype=float),
            "base_linvel": np.array(d.qvel[idx:idx + 3], dtype=float),
            "base_angvel": np.array(d.qvel[idx + 3:idx + 6], dtype=float),
            "joint_qpos": np.array(d.qpos[m.jnt_qposadr[self.free] + 7:], float),
            "joint_qvel": np.array(d.qvel[m.jnt_dofadr[self.free] + 6:], float),
            "feet_pos": np.array(feet_p, dtype=float),
            "time": float(self.t),
        }
        if self.noise_scale > 0.0:
            for key in ("base_linvel", "base_angvel", "joint_qpos", "joint_qvel"):
                obs[key] = obs[key] + self.rng.normal(0, self.noise_scale, obs[key].shape)
        return obs

    # ------------------------------------------------------------------ step
    def step(self, controller) -> dict:
        """One control period. Controller decides at the period start, its
        torque is held for ``n_sub`` physics substeps."""
        m, d = self.model, self.data
        tau = controller.step(m, d, self.observe())
        for i in range(self.n_sub):
            mujoco.mj_step1(m, d)                     # nonlinear dynamics (caches bias)
            if i == 0:
                d.ctrl[:] = tau                       # apply held torque
            mujoco.mj_step2(m, d)                     # contact/constraint solve
        self.t = float(d.time)
        return self.observe()

    # ------------------------------------------------------------- RL-style hooks
    def reward(self, obs, info):
        """Generic shaping: forward progress - tilt - fall penalty."""
        vx = obs["base_linvel"][0]
        tilt = abs(obs["base_euler"][0]) + abs(obs["base_euler"][1])
        return vx - 1.0 * tilt - (10.0 if info.get("fallen") else 0.0)

    def done(self, obs) -> bool:
        """Terminate when the base pitches/rolls past the stability limit."""
        return max(abs(obs["base_euler"][0]), abs(obs["base_euler"][1])) > math.radians(50)

    def success(self, obs) -> bool:
        """A run 'succeeded' if it neither fell nor lost contact with the floor."""
        m, d = self.model, self.data
        idx = m.jnt_qposadr[self.free]
        height = d.qpos[idx + 2]
        return not self.done(obs) and height > self._start_z * 0.25