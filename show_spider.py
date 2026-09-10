import math
import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = r".\robot_studio\models\spider_mjcf.xml"

# ---------------------------------------------------------------------------
# Gait timing
# ---------------------------------------------------------------------------
STRIDE_PERIOD = 1.0          # seconds per full leg cycle
STANCE_FRACTION = 0.65       # higher duty factor -> more feet on the ground
                              # at once -> more static stability now that the
                              # body is no longer force-positioned.
SWING_FRACTION = 1.0 - STANCE_FRACTION

STRIDE_LENGTH = 0.06         # fore-aft foot excursion (m)
GROUND_CLEARANCE = 0.03      # swing foot lift height (m)
SETTLE_TIME = 1.0            # seconds spent standing on all 6 feet before
                              # the gait starts, so the robot doesn't collapse
                              # from an arbitrary initial pose.
UPRIGHT_THRESHOLD = 0.2      # body-frame "up" dotted with world-up; below
                              # this the robot is considered fallen/flipped

# Turning: bias stride length between the left/right leg sets.
# 0.0 = straight ahead, positive curves one way, negative the other.
TURN_BIAS = 0.0

LEG_NAMES = ["FL", "ML", "RL", "FR", "MR", "RR"]

# Ant-like wave gait: each leg cycles in sequence rather than as a tripod
# pair, giving a rear-to-front metachronal pattern.
PHASE_OFFSET = {
    "FL": 0.00,
    "ML": 0.18 * 2.0 * math.pi,
    "RL": 0.36 * 2.0 * math.pi,
    "FR": 0.50 * 2.0 * math.pi,
    "MR": 0.68 * 2.0 * math.pi,
    "RR": 0.86 * 2.0 * math.pi,
}

SIDE = {"FL": 1.0, "ML": 1.0, "RL": 1.0, "FR": -1.0, "MR": -1.0, "RR": -1.0}

# Nominal standing foot position, expressed directly in the body frame
# (this is also the IK "home" target used during the settle phase).
FOOT_HOME = {
    "FL": (0.10, 0.08, -0.18),
    "ML": (0.00, 0.10, -0.18),
    "RL": (-0.10, 0.08, -0.18),
    "FR": (0.10, -0.08, -0.18),
    "MR": (0.00, -0.10, -0.18),
    "RR": (-0.10, -0.08, -0.18),
}

# Fallback ctrl clamp ranges, only used if a joint has no <limit> in the XML.
FALLBACK_RANGE = {
    "coxa": (-1.2, 1.2),
    "femur": (-0.34906585, 0.34906585),
    "tibia": (-2.5, 0.4),
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class HopfCPG:
    """Central pattern generator giving a smooth, self-stabilizing phase clock."""

    def __init__(self, mu=1.0, omega=2.0 * math.pi, gamma=5.0):
        self.mu = mu
        self.omega = omega
        self.gamma = gamma
        self.x = math.sqrt(mu)
        self.y = 0.0

    def step(self, dt):
        r2 = self.x * self.x + self.y * self.y
        dx = self.gamma * (self.mu - r2) * self.x - self.omega * self.y
        dy = self.gamma * (self.mu - r2) * self.y + self.omega * self.x
        self.x += dx * dt
        self.y += dy * dt

    def phase(self):
        return math.atan2(self.y, self.x)


cpg = HopfCPG(mu=1.0, omega=2.0 * math.pi / STRIDE_PERIOD, gamma=5.0)

BASE_BODY_HEIGHT = 0.20
NOMINAL_STANCE_HEIGHT = 0.22  # fixed reference height used for foot targets.
                                # CRITICAL: this must NOT be read from the
                                # live body pose - if it were, a body that
                                # sags would drag the stance targets down
                                # with it, giving no restoring force at all.
MINIMUM_STRIDE_ADVANCE = 0.02


def yaw_only_rotation(quat_wxyz):
    """2D rotation matrix from just the yaw component of a body's
    orientation. Using full 3D orientation (including roll/pitch) to place
    foot targets creates bad feedback - a body that tips forward would tilt
    the whole foot-target pattern with it, amplifying the tip instead of
    correcting it."""
    w, x, y, z = quat_wxyz
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def objective_fn(successful_strides, walking_distance, body_height):
    height_margin = max(body_height - BASE_BODY_HEIGHT, 0.0)
    low_height_penalty = max(BASE_BODY_HEIGHT - body_height, 0.0)
    return (200.0 * successful_strides + 10.0 * walking_distance
            + 150.0 * height_margin - 400.0 * low_height_penalty)


def leg_phase(leg_name, walk_time, cpg_phase):
    return (cpg_phase + PHASE_OFFSET[leg_name]) % (2.0 * math.pi)


def foot_target_in_local(leg_name, cpg_phase, turn_bias=0.0):
    """Reference trajectory generator: smooth swing arc forward, then a
    straight-line stance sweep backward at ground height. The stance sweep
    is what actually propels the body - the planted foot resists relative
    to the ground while the body moves forward over it."""
    px, py, pz = FOOT_HOME[leg_name]
    phase = leg_phase(leg_name, 0.0, cpg_phase)
    swing_end = 2.0 * math.pi * SWING_FRACTION
    stride = STRIDE_LENGTH * (1.0 + SIDE[leg_name] * turn_bias)

    if phase < swing_end:
        s = clamp(phase / swing_end, 0.0, 1.0)
        ease = 0.5 - 0.5 * math.cos(math.pi * s)  # zero velocity at the ends
        x = px - stride / 2.0 + stride * ease
        z = pz + GROUND_CLEARANCE * math.sin(math.pi * s)
        y = py
        in_swing = True
    else:
        s = clamp((phase - swing_end) / (2.0 * math.pi - swing_end), 0.0, 1.0)
        x = px + stride / 2.0 - stride * s
        z = pz
        y = py
        in_swing = False
    return (x, y, z), in_swing


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

actuator_ids = {}
foot_body_ids = {}
leg_dof_idx = {}    # per leg: [coxa_dof, femur_dof, tibia_dof] into qvel/jac
leg_qpos_idx = {}   # per leg: [coxa_qposadr, femur_qposadr, tibia_qposadr]
leg_jnt_range = {}  # per leg: [(lo,hi) or None, ...] matching the order above

for leg in LEG_NAMES:
    aids = {}
    dof_idx, qpos_idx, ranges = [], [], []
    for axis in ("coxa", "femur", "tibia"):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_{axis}_act")
        aids[axis] = aid
        if aid >= 0:
            joint_id = int(model.actuator_trnid[aid, 0])
            dof_idx.append(int(model.jnt_dofadr[joint_id]))
            qpos_idx.append(int(model.jnt_qposadr[joint_id]))
            if model.jnt_limited[joint_id]:
                ranges.append(tuple(model.jnt_range[joint_id]))
            else:
                ranges.append(FALLBACK_RANGE[axis])
        else:
            dof_idx.append(None)
            qpos_idx.append(None)
            ranges.append(FALLBACK_RANGE[axis])
    actuator_ids[leg] = aids
    leg_dof_idx[leg] = dof_idx
    leg_qpos_idx[leg] = qpos_idx
    leg_jnt_range[leg] = ranges
    foot_body_ids[leg] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot")

# Locate the free-jointed root body (the torso) so foot targets, which are
# expressed in the body frame, can be converted to world coordinates.
root_body_id = 1
for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
        root_body_id = int(model.jnt_bodyid[j])
        break

print(f"Loaded model: {MODEL_PATH}")
print(f"nq={model.nq}, njnt={model.njnt}, nbody={model.nbody}, root_body={root_body_id}")

# ---------------------------------------------------------------------------
# Contact hardening: reduce foot/floor penetration without editing the XML.
# ---------------------------------------------------------------------------
FLOOR_Z = 0.0          # assumed floor height; adjust if your floor plane sits
                        # somewhere other than z=0
FOOT_CLEAR_MARGIN = 0.004

model.opt.iterations = 100
model.opt.tolerance = 1e-10
model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

# Stiffer, less "mushy" solref/solimp on foot geoms and the floor geom means
# the solver resolves overlap faster and allows less sinking-in before it
# pushes back.
STIFF_SOLREF = np.array([0.01, 1.0])           # (timeconst, dampratio)
STIFF_SOLIMP = np.array([0.9, 0.95, 0.001, 0.5, 2.0])  # softer->stiffer d(x)
GRIP_FRICTION = np.array([1.2, 0.006, 0.0002])  # (sliding, torsional, rolling)

foot_geom_ids = []
for leg, body_id in foot_body_ids.items():
    if body_id < 0:
        continue
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == body_id:
            foot_geom_ids.append(g)

floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
if floor_geom_id < 0:
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            floor_geom_id = g
            break

for g in foot_geom_ids + ([floor_geom_id] if floor_geom_id >= 0 else []):
    model.geom_solref[g, :2] = STIFF_SOLREF
    model.geom_solimp[g, :5] = STIFF_SOLIMP
    model.geom_condim[g] = max(model.geom_condim[g], 3)  # ensure friction, not just normal contact
    model.geom_friction[g, :3] = GRIP_FRICTION

if not foot_geom_ids:
    print("WARNING: no foot geoms found under the *_foot bodies - "
          "contact hardening was skipped for feet.")
if floor_geom_id < 0:
    print("WARNING: could not find a floor/plane geom named 'floor' - "
          "contact hardening was skipped for the ground. Rename your "
          "ground geom to 'floor' or adjust the lookup above.")

mujoco.mj_forward(model, data)


def contact_force_summary():
    total = np.zeros(6, dtype=float)
    count = 0
    for i in range(data.ncon):
        force = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, i, force)
        total += force
        count += 1
    return count, total


def leg_ik_step(leg, target_world, max_step=0.15, damping=0.05):
    """One damped-least-squares Newton step toward target_world, expressed
    as a joint-angle *target* rather than a qpos write. Only data.ctrl is
    touched here - the physical state (data.qpos) is left alone and the
    position actuators' own PD tracking closes the gap over the next few
    physics steps, respecting contacts and dynamics along the way."""
    dof_idx = leg_dof_idx[leg]
    qpos_idx = leg_qpos_idx[leg]
    foot_id = foot_body_ids[leg]
    if foot_id < 0 or any(i is None for i in dof_idx):
        return

    target_world = np.asarray(target_world, dtype=float).copy()
    target_world[2] = max(target_world[2], FLOOR_Z + FOOT_CLEAR_MARGIN)

    cur = data.xpos[foot_id]
    err = target_world - cur

    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, None, foot_id)
    J = jacp[:, dof_idx]

    lam2 = damping * damping
    try:
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(3), err)
    except np.linalg.LinAlgError:
        return
    dq = np.clip(dq, -max_step, max_step)

    for k, (axis, qi) in enumerate(zip(("coxa", "femur", "tibia"), qpos_idx)):
        q_target = data.qpos[qi] + dq[k]
        lo, hi = leg_jnt_range[leg][k]
        q_target = clamp(q_target, lo, hi)
        aid = actuator_ids[leg][axis]
        if aid >= 0:
            data.ctrl[aid] = q_target


def reset_robot():
    """Place the robot upright, already at standing height, with zero
    velocity - no free-fall drop, no impact bounce. Leg joint angles reset
    to whatever the model's default qpos defines; the settle phase then
    eases them into the FOOT_HOME stance via IK rather than snapping."""
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, NOMINAL_STANCE_HEIGHT + 0.02]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


reset_robot()

# Start standing on all six feet at the home pose.
for leg in LEG_NAMES:
    Ryaw = yaw_only_rotation(data.xquat[root_body_id])
    xy_world = data.xpos[root_body_id, :2] + Ryaw @ np.array(FOOT_HOME[leg][:2])
    z_world = NOMINAL_STANCE_HEIGHT + FOOT_HOME[leg][2]
    leg_ik_step(leg, np.array([xy_world[0], xy_world[1], z_world]))
mujoco.mj_forward(model, data)


with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 1.8
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -18.0
    viewer.cam.lookat[:] = [0.0, 0.0, 0.12]

    print("Viewer open. Close the window to exit.")
    t = 0.0
    dt = 0.002
    next_force_log = 0.0
    last_reset_t = 0.0
    fall_count = 0
    start_x = None
    last_stride_x = None
    successful_strides = 0

    while viewer.is_running():
        settling = (t - last_reset_t) < SETTLE_TIME

        up_z = data.xmat[root_body_id][8]  # body's local z-axis, dotted with world z
        if up_z < UPRIGHT_THRESHOLD:
            fall_count += 1
            print(f"t={t:.2f}s  ROBOT FALLEN/FLIPPED (up_z={up_z:.2f}) - "
                  f"resetting to standing pose (fall #{fall_count})")
            reset_robot()
            last_reset_t = t
            start_x = None
            last_stride_x = None
            continue

        Ryaw = yaw_only_rotation(data.xquat[root_body_id])
        root_xy = data.xpos[root_body_id, :2].copy()

        for leg in LEG_NAMES:
            if settling:
                local_target = FOOT_HOME[leg]
            else:
                local_target, _ = foot_target_in_local(leg, cpg.phase(), TURN_BIAS)
            xy_world = root_xy + Ryaw @ np.array(local_target[:2])
            z_world = NOMINAL_STANCE_HEIGHT + local_target[2]
            world_target = np.array([xy_world[0], xy_world[1], z_world])
            leg_ik_step(leg, world_target)

        if not settling:
            cpg.step(dt)

        mujoco.mj_step(model, data)
        viewer.sync()

        body_height = data.qpos[2]
        current_x = data.qpos[0]

        if start_x is None and not settling:
            start_x = current_x
            last_stride_x = current_x

        if start_x is not None:
            if current_x - last_stride_x > MINIMUM_STRIDE_ADVANCE and body_height >= BASE_BODY_HEIGHT:
                successful_strides += 1
                last_stride_x = current_x
            objective = objective_fn(successful_strides, current_x - start_x, body_height)
        else:
            objective = 0.0

        if t >= next_force_log:
            count, total = contact_force_summary()
            dist = (current_x - start_x) if start_x is not None else 0.0
            state = "settling" if settling else "walking"
            n_in_contact = sum(1 for leg in LEG_NAMES
                                if any(data.contact[i].geom1 == foot_geom
                                       or data.contact[i].geom2 == foot_geom
                                       for i in range(data.ncon)
                                       for foot_geom in [g for g in foot_geom_ids
                                                          if model.geom_bodyid[g] == foot_body_ids[leg]]))
            if count:
                print(f"t={t:.2f}s [{state}] body_z={body_height:.3f}  x={dist:.3f}  "
                      f"strides={successful_strides}  objective={objective:.1f}  "
                      f"contacts={count}  feet_touching={n_in_contact}/6  "
                      f"GRF={total[2]:.2f} N  net_xy={np.linalg.norm(total[:2]):.2f}")
            else:
                print(f"t={t:.2f}s [{state}] body_z={body_height:.3f}  x={dist:.3f}  "
                      f"NO CONTACTS AT ALL - feet are not reaching the ground")
            next_force_log += 0.1

        t += dt
        time.sleep(dt)