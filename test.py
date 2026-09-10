"""
Spider CPG demo — dynamic mode with gravity.

Base is welded to world (via MJCF). Legs are driven by position
actuators, so they hold their CPG-commanded trajectory against gravity.

Mode:
    KINEMATIC = True   -> mj_forward teleport (no dynamics, gravity ignored)
    KINEMATIC = False  -> mj_step with actuators (gravity ON, real dynamics)

Requires:  pip install mujoco numpy
"""

import os
import time
import csv
import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# CONFIG
# ============================================================
XML_PATH = r"C:\Users\megah\Downloads\spider.xml"

ALL_LEGS    = ["FL", "ML", "RL", "FR", "MR", "RR"]
ACTIVE_LEGS = ALL_LEGS

# --- CPG ---
CPG_MU    = 1.0
CPG_OMEGA = 2.0
CPG_GAMMA = 5.0

# --- Leg mount angles (must match MJCF euler) ---
LEG_MOUNT_ANGLE = {
    "FL": np.deg2rad( 60),
    "ML": np.deg2rad( 90),
    "RL": np.deg2rad(120),
    "FR": np.deg2rad(-60),
    "MR": np.deg2rad(-90),
    "RR": np.deg2rad(-120),
}

# --- Tripod gait ---
LEG_PHASE_OFFSET = {
    "FL": 0.0, "MR": 0.0, "RL": 0.0,
    "FR": 0.5, "ML": 0.5, "RR": 0.5,
}

# --- Ellipse geometry ---
PUSH_OUT   = 0.04
A_SEMI     = 0.06
B_SEMI     = 0.035
SWING_DUTY = 0.4

# --- Femur posture bias ---
# Keep the femur horizontal at the initial pose; set to 0.0 for a flat start.
FEMUR_BIAS = 0.0

# --- IK ---
IK_ITERATIONS = 6
IK_LAMBDA     = 5e-3
IK_GAIN       = 0.7
JOINT_WEIGHTS = np.array([0.5, 1.0, 1.0])

# --- Joint limits ---
LIMITS = [
    (-1.2, 1.2),
    (-1.5, 1.2),
    (-2.5, 0.4),
]

# --- Loop ---
KINEMATIC = False       # <-- DYNAMIC MODE with gravity
DT        = 0.002

# --- CSV ---
SAVE_CSV = True
CSV_DIR  = "."

# --- Camera ---
CAM_ELEVATION = -20.0
CAM_AZIMUTH   = 135.0
CAM_DISTANCE  = 1.4


def clamp_q(q):
    return np.array([np.clip(q[k], *LIMITS[k]) for k in range(3)])


# ============================================================
# LOAD MJCF
# ============================================================
if not os.path.exists(XML_PATH):
    raise FileNotFoundError(f"MJCF not found: {XML_PATH}")

print(f"Loading: {XML_PATH}")
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)
print(f"nq={model.nq}  nu={model.nu}  nbody={model.nbody}  njnt={model.njnt}")

# Gravity: keep whatever is in the MJCF (usually 0 0 -9.81).
# Print it so we know what we're working with.
print(f"gravity = {model.opt.gravity}")

if KINEMATIC:
    model.opt.gravity[:] = 0.0
    print("kinematic mode: gravity overridden to 0")
else:
    print("dynamic mode: gravity active")


# ============================================================
# DIAGNOSTIC
# ============================================================
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
print(f"\nbase xpos  = {data.xpos[base_id]}")
print(f"base xquat = {data.xquat[base_id]}")
xmat = data.xmat[base_id].reshape(3, 3)
print(f"base local +Z in world = {xmat[:, 2]}   (should be (0,0,1))")

print("\n=== FOOT POSITIONS AT ZERO POSE (world) ===")
for leg in ALL_LEGS:
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot")
    if fid < 0:
        print(f"  {leg}_foot: NOT FOUND")
        continue
    p = data.xpos[fid]
    tag = "DOWN (good)" if p[2] < data.xpos[base_id][2] else "UP (bad)"
    print(f"  {leg}_foot: ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})  {tag}")

print("\n=== ACTUATORS ===")
for a in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    jid = model.actuator_trnid[a, 0]
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    print(f"  [{a}] {name} -> joint {jname}")


# ============================================================
# LEG OBJECT
# ============================================================
class Leg:
    def __init__(self, name):
        self.name = name
        self.joint_names = [
            f"{name}_coxa_joint",
            f"{name}_femur_joint",
            f"{name}_tibia_joint",
        ]
        self.joint_ids, self.qpos_adr, self.dof_adr, self.act_ids = [], [], [], []
        for jn in self.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                raise RuntimeError(f"Joint '{jn}' not found.")
            self.joint_ids.append(jid)
            self.qpos_adr.append(model.jnt_qposadr[jid])
            self.dof_adr.append(model.jnt_dofadr[jid])
            aid = -1
            for a in range(model.nu):
                if model.actuator_trnid[a, 0] == jid:
                    aid = a
                    break
            self.act_ids.append(aid)

        self.foot_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_foot"
        )
        if self.foot_body_id < 0:
            raise RuntimeError(f"Body '{name}_foot' not found.")

        ang = LEG_MOUNT_ANGLE[name]
        self.r_hat = np.array([np.cos(ang), np.sin(ang), 0.0])
        self.t_hat = np.array([-np.sin(ang), np.cos(ang), 0.0])
        self.z_hat = np.array([0.0, 0.0, 1.0])

        self.u_rest = self.v_rest = 0.0
        self.U_CENTER = self.V_CENTER = 0.0
        self.phase_offset = LEG_PHASE_OFFSET.get(name, 0.0)
        self.q = np.zeros(3)
        self.trace = []

    def read_rest_pose(self):
        p = data.xpos[self.foot_body_id].copy()
        self.u_rest = float(p @ self.r_hat)
        self.v_rest = float(p @ self.z_hat)
        self.U_CENTER = self.u_rest + PUSH_OUT
        self.V_CENTER = self.v_rest

    def target(self, phi):
        u = self.U_CENTER + A_SEMI * np.cos(phi)
        v = self.V_CENTER + B_SEMI * np.sin(phi)
        return u * self.r_hat + v * self.z_hat

    def solve_ik(self, target_pos):
        """
        IK in kinematic mode uses mj_forward.
        In dynamic mode, we use the CURRENT pose as the initial guess,
        which is much more stable because the actuators keep the leg
        near the previous solution.
        """
        q = self.q.copy()
        err_norm = np.inf
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        for _ in range(IK_ITERATIONS):
            if KINEMATIC:
                for k in range(3):
                    data.qpos[self.qpos_adr[k]] = q[k]
                mujoco.mj_forward(model, data)
            else:
                # In dynamic mode, use current sim state for Jacobian,
                # but solve for the desired q.
                mujoco.mj_forward(model, data)

            cur = data.xpos[self.foot_body_id].copy()
            err = target_pos - cur
            err_norm = np.linalg.norm(err)

            jacp[:] = 0.0
            jacr[:] = 0.0
            mujoco.mj_jacBody(model, data, jacp, jacr, self.foot_body_id)
            J = jacp[:, self.dof_adr]

            W = np.diag(JOINT_WEIGHTS)
            A = J.T @ J + (IK_LAMBDA ** 2) * (W.T @ W)
            dq = np.linalg.solve(A, J.T @ err)

            q = clamp_q(q + IK_GAIN * dq)
            q[1] += FEMUR_BIAS
            q = clamp_q(q)

            if not KINEMATIC:
                # Recompute the foot position with the new q
                # so subsequent iterations use the updated guess.
                for k in range(3):
                    data.qpos[self.qpos_adr[k]] = q[k]
                mujoco.mj_forward(model, data)

        self.q = q
        return err_norm

    def write_qpos(self):
        for k in range(3):
            data.qpos[self.qpos_adr[k]] = self.q[k]

    def write_ctrl(self):
        for k in range(3):
            if self.act_ids[k] >= 0:
                data.ctrl[self.act_ids[k]] = self.q[k]

    def log(self, t):
        # log the ACTUAL foot pos and ACTUAL joint angles
        p = data.xpos[self.foot_body_id]
        self.trace.append((t, p[0], p[1], p[2],
                           data.qpos[self.qpos_adr[0]],
                           data.qpos[self.qpos_adr[1]],
                           data.qpos[self.qpos_adr[2]]))


# ============================================================
# BUILD LEGS
# ============================================================
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

legs = {}
for name in ACTIVE_LEGS:
    leg = Leg(name)
    leg.read_rest_pose()
    print(f"\n=== LEG {name} ===")
    print(f"  rest foot = {data.xpos[leg.foot_body_id]}")
    print(f"  rest (u,v) = ({leg.u_rest:+.4f}, {leg.v_rest:+.4f})")
    print(f"  ellipse center = ({leg.U_CENTER:+.4f}, {leg.V_CENTER:+.4f})")
    print(f"  actuator ids = {leg.act_ids}")
    legs[name] = leg


# ============================================================
# CPG
# ============================================================
class HopfCPG:
    def __init__(self, mu, omega, gamma):
        self.mu, self.omega, self.gamma = mu, omega, gamma
        self.x, self.y = np.sqrt(mu), 0.0

    def step(self, dt):
        r2 = self.x**2 + self.y**2
        dx = self.gamma * (self.mu - r2) * self.x - self.omega * self.y
        dy = self.gamma * (self.mu - r2) * self.y + self.omega * self.x
        self.x += dx * dt
        self.y += dy * dt

    def phase(self):
        return np.arctan2(self.y, self.x)


cpg = HopfCPG(CPG_MU, CPG_OMEGA, CPG_GAMMA)


def warped_phase(phi, duty=0.4):
    p = (phi % (2 * np.pi)) / (2 * np.pi)
    w = p / duty if p < duty else 1.0 - (p - duty) / (1.0 - duty)
    return w * 2 * np.pi


# ============================================================
# INITIAL POSE — send initial ctrl so actuators start near target
# ============================================================
print("\n=== INITIAL IK ===")
for leg in legs.values():
    leg.q = np.array([data.qpos[leg.qpos_adr[k]] for k in range(3)])
    leg.q[1] = 0.0  # keep femur horizontal at startup
    res = leg.solve_ik(leg.target(0.0))
    print(f"  {leg.name}: q={leg.q}, residual={res:.5f}")
    if KINEMATIC:
        leg.write_qpos()
    else:
        # set both qpos (so sim starts at target) and ctrl (so actuators hold)
        for k in range(3):
            data.qpos[leg.qpos_adr[k]] = leg.q[k]
        leg.write_ctrl()

mujoco.mj_forward(model, data)


# ============================================================
# SUMMARY + CSV
# ============================================================
def summarize_leg(leg):
    if len(leg.trace) < 20:
        return
    arr = np.array(leg.trace)
    xyz = arr[:, 1:4]
    print(f"\n=== LEG {leg.name} TRAJECTORY (ACTUAL) ===")
    for i, ax in enumerate("XYZ"):
        print(f"  {ax}: span={xyz[:, i].ptp():.4f}  "
              f"min={xyz[:, i].min():+.4f}  max={xyz[:, i].max():+.4f}")
    u = xyz @ leg.r_hat
    v = xyz @ leg.z_hat
    t = xyz @ leg.t_hat
    print(f"  side-plane: u={u.ptp():.4f}  v={v.ptp():.4f}  t={t.ptp():.4f}")


def save_leg_csv(leg):
    fname = os.path.join(CSV_DIR, f"foot_trace_{leg.name}.csv")
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z", "q_coxa", "q_femur", "q_tibia"])
        w.writerows(leg.trace)
    print(f"  saved: {fname}")


# ============================================================
# MAIN
# ============================================================
def main():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance  = CAM_DISTANCE
        viewer.cam.azimuth   = CAM_AZIMUTH
        viewer.cam.elevation = CAM_ELEVATION
        viewer.cam.lookat[:] = [0.0, 0.0, 0.10]

        t = 0.0
        print("\nRunning. Close viewer to exit.\n")

        try:
            while viewer.is_running():
                loop_start = time.time()

                cpg.step(DT)
                phi_base = cpg.phase()

                for leg in legs.values():
                    phi   = phi_base + 2 * np.pi * leg.phase_offset
                    phi_w = warped_phase(phi, SWING_DUTY)
                    tgt   = leg.target(phi_w)
                    leg.solve_ik(tgt)
                    if KINEMATIC:
                        leg.write_qpos()
                    else:
                        leg.write_ctrl()

                if KINEMATIC:
                    mujoco.mj_forward(model, data)
                else:
                    mujoco.mj_step(model, data)

                for leg in legs.values():
                    leg.log(t)

                viewer.sync()
                elapsed = time.time() - loop_start
                if elapsed < DT:
                    time.sleep(DT - elapsed)
                t += DT

        except KeyboardInterrupt:
            print("\nInterrupted.")

    print("\n=== SUMMARY ===")
    for leg in legs.values():
        summarize_leg(leg)
        if SAVE_CSV:
            save_leg_csv(leg)


if __name__ == "__main__":
    main()