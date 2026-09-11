"""MuJoCo control demonstration — steps 1..6 of "show my controller works".

A single self-contained script.  It only depends on ``mujoco`` and ``numpy``
(plus optionally ``matplotlib`` for the comparison figure).  You can point it
at anything that reads like a robot + environment; the bundled aircraft
acrobot example is defined inline below as an MJCF string.

It walks through the 6 steps:

    1. describe the environment + robot in an MJCF/URDF XML
    2. load it into an MjModel + MjData and choose a start state
    3. close the loop: read state -> run the algorithm -> set ctrl -> step
    4. run physics at a high rate, your controller at a lower rate
    5. record & compare (baseline vs. your algorithm) into a metric table
    6. show it live in the native MuJoCo viewer

Run it headless (no window; plain python):

    .venv/bin/python prototype_gui/mujoco_control_demo.py --duration 4.0
    .venv/bin/python prototype_gui/mujoco_control_demo.py --duration 4.0 --plot

Or with a live 3D window (macOS needs jmpython for the native viewer):

    .venv/bin/mjpython prototype_gui/mujoco_control_demo.py --viewer
"""

import argparse
import math

import numpy as np
import mujoco

# --------------------------------------------------------------------------
# STEP 1 — the environment + robot as XML ("plant" description)
#
#   * <worldbody>: ground plane, light; the robot body tree hangs from a
#     fixed shoulder at (0,0,1),
#   * joints: revolute hinge joints (shoulder q1, elbow q2) -> 2 DOF,
#   * geoms: collision, radius/mlarc (mass, not density, so the inertia is
#     inferred from the shape), color per arm,
#   * <option>: physics timestep + integrator; gravity defaults to -9.81,
#   * <actuator>: two motors, one per joint, torques go to data.ctrl,
#   * <sensor>: optional joint-position sensors (we also read qpos directly).
#
# The same XML is what you would hand to load in the laboratory with a real
# robot: masses/inertias from CAD, ground friction from contact experiments,
# joint limits from the datasheet.
ENV_XML = """
<mujoco model="acrobot_control_demo">
  <option timestep="0.002" integrator="RK4">
    <flag energy="enable" contact="enable"/>
  </option>

  <asset>
    <material name="ground_mat" rgba="0.42 0.43 0.46 1"/>
    <material name="arm1_mat" rgba="0.85 0.20 0.20 1"/>
    <material name="arm2_mat" rgba="0.25 0.55 0.95 1"/>
  </asset>

  <worldbody>
    <light name="key" pos="0 2 4" dir="0 -1 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="ground" type="plane" size="5 5 0.1" material="ground_mat"/>

    <!-- shoulder: a hinge joint on the first body, axis along +y -->
    <body name="link1" pos="0 0 1.0">
      <joint name="joint1" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom name="g1" type="cylinder" size="0.05" mass="2.0"
            fromto="0 0 -0.25 0 0 0.25" material="arm1_mat"/>

      <!-- elbow: a hinge joint that hangs link2 from the tip of link1 -->
      <body name="link2" pos="0 0 -0.5">
        <joint name="joint2" type="hinge" axis="0 1 0" damping="0.05"/>
        <geom name="g2" type="capsule" size="0.045" mass="1.0"
              fromto="0 0 -0.25 0 0 0.25" material="arm2_mat"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor1" joint="joint1" gear="1.0"/>
    <motor name="motor2" joint="joint2" gear="1.0"/>
  </actuator>

  <sensor>
    <jointpos joint="joint1" name="q1_sensor"/>
    <jointpos joint="joint2" name="q2_sensor"/>
  </sensor>
</mujoco>
"""

# Physics runs at model.opt.timestep; the controller below is evaluated at a
# lower rate (a control "spacing" of 100 Hz here -> 5 physics steps between
# two controller evaluations).
CONTROL_PERIOD = 0.01   # seconds between two evaluations of the controller
START_QPOS = [0.3, -0.7]  # perturbed start so the open-loop baseline swings
START_QVEL = [0.0, 0.0]


# --------------------------------------------------------------------------
# STEP 3 — the control algorithm (the object of the demonstration)
#
# A very standard diagonal PD whose reference is a per-joint sinusoid, plus a
# gravity/Coriolis bias feedforward.  ``data.qfrc_bias`` carries the bias
# forces the robot currently feels (gravity, Coriolis, centrifugal) — adding
# it to the PD torque "cancels" the plant nonlinearity so the closed loop
# behaves almost like a linear 2nd-order system.
class SinusoidPD:
    """Per-joint sinusoid PD + bias feedforward. Torques -> data.ctrl."""

    def __init__(self, amp=0.8, freq=0.30, phase2=2.0, kp=20.0, kd=4.0):
        self.amp = amp
        self.omega = 2.0 * math.pi * freq
        self.phase2 = phase2
        self.kp = kp
        self.kd = kd

    def reference(self, t):
        """Desired joint positions (and velocities for the D-term)."""
        q = np.array([self.amp * math.sin(self.omega * t),
                      self.amp * math.sin(self.omega * t + self.phase2)])
        dq = np.array([self.amp * self.omega * math.cos(self.omega * t),
                       self.amp * self.omega * math.cos(self.omega * t + self.phase2)])
        return q, dq

    def step(self, model, data):
        """Compute the torque for the current state.

        Called BETWEEN mj_step1 and mj_step2, so ``data.qfrc_bias`` is the
        freshly-computed bias of this sub-step's configuration.
        """
        q = np.asarray(data.qpos[0:2])
        dq = np.asarray(data.qvel[0:2])
        qref, dqref = self.reference(data.time)
        tau = self.kp * (qref - q) + self.kd * (dqref - dq) \
            + np.asarray(data.qfrc_bias[0:2])
        return tau


class NullController:
    """Baseline: no intelligence, zero torque. The plant swings on its own."""

    def reference(self, t):
        return self.ref_callback(t) if hasattr(self, "ref_callback") else (np.zeros(2), np.zeros(2))

    def step(self, model, data):
        return np.zeros(model.nu)


# STEP 2 + control step 4 — one physics tick, then STEP 5 recording.
def _build_env():
    """Load the XML (step 1) into an MjModel/MjData pair and probe it."""
    model = mujoco.MjModel.from_xml_string(ENV_XML)
    data = mujoco.MjData(model)
    data.qpos[:] = START_QPOS        # choose a start state (step 2)
    data.qvel[:] = START_QVEL
    mujoco.mj_forward(model, data)   # initialize contact/sensed quantities
    root_name = bytes(model.names).split(b"\x00")[0].decode() or "unnamed"
    print("[step 2] model: %s | nq=%d nv=%d nu=%d dt=%.4fs sensors=%s" % (
        root_name, model.nq, model.nv, model.nu, model.opt.timestep,
        [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
         for i in range(model.nsensor)]))
    return model, data


def _run_one(model, controller, duration):
    """Run one episode, return a dict of recorded time-series (step 5).

    Implements step 4 (controller at CONTROL_PERIOD, physics at timestep) and
    step 3 (read -> decide -> act, once per control period).
    """
    data = mujoco.MjData(model)
    data.qpos[:] = START_QPOS
    data.qvel[:] = START_QVEL
    mujoco.mj_forward(model, data)

    n_sub = max(1, round(CONTROL_PERIOD / model.opt.timestep))  # physics steps
    n_phys = int(round(duration / model.opt.timestep))          # per control period

    rec = {"t": [], "q": [], "dq": [], "tau": [], "ek": [], "ep": []}
    tau = np.zeros(model.nu)
    for _ in range(n_phys):
        for i in range(n_sub):
            mujoco.mj_step1(model, data)          # advance nonlinearly, cache bias
            if i == 0:
                tau = controller.step(model, data)  # decide ONCE per period …
            data.ctrl[:] = tau                      # … and hold it until the next
            mujoco.mj_step2(model, data)            # solve contacts + integrate
        rec["t"].append(data.time)
        rec["q"].append(np.array(data.qpos[0:2]))
        rec["dq"].append(np.array(data.qvel[0:2]))
        rec["tau"].append(np.array(data.ctrl[0:2]))
        rec["ek"].append(data.energy[0])
        rec["ep"].append(data.energy[1])
    for k, v in rec.items():
        rec[k] = np.asarray(v)
    return rec


# --------------------------------------------------------------------------
# STEP 5 — quantify and compare.  Numbers first, picture second.
def _metrics(rec, controller):
    t = rec["t"]
    err = []
    for i in range(len(t)):
        qref, _ = controller.reference(t[i])
        err.append(qref - rec["q"][i])
    err = np.asarray(err)
    rms_q = np.sqrt(np.mean(err ** 2, axis=0))          # per-joint RMS error (rad)
    rms_tau = np.sqrt(np.mean(rec["tau"] ** 2, axis=0))  # per-joint RMS torque (Nm)
    total = rec["ek"] + rec["ep"]
    return {
        "rms_q1": rms_q[0], "rms_q2": rms_q[1],
        "rms_tau1": rms_tau[0], "rms_tau2": rms_tau[1],
        "mean_E": np.mean(total),
        "final_q": rec["q"][-1].copy(),
    }


def _print_table(rows):
    print()
    hdr = " %-22s %22s %22s" % ("metric", "open-loop", "PD control")
    print(hdr); print("-" * len(hdr))
    for name, bl, ctl in rows:
        print(" %-22s %22s %22s" % (name, _fmt(bl), _fmt(ctl)))
    print()


def _fmt(x):
    if isinstance(x, str):
        return x
    return "%.4f" % x


def _maybe_plot(baseline, controlled, ctl):
    """Optional matplotlib comparison figure (guarded import)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping comparison figure)")
        return
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for axx, name, rec in ((ax[0], "open-loop", baseline), (ax[1], "PD", controlled)):
        q = rec["q"]
        for j in range(2):
            axx.plot(rec["t"], q[:, j], label="q%d %s" % (j + 1, name))
        qref = np.array([ctl.reference(tt)[0] for tt in rec["t"]])
        for j in range(2):
            axx.plot(rec["t"], qref[:, j], "--", color="k", lw=0.8, alpha=0.5,
                     label=("qref%d" % (j + 1)) if j == 0 and name == "PD" else None)
        axx.set_ylabel("joint [rad]"); axx.grid(True, alpha=0.3)
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    ax[2].plot(baseline["t"], baseline["ek"] + baseline["ep"], label="E open-loop")
    ax[2].plot(controlled["t"], controlled["ek"] + controlled["ep"], label="E PD")
    ax[2].set_ylabel("total energy [J]"); ax[2].set_xlabel("time [s]")
    ax[2].legend(fontsize=8); ax[2].grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# STEP 6 — watch it live in the native MuJoCo viewer.
#
# mjpython owns the macOS main thread for the viewer; the physics loop below
# runs on the background thread MuJoCo created.  ESC closes the window.
def _run_viewer(model, controller, duration):
    from mujoco import viewer
    data = mujoco.MjData(model)
    data.qpos[:] = START_QPOS
    data.qvel[:] = START_QVEL
    mujoco.mj_forward(model, data)

    n_sub = max(1, round(CONTROL_PERIOD / model.opt.timestep))
    with viewer.launch_passive(model, data) as v:
        tau = np.zeros(model.nu)
        while v.is_running() and data.time < duration:
            for i in range(n_sub):
                mujoco.mj_step1(model, data)
                if i == 0:
                    tau = controller.step(model, data)
                data.ctrl[:] = tau
                mujoco.mj_step2(model, data)
                v.sync()
    print("[step 6] viewer closed after %.2f s of sim time" % data.time)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--duration", type=float, default=4.0, help="sim seconds")
    ap.add_argument("--viewer", action="store_true",
                    help="open the native 3D viewer (needs mjpython on macOS)")
    ap.add_argument("--plot", action="store_true", help="matplotlib comparison figure")
    args = ap.parse_args()

    model, data = _build_env()   # steps 1 + 2

    if args.viewer:
        _run_viewer(model, SinusoidPD(), args.duration)
        return

    # steps 3 + 4 + 5: two episodes, then compare.
    ctl = SinusoidPD()
    baseline = _run_one(model, NullController(), args.duration)
    controlled = _run_one(model, ctl, args.duration)

    mb = _metrics(baseline, ctl)
    mc = _metrics(controlled, ctl)
    qbl = ", ".join("%.2f" % x for x in mb["final_q"])
    qcl = ", ".join("%.2f" % x for x in mc["final_q"])
    print("\n[step 5] comparison over %.1f s sim time" % args.duration)
    _print_table([
        ("RMS joint-1 error (rad)", mb["rms_q1"], mc["rms_q1"]),
        ("RMS joint-2 error (rad)", mb["rms_q2"], mc["rms_q2"]),
        ("RMS torque-1 (Nm)", mb["rms_tau1"], mc["rms_tau1"]),
        ("RMS torque-2 (Nm)", mb["rms_tau2"], mc["rms_tau2"]),
        ("mean total energy (J)", mb["mean_E"], mc["mean_E"]),
        ("final pose q1,q2", qbl, qcl),
    ])

    if args.plot:
        _maybe_plot(baseline, controlled, ctl)


if __name__ == "__main__":
    main()