# legged — a systematic MuJoCo control study for floating-base robots

A framework for *formulating and measuring* the problem of simulating
closed-loop legged control, in the six-step recipe also shown the
self-contained `prototype_gui/mujoco_control_demo.py` (acrobot).  Here it is
instantiated on a real-ish robot: the PLA hexapod URDF.

The point is not "make the gaunt hexapod walk" — it is the *contract*, the
measurement, and the reproducibility: any controller you plug in gets the same
env, the same observer, the same numbers.

## The six-step formulation

| Step | Where | What is decided |
|------|-------|-----------------|
| 1 | `robot_studio/model_loader.load_model` | the plant: URDF→MJCF, free joint, motor limits, contact tuning, `opt.enableflags` (energy) |
| 2 | `legged/env.py` `LeggedEnv.reset` | the initial condition: nominal legs-out stance placed just above the floor (a *straight-leg* spawn makes legs start through the ground → guaranteed tumble) |
| 3 | `legged/env.py` `observe/step` + `legged/gait.py` | the interface contract: the exact observation dict a controller may read, and the `step(model, data, obs) -> tau` signature it must implement |
| 4 | `legged/env.py` `LeggedEnv.step` | the two-rate realtime loop: controller at `CONTROL_PERIOD` (0.02 s), physics at `model.opt.timestep`, using the `mj_step1`/`mj_step2` split (motor holds torque; no mid-period recompute) |
| 5 | `legged/metrics.py` | efficacy numbers computed **only from recorded observations** → apples-to-apples comparisons: falls, forward speed, tilt envelope, energy, cost-of-transport |
| 6 | `legged/evaluate.py` | the experiment: seeds × gait parameters, aggregated table + a live native-MuJoCo viewer running the exact same `step()` |

## The layered controller (step 3 reference design)

`LayeredController` splits *reference generation* from *stabilization*, so
either layer can be swapped and studied independently:

- **Layer 1 — `PhaseGait`**: phase oscillator → joint-space targets `q_ref(t)`,
  tripod gait = two anti-phase leg groups (`GROUP_A`/`GROUP_B`).
- **Layer 2 — `JointPD`**: `tau = kp·(q_ref−q) − kd·dq + qfrc_bias`
  (bias feed-forward ≈ gravity/Coriolis cancellation keeps the PD linear).

`LEGS`, groups and `DEFAULT_PARAMS` are imported from the repo app conventions.

## Usage

    # headless metric card (default: 6 s, seeds 0,1,2)
    .venv/bin/python -m legged.run_experiment

    # more statistical power
    .venv/bin/python -m legged.run_experiment --duration 8 --seeds 0 1 2 3 4 5

    # stride-frequency grid (slow/default/fast gait authoring)
    .venv/bin/python -m legged.run_experiment --speed-grid

    # live viewer of the same controller — macOS requires mjpython:
    .venv/bin/mjpython -m legged.run_experiment --viewer

## What the numbers mean (the efficacy card)

- `falls` — runs that exceeded the 50° tilt fall threshold (the very frame
  that violates it is recorded, so a mid-run fall shows up — not masked).
- `v_mean` — mean forward (x) speed; `v_rms_err` — RMS deviation from the
  0.2 m/s command; `v_side_rms` — lateral drift (should be ~0).
- `max_tilt_deg` / `mean_tilt_deg` / `height_std` — body stability: how close
  to the fall envelope the gait gets, and how level it travels.
- `ctrl_rms` — torque effort (actuator use); `mean_energy` — KE+PE excursion.
- `cot` — cost of transport: mechanical work / (mg·distance).  ~0.5–2 is
  good for legged robots; this simple tripod PD lands far above that, which is
  itself a finding (the gait burns torque on legs that slide).

## Notes

- **macOS / Intel**: `mujoco.viewer.launch_passive` requires `mjpython`; use
  `.venv/bin/mjpython` for the `--viewer` path.  The batch path runs under
  plain python.
- The hexapod's stock straight-legged stance cannot walk (no stance leverage).
  `LeggedEnv.reset` therefore pre-bends legs for a nominal stance before
  computing the spawn height — see `_apply_nominal_stance` / `_z_lift`.
- A 0.5° tilt margin vs. the 50° threshold during early settling is
  characteristic of this model + tripod PD without body-orientation feedback;
  adding a body roll/pitch stabilizer layer is the natural step-3 extension.