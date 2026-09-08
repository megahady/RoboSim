# Robot Studio

A standalone desktop workbench for testing and tuning **legged-robot and
double-pendulum control scripts** against real physics, with a **live 3D
viewport** and **realtime telemetry**.

Physics run in a background thread with [MuJoCo](https://mujoco.org/) 3.x; the
3D scene is rendered with [PyVista](https://pyvista.org/); the UI is PyQt5 with
pyqtgraph plots. Control code is plain Python you edit and **apply without
restarting** — the running simulation keeps stepping while you tune gains.

```
robot_studio/
  app.py                    CLI / Qt entry point
  main_window.py            MDI window: scene + telemetry + console child forms
  loader_worker.py          background thread: model compile + mesh building
  scene_viewer.py           PyVista actor builder + contact/animation
  simulation_worker.py      MuJoCo physics thread + spawn lift + throttles
  controller_manager.py     control-script loader, hot reload, stdout capture
  model_loader.py           URDF -> MJCF converter (motors, inertias, meshes)
  plots.py                  live telemetry plots (q, dq, tau, energy) + refs
  telemetry.py              CSV recording of snapshots
  editor.py                 embedded script editor + hot reload
  style.py                  dark theme
  models/acrobot.urdf       example double pendulum
  models/pla_hexapod.urdf   example 6-legged robot (floating base + ground)
  controllers/              example controllers (PD, inverse dynamics, gait)
  smoke_test.py             headless physics/render check
  widgets_smoke.py          headless Qt widgets check (offscreen)
  gui_smoke.py              best-effort headless MainWindow check
```

---

## Quick start

```bat
cd RoboSim
run_robot_studio.bat                    rem uses .venv, opens acrobot by default
.venv\Scripts\python.exe -m robot_studio
.venv\Scripts\python.exe -m robot_studio --model path\to\robot.urdf --controller controllers\acrobot_inverse_dynamics.py
```

Requirements are pinned in `requirements.txt`
(mujoco 3.12, pyvista 0.48, pyvistaqt, PyQt5, pyqtgraph, numpy).
To build the environment from scratch:

```bat
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

---

## Controls

| Key           | Action                    |
|---------------|---------------------------|
| `Space`       | Run / Pause               |
| `R`           | Reset simulation          |
| `C`           | Toggle CSV recording      |
| `F`           | Follow selected body      |
| `S`           | Screenshot (PNG)          |
| `Ctrl+Shift+N`| Open model                |
| `Ctrl+Shift+O`| Open controller           |
| `Ctrl+S` in editor | Apply script (hot reload) |

Transport bar: **Run/Step/Reset/Record**, model+controller pickers, a
realtime-speed dial, and a **seek slider** (drag to scrub back in time).
The controller panel shows **parameter sliders** declared by the script;
moving a slider updates the value live.

### Layout: MDI workbench

The main window is an **MDI (multi-document interface)**. The 3D Scene, the
Telemetry plots and the Console are **child forms** of the central window:

- **Plots** are a child form, not a separate top-level window. They are hidden
  until you press Run / `Space`, at which point the Telemetry form pops up and
  becomes active inside the MDI area. Reopen anytime via **View ▸ Plots**.
- Every child form can be **tiled, cascaded, floated into its own window** (the
  MDI system menu) or **docked back** into the workbench.
- **View** menu: `Plots` / `Console` / `3D Scene` toggles plus `Cascade`,
  `Tile` and `Close All Windows`. The right **Controls** panel and the
  **Script editor** stay as fixed docks.
- The pattern makes room for further child forms later (scope view, planner,
  log inspector, ...) without re-layout.

Telemetry charts: joint angles, velocities, actuation torque and energy
(KE/PE/Total), every series sliding-windowed to the selected time span.

```bat
rem launch with a robot of your own
.venv\Scripts\python.exe -m robot_studio --model "%USERPROFILE%\Downloads\XXXXX.urdf" --controller robot_studio\controllers\hexapod_tripod_gait.py
```

`models/pla_hexapod.urdf` is a copy of that robot so you can also pick it from
the model combo in the toolbar.

---

## Writing a controller

Drop a `.py` file into `controllers/` (or File > Open Controller). It declares
an `init` and a `step` over the Mujoco objects:

```python
import numpy as np

TITLE = "PD + gravity compensation"
PARAMS = {"Kp": (1.0, 200.0, 40.0), "Kd": (0.1, 40.0, 6.0)}   # (min, max, default)
HELP  = "what this controller does and how to tune it"          # optional

def init(model, data, params):
    pass                                # optional; called on load / re-init

def step(model, data, params):          # called once per physics substep
    params["Kp"]
    data.ctrl[...] = ...                # write motor commands

def reference(data, params):            # optional; returns target q (drawn dashed)
    return np.array([...])
```

- `model` / `data` are the real `mujoco.MjModel` / `MjData` owned by the worker.
- `print()` output from your script is captured and shown in the Console panel.
- **A raised exception only pauses the simulation** — it never takes the UI
  down, and the previous controller remains active until you reload.
- Existing examples: `controllers/simple_pd_tracking.py` (robust starter) and
  `controllers/acrobot_inverse_dynamics.py` (computed-torque).

### Motor mapping

The URDF loader adds **one torque motor per movable revolute joint** in URDF
joint order, so `data.ctrl[i]` drives joint `i`. `models/acrobot.urdf` -> 2
motors (`data.ctrl[0:2]`).

---

## Loading models

- **URDF** (`*.urdf`): converted to MJCF on disk (`model_loader.py`):
  - one `hinge` joint + motor per revolute/continuous/prismatic joint;
  - **floating bases are automatic**: a root body with mass and no joint to a
    world link becomes a 6-DOF `freejoint`, so legged/wheeled robots fall and
    stand/roll properly instead of being welded to the origin;
  - for such mobile robots a **ground plane is added automatically**
    (otherwise feet fall through the floor in MuJoCo);
  - `floating` joints become a sphere-body `freejoint` (6-DOF base);
  - link inertias are rotated into the joint frame and **sanitized** to a
    physically valid tensor, so sloppy URDFs still load;
  - `<mesh>` geometries are referenced by file (meshdir = the URDF folder);
  - `radian` units are forced.
- **MJCF** (`*.xml`): loaded straight from MuJoCo (ground, friction, etc. is
  whatever the XML already declares).

Open from the toolbar combo (bundled `models/`) or File > Open Model.

### Floor check & justified spawn

Before a floating-base robot runs, the loader **measures its rest pose and
justifies the starting height** (the console prints the numbers):

- one `mj_forward` computes the lowest body point at rest (conservative
  bounding sphere per geom);
- if that point is below (or too close to) the ground plane, the free base is
  lifted so the robot starts **hovering ~2 cm above the floor** and visibly
  settles onto its feet on Run — it never spawns half-buried;
- the lift is re-applied after every **Reset** (`R`) and **Seek**, and is kept
  out of the model file itself, so a fixed-base robot (e.g. the acrobot) is
  untouched.

Menu actions in **File ▸ Open Model** are awaited with a status-bar note while
the robot compiles; picking from the toolbar combo loads in the background.

### Legged robot workflow (hexapod-style)

1. Launch with your robot, e.g. `--model C:\Users\megah\Downloads\XXXXX.urdf`
   (or pick `models/pla_hexapod.urdf` from the combo).
2. The loader auto-detects the floating body, adds a ground plane, and lifts
   the robot **just above the floor** (see "Floor check" above). Console shows
   the justified spawn height before you press Run.
3. Load **`controllers/hexapod_tripod_gait.py`** — a tripod gait tuned by the
   sliders: stride frequency, step length, step height, body pitch, Kp/Kd.
4. Press `Space`. The Telemetry child form pops up; watch joint angles/torque
   and energy. Raise frequency to walk faster, step height if feet clip the
   floor.
5. `F` + a body in **Track body** keeps the camera on the robot as it walks;
   the seek slider rewinds time with plots kept in sync.

A robot like the arm in `Downloads\ur10.urdf` works the same way — loader
welds it to the world and injects joint motors — but it has no ground support
needs, so just skip the gait controller and write joint-space controllers.

---

## Architecture, threading & optimization

```
MainWindow (Qt / GUI thread)
 ├─ QMdiArea
 │    ├─ Scene child form  (SceneViewer, PyVista actors)  <- reads snapshots only
 │    ├─ Telemetry child form (pyqtgraph)                 <- pops up on Run
 │    └─ Console child form
 ├─ Controls dock  +  Script editor dock
 ├─ LoadWorker (QThread, spawned per load)
 │    └─ model compile + PyVista mesh building (pure data) -> emits model+items
 └─ SimulationWorker (QThread, long-lived)   owns MjModel/MjData, control loop
      └─ ControllerManager                   compiles + hot-reloads scripts
```

**Three threads, each with one responsibility:**

1. **Physics thread** (`SimulationWorker`): steps MuJoCo with realtime pacing,
   applies the controller, re-applies the spawn lift on reset/seek, and
   publishes **immutable numpy snapshots** (~30 Hz signal) plus a live copy for
   the 30 ms render tick. The GUI never touches live physics memory, so no
   locks are needed on the display side and a slow render can never stall
   physics.
2. **GUI thread**: renders, edits the script, and reads snapshots only.
3. **Loader thread** (`LoadWorker`): compiling a big URDF and converting its
   meshes to polygons happens off the GUI thread; only the VTK `add_mesh`
   calls (which require the GUI thread) are done afterwards.

**Why not more parallel work?** Qt and OpenGL (VTK, pyqtgraph) are confined to
the GUI thread; MuJoCo and numpy release the GIL while stepping, so the
physics thread already runs in true parallel. The remaining UI costs were
**cut without extra threads**:

- `mj_fullM` (kinetic energy) is `O(nv^3)`; it is now sampled every 4th
  snapshot instead of every physics step.
- Contact-force solves are only run when **Contact forces** is checked —
  otherwise the contact list stays empty and the solver cost is zero.
- Snapshots are published at ~30 Hz even when physics runs at up to 400 Hz, so
  the GUI work is bounded regardless of step rate.

---

## Validation

Run the bundled checks (no display needed):

```bat
.venv\Scripts\python.exe -m robot_studio.smoke_test        rem physics + controllers (+ offscreen render with ROBOT_STUDIO_RENDER_TEST=1)
.venv\Scripts\python.exe -m robot_studio.widgets_smoke     rem telemetry/editor/worker/csv under offscreen Qt
.venv\Scripts\python.exe -m robot_studio.gui_smoke         rem full MainWindow construction (offscreen)
```

---

## Critical UX Mistakes to Avoid

This tool exists to make **iterating on control code fast and safe**. The
following mistakes break that goal; we designed around each one — keep them in
mind when changing the UI or adding features.

1. **Don't let a control-script exceptions kill the app.** A typo in a
   controller must pause the sim and print a traceback, never crash the window.
   Controllers run in the worker thread and are wrapped so a failure is
   recoverable (`ControllerProxy`).

2. **Never block or pause the physics with the UI.** Long editor saves, dialog
   `exec_()` loops, or heavy render work on the GUI thread cause stutter and
   dropped steps. All simulation work lives in the worker thread; the GUI only
   consumes snapshots.

3. **Keep the viewport large and the transport controls obvious.** Run/Pause is
   the primary action; it is a toolbar button *and* `Space`. Don't bury reset,
   record, and speed behind menus. Cramped viewports make tuning miserable.

4. **Hot reload must not reset the world.** Re-applying a controller re-runs
   `init` and keeps current `qpos` — restarting the scene makes gain tuning
   useless. Only "Reset" returns to the start pose.

5. **An exception must never leave the sim half-initialized.** Controllers
   (re)load transactionally: on failure the previous controller stays active,
   so you always have a working baseline to recover from.

6. **Keep plots and 3D in sync — same time axis, same data source.** Telemetry
   and the scene are fed from the *same* snapshot, never two parallel loops
   that drift apart. The seek slider scrubs both together.

7. **Bound your buffers.** Plot and data history are sliding windows (plots
   trim to the selected window; console caps at 5000 lines; contact arrows are
   capped). An unbounded buffer turns a 10-minute tuning session into a UI
   freeze or memory leak.

8. **Smooth the render, not the physics.** Actors are created once and only get
   4x4 `user_matrix` updates (no mesh rebuilds per frame). If a model is slow
   to render, optimize the actor path — don't drop the physics rate.

9. **Give feedback the moment a value changes.** Parameter sliders update the
   live controller immediately, and `print()` from scripts appears in the
   console in real time. Silent controls feel broken.

10. **Safe camera and follow behavior.** Follow-camera preserves its offset
    relative to the body instead of snapping the view to origin, and camera
    presets are stable. Jerky camera jumps are the fastest way to disorient a
    user mid-tuning.

11. **Reuse your environment.** The app is tuned against the exact versions in
    `requirements.txt` (MuJoCo 3.12 changed several APIs: full inertia order,
    `mj_fullM` shape, field renames). A fresh `pip install` of whatever is
    latest is how subtle regressions creep in.

12. **Guard the tech landmine: import order on Windows.** MuJoCo's DLL fails to
    initialize if loaded *after* Qt on this machine (`WinError 1114`). `app.py`
    imports `mujoco` before any Qt import — preserve that ordering in new entry
    points and test scripts.