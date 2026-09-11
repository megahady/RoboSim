# Option C prototype — Qt GUI augmenting the native MuJoCo viewer

Two cooperative processes. The **Qt GUI** (plain python) owns the macOS main
thread; a **sim_host** process (under `mjpython`) owns physics, the controller,
and MuJoCo's **native 3D viewer** in its own window. They talk over a localhost
TCP socket.

```
┌──────────────────────────────┐        TCP/JSON          ┌──────────────────────────────┐
│  Qt GUI  (plain python)      │ ◄──── frames/log ──────► │  sim_host (mjpython)         │
│  · robot / controller pickers│        ──── commands ──► │  · physics loop              │
│  · run/pause/reset, speed    │                          │  · ControllerManager         │
│  · headless toggle           │                          │  · native MuJoCo viewer (win)│
│  · pyqtgraph telemetry       │                          └──────────────────────────────┘
└──────────────────────────────┘
```

Why two processes: on macOS `mujoco.viewer.launch_passive` **requires**
`mjpython` (Python runs off the main thread, native viewer owns it), but *
`QApplication` must be created on the main thread. A single process can't give
both the Qt event loop and the native viewer the main thread — splitting them
sidesteps the clash entirely.

## Run

```bash
# GUI + native viewer window (default acrobot, autostarts)
.venv/bin/python prototype_gui/app.py

# add your own robot / controller to the combo boxes
.venv/bin/python prototype_gui/app.py --model path/to/robot.urdf --controller path/to/ctrl.py

# headless: tick the "Headless (no viewer)" box, or force it:
ROBOT_STUDIO_HEADLESS_DEFAULT=1 .venv/bin/python prototype_gui/app.py
```

### Headless simulation (no window)

`sim_host.py` supports running the simulation with no native viewer window at
all — the physics + controller run and telemetry frames are still served over
TCP. In headless mode no `mjpython` is needed (plain python is enough).

```bash
# headless server (paused until the GUI/CLI sends "run")
.venv/bin/python prototype_gui/sim_host.py --model M.urdf --headless --port 61234

# headless server that starts stepping immediately, stops after 2 s of sim time
.venv/bin/python prototype_gui/sim_host.py --model M.urdf --controller C.py \
    --headless --run --duration 2.0

# pure batch, no TCP at all (prints SIMHOST_PORT-style logs to stdout)
.venv/bin/python prototype_gui/sim_host.py --model M.urdf --controller C.py \
    --headless --run --duration 5.0 --no-server
```

Flags:

| flag            | effect                                             |
|-----------------|----------------------------------------------------|
| `--headless`    | disable the native viewer window                   |
| `--run`         | start stepping immediately (default: paused)       |
| `--speed`       | initial realtime speed multiplier (default 1.0)    |
| `--duration`    | stop after N simulated seconds (0 = unlimited)     |
| `--no-server`   | batch mode: don't open the TCP server              |
| `--port`        | TCP port for the GUI (default: ephemeral)          |

## TCP protocol

Commands (GUI → sim):

```json
{"cmd":"run"}  {"cmd":"pause"}  {"cmd":"reset"}
{"cmd":"set_rate","value":2.0}
{"cmd":"set_controller","path":"/abs/path.py"}
{"cmd":"quit"}
```

Frames (sim → GUI):

```json
{"type":"ready","summary":"..."}
{"type":"frame","time":0.0,"qpos":[...],"qvel":[...],"ctrl":[...],"ek":..,"ep":..,"qref":[...]}
{"type":"log","text":"..."}
{"type":"closed"}
```

## Files

| file | role |
|------|------|
| `app.py`        | GUI launcher (plain python) |
| `main_window.py`| loads the full window form from `ui/main_window.ui`; wires transport + telemetry + console |
| `ui/main_window.ui` | **entire window** in Qt Designer form: grouped toolbar (promoted `TopToolBar`) + telemetry (promoted `TelemetryPanel`) + params/console docks + status bar |
| `toolbar.py`    | top toolbar wrapper; loads `ui/main_toolbar.ui` (Model \| Controller \| Simulation \| View groups) |
| `ui/main_toolbar.ui` | Qt Designer form for the toolbar section |
| `sim_host.py`   | physics + controller + native viewer (or headless) + TCP server |
| `viewer_thread.py` | earlier single-process attempt (kept for reference; superseded) |

> Everything the user sees is a Designer form now — `ui/main_window.ui`
> (whole window) and `ui/main_toolbar.ui` (toolbar inside it). Open either in
> Designer to tweak layout/order without touching code.