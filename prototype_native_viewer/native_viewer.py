"""Standalone prototype: native MuJoCo viewer vs. the app's PyVista strategy.

This does NOT touch the working app.  It reuses the app's own model loader and
controller manager, but renders with MuJoCo's *native* OpenGL viewer instead of
PyVista, to give you a side-by-side comparison before committing to a switch.

Run:
    .venv/bin/python prototype_native_viewer/native_viewer.py [model] [controller]

Examples:
    .venv/bin/python prototype_native_viewer/native_viewer.py
    .venv/bin/python prototype_native_viewer/native_viewer.py \
        robot_studio/models/pla_hexapod.urdf \
        robot_studio/controllers/hexapod_tripod_gait.py

Two backends are offered (switch with --mode):

  --mode classic   mujoco.viewer.launch_passive        (default)
                   Our own physics+controller loop on a thread; the native
                   viewer window syncs to it.  Closest to the app's
                   "physics thread + separate viewer" separation.
  --mode simulate  mujoco.viewer.launch (built-in loop)
                   MuJoCo owns model+data+stepping entirely; we only hook the
                   controller via a step callback.  Simplest, but the physics
                   and the viewer are tied together in one loop.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import mujoco
import mujoco.viewer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from robot_studio import model_loader  # noqa: E402
from robot_studio.controller_manager import ControllerManager  # noqa: E402


def _resolve_path(value, default):
    if value:
        return os.path.abspath(value)
    return os.path.join(ROOT, default)


def run_passive(model, data, controller_path, log=print):
    """Own physics loop + controller; native viewer synced from our loop."""
    cm = ControllerManager(log=log)
    if controller_path:
        cm.load_file(controller_path, model, data)

    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)

    paused = False
    stop = False

    def key_cb(keycode):
        nonlocal paused
        if chr(keycode) in (" ", "P", "p"):
            paused = not paused
        elif chr(keycode) in ("R", "r"):
            mujoco.mj_resetData(model, data)
            cm.reset_controller(model, data)

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_cb, show_left_ui=True, show_right_ui=True
    ) as handle:
        log("[native] viewer open (Space = pause, R = reset)")
        mujoco.mj_forward(model, data)

        while handle.is_running() and not stop:
            if not paused:
                proxy = cm.active()
                for _ in range(4):
                    if proxy is not None:
                        proxy.step(model, data)
                    mujoco.mj_step(model, data)
            handle.sync()

    log("[native] viewer closed")


def run_simulate(model, controller_path, log=print):
    """Let MuJoCo own the sim loop.  NOTE: the built-in simulate launch does
    not expose a per-step callback, so the controller cannot be injected here.
    Use --mode classic (under mjpython) to drive a controller."""
    if controller_path:
        log("[warn] controller %s loaded but built-in simulate loop cannot "
            "run it; use --mode classic." % os.path.basename(controller_path))
    log("[native] launching built-in simulate viewer...")
    mujoco.viewer.launch(
        model, mujoco.MjData(model), show_left_ui=True, show_right_ui=True
    )
    log("[native] viewer closed")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native MuJoCo viewer prototype")
    parser.add_argument("model", nargs="?", default=None)
    parser.add_argument("controller", nargs="?", default=None)
    parser.add_argument(
        "--mode", choices=["classic", "simulate"], default="classic",
        help="classic = own physics+controller thread; simulate = MuJoCo's built-in loop",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args(argv)

    model = model_loader.load_model(
        _resolve_path(args.model, "robot_studio/models/acrobot.urdf")
    )
    print(model_loader.model_summary(model))

    controller_path = None
    if args.controller:
        controller_path = _resolve_path(args.controller, "")
    elif not args.model:
        controller_path = os.path.join(
            ROOT, "robot_studio/controllers/acrobot_inverse_dynamics.py"
        )

    log = lambda *a: print(*a, flush=True)

    if args.mode == "simulate":
        run_simulate(model, controller_path, log=log)
        return 0

    data = mujoco.MjData(model)
    spawn = model_loader.spawn_offset(model)
    if spawn["offset"]:
        mujoco.mj_resetData(model, data)
        data.qpos[2] = spawn["offset"]
        log("[native] spawn lift %.3f applied" % spawn["offset"])

    run_passive(model, data, controller_path, log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
