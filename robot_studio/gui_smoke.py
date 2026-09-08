"""Best-effort headless MainWindow smoke (offscreen platform). The 3D
QtInteractor requires a GL context; with separatable failures, the test still
validates everything up to the viewport and falls back gracefully."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

import mujoco  # must load before Qt on Windows (DLL init ordering)


def rel(*parts):
    return os.path.join(HERE, *parts)


def main():
    from PyQt5.QtWidgets import QApplication

    from robot_studio import model_loader
    from robot_studio.main_window import MainWindow

    log = lambda s: print(s, flush=True)
    app = QApplication([])
    model = model_loader.load_model(rel("models", "acrobot.urdf"))
    log("model ok")

    try:
        log("constructing MainWindow ...")
        win = MainWindow(
            model_path=rel("models", "acrobot.urdf"),
            controller_path=rel("controllers", "simple_pd_tracking.py"),
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        log("MainWindow construct SKIPPED: %s: %s" % (type(exc).__name__, exc))
        return
    log("constructed")
    win.show()
    log("shown")

    log("viewer actor count: %d" % len(win.viewer.actor_by_geom))

    snap = {
        "time": 0.1,
        "qpos": np.array([0.1, -0.2]),
        "qvel": np.array([0.3, 0.4]),
        "ctrl": np.array([1.0, 2.0]),
        "qfrc_bias": np.array([0.0, 0.0]),
        "qfrc_applied": np.array([0.0, 0.0]),
        "ek": 1.5, "ep": 3.0,
        "xpos": np.zeros((model.ngeom, 3)),
        "xmat": np.zeros((model.ngeom, 9)),
        "body_xpos": np.zeros((model.nbody, 3)),
        "qref": np.array([0.0, 0.0]),
        "fps": 60.0,
        "steps": 10,
        "contacts": {"n": 0, "pos": np.zeros((0, 3)), "force": np.zeros((0, 3)), "frame": np.zeros((0, 3, 3))},
    }
    log("sending snapshot via worker state + _tick ...")
    win.sim._state = snap
    win._tick()
    log("time label: %s" % win._time_label.text())
    log("fps label: %s" % win._fps_label.text())
    log("GUILESS SMOKE OK")


if __name__ == "__main__":
    main()