"""Option C prototype launcher: PyQt GUI (plain python) augmenting the native
MuJoCo viewer (a separate mjpython process it spawns).

The two processes each own their own macOS main thread, which is required:
Qt must init on the main thread and the native passive viewer requires
mjpython to dispatch to the main thread.  Splitting them avoids that clash.

Run (plain python — no mjpython needed for THIS process):

    .venv/bin/python prototype_gui/app.py
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mujoco  # noqa: E402  (load before Qt on Windows)

from PyQt5.QtWidgets import QApplication  # noqa: E402

from robot_studio import style  # noqa: E402
from prototype_gui.main_window import MainWindow  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Option C prototype GUI")
    parser.add_argument("--model", default=None, help="URDF/MJCF path to add")
    parser.add_argument("--controller", default=None, help="controller .py to add")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    app.setApplicationName("Robot Studio (native prototype)")
    style.apply(app)

    win = MainWindow()
    if args.model:
        key = os.path.basename(args.model)
        win._available_models[key] = os.path.abspath(args.model)
        win.model_combo.addItem(key)
    if args.controller:
        key = os.path.basename(args.controller)
        win._available_controllers[key] = os.path.abspath(args.controller)
        win.ctrl_combo.addItem(key)

    win.model_combo.setCurrentIndex(0)   # spawns sim + native viewer at launch
    win.resize(1360, 820)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()