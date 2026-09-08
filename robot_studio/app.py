"""Command line entry point for robot_studio."""

from __future__ import annotations

import argparse
import os
import sys

import mujoco  # must load before Qt on Windows (DLL init ordering)

from PyQt5.QtWidgets import QApplication

from robot_studio import style


def _base_dirs():
    here = os.path.dirname(os.path.abspath(__file__))
    return {
        "models": os.path.join(here, "models"),
        "controllers": os.path.join(here, "controllers"),
        "runtime": os.path.join(here, "_runtime"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="robot_studio",
        description="MuJoCo physics + PyVista rendering workbench for control scripts.",
    )
    parser.add_argument("--model", default=None, help="URDF or MJCF model file (default: bundled acrobot.urdf)")
    parser.add_argument("--controller", default=None, help="control script file (.py)")
    parser.add_argument("--speed", type=float, default=1.0, help="initial realtime speed multiplier")
    args = parser.parse_args(argv)

    from robot_studio.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Robot Studio")
    style.apply(app)

    win = MainWindow(
        model_path=args.model,
        controller_path=args.controller,
        initial_speed=args.speed,
    )
    win.resize(1560, 940)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()