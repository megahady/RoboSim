"""Headless Qt widgets smoke test (offscreen platform).

Run:
    .venv\\Scripts\\python.exe -m robot_studio.widgets_smoke
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import mujoco  # must load before Qt on Windows (DLL init ordering)


def rel(*parts):
    return os.path.join(HERE, *parts)


def main():
    from PyQt5.QtCore import QEventLoop, QTimer
    from PyQt5.QtWidgets import QApplication

    from robot_studio import model_loader
    from robot_studio.controller_manager import ControllerManager
    from robot_studio.simulation_worker import SimulationWorker
    from robot_studio.telemetry import CsvRecorder, header_columns, row_from_state

    app = QApplication([])

    print("== model ==")
    model = model_loader.load_model(rel("models", "acrobot.urdf"))

    print("== telemetry panel ==")
    from robot_studio.plots import TelemetryPanel
    panel = TelemetryPanel()
    panel.configure(model)
    for i in range(40):
        st = {
            "time": i * 0.05,
            "qpos": np.array([0.1 * np.sin(i * 0.1), 0.2 * np.cos(i * 0.1)]),
            "qvel": np.zeros(2),
            "ctrl": np.array([1.0, -1.0]),
            "qfrc_applied": np.zeros(2),
            "ek": 0.5, "ep": 1.0,
            "qref": np.array([0.0, 0.0]),
        }
        panel.update(st)
    print("   curves:", sorted(panel.signals.keys()))
    assert "ref q1" in panel.signals

    print("== editor widget ==")
    from robot_studio.editor import EditorWidget
    ed = EditorWidget()
    ed.editor.setPlainText("print('hello')\n# comment\ndef f():\n    return 1\n")
    got = {}
    ed.saved.connect(lambda f: got.update(path=f))
    tmpf = os.path.join(tempfile.gettempdir(), "rs_editor_save.py")
    ed._path = tmpf
    ed.save()
    assert os.path.exists(tmpf), "editor save failed"
    print("   save signal fired:", got.get("path") is not None)

    print("== simulation worker (thread) ==")
    import mujoco
    mgr = ControllerManager()
    data0 = mujoco.MjData(model)
    mgr.load_file(rel("controllers", "simple_pd_tracking.py"), model, data0)
    worker = SimulationWorker(model, mgr)
    snaps = []
    worker.snapshot_ready.connect(lambda s: snaps.append(s))
    worker.log.connect(lambda s: print("   [log]", s))
    hook = {"t": None}
    worker.start()
    worker.resume()

    loop = QEventLoop()
    QTimer.singleShot(900, loop.quit)
    loop.exec_()
    worker.pause()
    worker.stop()
    t_max = max(s["time"] for s in snaps)
    print("   snapshots:", len(snaps), "sim time:", round(t_max, 3))
    assert len(snaps) > 5 and t_max > 0.2, "worker did not step"
    last = worker.latest_state()
    assert np.all(np.isfinite(last["qpos"]))

    print("== csv recorder ==")
    rec = CsvRecorder(tempfile.gettempdir())
    path = rec.start(["h"] + header_columns(model), prefix="rs_widgets_smoke")
    d = worker.data
    row = row_from_state({"time": float(d.time), "qpos": np.asarray(d.qpos),
                          "qvel": np.asarray(d.qvel), "ctrl": np.asarray(d.ctrl),
                          "qfrc_applied": np.asarray(d.qfrc_applied),
                          "ek": 0.1, "ep": 0.2})
    rec.write(row)
    rec.stop()
    with open(path) as fh:
        head = fh.readline().strip()
    print("   header cols:", len(head.split(",")))
    assert len(head.split(",")) == 1 + len(header_columns(model))

    print("== WIDGETS SMOKE OK ==")


if __name__ == "__main__":
    main()