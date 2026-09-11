"""Prototype GUI (option C, two processes).

Qt GUI runs under plain python and owns the main thread.  It spawns a separate
``sim_host`` process (under mjpython) that owns physics + the native MuJoCo
viewer.  The two talk over a localhost TCP socket: the GUI sends commands
(run/pause/reset/set-rate/set-controller) and receives JSON frames that drive
the pyqtgraph telemetry.

3D scene:       native MuJoCo viewer (separate window, sim_host process)
Physics:        sim_host physics loop (ControllerManager)
Telemetry:      pyqtgraph here (fed from the same frames the viewer renders)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

from PyQt5.QtCore import QThread, pyqtSignal as Signal

from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal as Signal
from PyQt5.QtWidgets import QMainWindow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(ROOT, ".venv", "bin", "mjpython")
VENV_PYTHON = os.path.join(ROOT, ".venv", "bin", "python")
MAIN_WINDOW_UI = os.path.join(ROOT, "prototype_gui", "ui", "main_window.ui")


def _rel(*parts):
    return os.path.join(ROOT, *parts)


class SimClient(QThread):
    """Reads JSON frames/commands from sim_host over TCP (GUI thread safe)."""

    frame = Signal(dict)
    ready = Signal(dict)
    log = Signal(str)
    closed = Signal(str)

    def __init__(self, port, parent=None):
        super().__init__(parent)
        self.port = port
        self._stop = False

    def run(self):
        import time
        deadline = time.time() + 8.0
        sock = None
        while time.time() < deadline and not self._stop:
            try:
                sock = socket.create_connection(("127.0.0.1", self.port), timeout=1.0)
                break
            except OSError:
                time.sleep(0.2)
        if sock is None:
            self.closed.emit("could not connect to sim process")
            return
        sock.settimeout(0.5)
        buf = b""
        try:
            while not self._stop:
                try:
                    chunk = sock.recv(1 << 16)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    kind = msg.get("type")
                    if kind == "frame":
                        self.frame.emit(msg)
                    elif kind == "ready":
                        self.ready.emit(msg)
                    elif kind == "log":
                        self.log.emit(msg.get("text", ""))
                    elif kind == "closed":
                        self.closed.emit("sim process stopped")
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def stop_client(self):
        self._stop = True
        self.wait(3000)


class MainWindow(QMainWindow):
    _log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Studio — native viewer prototype (two-process)")
        self.model = None
        self.proc = None
        self.client = None
        self._running = False
        self._params_sliders = {}

        self._build_ui()
        self._log_signal.connect(self._log)
        self._available_models = {
            "acrobot": _rel("robot_studio/models/acrobot.urdf"),
            "pla_hexapod": _rel("robot_studio/models/pla_hexapod.urdf"),
        }
        self._available_controllers = {
            "(none)": None,
            "acrobot_inverse_dynamics.py": _rel("robot_studio/controllers/acrobot_inverse_dynamics.py"),
            "simple_pd_tracking.py": _rel("robot_studio/controllers/simple_pd_tracking.py"),
            "hexapod_tripod_gait.py": _rel("robot_studio/controllers/hexapod_tripod_gait.py"),
        }
        self._load_combos()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        uic.loadUi(MAIN_WINDOW_UI, self)

        tb = self.toolbar
        self.model_combo = tb.model_combo
        self.ctrl_combo = tb.ctrl_combo
        self.btn_run = tb.btn_run
        self.btn_reset = tb.btn_reset
        self.speed = tb.speed
        self.chk_headless = tb.chk_headless

        tb.model_combo.currentIndexChanged.connect(self._on_model_change)
        tb.ctrl_combo.currentIndexChanged.connect(self._on_ctrl_change)
        tb.btn_run.clicked.connect(self._toggle_run)
        tb.btn_reset.clicked.connect(self._send_reset)
        tb.speed.valueChanged.connect(self._send_rate)
        if os.environ.get("ROBOT_STUDIO_HEADLESS_DEFAULT", "") == "1":
            tb.chk_headless.setChecked(True)

        self.statusBar().showMessage("Pick a robot to launch the sim + viewer.")

    def _load_combos(self):
        for k in self._available_models:
            self.model_combo.addItem(k)
        for k in self._available_controllers:
            self.ctrl_combo.addItem(k)

    # ------------------------------------------------------------ launch/quit
    def _spawn_sim(self, model_name):
        self._stop_sim()
        self.telemetry.clear()
        self._log("""--- launching native viewer + sim for %s ---""" % model_name)

        port = self._free_port()
        interpreter = VENV_PYTHON if self.chk_headless.isChecked() else VENV_PY
        cmd = [
            interpreter, _rel("prototype_gui/sim_host.py"),
            "--model", self._available_models[model_name],
            "--port", str(port),
        ]
        if self.chk_headless.isChecked():
            cmd.append("--headless")
        ctrl_path = self._available_controllers.get(self.ctrl_combo.currentText())
        if ctrl_path:
            cmd += ["--controller", ctrl_path]

        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        self._drain_stdout()

        self.client = SimClient(port, self)
        self.client.frame.connect(self._on_frame)
        self.client.ready.connect(self._on_ready)
        self.client.log.connect(self._log)
        self.client.closed.connect(self._on_closed)
        self.client.start()
        self.btn_run.setText("Run")

    def _stop_sim(self):
        if self.client is not None:
            self._send({"cmd": "quit"})
            self.client.stop_client()
            self.client = None
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None
        self._running = False
        self.btn_run.setText("Run")

    def _drain_stdout(self):
        proc = self.proc

        def pump():
            for line in proc.stdout:
                line = line.strip()
                if line:
                    self._log_signal.emit("[sim] %s" % line)

        import threading
        t = threading.Thread(target=pump, daemon=True)
        t.start()

    # -------------------------------------------------------------- send cmds
    def _send(self, msg):
        if self.client is not None:
            import socket as _s
            try:
                s = socket.create_connection(("127.0.0.1", self.client.port), timeout=0.5)
                s.sendall((json.dumps(msg) + "\n").encode())
                s.close()
            except OSError as exc:
                self._log("[send] %s" % exc)

    def _toggle_run(self):
        if self.client is None:
            return
        if self._running:
            self._send({"cmd": "pause"})
            self.btn_run.setText("Run")
            self._running = False
        else:
            self._send({"cmd": "run"})
            self.btn_run.setText("Pause")
            self._running = True

    def _send_reset(self):
        self._send({"cmd": "reset"})

    def _send_rate(self, v):
        self._send({"cmd": "set_rate", "value": v})

    def _on_model_change(self, idx):
        if idx >= 0:
            self._spawn_sim(self.model_combo.currentText())

    def _on_ctrl_change(self, idx):
        name = self.ctrl_combo.currentText()
        path = self._available_controllers[name]
        if path:
            self._send({"cmd": "set_controller", "path": path})

    # ------------------------------------------------------------------ events
    def _on_frame(self, msg):
        st = {
            "time": msg["time"],
            "qpos": msg["qpos"],
            "qvel": msg["qvel"],
            "ctrl": msg["ctrl"],
            "ek": msg["ek"],
            "ep": msg["ep"],
            "qref": msg.get("qref"),
        }
        self.telemetry.update(st)

    def _on_ready(self, msg):
        self._log("[sim] ready: %s" % msg.get("summary", ""))
        self.statusBar().showMessage("Native viewer open — press Run to simulate.")

    def _on_closed(self, text):
        self._log("[sim] %s" % text)
        self.btn_run.setText("Run")
        self._running = False

    # ------------------------------------------------------------------ misc
    def _free_port(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def _log(self, text):
        if text:
            self.console.appendPlainText(text)
            print(text, flush=True)

    def closeEvent(self, event):
        self._stop_sim()
        event.accept()