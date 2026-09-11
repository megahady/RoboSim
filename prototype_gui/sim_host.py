"""muJocoCo sim host for the two-process native-viewer prototype.

Runs under ``mjpython`` (required by ``launch_passive`` on macOS).  It owns:

  - the physics loop + controller (reuses robot_studio's ControllerManager)
  - the native MuJoCo viewer window (``launch_passive``)
  - a JSON-over-TCP server that lets the GUI:
        send commands:   {"cmd":"set_controller","path":...}
                         {"cmd":"run"} {"cmd":"pause"} {"cmd":"reset"}
                         {"cmd":"set_rate","value":...}  {"cmd":"quit"}
        receive frames:  {"type":"frame", "time":.., "qpos":[...], "qvel":[...],
                          "ctrl":[...], "ek":.., "ep":.., "qref":[...]}
                         {"type":"log", "text":...}
                         {"type":"ready"} {"type":"closed"}

The GUI must send "run" to begin stepping; the viewer window opens immediately
so you see the robot while it is paused.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback
import numpy as np

import mujoco
import mujoco.viewer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from robot_studio import model_loader  # noqa: E402
from robot_studio.controller_manager import ControllerManager  # noqa: E402

_TICK = 1.0 / 60.0

_EMPTY_SNAP_KEYS = ("qpos", "qvel", "ctrl", "qref")


def _to_list(a):
    if a is None:
        return None
    return [float(x) for x in a]


class SimHost:
    def __init__(self, model_path, port, log=print, headless=False,
                 initial_run=False, initial_rate=1.0, duration=0.0):
        self.headless = headless
        self.model = model_loader.load_model(model_path)
        self.data = mujoco.MjData(self.model)
        spawn = model_loader.spawn_offset(self.model)
        self.spawn_z = spawn["offset"]
        if self.spawn_z:
            self._lift()
        mujoco.mj_forward(self.model, self.data)

        self.manager = ControllerManager(log=log)
        self._running = False
        self._paused = not initial_run
        self._stop = False
        self._rate = initial_rate
        self._duration = float(duration)
        self._sock = None
        self._send_lock = threading.Lock()
        self._port = port
        self._state = self._grab()

        log("model: %s" % model_loader.model_summary(self.model))
        if headless:
            log("headless mode: native viewer disabled")

    # ------------------------------------------------------------ model utils
    def _lift(self):
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
                self.data.qpos[int(self.model.jnt_qposadr[j]) + 2] += self.spawn_z

    # ------------------------------------------------------------ snapshotting
    def _grab(self):
        d, m = self.data, self.model
        proxy = self.manager.active()
        qref = getattr(proxy, "last_ref", None)
        return {
            "time": float(d.time),
            "qpos": _to_list(d.qpos),
            "qvel": _to_list(d.qvel),
            "ctrl": _to_list(d.ctrl),
            "ek": float(d.energy[0]),
            "ep": float(d.energy[1]),
            "qref": _to_list(qref),
        }

    def _send_frame(self):
        if self._sock is None:
            return
        try:
            self._send({"type": "frame", **self._state})
        except OSError:
            pass

    def _send(self, obj):
        if self._sock is None:
            return
        try:
            raw = (json.dumps(obj) + "\n").encode()
            with self._send_lock:
                self._sock.sendall(raw)
        except OSError:
            pass

    def _log(self, text):
        if text:
            if self._sock is None:
                print(text, flush=True)
            else:
                self._send({"type": "log", "text": text})

    # ------------------------------------------------------------ command loop
    def run_server(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", self._port))
        srv.listen(1)
        self._sock = srv.accept()[0]
        self._sock.settimeout(0.2)
        self._send({"type": "ready", "summary": model_loader.model_summary(self.model)})
        self._send_frame()
        try:
            while not self._stop:
                try:
                    line = self._sock.recv(1 << 16)
                except socket.timeout:
                    self._send_frame()
                    continue
                if not line:
                    break
                for msg in line.decode().splitlines():
                    if msg.strip():
                        self._handle(json.loads(msg))
        finally:
            try:
                self._sock.close()
            except OSError:
                pass
            srv.close()

    def _handle(self, msg):
        cmd = msg.get("cmd")
        if cmd == "run":
            self._paused = False
        elif cmd == "pause":
            self._paused = True
        elif cmd == "reset":
            self._do_reset()
        elif cmd == "set_rate":
            self._rate = max(0.01, float(msg["value"]))
        elif cmd == "set_controller":
            self._load_controller(msg.get("path"))
        elif cmd == "quit":
            self._stop = True
        else:
            self._log("unknown cmd %r" % cmd)

    def _load_controller(self, path):
        if not path or not os.path.isfile(path):
            self._log("controller not found: %s" % path)
            return
        try:
            self.manager.load_file(path, self.model, self.data)
            self._log("controller loaded: %s" % os.path.basename(path))
        except Exception as exc:
            self._log("[controller error] %s" % exc)

    def _do_reset(self):
        mujoco.mj_resetData(self.model, self.data)
        if self.spawn_z:
            self._lift()
        self.manager.reset_controller(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._state = self._grab()
        self._send_frame()

    # ------------------------------------------------------------ physics loop
    def physics_loop(self):
        dt = float(self.model.opt.timestep)
        while not self._stop:
            if self._paused:
                time.sleep(0.004)
                continue
            t0 = time.perf_counter()
            proxy = self.manager.active()
            if proxy is not None:
                try:
                    proxy.step(self.model, self.data)
                except Exception:
                    self._paused = True
                    self._log("control script raised an error - paused")
            mujoco.mj_step(self.model, self.data)
            self._state = self._grab()
            self._send_frame()
            if self._duration > 0 and self.data.time >= self._duration:
                self._log("[sim] reached duration %.2f s - stopping" % self._duration)
                self._stop = True
                break
            elapsed = time.perf_counter() - t0
            sleep_s = dt / max(self._rate, 1e-6) - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    def run_viewer(self):
        """Native MuJoCo viewer. Must be on the mjpython UI thread; when run
        from mjpython, launch_passive dispatches the render loop there."""
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=True,
                show_right_ui=True,
            ) as handle:
                self._log("[viewer] window open")
                while not self._stop and handle.is_running():
                    handle.sync()
                    time.sleep(_TICK)
                self._log("[viewer] window closed")
        except Exception as exc:
            self._log("[viewer] error: %s" % exc)
            traceback.print_exc()


def _alloc_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--controller", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--headless", action="store_true",
                    help="no native viewer window (plain python is enough)")
    ap.add_argument("--run", action="store_true",
                    help="start stepping immediately (default: paused)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="initial realtime speed multiplier")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after this many simulated seconds (0 = unlimited)")
    ap.add_argument("--no-server", action="store_true",
                    help="don't open the TCP server (batch mode, requires --duration)")
    args = ap.parse_args(argv)
    port = None if args.no_server else (args.port or _alloc_port())

    host = SimHost(
        args.model, port or -1, headless=args.headless,
        initial_run=args.run, initial_rate=args.speed, duration=args.duration,
    )
    if args.controller:
        host._load_controller(args.controller)

    if port is not None:
        print("SIMHOST_PORT=%d" % port, flush=True)

    t_physics = threading.Thread(target=host.physics_loop, daemon=True)
    t_physics.start()

    if not args.headless:
        t_viewer = threading.Thread(target=host.run_viewer, daemon=True)
        t_viewer.start()
        # give the viewer a moment to appear before serving
        time.sleep(1.0)

    if args.no_server:
        # batch: run until --duration elapses (or ctrl-C)
        while not host._stop:
            time.sleep(0.1)
    else:
        host.run_server()

    host._stop = True
    if port is not None:
        host._send({"type": "closed"})
    time.sleep(0.5)


if __name__ == "__main__":
    main()