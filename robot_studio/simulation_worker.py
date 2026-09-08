"""Physics worker thread: MuJoCo stepping with realtime pacing.

The worker owns the ``MjData`` object and runs the control script inside the
physics loop. Everything it touches on live data happens on this thread; the GUI
reads immutable snapshots (plain numpy copies) taken after each ``mj_step``.
"""

from __future__ import annotations

import math
import queue
import threading
import time
import traceback
import numpy as np
import mujoco

from PyQt5.QtCore import QThread, pyqtSignal as Signal

_SNAPSHOT_HZ = 30.0
_ENERGY_EVERY = 4      # mj_fullM is O(nv^3); sample it a few times per second
_CONTACT_EVERY = 2     # contact-force solves are costly; sample at ~half step rate

_EMPTY_CONTACTS = {"n": 0, "pos": np.zeros((0, 3)), "force": np.zeros((0, 3)), "frame": np.zeros((0, 3, 3))}


class SimulationWorker(QThread):
    snapshot_ready = Signal(dict)
    log = Signal(str)
    status = Signal(str)

    def __init__(self, model, manager, parent=None):
        super().__init__(parent)
        self.model = model
        self.manager = manager
        self.data = mujoco.MjData(model)
        mujoco.mj_forward(model, self.data)
        self._free_qpos = [
            int(model.jnt_qposadr[j]) + 2
            for j in range(model.njnt)
            if int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE
        ]
        self._spawn_z = 0.0
        self._want_contacts = False
        self._sample_tick = 0
        self._state_lock = threading.Lock()
        self._running = False
        self._paused = True
        self._rate = 1.0
        self._commands = queue.Queue()
        self._steps = 0
        self._fps = 0.0
        self._fps_clock = time.time()
        self._last_ek = 0.0
        self._last_ep = 0.0
        self._last_contacts = _EMPTY_CONTACTS
        self._apply_spawn()
        self._state = self._grab_state()

    def set_spawn_offset(self, z: float):
        """Lift floating bases so the robot starts just above the floor.
        Applied immediately and again after every reset / seek."""
        self._spawn_z = max(0.0, float(z))
        self._apply_spawn()
        mujoco.mj_kinematics(self.model, self.data)
        with self._state_lock:
            self._state = self._grab_state()
        self._publish_current()

    def set_contacts_enabled(self, on: bool):
        """Only sample contact forces when they are being visualized."""
        self._want_contacts = bool(on)

    def _apply_spawn(self):
        if self._spawn_z:
            for adr in self._free_qpos:
                self.data.qpos[adr] += self._spawn_z

    # ------------------------------------------------------------ public API
    @property
    def paused(self):
        return self._paused

    def start(self, priority=QThread.InheritPriority):
        self._running = True
        self._fps_clock = time.time()
        super().start(priority)

    def stop(self):
        self._running = False
        self.wait(5000)

    def resume(self):
        self._paused = False

    def pause(self):
        self._paused = True

    def set_rate(self, rate: float):
        self._rate = max(0.01, float(rate))

    def reset(self, reinit_controller=True):
        self.post(lambda: self._do_reset(reinit_controller))

    def step_forward(self):
        self.post(self._do_step)

    def seek(self, target_time: float):
        self.post(lambda: self._do_seek(target_time))

    def post(self, fn):
        self._commands.put(fn)

    def latest_state(self):
        with self._state_lock:
            return self._state

    # ------------------------------------------------------------- internals
    def run(self):
        model = self.model
        data = self.data
        dt = float(model.opt.timestep)
        emit_every = max(1, int(round(1.0 / (dt * _SNAPSHOT_HZ))))
        step_index = 0
        while self._running:
            self._drain_commands()
            if self._paused:
                time.sleep(0.004)
                continue
            t0 = time.perf_counter()
            self._step_once()
            step_index += 1
            if step_index % emit_every == 0:
                self._update_fps()
                with self._state_lock:
                    snap = self._state
                self.snapshot_ready.emit(
                    {
                        "time": snap["time"],
                        "qpos": snap["qpos"],
                        "qvel": snap["qvel"],
                        "ctrl": snap["ctrl"],
                        "ek": snap["ek"],
                        "ep": snap["ep"],
                        "qref": snap.get("qref"),
                    }
                )
            elapsed = time.perf_counter() - t0
            sleep_s = dt / max(self._rate, 1e-6) - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _drain_commands(self):
        while True:
            try:
                fn = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception:
                traceback.print_exc()
                self._running = False
                break

    def _step_once(self):
        data = self.data
        proxy = self.manager.active()
        if proxy is not None:
            try:
                proxy.step(self.model, data)
            except Exception:
                traceback.print_exc()
                self._paused = True
                self.status.emit("Control script raised an error - simulation paused.")
                return
        mujoco.mj_step(self.model, data)
        self._steps += 1
        with self._state_lock:
            self._state = self._grab_state()

    def _do_reset(self, reinit_controller=True):
        mujoco.mj_resetData(self.model, self.data)
        self._apply_spawn()
        if reinit_controller:
            self.manager.reset_controller(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._sample_tick = 0
        with self._state_lock:
            self._state = self._grab_state()
        self._publish_current()

    def _do_step(self):
        if not self._paused:
            return
        self._step_once()

    def _do_seek(self, target_time: float):
        self._paused = True
        mujoco.mj_resetData(self.model, self.data)
        self._apply_spawn()
        self.manager.reset_controller(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        t = float(target_time)
        max_iter = int(t / self.model.opt.timestep) + 8
        guard = 0
        while self.data.time < t - self.model.opt.timestep * 0.5 and guard < max_iter:
            self._step_physics_only()
            guard += 1
        self._steps += guard
        with self._state_lock:
            self._state = self._grab_state()
        self._publish_current()

    def _step_physics_only(self):
        proxy = self.manager.active()
        if proxy is not None:
            try:
                proxy.step(self.model, self.data)
            except Exception:
                self._paused = True
                self.status.emit("Seek stopped: control script raised an error.")
                return
        mujoco.mj_step(self.model, self.data)

    def _publish_current(self):
        with self._state_lock:
            snap = self._state
        self.snapshot_ready.emit(
            {
                "time": snap["time"],
                "qpos": snap["qpos"],
                "qvel": snap["qvel"],
                "ctrl": snap["ctrl"],
                "ek": snap["ek"],
                "ep": snap["ep"],
                "qref": snap.get("qref"),
            }
        )

    def _update_fps(self):
        now = time.time()
        dt = now - self._fps_clock
        if dt > 0.6:
            self._fps = self._steps / dt
            self._steps = 0
            self._fps_clock = now

    def _grab_state(self):
        d = self.data
        m = self.model
        proxy = self.manager.active()
        qref = getattr(proxy, "last_ref", None)
        self._sample_tick += 1
        st = {
            "time": float(d.time),
            "qpos": np.array(d.qpos, copy=True),
            "qvel": np.array(d.qvel, copy=True),
            "ctrl": np.array(d.ctrl, copy=True),
            "qfrc_bias": np.array(d.qfrc_bias, copy=True),
            "qfrc_applied": np.array(d.qfrc_applied, copy=True),
            "xpos": np.array(d.geom_xpos, copy=True),
            "xmat": np.array(d.geom_xmat, copy=True),
            "body_xpos": np.array(d.xpos, copy=True),
            "xipos": np.array(d.xipos, copy=True),
            "fps": self._fps,
            "steps": self._steps,
            "qref": qref,
        }
        if self._sample_tick % _ENERGY_EVERY == 0:
            st["ek"], st["ep"] = self._energies()
        else:
            st["ek"], st["ep"] = self._last_ek, self._last_ep
        if self._want_contacts and self._sample_tick % _CONTACT_EVERY == 0:
            st["contacts"] = self._contacts()
        else:
            st["contacts"] = getattr(self, "_last_contacts", _EMPTY_CONTACTS)
        self._last_ek, self._last_ep = st["ek"], st["ep"]
        self._last_contacts = st["contacts"]
        return st

    def _energies(self):
        m = self.model
        d = self.data
        nv = m.nv
        ek = 0.0
        if nv:
            flat = np.zeros((nv, nv))
            try:
                mujoco.mj_fullM(m, d, flat)
                ek = 0.5 * float(np.asarray(d.qvel) @ flat @ np.asarray(d.qvel))
            except Exception:
                ek = 0.0
        g = abs(float(m.opt.gravity[2]))
        ep = float(np.dot(np.asarray(m.body_mass), np.asarray(d.xipos)[:, 2])) * g
        return ek, ep

    def _contacts(self):
        d = self.data
        pos, force, frame = [], [], []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > 0:
                continue
            res = np.zeros(6)
            try:
                mujoco.mj_contactForce(self.model, d, i, res)
            except Exception:
                continue
            pos.append(np.array(c.pos, copy=True))
            force.append(np.array(res[0:3], copy=True))
            frame.append(np.asarray(c.frame, dtype=float).reshape(3, 3).copy())
        n = len(pos)
        if not n:
            return {"n": 0, "pos": np.zeros((0, 3)), "force": np.zeros((0, 3)), "frame": np.zeros((0, 3, 3))}
        return {"n": n, "pos": np.array(pos), "force": np.array(force), "frame": np.array(frame)}