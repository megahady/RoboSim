"""Native-MuJoCo-viewer thread (option C) for the prototype.

Owns a ``mujoco.viewer.launch_passive`` session and renders whichever physics
state the GUI hands it.  It does NOT step physics itself — it only mirrors the
latest snapshot (qpos/qvel/ctrl) from the physics worker into a private
``MjData`` and calls ``handle.sync()`` so the native viewport shows the same
state the telemetry plots see (single source of truth).

Lifecycle is driven from the GUI: ``start`` opens the window, ``close`` shuts
it down, and pausing the worker simply freezes the mirror (no stepping).
"""

from __future__ import annotations

import threading
import time

import mujoco
import mujoco.viewer

from PyQt5.QtCore import QThread, pyqtSignal as Signal


class NativeViewerThread(QThread):
    """Background thread that opens/closes the native MuJoCo viewer window."""

    opened = Signal()
    closed = Signal(str)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.data = mujoco.MjData(model)
        mujoco.mj_forward(model, self.data)

        self._lock = threading.Lock()
        self._mirror = None          # latest snapshot dict to render
        self._open = False
        self._stop = False
        self._handle = None
        self._follow_body = None

    # ------------------------------------------------------------- public API
    def start(self, priority=QThread.InheritPriority):
        self._stop = False
        super().start(priority)

    def shutdown(self):
        """Request a clean close of the viewer + thread. Thread-safe."""
        self._stop = True
        h = None
        with self._lock:
            h = self._handle
        if h is not None:
            try:
                h.close()
            except Exception:
                pass
        if not self.wait(4000):
            self.terminate()
            self.wait(2000)

    def push_state(self, st):
        """Copy a physics snapshot into the render mirror (GUI thread)."""
        if st is None:
            return
        flat = dict(st)
        flat["qpos"] = _copy(st["qpos"])
        flat["qvel"] = _copy(st["qvel"])
        flat["ctrl"] = _copy(st["ctrl"])
        with self._lock:
            self._mirror = flat

    def set_follow(self, body_name):
        with self._lock:
            self._follow_body = body_name

    # ------------------------------------------------------------- thread run
    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                key_callback=None,
                show_left_ui=True,
                show_right_ui=True,
            ) as handle:
                with self._lock:
                    self._handle = handle
                    self._open = True
                self.opened.emit()
                while not self._stop and handle.is_running():
                    self._render_once()
                    # stdlib sleep; QThread event loop is not running here
                    time.sleep(0.008)
        except Exception as exc:  # e.g. dylib/framework error on plain python
            self.closed.emit("viewer error: %s" % exc)
            return
        finally:
            with self._lock:
                self._open = False
                self._handle = None
        self.closed.emit("viewer closed")

    def _render_once(self):
        with self._lock:
            st = self._mirror
            follow = self._follow_body
        if st is not None:
            self.data.qpos[:] = st["qpos"]
            self.data.qvel[:] = st["qvel"]
            self.data.ctrl[:] = st["ctrl"]
            mujoco.mj_forward(self.model, self.data)
        if follow is not None:
            self._apply_follow(follow)
        self._handle.sync()

    def _apply_follow(self, body_name):
        if self.model is None or self._handle is None:
            return
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        except Exception:
            return
        if body_id < 0:
            return
        cam = self._handle.cam
        pos = self.data.xpos[body_id].copy()
        cam.lookat = pos.tolist()


def _copy(a):
    import numpy as np
    return np.array(a, copy=True)
