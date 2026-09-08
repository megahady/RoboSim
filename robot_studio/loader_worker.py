"""Background model-loading thread.

Compiling a physics model from a large URDF (and converting its meshes into
PyVista polygons) can take a few seconds. That work runs here, off the GUI
thread; only the resulting ``(model, mesh_items)`` crosses the event boundary.
The VTK widgets themselves are only touched later, on the GUI thread, during
``main_window``'s ``_apply_loaded_model``.
"""

from __future__ import annotations

import traceback

import mujoco

from PyQt5.QtCore import QThread, pyqtSignal as Signal

from robot_studio import model_loader
from robot_studio.scene_viewer import build_geom_data


class LoadWorker(QThread):
    """Builds physics model + scene mesh data; emits ``ready`` or ``failed``."""

    ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path
        self._abort = False

    def cancel(self):
        self._abort = True

    def run(self):
        try:
            model = model_loader.load_model(self.path)
            if self._abort:
                return
            items = build_geom_data(model)
            if self._abort:
                return
            spawn = model_loader.spawn_offset(model)
            payload = {
                "model": model,
                "items": items,
                "summary": model_loader.model_summary(model),
                "warnings": model_loader.captured_warnings(),
                "spawn": spawn,
                "path": self.path,
            }
            self.ready.emit(payload)
        except Exception:
            self.failed.emit(traceback.format_exc())