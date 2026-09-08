"""Control-script lifecycle. Scripts are plain Python modules loaded fresh on
every reload, so hot-reload restarts controller state and never accumulates
stale module globals. A script defines:

    def init(model, data, params): ...      # called on load / re-init
    def step(model, data, params): ...      # called once per physics substep

Optional extras:
    PARAMS = {"Name": (min, max, value), ...}   # auto-generated sliders in the UI
    TITLE / HELP                                # shown in the controller panel
    def reference(data, params): ...            # returns target q vector (drawn dashed)

Any ``print()`` in a script is captured and forwarded to the app console.
"""

from __future__ import annotations

import io
import os
import threading
import traceback
from contextlib import redirect_stdout


class ControllerError(Exception):
    pass


class ControllerSpec:
    def __init__(self, title: str, path: str):
        self.title = title
        self.path = path
        self.init = None
        self.step = None
        self.reference = None
        self.params_def = {}
        self.help_text = ""


class ControllerProxy:
    """A loaded controller instance bound to one simulation."""

    def __init__(self, spec, params, model, data, on_log=None):
        self.spec = spec
        self.params = params
        self._on_log = on_log
        self.last_ref = None
        self._call_init(model, data)

    def _emit(self, text: str):
        if text and self._on_log:
            self._on_log(text.rstrip())

    def _call_init(self, model, data):
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                self.spec.init(model, data, self.params)
        except Exception:
            self.last_ref = None
            raise
        self._emit(buf.getvalue())
        self.last_ref = self._compute_ref(data)

    def _compute_ref(self, data):
        if self.spec.reference is None:
            return None
        try:
            ref = self.spec.reference(data, self.params)
            if ref is None:
                return None
            return ref
        except Exception:
            return None

    def reset(self, model, data):
        self._call_init(model, data)

    def step(self, model, data):
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                self.spec.step(model, data, self.params)
        except Exception:
            self._emit("Safely stopped: control script raised an exception.")
            self.last_ref = None
            raise
        self._emit(buf.getvalue())
        self.last_ref = self._compute_ref(data)


class ControllerManager:
    """Thread-safe holder of the active controller; hot-replaceable."""

    def __init__(self, log=None):
        self.log = log or (lambda *_: None)
        self._lock = threading.RLock()
        self._spec = None
        self._proxy = None

    # ------------------------------------------------------------------ read
    def active(self):
        with self._lock:
            return self._proxy

    def active_spec(self):
        with self._lock:
            return self._spec

    def params(self):
        proxy = self.active()
        return proxy.params if proxy is not None else {}

    def set_param(self, name: str, value: float):
        proxy = self.active()
        if proxy is not None:
            proxy.params[name] = value

    # ------------------------------------------------------------------ write
    def load_file(self, path, model, data):
        """Compile and activate a controller script. On failure the previous
        controller stays active and a ControllerError is raised."""
        path = os.path.abspath(path)
        spec = self._compile(path)
        params = {name: cfg[2] for name, cfg in spec.params_def.items()}
        proxy = ControllerProxy(spec, params, model, data, self.log)
        with self._lock:
            self._spec = spec
            self._proxy = proxy
        self.log("Controller loaded: %s" % os.path.basename(path))
        return spec

    def reload(self, model, data):
        spec = self.active_spec()
        if spec is None:
            return None
        return self.load_file(spec.path, model, data)

    def reset_controller(self, model, data):
        proxy = self.active()
        if proxy is not None:
            proxy.reset(model, data)

    # ---------------------------------------------------------------- internal
    def _compile(self, path) -> ControllerSpec:
        if not os.path.isfile(path):
            raise ControllerError("Controller file not found: %s" % path)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                source = fh.read()
        except OSError as exc:
            raise ControllerError("Cannot read %s: %s" % (path, exc)) from exc
        ns = {"__name__": "__robot_studio_controller__", "__file__": path}
        try:
            code = compile(source, path, "exec")
            exec(code, ns)
        except SyntaxError as exc:
            raise ControllerError("Syntax error: %s" % exc) from exc

        init = ns.get("init")
        step = ns.get("step")
        if not callable(init) or not callable(step):
            raise ControllerError(
                "Controller must define callables init(model, data, params) and step(model, data, params)."
            )
        spec = ControllerSpec(ns.get("TITLE") or os.path.splitext(os.path.basename(path))[0], path)
        spec.init = init
        spec.step = step
        spec.reference = ns.get("reference") if callable(ns.get("reference")) else None
        params_def = ns.get("PARAMS", {})
        if not isinstance(params_def, dict):
            params_def = {}
        spec.params_def = params_def
        spec.help_text = ns.get("HELP", "") or ""
        return spec