"""Live telemetry plots built with pyqtgraph.

Plots are grouped automatically from the model dimensions:
    - Joint state      (q0..qN)        measured solid + optional reference dashed
    - Velocities       (dq0..dqN)
    - Actuation        (tau0..tauK)    ``ctrl`` or applied forces when no motors
    - Energy           (KE, PE, Total)

Buffers are sliding-window ring lists trimmed to the selected time window.
"""

from __future__ import annotations

import pyqtgraph as pg
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

pg.setConfigOptions(antialias=True, background="#17181a", foreground="#c2c8d0")
pg.setConfigOptions(useOpenGL=False)

_COLORS = ["#4fc3f7", "#ffb74d", "#81c784", "#f06292", "#ba68c8", "#fff176", "#4dd0e1", "#a5d6a7"]
_REF_COLOR = "#7fd0ff50"
_MAX_POINTS = 60000


class TelemetryPanel(QWidget):
    WINDOW_OPTIONS = ["5 s", "10 s", "30 s", "all"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._window = 10.0
        self._window_all = False
        self.plots = []
        self.curves = {}
        self.signals = {}
        self._plotted_q = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(3)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Telemetry"))
        bar.addStretch(1)
        bar.addWidget(QLabel("Time window"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(self.WINDOW_OPTIONS)
        self.window_combo.setCurrentIndex(1)
        self.window_combo.currentIndexChanged.connect(self._on_window)
        bar.addWidget(self.window_combo)
        root.addLayout(bar)
        self._glw = pg.GraphicsLayoutWidget()
        root.addWidget(self._glw, 1)

    # ------------------------------------------------------------- lifecycle
    def configure(self, model):
        self._glw.clear()
        self.plots = []
        self.curves = {}
        self.signals = {}
        self._plotted_q = 0
        nv = model.nv
        nu = model.nu
        self._add_plot("Joint state  q", ["q%d" % i for i in range(nv)])
        self._add_plot("Velocities  dq", ["dq%d" % i for i in range(nv)])
        prefix = "Actuation  tau" if nu else "Applied forces  f"
        self._add_plot(prefix, ["tau%d" % i for i in range(max(nu, 1))])
        self._add_plot("Energy", ["KE", "PE", "Total"])

    def clear(self):
        for name in self.signals:
            self.signals[name]["x"] = []
            self.signals[name]["y"] = []
            self.curves[name].setData([], [])

    def _add_plot(self, title, names):
        plot = self._glw.addPlot(row=len(self.plots), col=0)
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.addLegend(offset=(8, 8), labelTextColor="#c2c8d0")
        plot.setMenuEnabled(False)
        for k, name in enumerate(names):
            curve = plot.plot(pen=pg.mkPen(_COLORS[k % len(_COLORS)], width=1.6), name=name)
            self.curves[name] = curve
            self.signals[name] = {"x": [], "y": []}
        plot.setLabel("left", title)
        plot.getAxis("left").setTextPen(pg.mkPen("#c2c8d0"))
        plot.getAxis("bottom").setTextPen(pg.mkPen("#c2c8d0"))
        if len(self.plots) > 0:
            plot.setXLink(self.plots[0])
        self.plots.append(plot)

    def _ensure_refs(self, nq):
        if hasattr(self, "_ref_count") and self._ref_count == nq:
            return
        self._ref_count = nq
        for i in range(nq):
            name = "ref q%d" % i
            if name in self.curves:
                continue
            plot = self.plots[0]
            curve = plot.plot(pen=pg.mkPen(_REF_COLOR, width=1.2,
                                           style=pg.QtCore.Qt.DashLine), name=name)
            self.curves[name] = curve
            self.signals[name] = {"x": [], "y": []}

    # ------------------------------------------------------------- live update
    def update(self, st):
        t = float(st["time"])
        qref = st.get("qref")
        if qref is not None and len(qref) > 0:
            self._ensure_refs(len(qref))
        for name, buf in self.signals.items():
            try:
                v = self._extract(name, st)
            except Exception:
                continue
            buf["x"].append(t)
            buf["y"].append(v)
            self._trim(name, buf)
            self.curves[name].setData(buf["x"], buf["y"])

    def _extract(self, name, st):
        if name.startswith("ref q"):
            idx = int(name[5:])
            return float(st["qref"][idx])
        if name.startswith("tau"):
            idx = int(name[3:])
            if st["ctrl"].size:
                return float(st["ctrl"][idx])
            return float(st["qfrc_applied"][idx])
        if name.startswith("dq"):
            return float(st["qvel"][int(name[2:])])
        if name.startswith("q"):
            return float(st["qpos"][int(name[1:])])
        if name == "KE":
            return float(st["ek"])
        if name == "PE":
            return float(st["ep"])
        if name == "Total":
            return float(st["ek"]) + float(st["ep"])
        raise KeyError(name)

    # ------------------------------------------------------------------ trims
    def _on_window(self, idx):
        opt = self.WINDOW_OPTIONS[idx]
        self._window_all = opt == "all"
        if not self._window_all:
            self._window = float(opt[:-2])
        for name in list(self.signals.keys()):
            self._trim(name, self.signals[name])
            self.curves[name].setData(self.signals[name]["x"], self.signals[name]["y"])

    def _trim(self, name, buf):
        x = buf["x"]
        if self._window_all:
            if len(x) > _MAX_POINTS:
                half = len(x) // 2
                buf["x"] = x[half:]
                buf["y"] = buf["y"][half:]
            return
        cutoff = x[-1] - self._window if x else 0.0
        i = 0
        n = len(x)
        while i < n - 1 and x[i] < cutoff:
            i += 1
        if i:
            buf["x"] = x[i:]
            buf["y"] = buf["y"][i:]