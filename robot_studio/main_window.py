"""Main application window: MDI workbench with transport bar, 3D scene, plots,
console, controller panel and embedded script editor.

The central widget is a ``QMdiArea``: the **3D Scene**, **Telemetry** and
**Console** are child forms (``QMdiSubWindow``) of it. Child forms can be
tiled, cascaded, floated into separate top-level windows or docked back, and
new instrument panels can be added later as further subwindows. Only the
**right Controls** dock and the **Script editor** dock are fixed to the window.
Models are compiled on a background thread (``LoadWorker``) so the UI never
freezes when opening large robots.
"""

from __future__ import annotations

import glob
import os
import time

import mujoco

from PyQt5.QtCore import QEvent, QSettings, Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QShortcut,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from robot_studio import model_loader
from robot_studio.controller_manager import ControllerManager
from robot_studio.editor import EditorWidget
from robot_studio.loader_worker import LoadWorker
from robot_studio.plots import TelemetryPanel
from robot_studio.scene_viewer import SceneViewer
from robot_studio.simulation_worker import SimulationWorker
from robot_studio.telemetry import CsvRecorder, header_columns, row_from_state

_SPEEDS = ["0.25x", "0.5x", "1x", "2x", "5x", "10x", "20x"]


class MainWindow(QMainWindow):
    def __init__(self, model_path=None, controller_path=None, initial_speed=1.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Robot Studio - MuJoCo + PyVista control workbench")

        self._base_dir = os.path.dirname(os.path.abspath(__file__))
        self._models_dir = os.path.join(self._base_dir, "models")
        self._controllers_dir = os.path.join(self._base_dir, "controllers")
        self._runtime_dir = os.path.join(self._base_dir, "_runtime")
        self._model_files = []
        self._controller_files = []
        self._body_ids = []
        self.model = None
        self.model_path = None
        self.sim = None
        self.manager = ControllerManager(log=self._log)
        self._recorder = CsvRecorder(os.path.join(self._runtime_dir, "recordings"))
        self._recording = False
        self._tl_dragging = False
        self._settings = QSettings("HadyLab", "robot_studio")
        self.plots = TelemetryPanel()
        self._spawn = {"free": False, "lowest": 0.0, "offset": 0.0}
        self._load_worker = None
        self._last_load_ok = None
        self._closing = False

        self._mdi = QMdiArea(self)
        self._build_scene()
        self._build_right_panel()
        self._build_console()
        self._build_editor_dock()
        self._build_toolbar()
        self._assemble_mdi()
        self._build_menus()
        self._install_key_filter()
        self._build_shortcuts()

        self._paint_timer = QTimer(self)
        self._paint_timer.timeout.connect(self._tick)
        self._paint_timer.start(33)

        self._restore_session(model_path, controller_path, initial_speed)

    # ------------------------------------------------------------- UI build
    def _build_scene(self):
        self._scene_sub = QMdiSubWindow()
        self._scene_sub.setWindowTitle("3D Scene")
        self._scene_sub.setAttribute(Qt.WA_DeleteOnClose, False)
        try:
            from pyvistaqt import QtInteractor
        except Exception as exc:
            self.viewer = None
            label = QLabel("pyvistaqt is not available:\n%s" % exc)
            self._scene_sub.setWidget(label)
            return
        self._viewer_shell = QtInteractor(self._scene_sub)
        self._scene_sub.setWidget(self._viewer_shell)
        self.viewer = SceneViewer(self._viewer_shell)
        self._scene_sub.setMinimumSize(420, 300)

    def _assemble_mdi(self):
        self._plots_sub = QMdiSubWindow()
        self._plots_sub.setWindowTitle("Telemetry")
        self._plots_sub.setWidget(self.plots)
        self._plots_sub.setAttribute(Qt.WA_DeleteOnClose, False)
        self._plots_sub.resize(760, 520)

        self._console_sub = QMdiSubWindow()
        self._console_sub.setWindowTitle("Console")
        self._console_sub.setWidget(self._console_group)
        self._console_sub.setAttribute(Qt.WA_DeleteOnClose, False)
        self._console_sub.resize(560, 240)

        self._mdi.addSubWindow(self._scene_sub)
        self._mdi.addSubWindow(self._plots_sub)
        self._mdi.addSubWindow(self._console_sub)
        self._scene_sub.show()
        self._console_sub.show()
        self._plots_sub.hide()
        self._controls_dock = QDockWidget("Controls", self)
        self._controls_dock.setObjectName("ControlsDock")
        self._controls_dock.setWidget(self._right_panel)
        self._controls_dock.setMinimumWidth(330)
        self.addDockWidget(Qt.RightDockWidgetArea, self._controls_dock)
        self.setCentralWidget(self._mdi)

    def _build_right_panel(self):
        panel = QWidget()
        root = QVBoxLayout(panel)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        sim_group = QGroupBox("Simulation")
        grid = QGridLayout(sim_group)
        self.btn_run = QPushButton("Run")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._toggle_run)
        self.btn_step = QPushButton("Step")
        self.btn_step.clicked.connect(self._step_once)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset)
        self.btn_record = QPushButton("Record")
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self._toggle_record)
        self.btn_shot = QPushButton("Snapshot")
        self.btn_shot.clicked.connect(self._screenshot)
        grid.addWidget(self.btn_run, 0, 0)
        grid.addWidget(self.btn_step, 0, 1)
        grid.addWidget(self.btn_reset, 0, 2)
        grid.addWidget(self.btn_record, 1, 0)
        grid.addWidget(self.btn_shot, 1, 1)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(_SPEEDS)
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(self._on_speed)
        grid.addWidget(QLabel("Realtime"), 2, 0)
        grid.addWidget(self.speed_combo, 2, 1, 1, 2)
        self._time_label = QLabel("t = 0.000 s")
        self._fps_label = QLabel("")
        grid.addWidget(self._time_label, 3, 0, 1, 1)
        grid.addWidget(self._fps_label, 3, 1, 1, 2, Qt.AlignRight)
        self._tl = QSlider(Qt.Horizontal)
        self._tl.setRange(0, 30000)
        self._tl.setValue(0)
        self._tl.sliderPressed.connect(self._on_tl_pressed)
        self._tl.sliderReleased.connect(self._on_tl_released)
        grid.addWidget(QLabel("Seek"), 4, 0)
        grid.addWidget(self._tl, 4, 1, 1, 2)
        root.addWidget(sim_group)

        ctrl_group = QGroupBox("Controller")
        croot = QVBoxLayout(ctrl_group)
        self._params_box = QWidget()
        self._params_layout = QVBoxLayout(self._params_box)
        self._params_layout.setContentsMargins(0, 0, 0, 0)
        self._params_layout.addWidget(QLabel("No tunable parameters in this script."))
        croot.addWidget(self._params_box)
        self._help_label = QLabel("")
        self._help_label.setWordWrap(True)
        self._help_label.setStyleSheet("color:#8fa3bd; font-size:11px;")
        croot.addWidget(self._help_label)
        btns = QHBoxLayout()
        self.btn_reload_ctrl = QPushButton("Reload script")
        self.btn_reload_ctrl.clicked.connect(self._reload_controller)
        self.btn_reinit_ctrl = QPushButton("Re-init state")
        self.btn_reinit_ctrl.clicked.connect(self._reinit_controller)
        btns.addWidget(self.btn_reload_ctrl)
        btns.addWidget(self.btn_reinit_ctrl)
        croot.addLayout(btns)
        root.addWidget(ctrl_group)

        view_group = QGroupBox("Rendering / Camera")
        vgrid = QGridLayout(view_group)
        presets = ["orbit", "front", "side", "top", "fit"]
        self._cam_buttons = {}
        for k, name in enumerate(presets):
            b = QPushButton(name.title())
            b.clicked.connect(lambda _=False, n=name: self._camera_action(n))
            self._cam_buttons[name] = b
            vgrid.addWidget(b, 0, k)
        vgrid.addWidget(QLabel("Track body"), 1, 0)
        self.follow_combo = QComboBox()
        vgrid.addWidget(self.follow_combo, 1, 1, 1, 4)
        self.follow_check = QCheckBox("Follow target")
        self.follow_check.toggled.connect(self._on_follow)
        vgrid.addWidget(self.follow_check, 2, 0, 1, 2)
        self.contacts_check = QCheckBox("Contact forces")
        self.contacts_check.toggled.connect(self._on_contacts)
        vgrid.addWidget(self.contacts_check, 2, 2, 1, 3)
        root.addWidget(view_group)
        root.addStretch(1)

        self._right_panel = panel

    def _build_console(self):
        self._console_group = QWidget()
        cg = QVBoxLayout(self._console_group)
        cg.setContentsMargins(4, 8, 4, 4)
        title = QLabel("<b>Console</b>")
        cg.addWidget(title)
        self._console = QPlainTextEdit()
        self._console.setReadOnly(True)
        self._console.setMaximumBlockCount(5000)
        cg.addWidget(self._console)

    def _build_toolbar(self):
        tb = self.addToolBar("Transport")
        tb.setMovable(False)
        self._ac_run = QAction("Run", self)
        self._ac_run.triggered.connect(self._toggle_run)
        ac_step = QAction("Step", self)
        ac_step.triggered.connect(self._step_once)
        ac_reset = QAction("Reset", self)
        ac_reset.triggered.connect(self._reset)
        ac_record = QAction("Record", self, checkable=True)
        ac_record.triggered.connect(self._toggle_record)
        self._ac_record = ac_record
        tb.addAction(self._ac_run)
        tb.addAction(ac_step)
        tb.addAction(ac_reset)
        tb.addAction(ac_record)
        tb.addSeparator()

        tb.addWidget(QLabel("Model  "))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo)
        tb.addWidget(self.model_combo)
        tb.addWidget(QLabel("  Controller  "))
        self.ctrl_combo = QComboBox()
        self.ctrl_combo.setMinimumWidth(220)
        self.ctrl_combo.currentIndexChanged.connect(self._on_ctrl_combo)
        tb.addWidget(self.ctrl_combo)
        self._rescan()

    def _build_menus(self):
        mfile = self.menuBar().addMenu("&File")
        open_model = QAction("Open Model...", self)
        open_model.triggered.connect(self._open_model_dialog)
        open_ctrl = QAction("Open Controller...", self)
        open_ctrl.triggered.connect(self._open_controller_dialog)
        mfile.addAction(open_model)
        mfile.addAction(open_ctrl)
        mfile.addSeparator()
        mfile.addAction("Exit", self.close)

        mview = self.menuBar().addMenu("&View")
        mview.addAction(self._editor_dock.toggleViewAction())
        mview.addAction(self._controls_dock.toggleViewAction())
        mview.addSeparator()
        mview.addAction("Plots", self._show_plots)
        mview.addAction("Console", self._show_console)
        mview.addAction("3D Scene", self._show_scene)
        mview.addSeparator()
        mview.addAction("Cascade", self._mdi.cascadeSubWindows)
        mview.addAction("Tile", self._mdi.tileSubWindows)
        mview.addAction("Close All Windows", self._mdi.closeAllSubWindows)

        mhelp = self.menuBar().addMenu("&Help")
        mhelp.addAction("Shortcuts", self._show_shortcuts)
        mhelp.addAction("About", self._show_about)

    def _build_editor_dock(self):
        self._editor_widget = EditorWidget()
        self._editor_widget.saved.connect(self._reload_controller_from)
        dock = QDockWidget("Controller Script", self)
        dock.setWidget(self._editor_widget)
        dock.setObjectName("ScriptDock")
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self._editor_dock = dock
        self._editor_dock.hide()

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Shift+N"), self, activated=self._open_model_dialog)
        QShortcut(QKeySequence("Ctrl+Shift+O"), self, activated=self._open_controller_dialog)

    def _install_key_filter(self):
        from PyQt5.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self)

    # ------------------------------------------------------------ key filter
    def eventFilter(self, obj, event):
        if event.type() != QEvent.KeyPress:
            return False
        if self._is_text_input(obj):
            return False
        app = self.sim
        if app is None or self._is_modal_shown():
            return False
        key = event.key()
        mods = event.modifiers()
        if mods == Qt.NoModifier:
            if key == Qt.Key_Space:
                self._toggle_run()
                return True
            if key == Qt.Key_R:
                self._reset()
                return True
            if key == Qt.Key_C:
                self._toggle_record(not self._recording)
                return True
            if key == Qt.Key_F:
                self.follow_check.setChecked(not self.follow_check.isChecked())
                return True
            if key == Qt.Key_S:
                self._screenshot()
                return True
        return False

    def _is_text_input(self, obj):
        from PyQt5.QtWidgets import QComboBox, QLineEdit, QPlainTextEdit, QSpinBox, QTextEdit
        return isinstance(obj, (QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox))

    def _is_modal_shown(self):
        from PyQt5.QtWidgets import QApplication
        return QApplication.activeModalWidget() is not None

    # ------------------------------------------------------------ session
    def _restore_session(self, model_path, controller_path, initial_speed):
        settings = self._settings
        if model_path is None:
            model_path = settings.value("last_model")
        if controller_path is None:
            controller_path = settings.value("last_controller")
        if initial_speed != 1.0:
            text = "%gx" % initial_speed
            if text in _SPEEDS:
                self.speed_combo.setCurrentIndex(_SPEEDS.index(text))
        self._on_speed()

        self.model_combo.setCurrentIndex(-1)
        if model_path and os.path.isfile(model_path):
            self_load = model_path
        else:
            self_load = None
        if self_load is None:
            fallback = os.path.join(self._models_dir, "acrobot.urdf")
            if os.path.isfile(fallback):
                self_load = fallback
        if self_load:
            ok = self.load_model(self_load, wait=True)
            if ok and self._last_load_ok and controller_path and os.path.isfile(controller_path):
                self.load_controller(controller_path)
        else:
            self._log("No model available. Open File > Open Model...")

    # ------------------------------------------------------------ scanning
    def _rescan(self):
        self._model_files = sorted(
            glob.glob(os.path.join(self._models_dir, "*.urdf"))
            + glob.glob(os.path.join(self._models_dir, "*.xml"))
        )
        self._controller_files = sorted(glob.glob(os.path.join(self._controllers_dir, "*.py")))
        with_model = self._settings.value("vendored_models", "", type=str)
        for p in (with_model.split(";") if with_model else []):
            if p and os.path.isfile(p) and p not in self._model_files:
                self._model_files.append(p)

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for p in self._model_files:
            self.model_combo.addItem(os.path.basename(p), p)
        self.model_combo.blockSignals(False)

        self.ctrl_combo.blockSignals(True)
        self.ctrl_combo.clear()
        for p in self._controller_files:
            self.ctrl_combo.addItem(os.path.basename(p), p)
        self.ctrl_combo.blockSignals(False)

    def _suggest_controller(self, model_path):
        base = os.path.splitext(os.path.basename(model_path))[0]
        for p in self._controller_files:
            if os.path.splitext(os.path.basename(p))[0].startswith(base):
                return p
        return self._controller_files[0] if self._controller_files else None

    # ------------------------------------------------------------ model load
    def load_model(self, path, wait=False):
        path = os.path.abspath(path)
        if self._load_worker is not None and self._load_worker.isRunning():
            self._load_worker.cancel()
        self._last_load_ok = None
        self._log("Loading %s ..." % os.path.basename(path))
        if self.viewer is not None:
            self.statusBar().showMessage("Compiling %s ..." % os.path.basename(path), 0)
        w = LoadWorker(path, parent=self)
        w.ready.connect(self._on_load_done)
        w.failed.connect(self._on_load_failed)
        self._load_worker = w
        w.start()
        if wait:
            self._wait_loader(w)
        return True

    def _wait_loader(self, w):
        guard = 0
        while w.isRunning() and not self._closing and guard < 2000:
            QApplication.processEvents()
            time.sleep(0.005)
            guard += 1
        if w.isRunning():
            w.wait(3000)
        guard = 0
        while self._last_load_ok is None and not self._closing and guard < 2000:
            QApplication.processEvents()
            time.sleep(0.005)
            guard += 1

    def _on_load_done(self, payload):
        self._load_worker = None
        self.statusBar().clearMessage()
        self._last_load_ok = True
        self._apply_loaded_model(payload)

    def _on_load_failed(self, trace):
        self._load_worker = None
        self.statusBar().clearMessage()
        self._last_load_ok = False
        self._log("Model load failed: %s" % trace)
        QMessageBox.warning(self, "Model", "Load failed:\n%s" % (trace or "unknown error"))
        return False

    def _apply_loaded_model(self, payload):
        model = payload["model"]
        items = payload["items"]
        path = payload["path"]
        if self.sim is not None:
            self.sim.stop()
            self.sim = None
        self.model = model
        self.model_path = path
        self._spawn = payload["spawn"]
        if self.viewer is not None:
            try:
                self.viewer.install_model(model, items)
            except Exception as exc:
                self._log("3D view build failed: %s" % exc)
        self.manager = ControllerManager(log=self._log)
        self.sim = SimulationWorker(model, self.manager, parent=self)
        self.sim.set_spawn_offset(float(self._spawn["offset"]))
        self.sim.set_contacts_enabled(self.contacts_check.isChecked())
        self.sim.snapshot_ready.connect(self._on_snapshot)
        self.sim.log.connect(self._log)
        self.sim.status.connect(lambda s: self.statusBar().showMessage(s, 5000))
        self.sim.start()
        self.plots.configure(model)
        self.plots.clear()
        self._populate_follow(model)
        self._reset_timeline()
        self._recorder = CsvRecorder(os.path.join(self._runtime_dir, "recordings"))
        self._settings.setValue("last_model", path)
        self._log(model_loader.model_summary(model))
        for w in model_loader.captured_warnings():
            self._log("  %s" % w)
        self._log_spawn()
        idx = self._find_combo_index(self.model_combo, path)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.statusBar().showMessage(model_loader.model_summary(model), 8000)

        controller = self._suggest_controller(path)
        if controller:
            self.load_controller(controller)
        return True

    def _log_spawn(self):
        """Justify the starting pose before the simulation runs: the requested
        hover height above the floor and the lift that achieves it."""
        sp = self._spawn
        if not sp["free"]:
            self._log("Floor check: fixed-base model; rest pose is used as-is.")
            return
        if sp["offset"]:
            low = sp["lowest"]
            self._log(
                "Floor check: lowest body point is %.0f mm below the ground plane at rest; "
                "raising the free base by %+.3f m so the robot starts %.0f mm above the floor "
                "and settles onto its feet on Run." % (low * 1000.0, sp["offset"], (low + sp["offset"]) * 1000.0)
            )
        else:
            self._log("Floor check: rest pose already clears the ground (lowest point %.0f mm)."
                      % (sp["lowest"] * 1000.0))

    def _populate_follow(self, model):
        self.follow_combo.blockSignals(True)
        self.follow_combo.clear()
        self.follow_combo.addItem("(world)", -1)
        self._body_ids = []
        names = [model_loader.name_of(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
        for i, name in enumerate(names):
            self.follow_combo.addItem(name if i else "(world)", i)
            self._body_ids.append(i)
        self.follow_combo.blockSignals(False)
        self.follow_check.setChecked(False)

    # --------------------------------------------------------- controller
    def load_controller(self, path):
        if self.model is None or self.sim is None:
            self._log("Cannot load controller before a model.")
            return False
        path = os.path.abspath(path)
        try:
            spec = self.manager.load_file(path, self.model, self.sim.data)
        except Exception as exc:
            self._log("Controller load failed: %s" % exc)
            QMessageBox.warning(self, "Controller", str(exc))
            return False
        self._build_params(spec)
        self._help_label.setText(spec.help_text)
        if hasattr(self, "_editor_widget"):
            self._editor_widget.open(path)
            self._editor_dock.show()
        idx = self._find_combo_index(self.ctrl_combo, path)
        if idx >= 0:
            self.ctrl_combo.blockSignals(True)
            self.ctrl_combo.setCurrentIndex(idx)
            self.ctrl_combo.blockSignals(False)
        self._settings.setValue("last_controller", path)
        return True

    def _reload_controller(self):
        if self.sim is None:
            return
        try:
            spec = self.manager.reload(self.model, self.sim.data)
        except Exception as exc:
            self._log("Reload failed: %s" % exc)
            QMessageBox.warning(self, "Controller", str(exc))
            return
        if spec:
            self._build_params(spec)
            self._help_label.setText(spec.help_text)
            self._log("Controller reloaded.")

    def _reload_controller_from(self, path):
        spec = self.manager.active_spec()
        if spec is None:
            self._log("No active controller to reload.")
            return
        try:
            self.manager.load_file(spec.path, self.model, self.sim.data)
        except Exception as exc:
            self._log("Reload failed: %s" % exc)
            return
        self._log("Saved + reloaded controller.")

    def _reinit_controller(self):
        if self.sim is None:
            return
        try:
            self.manager.reset_controller(self.model, self.sim.data)
        except Exception as exc:
            self._log("Re-init failed: %s" % exc)
            return
        self._log("Controller state re-initialized.")

    def _build_params(self, spec):
        while self._params_layout.count():
            item = self._params_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        defaults = spec.params_def
        if not defaults:
            self._params_layout.addWidget(QLabel("No tunable parameters in this script."))
            return
        for name, (vmin, vmax, val) in defaults.items():
            row = QHBoxLayout()
            label = QLabel(name)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 1000)
            slider.setValue(int((float(val) - vmin) / max(vmax - vmin, 1e-9) * 1000.0))
            val_label = QLabel("%.3f" % float(val))
            val_label.setMinimumWidth(52)
            val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            slider.valueChanged.connect(
                lambda v, n=name, lo=float(vmin), hi=float(vmax), sl=slider, vl=val_label: self._param_changed(
                    n, lo, hi, sl, vl
                )
            )
            row.addWidget(label, 1)
            row.addWidget(slider, 3)
            row.addWidget(val_label)
            self._params_layout.addLayout(row)
        self._params_layout.addStretch(1)

    def _param_changed(self, name, vmin, vmax, slider, val_label):
        v = vmin + (vmax - vmin) * slider.value() / 1000.0
        val_label.setText("%.3f" % v)
        self.manager.set_param(name, v)

    # ------------------------------------------------------------ transport
    def _toggle_run(self, checked=None):
        if self.sim is None:
            return
        if self.sim.paused:
            self.sim.resume()
            self._show_plots()
        else:
            self.sim.pause()
        self._sync_run_button()

    def _sync_run_button(self):
        paused = self.sim is None or self.sim.paused
        self.btn_run.setText("Run" if paused else "Pause")
        self.btn_run.setChecked(not paused)
        self._ac_run.setText("Run" if paused else "Pause")

    def _step_once(self):
        if self.sim is not None:
            self.sim.pause()
            self._sync_run_button()
            self.sim.step_forward()

    def _reset(self):
        if self.sim is None:
            return
        self.sim.reset(reinit_controller=True)
        self.plots.clear()
        self._sync_run_button()

    def _on_speed(self):
        if self.sim is not None:
            text = self.speed_combo.currentText()
            mult = float(text[:-1]) if text.endswith("x") else 1.0
            self.sim.set_rate(mult)

    def _toggle_record(self, checked=None):
        on = bool(self._recording) if checked is None else bool(checked)
        if self._recording and not on:
            self._recorder.stop()
            self._log("Recorded %d rows -> %s" % (self._recorder.rows, self._recorder.path))
            self._recording = False
            self.btn_record.setChecked(False)
            self._ac_record.setChecked(False)
            return
        if self.model is None:
            return
        self._recording = True
        self.btn_record.setChecked(True)
        self._ac_record.setChecked(True)
        prefix = os.path.splitext(os.path.basename(self.model_path))[0]
        path = self._recorder.start(header_columns(self.model), prefix=prefix)
        self._log("Recording -> %s" % path)

    def _screenshot(self):
        if self.viewer is None:
            return
        outdir = os.path.join(self._runtime_dir, "shots")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "shot_%s.png" % time.strftime("%Y%m%d_%H%M%S"))
        try:
            self.viewer.plotter.screenshot(path)
            self._log("Snapshot -> %s" % path)
        except Exception as exc:
            self._log("Snapshot failed: %s" % exc)

    # ------------------------------------------------------------ rendering
    def _camera_action(self, name):
        if self.viewer is None:
            return
        if name == "fit":
            self.viewer.plotter.reset_camera()
        else:
            self.viewer.set_camera_preset(name)

    def _on_follow(self, on):
        if self.viewer is None:
            return
        if on:
            self.viewer.enable_follow(int(self.follow_combo.currentData()))
        else:
            self.viewer.disable_follow()

    def _on_contacts(self, on):
        if self.sim is not None:
            self.sim.set_contacts_enabled(on)
        if self.viewer is not None:
            self.viewer.toggle_contacts(on)

    # ------------------------------------------------------------ MDI actions
    def _show_scene(self):
        self._reveal_sub(self._scene_sub)

    def _show_plots(self):
        self._reveal_sub(self._plots_sub)

    def _show_console(self):
        self._reveal_sub(self._console_sub)

    def _reveal_sub(self, sub):
        sub.show()
        sub.raise_()
        self._mdi.setActiveSubWindow(sub)

    # ------------------------------------------------------------ timeline
    def _reset_timeline(self):
        self._tl_dragging = False
        if hasattr(self, "_time_label"):
            self._time_label.setText("t = 0.000 s")
        if hasattr(self, "_tl"):
            self._tl.setValue(0)
            self._tl.setRange(0, 30000)

    def _on_tl_pressed(self):
        self._tl_dragging = True

    def _on_tl_released(self):
        self._tl_dragging = False
        target = self._tl.value() / 1000.0
        if self.sim is not None:
            self.sim.pause()
            self._sync_run_button()
            self.sim.seek(target)

    # ------------------------------------------------------------ snapshot
    def _on_snapshot(self, st):
        if self.plots is not None:
            self.plots.update(st)
        if self._recording and self._recorder is not None:
            self._recorder.write(row_from_state(st))
            if self._recorder.rows % 200 == 0:
                self._recorder.flush()

    def _tick(self):
        if self.sim is None or self.viewer is None:
            return
        st = self.sim.latest_state()
        if st is None:
            return
        self.viewer.apply_state(st)
        ms = int(st["time"] * 1000.0)
        if not self._tl_dragging:
            self._tl.setMaximum(max(self._tl.maximum(), ms + 8000))
            self._tl.setValue(ms)
        self._time_label.setText("t = %8.3f s" % st["time"])
        self._fps_label.setText("sim %4.0f Hz" % st.get("fps", 0))
        self._sync_run_button()

    # ------------------------------------------------------------ dialogs
    def _open_model_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open model", self._models_dir, "Models (*.urdf *.xml);;All files (*)"
        )
        if path and self.load_model(path, wait=True) and self._last_load_ok:
            parts = self._settings.value("vendored_models", "", type=str)
            entries = [p for p in parts.split(";") if p] if parts else []
            if path not in entries:
                entries.append(path)
            self._settings.setValue("vendored_models", ";".join(entries))
            self._rescan()

    def _open_controller_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open controller", self._controllers_dir, "Python scripts (*.py);;All files (*)"
        )
        if path:
            self.load_controller(path)

    def _show_shortcuts(self):
        text = (
            "Modifier-free shortcuts are ignored while typing in an editor or combo box.\n\n"
            "  Space          Run / Pause\n"
            "  R              Reset simulation\n"
            "  C              Start / stop CSV recording\n"
            "  S              Save 3D snapshot (PNG)\n"
            "  F              Toggle follow camera\n"
            "  Ctrl+S         Save controller script + hot reload\n"
            "  Ctrl+Shift+N   Open model\n"
            "  Ctrl+Shift+O   Open controller\n"
        )
        QMessageBox.information(self, "Shortcuts", text)

    def _show_about(self):
        QMessageBox.about(
            self,
            "About Robot Studio",
            "<b>Robot Studio</b><br>MuJoCo physics + PyVista rendering "
            "workbench for testing legged-robot control scripts.<br><br>"
            "Physics: MuJoCo %s<br>3D: pyvista<br>GUI: Qt5" % getattr(self.model, "names", ""),
        )

    # ------------------------------------------------------------ helpers
    def _log(self, text):
        self._console.appendPlainText("[%s]  %s" % (time.strftime("%H:%M:%S"), text))

    def _find_combo_index(self, combo, path):
        for i in range(combo.count()):
            if combo.itemData(i) == path:
                return i
        return -1

    def _on_model_combo(self, idx):
        if idx < 0:
            return
        p = self.model_combo.itemData(idx)
        if p != self.model_path:
            self.load_model(p)

    def _on_ctrl_combo(self, idx):
        if idx < 0:
            return
        p = self.ctrl_combo.itemData(idx)
        active = self.manager.active_spec()
        if active is None or active.path != p:
            self.load_controller(p)

    def closeEvent(self, event):
        self._closing = True
        self._paint_timer.stop()
        if self._load_worker is not None:
            self._load_worker.cancel()
        if self.sim is not None:
            self.sim.stop()
        super().closeEvent(event)