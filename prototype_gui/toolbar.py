"""Top toolbar built from a Qt Designer .ui file.

Waits for nothing at import time -- the .ui is loaded once per instance so the
file can be edited in Qt Designer and picked up on the next GUI launch.
"""

import os

from PyQt5 import uic
from PyQt5.QtWidgets import QWidget

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_PATH = os.path.join(ROOT, "prototype_gui", "ui", "main_toolbar.ui")


class TopToolBar(QWidget):
    """Grouped toolbar: Model | Controller | Simulation | View.

    Child widgets (``model_combo``, ``ctrl_combo``, ``btn_run``, ``btn_reset``,
    ``speed``, ``chk_headless``) become attributes once ``uic.loadUi`` runs.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(UI_PATH, self)