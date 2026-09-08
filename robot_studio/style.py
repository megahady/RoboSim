"""Dark IDE theme applied to the whole application."""

from __future__ import annotations

from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

_QSS = """
QWidget {
    background-color: #1b1d21;
    color: #d5d8dc;
    font-size: 12px;
}

QToolBar {
    background: #23262c;
    border-bottom: 1px solid #3a3f47;
    padding: 3px;
    spacing: 4px;
}

QMenuBar {
    background: #23262c;
    border-bottom: 1px solid #3a3f47;
}
QMenuBar::item { padding: 4px 8px; }
QMenuBar::item:selected { background: #333a44; }

QMenu { background: #23262c; border: 1px solid #3a3f47; }
QMenu::item { padding: 4px 18px; }
QMenu::item:selected { background: #2f6f97; }

QGroupBox {
    border: 1px solid #3a3f47;
    border-radius: 6px;
    margin-top: 12px;
    padding: 8px 6px 6px 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #9fb6d0;
    font-weight: bold;
}

QPushButton {
    background: #2b2f37;
    border: 1px solid #3a3f47;
    border-radius: 4px;
    padding: 5px 10px;
}
QPushButton:hover { background: #333a44; }
QPushButton:pressed { background: #262a31; }
QPushButton:checked { background: #2e4b5e; border-color: #4a90d9; }
QPushButton#primary { background: #2f6f97; border-color: #4a90d9; color: #ffffff; }
QPushButton#primary:hover { background: #3a84b4; }
QPushButton#danger { background: #5c2c33; border-color: #a05a63; color: #ffd9d9; }

QComboBox, QLineEdit {
    background: #262a31;
    border: 1px solid #3a3f47;
    border-radius: 4px;
    padding: 4px 6px;
}
QComboBox:hover { border-color: #4a90d9; }
QComboBox QAbstractItemView { background: #23262c; border: 1px solid #3a3f47; selection-background-color: #2f6f97; }

QPlainTextEdit {
    background: #14151a;
    border: 1px solid #33383f;
    border-radius: 4px;
    font-family: "Consolas", "Cascadia Code", monospace;
    font-size: 11px;
}

QLabel { background: transparent; }

QSlider::groove:horizontal { height: 4px; background: #33383f; border-radius: 2px; }
QSlider::handle:horizontal { background: #6aa9dc; width: 12px; margin: -5px 0; border-radius: 6px; }
QSlider::sub-page:horizontal { background: #4a90d9; border-radius: 2px; }

QStatusBar { background: #23262c; border-top: 1px solid #3a3f47; }
QStatusBar QLabel { color: #aab2bf; }

QSplitter::handle { background: #33383f; }

QDockWidget { color: #d5d8dc; }
QDockWidget::title { background: #23262c; padding: 4px 8px; border: 1px solid #3a3f47; }

QTabWidget::pane { border: 1px solid #3a3f47; }
QTabBar::tab { background: #23262c; padding: 5px 12px; border: 1px solid #3a3f47; }
QTabBar::tab:selected { background: #2f6f97; color: #ffffff; }

QToolBar QToolButton { background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 4px; }
QToolBar QToolButton:hover { background: #333a44; }
QToolBar QToolButton:checked { background: #2e4b5e; }

QScrollBar:vertical { background: #1b1d21; width: 12px; }
QScrollBar::handle:vertical { background: #3a3f47; border-radius: 6px; min-height: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
"""


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#1b1d21"))
    palette.setColor(QPalette.WindowText, QColor("#d5d8dc"))
    palette.setColor(QPalette.Base, QColor("#14151a"))
    palette.setColor(QPalette.AlternateBase, QColor("#1f2227"))
    palette.setColor(QPalette.ToolTipBase, QColor("#23262c"))
    palette.setColor(QPalette.ToolTipText, QColor("#d5d8dc"))
    palette.setColor(QPalette.Text, QColor("#d5d8dc"))
    palette.setColor(QPalette.Button, QColor("#2b2f37"))
    palette.setColor(QPalette.ButtonText, QColor("#d5d8dc"))
    palette.setColor(QPalette.Highlight, QColor("#2f6f97"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(_QSS)