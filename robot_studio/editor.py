"""Minimal Python script editor with light syntax highlighting and save+reload.

Ctrl+S writes the buffer to disk and emits ``saved`` so the main window can
trigger a live controller reload - this is the heart of the tuning loop:
edit -> save -> watch the robot react without restarting anything.
"""

from __future__ import annotations

import os
import time
import re

from PyQt5.QtCore import QRegularExpression, QRegularExpressionMatch, Qt, pyqtSignal as Signal
from PyQt5.QtGui import (
    QColor,
    QFont,
    QKeySequence,
    QSyntaxHighlighter,
    QTextCharFormat,
)
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_KEYWORDS = {
    "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "False", "finally", "for",
    "from", "global", "if", "import", "in", "is", "lambda", "None", "nonlocal",
    "not", "or", "pass", "raise", "return", "True", "try", "while", "with", "yield",
}
_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "getattr",
    "hasattr", "int", "isinstance", "len", "list", "max", "min", "print",
    "range", "set", "sorted", "sum", "str", "tuple", "zip",
}

_COM = QColor("#7a8a9a")
_STR = QColor("#a5d6a7")
_KW = QColor("#82aaff")
_BLT = QColor("#c792ea")
_NUM = QColor("#f78c6c")
_DEF = QColor("#ffcb6b")


class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self._in_triple = False
        self._triple_fmt = self._fmt(_STR)

    def _fmt(self, color, bold=False, italic=False):
        f = QTextCharFormat()
        f.setForeground(color)
        if bold:
            f.setFontWeight(QFont.Bold)
        f.setFontItalic(italic)
        return f

    def highlightBlock(self, text: str):
        state = self.previousBlockState()
        i = 0
        n = len(text)
        while i < n:
            if state == 1:
                close = text.find('"""', i)
                if close == -1:
                    self.setFormat(i, n - i, self._triple_fmt)
                    break
                end = close + 3
                self.setFormat(i, end - i, self._triple_fmt)
                state = 0
                i = end
                continue
            ch = text[i]
            if text.startswith("#", i):
                self.setFormat(i, n - i, self._fmt(_COM, italic=True))
                break
            if text.startswith('"""', i):
                self.setFormat(i, 3, self._triple_fmt)
                state = 1
                i += 3
                continue
            if ch in "\"'":
                j = self._matched_end(text, i)
                self.setFormat(i, j - i + 1, self._fmt(_STR))
                i = j + 1
                continue
            if ch.isdigit() and (i == 0 or text[i - 1].isalnum() or text[i - 1] in " _("):
                j = i
                while j < n and (text[j].isdigit() or text[j] in "."):
                    j += 1
                self.setFormat(i, j - i, self._fmt(_NUM))
                i = j
                continue
            m = re.match(r"[A-Za-z_]\w*", text[i:])
            if m:
                word = m.group(0)
                if word in _KEYWORDS:
                    self.setFormat(i, len(word), self._fmt(_KW, bold=True))
                elif word in _BUILTINS:
                    self.setFormat(i, len(word), self._fmt(_BLT))
                elif word in ("init", "step", "reference", "params", "model", "data"):
                    self.setFormat(i, len(word), self._fmt(_DEF))
                i += len(word)
                continue
            i += 1
        if state == 1:
            self.setCurrentBlockState(1)

    def _matched_end(self, text, start):
        quote = text[start]
        i = start + 1
        n = len(text)
        while i < n:
            if text[i] == "\\":
                i += 2
                continue
            if text[i] == quote:
                return i
            i += 1
        return n - 1


class EditorWidget(QWidget):
    saved = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._path = None
        self._modified = False

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        bar = QHBoxLayout()
        self.file_label = QLabel("(no file)")
        self.status_label = QLabel("")
        self.apply_btn = QPushButton("Apply  (Ctrl+S)")
        self.apply_btn.setObjectName("primary")
        self.apply_btn.clicked.connect(self.save)
        bar.addWidget(self.file_label, 1)
        bar.addWidget(self.status_label)
        bar.addWidget(self.apply_btn)
        root.addLayout(bar)

        self.editor = QPlainTextEdit(self)
        self.editor.setFont(QFont("Consolas", 10))
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.editor.setTabStopDistance(4 * self.editor.fontMetrics().horizontalAdvance(" "))
        self.highlighter = PythonHighlighter(self.editor.document())
        root.addWidget(self.editor, 1)

        self.editor.textChanged.connect(self._on_change)

    def open(self, path):
        self._path = os.path.abspath(path)
        with open(self._path, "r", encoding="utf-8") as fh:
            self.editor.setPlainText(fh.read())
        self._modified = False
        self.file_label.setText(os.path.basename(path))
        self.status_label.setText("")
        self.apply_btn.setEnabled(False)

    def save(self):
        if self._path is None:
            return
        try:
            with open(self._path, "w", encoding="utf-8") as fh:
                fh.write(self.editor.toPlainText())
        except OSError as exc:
            self.status_label.setText("save failed: %s" % exc)
            return
        self._modified = False
        self.apply_btn.setEnabled(False)
        self.status_label.setText("saved %s" % time.strftime("%H:%M:%S"))
        self.saved.emit(self._path)

    def _on_change(self):
        self._modified = True
        self.apply_btn.setEnabled(True)

    @property
    def path(self):
        return self._path

    @property
    def is_modified(self):
        return self._modified

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Save):
            self.save()
            event.accept()
            return
        super().keyPressEvent(event)