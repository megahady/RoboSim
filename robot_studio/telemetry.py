"""CSV recording of per-step simulation snapshots."""

from __future__ import annotations

import csv
import os
import time


class CsvRecorder:
    def __init__(self, outdir: str):
        self.outdir = outdir
        self.path = None
        self._fh = None
        self._writer = None
        self._rows = 0

    def start(self, columns, prefix="run"):
        os.makedirs(self.outdir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self.outdir, "%s_%s.csv" % (prefix, stamp))
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(columns)
        self._columns = columns
        self._rows = 0
        return self.path

    def write(self, row):
        if self._writer is not None:
            self._writer.writerow(row)
            self._rows += 1

    def flush(self):
        if self._fh is not None:
            self._fh.flush()

    @property
    def rows(self):
        return self._rows

    def stop(self):
        if self._fh is not None:
            self.flush()
            self._fh.close()
            self._fh = None
            self._writer = None


def header_columns(model) -> list[str]:
    cols = ["time"]
    cols += ["q%d" % i for i in range(model.nq)]
    cols += ["dq%d" % i for i in range(model.nv)]
    if model.nu:
        cols += ["tau%d" % i for i in range(model.nu)]
    else:
        cols += ["f%d" % i for i in range(model.nv)]
    cols += ["KE", "PE"]
    return cols


def row_from_state(st) -> list:
    row = [st["time"]]
    row += list(st["qpos"])
    row += list(st["qvel"])
    if st["ctrl"].size:
        row += list(st["ctrl"])
    else:
        row += list(st["qfrc_applied"])
    row += [st["ek"], st["ep"]]
    return ["%.6g" % v for v in row]