"""CLI for the legged-robot control experiments (steps 1-6).

Examples:

    # headless metric sweep on the PLA hexapod (defaults):
    .venv/bin/python -m legged.run_experiment

    # longer runs, more seeds (more statistical power):
    .venv/bin/python -m legged.run_experiment --duration 8 --seeds 0 1 2 3 4 5

    # stride-frequency grid to show gait-speed control authoring:
    .venv/bin/python -m legged.run_experiment --speed-grid

    # live 3D viewer of the same controller (macOS: use mjpython):
    .venv/bin/mjpython -m legged.run_experiment --viewer
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from robot_studio.model_loader import load_model  # noqa: E402

from legged.gait import LayeredController  # noqa: E402
from legged import evaluate  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="robot_studio/models/pla_hexapod.urdf",
                    help="URDF/MJCF of a floating-base legged robot")
    ap.add_argument("--duration", type=float, default=6.0, help="sim seconds / episode")
    ap.add_argument("--seeds", type=int, nargs="*", default=(0, 1, 2),
                    help="random seeds to average over")
    ap.add_argument("--speed-grid", action="store_true",
                    help="sweep stride frequency + step length and print table")
    ap.add_argument("--viewer", action="store_true",
                    help="open the native MuJoCo viewer instead of a batch run")
    args = ap.parse_args(argv)

    path = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    if not os.path.isfile(path):
        sys.exit("model not found: %s" % path)
    model = load_model(path)

    if args.viewer:
        evaluate.run_viewer(model, LayeredController(), duration=args.duration)
        return

    if args.speed_grid:
        speeds = {
            "slow": {"stride_freq_hz": 0.35, "step_length_m": 0.12},
            "default": {},
            "fast": {"stride_freq_hz": 0.9, "step_length_m": 0.30},
        }
    else:
        speeds = None

    print("=" * 72)
    print("LayeredController | model: %s" % os.path.basename(path))
    print("=" * 72)
    print(LayeredController().describe())

    rows, combos = evaluate.grid(model, LayeredController(),
                                 duration=args.duration, seeds=args.seeds,
                                 speed_options=speeds)
    table = "\n".join(evaluate.pretty_lines(rows))
    print(table)
    print("runs/episode: %d seeds: %s" % (len(combos), ", ".join(map(str, args.seeds))))


if __name__ == "__main__":
    main()