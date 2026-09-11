"""Step 6 — run the systematic experiment grid and print the efficacy card.

Two entry points:

    evaluate.single(model, controller, duration, seed, record=True, render=False)
        runs one episode; returns (recording, metrics) or (info, state) when
        ``record`` is False (viewer / interactive mode).

    evaluate.grid(model, controller, ...) 
        sweeps seeds (and optionally stride params), aggregates success rate
        and mean metric values, and prints a comparison table — the
        "reproducible number" that stands in for "it works".

Also contains the live-viewer driver (step 6) that reuses the exact same
``controller.step(model, data, obs)`` call as the batch path, so the visual
demo is faithful to the measured one.
"""

from __future__ import annotations

import copy

import numpy as np

import mujoco

from legged.env import LeggedEnv
from legged.metrics import Metrics


def assert_model(model) -> LeggedEnv:
    try:
        return LeggedEnv(model)
    except ValueError as exc:
        raise SystemExit(str(exc))


def single(model, controller, duration=6.0, seed=0, record=True):
    """One episode. Returns (recording, metrics) or (env, last_obs) in viewer mode."""
    env = assert_model(model)

    # A stateless controller takes params; if it's the layered one expose it.
    if hasattr(controller, "reset"):
        controller.reset(model, env.data, env.observe())
    obs = env.reset(seed=seed)

    rec: list[dict] = []
    n_steps = int(duration / env.CONTROL_PERIOD)
    fell = False
    for _ in range(n_steps):
        obs = env.step(controller)
        # Append AFTER stepping so the very frame that violated stability is
        # recorded too (otherwise a fall inside the final period is missed and
        # the 'falls' metric silently reads zero).
        snap = dict(obs)
        snap["ctrl"] = np.array(env.data.ctrl)
        snap["ek"] = float(env.data.energy[0])
        snap["ep"] = float(env.data.energy[1])
        rec.append(snap)
        if env.done(obs):
            fell = True
            break

    if record:
        metrics = Metrics(rec, dt=env.CONTROL_PERIOD).summary(sum(model.body_mass))
        metrics["fell"] = fell
        return rec, metrics
    return env, obs


def grid(model, controller, duration=6.0, seeds=(0, 1, 2, 3),
         speed_options=None, params=None):
    """Batch sweep: seeds x optional stride settings -> success + metric table.

    Returns (rows, stats) where rows is a list of (label, metrics-dict).
    """
    base = dict(getattr(controller, "params", {}) or {})
    if params:
        base.update(params)

    combos = []
    for label, extra in (speed_options or {}).items():
        use = dict(base); use.update(extra or {})
        combos.append((label, use))
    if not combos:
        combos = [("default", dict(base))]

    rows = []
    for label, use in combos:
        # per-label controller clone keeps sweep states independent
        ctl = copy.deepcopy(controller)
        if hasattr(ctl, "params"):
            ctl.params.update(use)
            ctl.gait.params.update(use)
            ctl.pd.kp = use.get("kp", ctl.pd.kp)
            ctl.pd.kd = use.get("kd", ctl.pd.kd)

        recs = []
        for seed in seeds:
            rec, _ = single(model, ctl, duration=duration, seed=seed)
            recs.append(rec)
        merged = [s for r in recs for s in r]
        m = Metrics(merged, dt=LeggedEnv.CONTROL_PERIOD)
        stats = m.summary(sum(model.body_mass))
        stats["label"] = label
        stats["falls"] = sum(1 for r in recs if Metrics(r).fallen)
        stats["runs"] = len(seeds)
        rows.append(stats)
    return rows, combos


def pretty_lines(rows) -> list[str]:
    """Format a metric table: columns are the sweep labels, rows the metrics."""
    width = max(len(r["label"]) for r in rows) + 2
    gutter = 14
    lines = [""]
    hdr = ("%-*s" % (width, "metric"))
    for r in rows:
        hdr += ("%*s" % (gutter, r["label"]))
    lines.append(hdr)
    lines.append("-" * len(hdr))

    def row(name, fmt):
        line = ("%-*s" % (width, name))
        for r in rows:
            v = r.get(name, float("nan"))
            line += ("%*s" % (gutter, ("%.3f" % v) if fmt == "f" else str(v)))
        lines.append(line)

    row("falls", "x")
    row("v_mean", "f")
    row("v_rms_err", "f")
    row("max_tilt_deg", "f")
    row("height_std", "f")
    row("mean_energy", "f")
    row("ctrl_rms", "f")
    row("cot", "f")
    lines.append("")
    return lines


def run_viewer(model, controller, duration=10.0):
    """Step 6 — live native viewer of the exact controller under test."""
    env = assert_model(model)
    obs = env.reset()
    if hasattr(controller, "reset"):
        controller.reset(model, env.data, obs)

    def tick():
        tau = controller.step(model, env.data, env.observe())
        env.data.ctrl[:] = tau
        return tau

    from mujoco.viewer import launch_passive
    with launch_passive(model, env.data) as v:
        print("viewer open - ESC to close")
        while v.is_running() and env.t < duration:
            tick()
            for _ in range(env.n_sub):
                mujoco.mj_step1(model, env.data)
                mujoco.mj_step2(model, env.data)
            v.sync()
    print("viewer closed after %.2f s" % env.t)