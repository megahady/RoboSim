from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from legged.evaluate import run_viewer, single
from legged.gait import LayeredController
from robot_studio.model_loader import load_model


MODEL_PATH = "robot_studio/models/pla_hexapod.urdf"
DEFAULT_STEP_LENGTH_M = 0.22
DEFAULT_STEP_HEIGHT_M = 0.07
DEFAULT_KP = 30.0
DEFAULT_KD = 4.0


def iter_stride_candidates(start_hz: float, end_hz: float, step_hz: float) -> list[float]:
    """Generate a frequency sweep between start and end using step_hz spacing."""
    start = float(start_hz)
    end = float(end_hz)
    step = abs(float(step_hz))
    if step <= 0.0:
        raise ValueError("step_hz must be positive")

    values: list[float] = []
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 4))
        current += step

    if not values:
        return [round(start, 4)]

    if not math.isclose(values[-1], end, abs_tol=1e-4):
        values.append(round(end, 4))

    deduped: list[float] = []
    for value in values:
        if not deduped or not math.isclose(deduped[-1], value, abs_tol=1e-4):
            deduped.append(value)
    return deduped


def iter_parameter_grid(
    start_hz: float,
    end_hz: float,
    step_hz: float,
    step_lengths: list[float] | tuple[float, ...] = (0.18, 0.22),
    step_heights: list[float] | tuple[float, ...] = (0.05, 0.07),
) -> list[dict]:
    """Create the full grid of gait parameters: frequency × stride length × stride height."""
    grid: list[dict] = []
    for hz in iter_stride_candidates(start_hz, end_hz, step_hz):
        for step_length in step_lengths:
            for step_height in step_heights:
                grid.append({
                    "stride_freq_hz": float(hz),
                    "step_length_m": float(step_length),
                    "step_height_m": float(step_height),
                })
    return grid


def _make_controller(
    stride_freq_hz: float,
    step_length_m: float = DEFAULT_STEP_LENGTH_M,
    step_height_m: float = DEFAULT_STEP_HEIGHT_M,
):
    return LayeredController(
        params={
            "stride_freq_hz": float(stride_freq_hz),
            "step_length_m": float(step_length_m),
            "step_height_m": float(step_height_m),
            "kp": DEFAULT_KP,
            "kd": DEFAULT_KD,
        }
    )


def _safe_score(freq_hz: float, metrics: dict) -> float:
    falls = int(metrics.get("falls", 0))
    success = bool(metrics.get("success", False))
    v_mean = float(metrics.get("v_mean", 0.0))
    max_tilt = float(metrics.get("max_tilt_deg", 0.0))
    height_std = float(metrics.get("height_std", 0.0))
    mean_energy = float(metrics.get("mean_energy", 0.0))
    ctrl_rms = float(metrics.get("ctrl_rms", 0.0))
    cot = float(metrics.get("cot", 10.0))

    if not success or falls > 0:
        return -1000.0 + 5.0 * float(freq_hz) - 100.0 * falls - 8.0 * max_tilt - 30.0 * height_std - 0.1 * mean_energy - 2.0 * ctrl_rms - 25.0 * cot

    return (
        12.0 * float(freq_hz)
        + 35.0 * v_mean
        - 0.8 * max_tilt
        - 120.0 * height_std
        - 0.05 * mean_energy
        - 1.8 * ctrl_rms
        - 22.0 * cot
    )


def evaluate_frequency(
    stride_freq_hz: float,
    duration: float = 2.0,
    seed: int = 0,
    settle: float = 0.5,
    model=None,
    step_length_m: float = DEFAULT_STEP_LENGTH_M,
    step_height_m: float = DEFAULT_STEP_HEIGHT_M,
) -> dict:
    """Evaluate one candidate gait setting and return a score plus summary metrics."""
    model = model or load_model(MODEL_PATH)
    controller = _make_controller(stride_freq_hz, step_length_m=step_length_m, step_height_m=step_height_m)
    _, metrics = single(model, controller, duration=duration, seed=seed, settle=settle)
    score = _safe_score(stride_freq_hz, metrics)
    report = {
        "stride_freq_hz": float(stride_freq_hz),
        "step_length_m": float(step_length_m),
        "step_height_m": float(step_height_m),
        "score": float(score),
        "metrics": metrics,
    }
    return report


def visualize_frequency(
    stride_freq_hz: float,
    duration: float = 2.5,
    settle: float = 0.5,
    model=None,
    until_space: bool = False,
    step_length_m: float = DEFAULT_STEP_LENGTH_M,
    step_height_m: float = DEFAULT_STEP_HEIGHT_M,
):
    """Open the native MuJoCo viewer for a single gait candidate."""
    model = model or load_model(MODEL_PATH)
    controller = _make_controller(stride_freq_hz, step_length_m=step_length_m, step_height_m=step_height_m)
    mode = "until SPACE" if until_space else f"for {duration:.2f}s"
    print(f"[viewer] stride={stride_freq_hz:.3f} Hz | step={step_length_m:.2f}m lift={step_height_m:.2f}m | {mode}")
    run_viewer(model, controller, duration=duration, settle=settle, realtime=True, until_space=until_space)


def load_best_result(path: str = "gair-score.json") -> dict | None:
    """Load the most recent saved best-gait result from disk."""
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    best = payload.get("best")
    if not isinstance(best, dict):
        return None
    return best


def save_best_result(best: dict | None, path: str = "gair-score.json") -> dict | None:
    """Persist the best optimization result to JSON in the project root."""
    if best is None:
        return None
    target = Path(path)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "best": best,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[save] best score written to {target.resolve()}")
    return payload


def optimize_stride_frequency(
    start_hz: float,
    end_hz: float,
    step_hz: float,
    duration: float = 2.0,
    settle: float = 0.5,
    seeds: Iterable[int] = (0, 1, 2),
    visualize_best: bool = True,
    viewer_duration: float = 2.5,
    results_path: str = "gair-score.json",
    step_lengths: list[float] | tuple[float, ...] = (0.18, 0.22),
    step_heights: list[float] | tuple[float, ...] = (0.05, 0.07),
):
    """Sweep the gait parameter grid headlessly and replay the best result in the viewer."""
    candidates = iter_parameter_grid(start_hz, end_hz, step_hz, step_lengths=step_lengths, step_heights=step_heights)
    model = load_model(MODEL_PATH)
    history: list[dict] = []
    best = None

    try:
        for idx, params in enumerate(candidates, start=1):
            seed_results = [
                evaluate_frequency(
                    params["stride_freq_hz"],
                    duration=duration,
                    seed=int(seed),
                    settle=settle,
                    model=model,
                    step_length_m=params["step_length_m"],
                    step_height_m=params["step_height_m"],
                )
                for seed in seeds
            ]
            mean_score = float(np.mean([r["score"] for r in seed_results]))
            v_mean = float(np.mean([r["metrics"].get("v_mean", 0.0) for r in seed_results]))
            max_tilt = float(np.max([r["metrics"].get("max_tilt_deg", 0.0) for r in seed_results]))
            safe = all(r["metrics"].get("success", False) for r in seed_results)

            result = {
                "iteration": idx,
                **params,
                "score": mean_score,
                "metrics": {
                    "v_mean": v_mean,
                    "max_tilt_deg": max_tilt,
                    "success": safe,
                    "falls": int(np.sum([r["metrics"].get("falls", 0) for r in seed_results])),
                },
                "seed_results": seed_results,
            }
            history.append(result)

            if best is None or result["score"] > best["score"]:
                best = result

            print(
                f"[opt] iter {idx}/{len(candidates)} | freq={params['stride_freq_hz']:.3f} Hz | "
                f"step={params['step_length_m']:.2f}m lift={params['step_height_m']:.2f}m | "
                f"score={mean_score:.3f} | v_mean={v_mean:.3f} | tilt={max_tilt:.2f}° | safe={safe}"
            )
    except KeyboardInterrupt:
        print("\n[stop] user interrupted optimization")
        payload = save_best_result(best, results_path)
        if visualize_best and best is not None:
            print(f"[viewer] replaying optimized parameter: {best['stride_freq_hz']:.3f} Hz")
            visualize_frequency(
                best["stride_freq_hz"],
                duration=viewer_duration,
                settle=settle,
                model=model,
                step_length_m=best["step_length_m"],
                step_height_m=best["step_height_m"],
            )
        return {"best": best, "history": history, "candidates": candidates, "saved": payload}

    save_best_result(best, results_path)
    if visualize_best and best is not None:
        print(f"[viewer] replaying optimized parameter: {best['stride_freq_hz']:.3f} Hz")
        visualize_frequency(
            best["stride_freq_hz"],
            duration=viewer_duration,
            settle=settle,
            model=model,
            step_length_m=best["step_length_m"],
            step_height_m=best["step_height_m"],
        )

    return {"best": best, "history": history, "candidates": candidates, "saved": save_best_result(best, results_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize robot stride frequency for maximum safe cadence.")
    parser.add_argument("--start", type=float, default=0.4, help="starting stride frequency in Hz")
    parser.add_argument("--end", type=float, default=1.4, help="ending stride frequency in Hz")
    parser.add_argument("--step", type=float, default=0.2, help="stride frequency sweep step in Hz")
    parser.add_argument("--duration", type=float, default=2.0, help="evaluation duration per seed")
    parser.add_argument("--settle", type=float, default=0.5, help="settle time before enabling controller")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="random seeds to test each frequency")
    parser.add_argument(
        "--viewer",
        dest="visualize_best",
        action="store_true",
        help="replay the optimized parameter in the MuJoCo viewer after the sweep",
    )
    parser.add_argument(
        "--no-viewer",
        dest="visualize_best",
        action="store_false",
        help="skip the final optimized replay viewer",
    )
    parser.add_argument(
        "--run-best",
        action="store_true",
        help="load the saved best stride frequency from gair-score.json and run only that gait",
    )
    parser.add_argument(
        "--until-space",
        action="store_true",
        help="keep the viewer running until the user presses Space instead of a fixed duration",
    )
    parser.add_argument("--viewer-duration", type=float, default=2.5, help="viewer duration for the final optimized replay")
    parser.set_defaults(visualize_best=False)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.run_best:
        best = load_best_result("gair-score.json")
        if best is None:
            raise SystemExit("No saved best result found in gair-score.json. Run the optimizer once first.")
        print(f"[load] replaying saved best gait from gair-score.json: {best['stride_freq_hz']:.3f} Hz")
        visualize_frequency(
            float(best["stride_freq_hz"]),
            duration=args.viewer_duration,
            settle=0.2,
            until_space=args.until_space,
            step_length_m=float(best.get("step_length_m", DEFAULT_STEP_LENGTH_M)),
            step_height_m=float(best.get("step_height_m", DEFAULT_STEP_HEIGHT_M)),
        )
        return

    result = optimize_stride_frequency(
        start_hz=args.start,
        end_hz=args.end,
        step_hz=args.step,
        duration=args.duration,
        settle=args.settle,
        seeds=tuple(args.seeds),
        visualize_best=args.visualize_best,
        viewer_duration=args.viewer_duration,
        results_path="gair-score.json",
        step_lengths=(0.18, 0.22),
        step_heights=(0.05, 0.07),
    )

    best = result["best"]
    if best is None:
        print("\n[opt] no valid candidate evaluated")
        return

    print("\n=== best candidate ===")
    print(f"stride_freq_hz: {best['stride_freq_hz']:.4f}")
    print(f"step_length_m: {best['step_length_m']:.4f}")
    print(f"step_height_m: {best['step_height_m']:.4f}")
    print(f"score: {best['score']:.4f}")
    print(f"v_mean: {best['metrics']['v_mean']:.4f}")
    print(f"max_tilt_deg: {best['metrics']['max_tilt_deg']:.2f}")
    print(f"success: {best['metrics']['success']}")


if __name__ == "__main__":
    main()
