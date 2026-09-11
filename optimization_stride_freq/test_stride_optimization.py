import math

from optimization_stride_freq.optimizer import (
    _safe_score,
    evaluate_frequency,
    iter_parameter_grid,
    iter_stride_candidates,
)


def test_iter_stride_candidates():
    vals = list(iter_stride_candidates(0.4, 1.4, 0.2))
    assert vals == [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]


def test_evaluate_frequency_returns_score_and_metrics():
    result = evaluate_frequency(0.6, duration=0.2, seed=0)
    assert "stride_freq_hz" in result
    assert "score" in result
    assert "metrics" in result
    assert math.isfinite(result["score"]) or result["score"] == float("-inf")


def test_iter_parameter_grid_contains_multiple_params():
    grid = iter_parameter_grid(
        start_hz=0.4,
        end_hz=0.8,
        step_hz=0.2,
        step_lengths=[0.18, 0.22],
        step_heights=[0.05, 0.07],
    )
    assert len(grid) == 12
    assert all("stride_freq_hz" in entry and "step_length_m" in entry and "step_height_m" in entry for entry in grid)


def test_safe_score_prefers_efficient_stable_candidates():
    speedy = {
        "success": True,
        "falls": 0,
        "v_mean": 0.9,
        "max_tilt_deg": 12.0,
        "height_std": 0.02,
        "mean_energy": 250.0,
        "ctrl_rms": 18.0,
        "cot": 1.3,
    }
    efficient = {
        "success": True,
        "falls": 0,
        "v_mean": 0.75,
        "max_tilt_deg": 8.0,
        "height_std": 0.015,
        "mean_energy": 30.0,
        "ctrl_rms": 7.0,
        "cot": 0.45,
    }

    assert _safe_score(1.0, efficient) > _safe_score(1.0, speedy)
