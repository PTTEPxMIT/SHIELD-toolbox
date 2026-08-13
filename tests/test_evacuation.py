"""Tests for pump-down fitting and prediction."""

import numpy as np
import pytest

from shield_toolbox.analysis import EvacuationFit, fit_evacuation
from shield_toolbox.plotting import plot_evacuation

# Ground truth: 700 Torr amplitude decaying with a 5-minute time constant
# onto a 1e-5 Torr base.
TRUE = EvacuationFit(
    amplitude_torr=700.0,
    rate_per_s=1.0 / 300.0,
    time_offset_s=0.0,
    base_pressure_torr=1e-5,
)


@pytest.fixture
def trace():
    time_s = np.linspace(0.0, 3600.0, 200)
    return time_s, TRUE.pressure_at(time_s)


def test_fit_recovers_known_parameters(trace):
    time_s, pressure = trace
    fit = fit_evacuation(time_s, pressure)
    # A and C are degenerate (only A·exp(−B·C) is identifiable) — compare
    # the effective amplitude at t = 0 and the independent parameters.
    effective_amplitude = fit.amplitude_torr * np.exp(
        -fit.rate_per_s * fit.time_offset_s
    )
    assert effective_amplitude == pytest.approx(TRUE.amplitude_torr, rel=1e-3)
    assert fit.rate_per_s == pytest.approx(TRUE.rate_per_s, rel=1e-3)
    assert fit.base_pressure_torr == pytest.approx(TRUE.base_pressure_torr, rel=0.05)
    np.testing.assert_allclose(fit.pressure_at(time_s), pressure, rtol=1e-3)


def test_time_to_reach_round_trips(trace):
    time_s, pressure = trace
    fit = fit_evacuation(time_s, pressure)
    target = 1e-3
    t = fit.time_to_reach(target)
    assert fit.pressure_at(t) == pytest.approx(target, rel=1e-6)
    # Analytic check: t = ln(A/(target-D))/B.
    assert t == pytest.approx(300.0 * np.log(700.0 / (target - 1e-5)), rel=1e-2)


def test_time_to_reach_rejects_unreachable_targets():
    with pytest.raises(ValueError, match="never reached"):
        TRUE.time_to_reach(1e-6)  # below the base pressure
    with pytest.raises(ValueError, match="already below"):
        TRUE.time_to_reach(1000.0)  # above the starting pressure


def test_volume_ratio_scales_pump_down_time_linearly():
    target = 1e-3
    t_small = TRUE.time_to_reach(target)
    t_large = TRUE.for_volume_ratio(100.0).time_to_reach(target)
    assert t_large == pytest.approx(100.0 * t_small)
    with pytest.raises(ValueError, match="positive"):
        TRUE.for_volume_ratio(0.0)


def test_fit_needs_enough_samples():
    with pytest.raises(ValueError, match="at least 5"):
        fit_evacuation([0, 1, 2, 3], [4, 3, 2, 1], skip_samples=1)


def test_plot_evacuation_smoke(trace):
    time_s, pressure = trace
    fit = fit_evacuation(time_s, pressure)
    ax = plot_evacuation(time_s, pressure, fit=fit, target_torr=1e-3)
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any("fit" in label for label in labels)
    assert any("target" in label and "min" in label for label in labels)
    assert ax.get_yscale() == "log"
