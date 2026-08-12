"""Tests for shield_toolbox.analysis.time_lag — τ, D, and S extraction."""

import numpy as np
import pytest
from uncertainties import ufloat

from shield_toolbox.analysis import (
    DownstreamFit,
    diffusivity_from_time_lag,
    downstream_baseline_torr,
    fit_downstream_rise,
    solubility_from_permeability,
    time_lag_from_fit,
)

THICKNESS_M = 0.00088


def _fit(slope: float, intercept: float) -> DownstreamFit:
    return DownstreamFit(
        slope_torr_per_s=slope, intercept_torr=intercept, used=np.array([True])
    )


def test_time_lag_from_fit_geometry():
    # Line P = 2e-4·t − 0.1 crosses baseline 0.02 Torr at t = 600 s;
    # permeation started at 100 s → τ = 500 s.
    tau = time_lag_from_fit(
        _fit(2e-4, -0.1), baseline_torr=0.02, permeation_start_s=100.0
    )
    assert tau == pytest.approx(500.0)


def test_time_lag_requires_rising_fit():
    with pytest.raises(ValueError, match="rising"):
        time_lag_from_fit(_fit(-1e-4, 0.5), baseline_torr=0.0, permeation_start_s=0.0)


def test_diffusivity_from_time_lag():
    # D = e²/(6τ) with e = 0.88 mm, τ = 1290.7 s → 1e-10 m²/s.
    tau = THICKNESS_M**2 / (6 * 1e-10)
    assert diffusivity_from_time_lag(tau, THICKNESS_M) == pytest.approx(1e-10)
    with pytest.raises(ValueError, match="positive"):
        diffusivity_from_time_lag(-5.0, THICKNESS_M)


def test_solubility_propagates_uncertainty():
    perm = ufloat(2.0e12, 4.0e11)
    sol = solubility_from_permeability(perm, 1e-10)
    assert sol.nominal_value == pytest.approx(2.0e22)
    assert sol.std_dev == pytest.approx(4.0e21)
    with pytest.raises(ValueError, match="positive"):
        solubility_from_permeability(perm, 0.0)


def test_downstream_baseline_is_median_of_window():
    time_s = np.arange(0.0, 100.0, 1.0)
    pressure = np.full_like(time_s, 0.02)
    pressure[15] = 5.0  # single spike survives a median
    baseline = downstream_baseline_torr(time_s, pressure, permeation_start_s=10.0)
    assert baseline == pytest.approx(0.02)


def test_downstream_baseline_interpolates_outside_samples():
    time_s = np.array([0.0, 10.0])
    pressure = np.array([0.0, 1.0])
    baseline = downstream_baseline_torr(
        time_s, pressure, permeation_start_s=200.0, window_s=1.0
    )
    assert baseline == pytest.approx(1.0)


def test_time_lag_recovers_diffusivity_from_fickian_transient():
    """End-to-end physics check: generate the exact Fickian downstream
    transient for a known D and recover it via the extraction chain
    (fit_downstream_rise → time_lag_from_fit → diffusivity_from_time_lag).

    The permeated amount for a membrane with constant upstream concentration
    is Q(t) ∝ Dt/e² − 1/6 − (2/π²)·Σ((−1)ⁿ/n²)·exp(−n²π²Dt/e²); its
    asymptote crosses zero at exactly τ = e²/6D.
    """
    d_true = 1e-10
    tau_true = THICKNESS_M**2 / (6 * d_true)  # ≈ 1290 s
    start_s = 50.0

    time_s = np.linspace(0.0, 12 * tau_true, 4000)
    theta = np.clip(time_s - start_s, 0.0, None) * d_true / THICKNESS_M**2
    n = np.arange(1, 60)[:, None]
    series = ((-1.0) ** n / n**2) * np.exp(-(n**2) * np.pi**2 * theta[None, :])
    q = theta - 1.0 / 6.0 - (2.0 / np.pi**2) * series.sum(axis=0)
    q[time_s < start_s] = 0.0
    baseline = 0.02
    scale = 0.5 / q[-1]  # top out around 0.52 Torr, inside the gauge range
    pressure_torr = baseline + scale * q

    # Fit only the late-time linear regime, as the pipeline does.
    late = time_s > start_s + 4 * tau_true
    fit = fit_downstream_rise(time_s[late], pressure_torr[late], max_reliable_torr=1.0)
    tau = time_lag_from_fit(fit, baseline, permeation_start_s=start_s)
    assert tau == pytest.approx(tau_true, rel=0.05)
    d_extracted = diffusivity_from_time_lag(tau, THICKNESS_M)
    assert d_extracted == pytest.approx(d_true, rel=0.05)
