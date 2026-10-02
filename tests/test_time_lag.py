"""Tests for shield_toolbox.analysis.time_lag — the background-subtracted
time-lag method, and τ → D → S."""

import numpy as np
import pytest
from uncertainties import ufloat

from shield_toolbox.analysis import (
    diffusivity_from_time_lag,
    find_noise_recording,
    fit_background,
    fit_steady_state,
    initial_time,
    rise_onset,
    solubility_from_permeability,
)

THICKNESS_M = 0.00088


def _fickian_rise(t_rel, tau_s, s_inf):
    """Exact permeated pressure for a step at t_rel = 0; the asymptote is
    s_inf·(t_rel − tau_s), i.e. it crosses zero at exactly τ = e²/6D.

    Q(t) ∝ Dt/e² − 1/6 − (2/π²)·Σ((−1)ⁿ/n²)·exp(−n²π²Dt/e²).
    """
    theta = np.clip(t_rel, 0.0, None) / (6.0 * tau_s)
    n = np.arange(1, 80)[:, None]
    series = ((-1.0) ** n / n**2) * np.exp(-(n**2) * np.pi**2 * theta[None, :])
    q = theta - 1.0 / 6.0 - (2.0 / np.pi**2) * series.sum(axis=0)
    q[t_rel <= 0] = 0.0
    return s_inf * 6.0 * tau_s * q


def test_initial_time_is_half_plateau_crossing():
    time_s = np.arange(0.0, 100.0)
    upstream = np.clip((time_s - 40.0) * 100.0, 0.0, 500.0)  # ramps 40–45 s
    assert initial_time(time_s, upstream) == 43.0  # first sample above 250
    with pytest.raises(ValueError, match="pressure step"):
        initial_time(time_s, np.zeros_like(time_s))


def test_rise_onset_finds_departure_from_background():
    rng = np.random.default_rng(1)
    t_rel = np.arange(0.0, 4 * 3600, 10.0)
    p = 1e-4 * t_rel + rng.normal(0, 0.01, len(t_rel))
    p[t_rel >= 5000] += 2e-4 * (t_rel[t_rel >= 5000] - 5000)
    onset_s, noise_sd = rise_onset(t_rel, p)
    # The onset is the start of the first block that rises: the one holding
    # the kink, or the next.
    assert 5000 - 600 < onset_s <= 5000 + 600
    assert onset_s % 600 == 60  # block starts are 60 s + k·600 s
    assert noise_sd == pytest.approx(0.01, rel=0.2)


def test_rise_onset_none_without_rise():
    rng = np.random.default_rng(2)
    t_rel = np.arange(0.0, 7 * 3600, 10.0)
    onset_s, _ = rise_onset(t_rel, rng.normal(0, 0.01, len(t_rel)))
    assert onset_s is None


def test_rise_onset_searches_the_whole_run():
    # Flat for 10 h, then a rise: found however late it starts.
    t_rel = np.arange(0.0, 14 * 3600, 10.0)
    p = np.where(t_rel < 36000, 0.0, 1e-3 * (t_rel - 36000))
    p = p + np.random.default_rng(5).normal(0, 0.01, len(t_rel))
    onset_s, _ = rise_onset(t_rel, p)
    assert 36000 - 600 < onset_s <= 36000 + 600


def test_noise_recording_falls_back_to_seed_stretch_without_onset():
    rng = np.random.default_rng(6)
    t_rel = np.arange(0.0, 3 * 3600, 10.0)
    noise = find_noise_recording(t_rel, rng.normal(0, 0.01, len(t_rel)))
    assert noise.onset_s is None
    assert noise.onset_from == "none detected"
    assert noise.end_s == 60.0 + 1800.0


def test_noise_recording_ends_margin_before_onset():
    rng = np.random.default_rng(3)
    t_rel = np.arange(-100.0, 6 * 3600, 10.0)
    p = 5.0 + 1e-4 * t_rel + _fickian_rise(t_rel, 3 * 3600, 1.5e-3)
    p += rng.normal(0, 0.01, len(t_rel))
    noise = find_noise_recording(t_rel, p)
    assert noise.onset_from == "10 min blocks"
    assert noise.end_s == pytest.approx(0.75 * noise.onset_s)
    np.testing.assert_array_equal(noise.used, (t_rel >= 60.0) & (t_rel <= noise.end_s))

    background = fit_background(t_rel, p, noise.used)
    assert background.slope.nominal_value == pytest.approx(1e-4, rel=0.1)
    assert background.level.nominal_value == pytest.approx(5.0, abs=0.02)
    assert background.evaluate(0.0) == pytest.approx(background.level.nominal_value)


def test_noise_recording_switches_to_one_minute_blocks_for_fast_rise():
    rng = np.random.default_rng(4)
    t_rel = np.arange(-100.0, 3 * 3600, 2.0)
    p = 5.0 + _fickian_rise(t_rel, 900, 1e-2) + rng.normal(0, 0.01, len(t_rel))
    noise = find_noise_recording(t_rel, p)
    assert noise.onset_from == "1 min blocks"
    assert noise.end_s == pytest.approx(0.75 * noise.onset_s)


def test_steady_state_fit_recovers_time_lag_from_fickian_transient():
    tau_true = THICKNESS_M**2 / (6 * 1e-10)  # ≈ 1290 s
    t_rel = np.arange(0.0, 12 * tau_true, 5.0)
    filtered = _fickian_rise(t_rel, tau_true, 2e-2)
    fit = fit_steady_state(t_rel, filtered, np.ones_like(t_rel, dtype=bool))
    assert fit.converged
    # The window starts at 3 τ_L (iterated) and runs to the end.
    assert t_rel[fit.used][0] == pytest.approx(3 * fit.time_lag_s, abs=5.0)
    assert t_rel[fit.used][-1] == t_rel[-1]
    assert fit.time_lag_s == pytest.approx(tau_true, rel=0.01)
    assert fit.slope == pytest.approx(2e-2, rel=0.005)
    assert diffusivity_from_time_lag(fit.time_lag_s, THICKNESS_M) == pytest.approx(
        1e-10, rel=0.01
    )
    np.testing.assert_allclose(fit.evaluate([fit.time_lag_s]), [0.0], atol=1e-12)


def test_steady_state_fit_respects_window_end_and_usable_mask():
    tau = 1000.0
    t_rel = np.arange(0.0, 15000.0, 5.0)
    filtered = _fickian_rise(t_rel, tau, 1e-2)
    usable = t_rel < 12000
    fit = fit_steady_state(t_rel, filtered, usable, start_taus=2, end_s=10000)
    assert t_rel[fit.used][-1] <= 10000
    assert t_rel[fit.used][0] == pytest.approx(2 * fit.time_lag_s, abs=5.0)


def test_steady_state_fit_keeps_last_window_when_next_is_past_the_data():
    t_rel = np.arange(0.0, 3000.0, 5.0)
    filtered = _fickian_rise(t_rel, 2000.0, 1e-2)  # 3 τ_L is past the end
    fit = fit_steady_state(t_rel, filtered, np.ones_like(t_rel, dtype=bool))
    assert not fit.converged
    assert fit.used.sum() >= 4
    assert np.isfinite(fit.slope) and np.isfinite(fit.time_lag_s)


def test_steady_state_fit_with_fixed_start():
    tau = 1000.0
    t_rel = np.arange(0.0, 15000.0, 5.0)
    filtered = _fickian_rise(t_rel, tau, 1e-2)
    fit = fit_steady_state(
        t_rel, filtered, np.ones_like(t_rel, dtype=bool), start_s=6000
    )
    assert t_rel[fit.used][0] == 6000
    assert fit.time_lag_s == pytest.approx(tau, rel=0.01)


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
