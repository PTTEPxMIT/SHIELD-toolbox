"""Tests for shield_toolbox.analysis.leak — background leak-rate fitting."""

import numpy as np
import pytest

from shield_toolbox.analysis import (
    LeakRateFit,
    fit_leak_rate,
    leak_molar_rate_mol_per_s,
)
from shield_toolbox.config import get_rig_config
from shield_toolbox.constants import TORR_TO_PA, R

RATE_TRUE = 2.5e-6  # Torr/s
INTERCEPT_TRUE = 0.1  # Torr


def _trace(duration_s: float = 600.0, n: int = 601):
    time_s = np.linspace(0.0, duration_s, n)
    return time_s, INTERCEPT_TRUE + RATE_TRUE * time_s


def test_fit_recovers_rate_and_intercept():
    time_s, pressure = _trace()
    fit = fit_leak_rate(time_s, pressure)
    assert fit.rate_torr_per_s == pytest.approx(RATE_TRUE)
    assert fit.intercept_torr == pytest.approx(INTERCEPT_TRUE)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.used.all()
    assert fit.mean_pressure_torr == pytest.approx(pressure.mean())


def test_fit_is_robust_to_noise():
    rng = np.random.default_rng(42)
    time_s, pressure = _trace()
    noisy = pressure + rng.normal(0.0, 5e-5, len(pressure))
    fit = fit_leak_rate(time_s, noisy)
    assert fit.rate_torr_per_s == pytest.approx(RATE_TRUE, rel=0.05)


def test_start_s_excludes_settling_transient():
    time_s, pressure = _trace()
    # Corrupt the first 60 s with a settling transient.
    pressure = pressure + np.where(time_s < 60.0, 0.05 * (60.0 - time_s), 0.0)
    fit = fit_leak_rate(time_s, pressure, start_s=60.0)
    assert fit.rate_torr_per_s == pytest.approx(RATE_TRUE)
    np.testing.assert_array_equal(fit.used, time_s >= 60.0)


def test_pressure_band_limits():
    time_s, pressure = _trace()
    fit = fit_leak_rate(time_s, pressure, min_torr=0.1005, max_torr=0.101)
    assert fit.used.sum() < len(time_s)
    assert fit.rate_torr_per_s == pytest.approx(RATE_TRUE)


def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="cannot fit"):
        fit_leak_rate([0.0, 1.0, 2.0], [0.1, 0.1, 0.1], start_s=1.5)


def test_flat_noise_has_low_r_squared():
    rng = np.random.default_rng(0)
    time_s = np.linspace(0.0, 600.0, 601)
    flat = 0.1 + rng.normal(0.0, 1e-4, len(time_s))
    fit = fit_leak_rate(time_s, flat)
    assert abs(fit.rate_torr_per_s) < 5e-8
    assert fit.r_squared < 0.5


def test_evaluate_is_the_fitted_line():
    fit = LeakRateFit(
        rate_torr_per_s=2e-6,
        intercept_torr=0.1,
        mean_pressure_torr=0.1,
        r_squared=1.0,
        used=np.ones(3, dtype=bool),
    )
    np.testing.assert_allclose(fit.evaluate([0.0, 100.0]), [0.1, 0.1002])


def test_molar_rate_matches_ideal_gas_law():
    rig = get_rig_config("v1")
    rate = leak_molar_rate_mol_per_s(RATE_TRUE, rig)
    expected = (
        RATE_TRUE
        * TORR_TO_PA
        * rig.downstream_volume_m3.nominal_value
        / (R * rig.ambient_temperature_K)
    )
    assert rate.nominal_value == pytest.approx(expected)
    # Volume uncertainty propagates through.
    assert rate.std_dev > 0
