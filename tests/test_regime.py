"""Tests for shield_toolbox.analysis.regime — pressure exponent, K_d, K_r."""

import numpy as np
import pytest
from uncertainties import ufloat

from shield_toolbox.analysis import (
    classify_exponent,
    dissociation_coeff,
    fit_pressure_exponent,
    fit_regime_transition,
    recombination_coeff,
)


def test_fit_pressure_exponent_recovers_power_law():
    pressure = np.logspace(0.5, 3, 8)
    flux = 4.2e15 * pressure**0.62
    exponent = fit_pressure_exponent(pressure, flux)
    assert exponent.nominal_value == pytest.approx(0.62)
    assert exponent.std_dev == pytest.approx(0.0, abs=1e-6)


def test_fit_pressure_exponent_error_reflects_scatter():
    pressure = np.array([10.0, 30.0, 100.0, 300.0])
    flux = 1e15 * pressure**0.5 * np.array([1.1, 0.9, 1.1, 0.9])
    exponent = fit_pressure_exponent(pressure, flux)
    assert exponent.nominal_value == pytest.approx(0.5, abs=0.1)
    assert exponent.std_dev > 0.01


def test_fit_pressure_exponent_input_guards():
    with pytest.raises(ValueError, match="three"):
        fit_pressure_exponent([1.0, 10.0], [1.0, 3.0])
    with pytest.raises(ValueError, match="positive"):
        fit_pressure_exponent([1.0, 10.0, 100.0], [1.0, -3.0, 10.0])


def test_dissociation_coeff_is_2j_over_p():
    assert dissociation_coeff(5e17, 1000.0) == pytest.approx(1e15)
    with pytest.raises(ValueError, match="positive"):
        dissociation_coeff(5e17, 0.0)


def test_dissociation_coeff_propagates_uncertainty():
    k_d = dissociation_coeff(ufloat(5e17, 1e17), 1000.0)
    assert k_d.nominal_value == pytest.approx(1e15)
    assert k_d.std_dev == pytest.approx(2e14)


def test_recombination_coeff_detailed_balance():
    # K_r = K_d/K_s²; K_s enters squared so its 10 % error dominates (~20 %).
    k_r = recombination_coeff(1e15, ufloat(1e23, 1e22))
    assert k_r.nominal_value == pytest.approx(1e-31)
    assert k_r.std_dev / k_r.nominal_value == pytest.approx(0.2, rel=0.05)
    with pytest.raises(ValueError, match="positive"):
        recombination_coeff(1e15, 0.0)


def test_classify_exponent_margins():
    assert classify_exponent(0.5) == "diffusion"
    assert classify_exponent(0.63) == "diffusion"  # common DL convention
    assert classify_exponent(0.7) == "mixed"
    assert classify_exponent(0.87) == "surface"
    assert classify_exponent(1.05) == "surface"


def _broken_power_law(crossover=50.0, n_low=1.0, n_high=0.5, scale=1e16):
    """Continuous two-regime sweep: J ∝ P below the crossover, ∝ √P above."""
    pressure = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0])
    low_scale = scale
    high_scale = scale * crossover ** (n_low - n_high)
    flux = np.where(
        pressure < crossover,
        low_scale * pressure**n_low,
        high_scale * pressure**n_high,
    )
    return pressure, flux


def test_fit_regime_transition_finds_the_crossover():
    pressure, flux = _broken_power_law(crossover=50.0)
    fit = fit_regime_transition(pressure, flux)

    assert fit.transition == pytest.approx(50.0, rel=0.05)
    low, high = fit.fits
    assert low.exponent.nominal_value == pytest.approx(1.0)
    assert low.regime == "surface"
    assert high.exponent.nominal_value == pytest.approx(0.5)
    assert high.regime == "diffusion"
    # prefactor + exponent reproduce the data, for plotting
    assert low.evaluate(10.0) == pytest.approx(1e16 * 10.0, rel=1e-6)


def test_fit_regime_transition_single_regime_stays_single():
    pressure = np.logspace(0.5, 3, 8)
    rng_scatter = np.array([1.02, 0.98, 1.01, 0.99, 1.02, 0.98, 1.01, 0.99])
    flux = 2e15 * pressure**0.55 * rng_scatter
    fit = fit_regime_transition(pressure, flux)

    assert fit.transition is None
    (single,) = fit.fits
    assert single.exponent.nominal_value == pytest.approx(0.55, abs=0.05)
    assert single.regime == "diffusion"


def test_fit_regime_transition_too_few_points_never_splits():
    pressure, flux = _broken_power_law()
    fit = fit_regime_transition(pressure[:5], flux[:5])
    assert fit.transition is None
    assert len(fit.fits) == 1


def test_fit_regime_transition_input_order_does_not_matter():
    pressure, flux = _broken_power_law(crossover=50.0)
    shuffled = [3, 0, 7, 2, 5, 1, 6, 4]
    fit = fit_regime_transition(pressure[shuffled], flux[shuffled])
    assert fit.transition == pytest.approx(50.0, rel=0.05)


def test_surface_limited_chain_recovers_k_d():
    """End-to-end physics check: in the surface-limited limit the steady
    flux is J = K_d·P/2, so a pressure sweep must fit exponent 1 and the
    extraction must return K_d at every pressure."""
    k_d_true = 3e14  # H/(m²·s·Pa)
    pressure_pa = np.array([1e3, 3e3, 1e4, 3e4])
    flux = k_d_true * pressure_pa / 2.0

    exponent = fit_pressure_exponent(pressure_pa, flux)
    assert exponent.nominal_value == pytest.approx(1.0)
    for p, j in zip(pressure_pa, flux):
        assert dissociation_coeff(j, p) == pytest.approx(k_d_true)
