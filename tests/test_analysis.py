"""Tests for shield_toolbox.analysis.

Parity values are pinned against ``shield_das.analysis`` (the live DAS
implementation) on synthetic data. The permeability differs from the DAS
value only through the Torr→Pa factor (133.322 here vs 133.3 legacy),
bounded below 1e-3 relative.
"""

import numpy as np
import pytest
from uncertainties import ufloat

from shield_toolbox import get_rig_config
from shield_toolbox.analysis import (
    fit_arrhenius,
    fit_downstream_rise,
    permeability_takaishi_sensui,
    run_window_mask,
    stable_upstream_pressure,
)

RIG = get_rig_config("v1")


@pytest.fixture
def synthetic_run():
    """Step-and-plateau upstream, linear downstream rise (201 samples)."""
    time_s = np.linspace(0, 1000, 201)
    upstream_torr = np.where(time_s < 50, 8.0 * time_s, 400.0)
    downstream_torr = 0.02 + 0.0008 * time_s
    return time_s, upstream_torr, downstream_torr


def test_run_window_all_true_when_never_saturated():
    voltage = np.array([0.1, 3.0, 9.9, 5.0])
    assert run_window_mask(voltage).all()


def test_run_window_starts_after_last_saturated_sample():
    voltage = np.array([0.1, 10.12, 10.12, 9.9, 10.05, 9.5, 8.0])
    np.testing.assert_array_equal(
        run_window_mask(voltage),
        [False, False, False, False, False, True, True],
    )


def test_stable_upstream_pressure_parity(synthetic_run):
    time_s, upstream_torr, _ = synthetic_run
    plateau = stable_upstream_pressure(time_s, upstream_torr)
    # Pinned against shield_das.average_pressure_after_increase.
    assert plateau.average_torr == pytest.approx(400.0, rel=1e-12)
    assert plateau.start_index > 0


def test_fit_downstream_rise_parity(synthetic_run):
    time_s, _, downstream_torr = synthetic_run
    fit = fit_downstream_rise(time_s, downstream_torr)
    # Pinned against shield_das.calculate_flux_from_sample.
    assert fit.slope_torr_per_s == pytest.approx(0.0008, rel=1e-9)
    # Only samples in [0.05, 0.95] Torr enter the fit.
    assert fit.used.sum() < len(time_s)
    inside = (downstream_torr >= 0.05) & (downstream_torr <= 0.95)
    np.testing.assert_array_equal(fit.used, inside)
    # The fitted line reproduces the underlying rise.
    np.testing.assert_allclose(fit.evaluate([0.0, 1000.0]), [0.02, 0.82], atol=1e-6)


def test_fit_downstream_rise_needs_reliable_samples():
    with pytest.raises(ValueError, match="reliable"):
        fit_downstream_rise([0.0, 1.0, 2.0], [1.5, 1.6, 1.7])


def test_permeability_takaishi_sensui_parity(synthetic_run):
    time_s, upstream_torr, downstream_torr = synthetic_run
    plateau = stable_upstream_pressure(time_s, upstream_torr)
    fit = fit_downstream_rise(time_s, downstream_torr)
    perm = permeability_takaishi_sensui(
        slope_torr_per_s=fit.slope_torr_per_s,
        temperature_K=500.0,
        sample_thickness_m=0.00088,
        downstream_pressure_torr=float(downstream_torr[-1]),
        upstream_pressure_torr=plateau.average_torr,
        rig=RIG,
    )
    # Pinned against shield_das.calculate_permeability_from_flux
    # (26702655614754.457 ± 5210003390681.288); rel tol covers the
    # 133.322-vs-133.3 Torr→Pa difference (~8e-5).
    assert perm.nominal_value == pytest.approx(26702655614754.457, rel=1e-3)
    assert perm.std_dev == pytest.approx(5210003390681.288, rel=2e-2)


def test_fit_arrhenius_recovers_known_line():
    # Construct perfect Arrhenius data: log10(P) = -2.0 * (1000/T) + 3.0
    temps = np.array([400.0, 500.0, 600.0, 700.0])
    perms = 10.0 ** (-2.0 * (1000.0 / temps) + 3.0)
    fit = fit_arrhenius(temps, list(perms))
    assert fit.slope == pytest.approx(-2.0)
    assert fit.intercept == pytest.approx(3.0)
    np.testing.assert_allclose(
        fit.fit_y[[0, -1]],
        10.0 ** (-2.0 * fit.fit_x_inverse_kK[[0, -1]] + 3.0),
    )
    # Ea = -slope * 1000 * ln(10) * R
    assert fit.activation_energy_J_per_mol == pytest.approx(
        2.0 * 1000.0 * np.log(10.0) * 8.314
    )


def test_fit_arrhenius_accepts_ufloats():
    temps = [500.0, 600.0]
    perms = [ufloat(1e12, 1e11), ufloat(5e12, 5e11)]
    fit = fit_arrhenius(temps, perms)
    assert np.isfinite(fit.slope)
    assert len(fit.fit_y) == 100
