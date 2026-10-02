"""Tests for shield_toolbox.analysis: run windows, Takaishi–Sensui
permeability, and Arrhenius fits.

The Takaishi–Sensui permeability deliberately departs from the legacy formula
(which dropped the hot downstream volume), so it is tested against physical
limits and an independent mole count.
"""

import dataclasses

import numpy as np
import pytest
from uncertainties import ufloat

from shield_toolbox import get_rig_config
from shield_toolbox.analysis import (
    downstream_window_mask,
    fit_arrhenius,
    permeability_takaishi_sensui,
    run_window_mask,
    takaishi_sensui_ratio,
)
from shield_toolbox.constants import N_A, TORR_TO_PA, R

RIG = get_rig_config("v1")


def test_run_window_all_true_when_never_saturated():
    voltage = np.array([0.1, 3.0, 9.9, 5.0])
    assert run_window_mask(voltage).all()


def test_run_window_starts_after_last_saturated_sample():
    voltage = np.array([0.1, 10.12, 10.12, 9.9, 10.05, 9.5, 8.0])
    np.testing.assert_array_equal(
        run_window_mask(voltage),
        [False, False, False, False, False, True, True],
    )


def test_downstream_window_all_true_when_never_saturated():
    assert downstream_window_mask(np.array([0.3, 5.0, 9.9])).all()


def test_downstream_window_ends_at_first_saturated_sample():
    # Readings after the first saturation are excluded even if they drop again.
    voltage = np.array([0.4, 6.0, 9.95, 10.12, 10.12, 9.0])
    np.testing.assert_array_equal(
        downstream_window_mask(voltage), [True, True, True, False, False, False]
    )


def test_takaishi_sensui_ratio_limits():
    # Free-molecular limit: p_hot/p_cold -> sqrt(T_hot/T_cold), no p dependence.
    ratio, p_dratio = takaishi_sensui_ratio(1e-9, 0.014, 600.0, 300.0)
    assert ratio == pytest.approx(np.sqrt(2.0), rel=1e-4)
    assert p_dratio == pytest.approx(0.0, abs=1e-4)
    # Continuum limit: equal pressures.
    ratio, p_dratio = takaishi_sensui_ratio(100.0, 0.014, 600.0, 300.0)
    assert ratio == pytest.approx(1.0, rel=1e-6)
    assert p_dratio == pytest.approx(0.0, abs=1e-6)


def test_takaishi_sensui_ratio_table_constants():
    # Hand evaluation of TS eqn (6) with the H2 constants of their Table 1:
    # X = 2·p·d/(T1+T2) = 2·0.3·14/900 Torr·mm/K.
    x = 2 * 0.3 * 14 / 900
    q = 1.24e5 * x**2 + 8.00e2 * x + 10.6 * np.sqrt(x)
    expected = (q + np.sqrt(600 / 300)) / (q + 1)
    ratio, _ = takaishi_sensui_ratio(0.3, 0.014, 600.0, 300.0)
    assert ratio == pytest.approx(expected, rel=1e-12)


def test_takaishi_sensui_ratio_derivative_matches_finite_difference():
    p, h = 0.01, 1e-7
    _, p_dratio = takaishi_sensui_ratio(p, 0.014, 567.0, 300.0)
    up, _ = takaishi_sensui_ratio(p + h, 0.014, 567.0, 300.0)
    down, _ = takaishi_sensui_ratio(p - h, 0.014, 567.0, 300.0)
    assert p_dratio == pytest.approx(p * (up - down) / (2 * h), rel=1e-5)


def _permeability(rig, p_down_torr=0.3, slope=1e-6, t_sample=567.0):
    return permeability_takaishi_sensui(
        slope_torr_per_s=slope,
        temperature_K=t_sample,
        sample_thickness_m=0.001,
        downstream_pressure_torr=p_down_torr,
        upstream_pressure_torr=500.0,
        rig=rig,
    )


def _phi_from_molar_flow(rig, molar_flow):
    flux = molar_flow / rig.sample_area_m2 * N_A
    return flux * 0.001 / (500.0 * TORR_TO_PA) ** 0.5


def test_permeability_without_hot_volume_is_ideal_gas():
    rig = dataclasses.replace(RIG, v1_v2_split_ratio=ufloat(0.0, 1e-9))
    molar = 1e-6 * TORR_TO_PA * RIG.downstream_volume_m3.nominal_value / (R * 300.0)
    assert _permeability(rig).nominal_value == pytest.approx(
        _phi_from_molar_flow(rig, molar), rel=1e-12
    )


def test_permeability_counts_hot_volume_in_continuum_limit():
    # At 100 Torr through a 14 mm tube the hot section sits at the gauge pressure,
    # so it holds V_hot·p/(R·T_sample).
    rig = dataclasses.replace(RIG, v1_v2_split_ratio=ufloat(0.5, 1e-9))
    v = RIG.downstream_volume_m3.nominal_value
    molar = 1e-6 * TORR_TO_PA * (0.5 * v / 300.0 + 0.5 * v / 567.0) / R
    assert _permeability(rig, p_down_torr=100.0).nominal_value == pytest.approx(
        _phi_from_molar_flow(rig, molar), rel=1e-6
    )


def test_permeability_matches_mole_count_derivative():
    # Independent check: differentiate n(p) = V_c·p/(R·T_c) + V_h·f(p)·p/(R·T_h).
    rig = dataclasses.replace(RIG, v1_v2_split_ratio=ufloat(0.5, 1e-9))
    v, t_c, t_h, d = (
        RIG.downstream_volume_m3.nominal_value,
        300.0,
        567.0,
        rig.sample_diameter_m,
    )

    def moles(p_torr):
        f, _ = takaishi_sensui_ratio(p_torr, d, t_h, t_c)
        p_pa = p_torr * TORR_TO_PA
        return (0.5 * v * p_pa / t_c + 0.5 * v * f * p_pa / t_h) / R

    for p in (0.001, 0.05, 0.3):
        h = p * 1e-4
        molar = 1e-6 * (moles(p + h) - moles(p - h)) / (2 * h)
        assert _permeability(rig, p_down_torr=p).nominal_value == pytest.approx(
            _phi_from_molar_flow(rig, molar), rel=1e-6
        )


def test_permeability_propagates_volume_uncertainty():
    perm = permeability_takaishi_sensui(
        slope_torr_per_s=8e-4,
        temperature_K=500.0,
        sample_thickness_m=0.00088,
        downstream_pressure_torr=0.82,
        upstream_pressure_torr=400.0,
        rig=RIG,
    )
    assert perm.nominal_value > 0
    # Volume (±12 %) and hot fraction both contribute.
    assert (
        perm.std_dev / perm.nominal_value
        > RIG.downstream_volume_m3.std_dev / RIG.downstream_volume_m3.nominal_value
    )


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
