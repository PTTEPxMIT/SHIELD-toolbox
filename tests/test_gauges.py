"""Calibration parity tests.

Expected values are pinned against the legacy implementation in
ShieldRunsAnalysis/analysis_functions.py so the migration cannot silently
change any conversion.
"""

import numpy as np
import pytest

from shield_toolbox.gauges import CVM211, WGM701, Baratron626D, TypeKThermocouple


def test_baratron_upstream_1000_torr_full_scale():
    gauge = Baratron626D(full_scale_torr=1000.0)
    # Legacy: voltage_to_torr_baratron_upstream(V) = V * 100
    assert gauge.to_torr(5.0) == pytest.approx(500.0)
    np.testing.assert_allclose(gauge.to_torr([0.0, 2.5, 10.0]), [0.0, 250.0, 1000.0])


def test_baratron_downstream_1_torr_full_scale():
    gauge = Baratron626D(full_scale_torr=1.0)
    # Legacy: voltage_to_torr_baratron_downstream(V) = V * 100 / 1000
    assert gauge.to_torr(5.0) == pytest.approx(0.5)


def test_wgm701_parity_with_legacy():
    gauge = WGM701()
    # Values computed with legacy voltage_to_torr_wasp_downstream:
    # 4.0 V -> low-range x2.4 branch, 5.5 V and 6.2 V -> polynomial branch.
    np.testing.assert_allclose(
        gauge.to_torr([4.0, 5.5, 6.2]),
        [2.4e-3, 1.39229628, 3.19536021e3],
        rtol=1e-8,
    )


def test_wgm701_branch_split_at_0p1_torr_indicated():
    gauge = WGM701()
    # Just below 5.0 V the indicated pressure is < 0.1 Torr (x2.4 branch);
    # just above it switches to the polynomial branch.
    below = float(gauge.to_torr(4.99))
    indicated_below = 10 ** ((4.99 - 5.5) / 0.5)
    assert below == pytest.approx(indicated_below * 2.4)


def test_cvm211_not_implemented():
    with pytest.raises(NotImplementedError):
        CVM211().to_torr(3.0)


def test_typek_parity_with_legacy():
    tc = TypeKThermocouple()
    # Values computed with legacy mv_to_temp_c, one per ITS-90 sub-range.
    assert tc.to_celsius(-1.0) == pytest.approx(-25.85739888077)
    assert tc.to_celsius(10.0) == pytest.approx(246.22195599999966)
    assert tc.to_celsius(30.0) == pytest.approx(720.8178399999997)


def test_typek_vector_and_kelvin():
    tc = TypeKThermocouple()
    result = tc.to_celsius(np.array([10.0, 30.0]))
    np.testing.assert_allclose(result, [246.22195599999966, 720.8178399999997])
    assert tc.to_kelvin(10.0) == pytest.approx(246.22195599999966 + 273.15)


def test_typek_out_of_range():
    tc = TypeKThermocouple()
    with pytest.raises(ValueError, match="valid Type K range"):
        tc.to_celsius(60.0)
    with pytest.raises(ValueError, match="valid Type K range"):
        tc.to_celsius(-6.0)
