"""Tests for the legacy original-rig method (analysis.legacy and
process_legacy_run)."""

import json

import numpy as np
import pytest

from shield_toolbox import process_legacy_run
from shield_toolbox.analysis import fit_tail_asymptote, tail_mean
from shield_toolbox.campaign import load_results
from test_processing import SAMPLE, _permeation_run


def test_tail_asymptote_fits_last_quarter_of_band():
    time_s = np.arange(0.0, 1000.0)
    # Flat at 0.02 Torr (below the band), then a straight rise from t = 200 s.
    pressure = np.where(time_s < 200, 0.02, 0.02 + 1e-3 * (time_s - 200))
    fit = fit_tail_asymptote(time_s, pressure)
    in_band = np.nonzero((pressure >= 0.05) & (pressure <= 0.95))[0]
    expected = in_band[int(len(in_band) * 0.75) :]
    np.testing.assert_array_equal(np.nonzero(fit.used)[0], expected)
    assert fit.slope_torr_per_s == pytest.approx(1e-3)
    # P = 0 crossing: 0.02 + 1e-3·(t − 200) = 0 → t = 180 s.
    assert fit.time_lag_s == pytest.approx(180.0)


def test_tail_mean_uses_last_three_quarters():
    assert tail_mean(np.array([100.0, 1.0, 2.0, 3.0])) == pytest.approx(2.0)


def test_process_legacy_run_results_and_storage(tmp_path):
    run = _permeation_run()
    processed = process_legacy_run(run, SAMPLE)
    fit = processed.fit
    assert processed.time_lag_s == pytest.approx(fit.time_lag_s)
    assert processed.diffusivity_m2_per_s == pytest.approx(
        SAMPLE.thickness_m**2 / (6 * fit.time_lag_s)
    )
    assert processed.upstream_pressure_torr == pytest.approx(500.0)
    assert processed.temperature_source == "thermocouple"
    assert processed.permeability.nominal_value > 0

    out_dir = processed.write(tmp_path)
    result = json.loads((out_dir / "result.json").read_text())
    assert result["provenance"]["method"] == "legacy tail asymptote"
    table = load_results(tmp_path)
    assert table.loc[0, "time_lag_s"] == pytest.approx(processed.time_lag_s)
    assert table.loc[0, "permeability"] == pytest.approx(
        processed.permeability.nominal_value
    )
