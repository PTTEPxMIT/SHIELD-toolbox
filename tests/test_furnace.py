"""Tests for the furnace-log loader, offset analysis, and plot."""

import numpy as np
import pytest

from shield_toolbox.analysis import furnace_temperature_offset
from shield_toolbox.io import load_furnace_log
from shield_toolbox.plotting import plot_furnace_log

_HEADER = (
    "Date,Time,Main_Controller_PV,QF,Main_Controller_Working_SP,QF,"
    "Running_Program_Data.Event1,QF,Running_Program_Data.Event2,QF,"
    "Running_Program_Data.ProgramNumber,QF,Running_Program_Data.SegmentNumber,QF"
)
# Modeled on a real Eurotherm export: integrity hash, header, MM/DD/YYYY
# dates, one QF quality-flag column per channel; rows cross midnight.
_LOG = "\n".join(
    [
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        _HEADER,
        "10/29/2025,23:59:30,18.2,0,0,0,0,0,0,0,3,0,1,0",
        "10/30/2025,00:00:30,250.1,0,300,0,0,0,0,0,3,0,1,0",
        "10/30/2025,00:01:30,499.5,0,500,0,0,0,0,0,3,0,2,0",
        "10/30/2025,00:02:30,500.2,0,500,0,0,0,0,0,3,0,2,0",
    ]
)


@pytest.fixture
def furnace_log(tmp_path):
    path = tmp_path / "TCCOMP410292025182243.csv"
    path.write_text(_LOG)
    return load_furnace_log(path)


def test_load_furnace_log_columns_and_time_axis(furnace_log):
    assert list(furnace_log.columns) == [
        "timestamp",
        "time_s",
        "furnace_temperature_C",
        "setpoint_C",
    ]
    # 60 s per row, continuous across the midnight rollover.
    np.testing.assert_allclose(furnace_log["time_s"], [0.0, 60.0, 120.0, 180.0])
    np.testing.assert_allclose(
        furnace_log["furnace_temperature_C"], [18.2, 250.1, 499.5, 500.2]
    )
    np.testing.assert_allclose(furnace_log["setpoint_C"], [0.0, 300.0, 500.0, 500.0])


def test_load_furnace_log_without_hash_line(tmp_path):
    path = tmp_path / "no_hash.csv"
    path.write_text(_LOG.split("\n", 1)[1])  # header first, no hash
    assert len(load_furnace_log(path)) == 4


def test_load_furnace_log_rejects_other_csv(tmp_path):
    path = tmp_path / "other.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Eurotherm"):
        load_furnace_log(path)


def test_furnace_temperature_offset_sign_and_window():
    # Transient first quarter (ignored), then settled at 482; furnace at 500.
    sample = np.concatenate([np.linspace(20, 482, 25), np.full(75, 482.0)])
    offset = furnace_temperature_offset(sample, furnace_final_temperature_c=500.0)
    assert offset == pytest.approx(-18.0, abs=0.1)

    with pytest.raises(ValueError, match="settled_fraction"):
        furnace_temperature_offset(sample, 500.0, settled_fraction=0.0)
    with pytest.raises(ValueError, match="empty"):
        furnace_temperature_offset([], 500.0)


def test_plot_furnace_log_with_sample_overlay(furnace_log):
    sample_t = np.linspace(0.0, 180.0, 50)
    sample_c = np.full(50, 482.0)
    ax = plot_furnace_log(
        furnace_log, sample_time_s=sample_t, sample_temperature_c=sample_c
    )
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert {"furnace", "setpoint", "sample TC"} <= set(labels)
    # ΔT annotation present (482 settled vs 500.2 final furnace PV).
    annotations = [child.get_text() for child in ax.texts]
    assert any("\\Delta T" in text and "-18" in text for text in annotations)
