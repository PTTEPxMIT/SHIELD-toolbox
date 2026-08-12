import dataclasses
import math
from datetime import date

import pytest

from shield_toolbox.config import (
    SHIELD_V1,
    SHIELD_V2,
    get_rig_config,
    get_rig_config_for_date,
    list_rig_versions,
)


def test_configs_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        SHIELD_V1.sample_diameter_m = 0.02


def test_v1_sample_area():
    assert SHIELD_V1.sample_area_m2 == pytest.approx(0.25 * math.pi * 0.0155**2)


def test_v1_values_match_legacy_notebooks():
    assert SHIELD_V1.downstream_volume_m3.nominal_value == pytest.approx(7.9e-5)
    assert SHIELD_V1.downstream_volume_m3.std_dev == pytest.approx(9.8e-6)
    assert SHIELD_V1.v1_v2_split_ratio.nominal_value == pytest.approx(0.35)
    assert SHIELD_V1.ambient_temperature_K == 300.0
    assert SHIELD_V1.furnace_setpoint_offset_K == -18.0


def test_get_rig_config():
    assert get_rig_config() is SHIELD_V1
    assert get_rig_config("v2") is SHIELD_V2
    assert list_rig_versions() == ("v1", "v2")
    with pytest.raises(KeyError, match="Unknown rig version"):
        get_rig_config("v99")


def test_get_rig_config_for_date_selects_v1():
    assert get_rig_config_for_date(date(2026, 1, 1)) is SHIELD_V1


def test_drafts_never_selected_by_date():
    assert SHIELD_V2.is_draft
    # Far-future date still resolves to the latest non-draft config.
    assert get_rig_config_for_date(date(2100, 1, 1)) is SHIELD_V1


def test_date_before_any_config_raises():
    with pytest.raises(ValueError, match="No rig configuration"):
        get_rig_config_for_date(date(1990, 1, 1))


def test_gauge_lookup():
    spec = SHIELD_V1.gauge("baratron_downstream")
    assert spec.full_scale_torr == 1.0
    assert spec.location == "downstream"
    with pytest.raises(KeyError, match="No gauge named"):
        SHIELD_V1.gauge("does_not_exist")
