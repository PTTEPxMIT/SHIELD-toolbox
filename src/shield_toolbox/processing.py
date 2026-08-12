"""End-to-end processing of a recorded run into a stored :class:`ProcessedRun`.

``process_run`` takes a loaded :class:`~shield_toolbox.io.run.PermeationRun`
plus the sample description, converts raw voltages to pressures and
temperature, restricts analysis to the valid run window (upstream gauge off
its saturation cap), extracts the steady-state permeability, and returns a
:class:`ProcessedRun` that can be written to disk as::

    <output_dir>/<substrate>/<coating>/<run_id>/
    ├── timeseries.parquet   full-resolution processed time series
    └── result.json          sample info, provenance, and scalar results

The parquet file keeps every raw voltage column alongside the processed
pressures so a stored run is re-analysable without the raw data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from uncertainties import UFloat

from shield_toolbox import __version__
from shield_toolbox.analysis import (
    DownstreamFit,
    UpstreamPlateau,
    fit_downstream_rise,
    permeability_takaishi_sensui,
    run_window_mask,
    stable_upstream_pressure,
)
from shield_toolbox.config import RigConfig, get_rig_config_for_date
from shield_toolbox.gauges import (
    Baratron626D,
    TypeKThermocouple,
    pressure_reading_error_torr,
)
from shield_toolbox.io import PermeationRun

TIMESERIES_FILENAME = "timeseries.parquet"
RESULT_FILENAME = "result.json"


@dataclass(frozen=True)
class SampleInfo:
    """The sample mounted for a run — not recorded by the DAS, so supplied at
    processing time."""

    substrate: str
    coating: str = "uncoated"
    thickness_m: float = 0.00088
    sample_id: str | None = None


@dataclass(frozen=True)
class ProcessedRun:
    """A fully processed run: time series plus scalar results and provenance."""

    run_id: str
    sample: SampleInfo
    rig: RigConfig
    timeseries: pd.DataFrame
    """Columns: ``timestamp``, ``time_s``, one ``<gauge>_voltage_V`` per
    gauge, ``upstream_torr``/``upstream_err_torr``,
    ``downstream_torr``/``downstream_err_torr``, ``temperature_K`` (NaN if
    unknown), ``in_run``, ``fit_used``."""
    upstream_plateau: UpstreamPlateau
    downstream_fit: DownstreamFit
    sample_temperature_K: float
    temperature_source: str
    """``"thermocouple"`` or ``"furnace_setpoint_offset"``."""
    permeability: UFloat
    """H/(m·s·Pa^0.5), with propagated uncertainty."""
    furnace_setpoint: float | None
    valve_times_s: dict[str, float]

    def result_dict(self) -> dict:
        """The scalar results and provenance, as written to ``result.json``."""
        window = self.timeseries["in_run"].to_numpy()
        time_s = self.timeseries["time_s"].to_numpy()
        return {
            "run_id": self.run_id,
            "sample": asdict(self.sample),
            "provenance": {
                "rig_version": self.rig.version,
                "toolbox_version": __version__,
                "processed_utc": datetime.now(UTC).isoformat(),
            },
            "run_info": {
                "furnace_setpoint": self.furnace_setpoint,
                "valve_times_s": self.valve_times_s,
                "duration_s": float(time_s[-1]),
                "n_samples": int(len(time_s)),
            },
            "window": {
                "rule": (
                    "upstream Baratron saturates at ~10.12 V raw; in-run = "
                    "after the last reading >= 10 V, through the final time"
                ),
                "in_run_start_s": float(time_s[window][0]) if window.any() else None,
                "n_in_run": int(window.sum()),
            },
            "temperature": {
                "sample_temperature_K": self.sample_temperature_K,
                "source": self.temperature_source,
            },
            "results": {
                "upstream_pressure_torr": self.upstream_plateau.average_torr,
                "downstream_rise_torr_per_s": self.downstream_fit.slope_torr_per_s,
                "downstream_fit_intercept_torr": self.downstream_fit.intercept_torr,
                "n_fit_samples": int(self.downstream_fit.used.sum()),
                "permeability": {
                    "nominal": self.permeability.nominal_value,
                    "std_dev": self.permeability.std_dev,
                    "units": "H/(m·s·Pa^0.5)",
                },
            },
        }

    def output_dir(self, base_dir: str | Path) -> Path:
        """``<base>/<substrate>/<coating>/<run_id>``, names made path-safe."""

        def safe(name: str) -> str:
            return name.strip().replace("/", "-").replace(" ", "_")

        return (
            Path(base_dir)
            / safe(self.sample.substrate)
            / safe(self.sample.coating)
            / safe(self.run_id)
        )

    def write(self, base_dir: str | Path) -> Path:
        """Write ``timeseries.parquet`` + ``result.json``; returns the run dir."""
        run_dir = self.output_dir(base_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.timeseries.to_parquet(run_dir / TIMESERIES_FILENAME, index=False)
        with open(run_dir / RESULT_FILENAME, "w") as f:
            json.dump(self.result_dict(), f, indent=2)
        return run_dir


def process_run(
    run: PermeationRun,
    sample: SampleInfo,
    rig: RigConfig | None = None,
) -> ProcessedRun:
    """Process a loaded run into a :class:`ProcessedRun`.

    Args:
        run: The loaded raw run.
        sample: Substrate / coating / thickness mounted for this run.
        rig: Rig configuration; defaults to the one in service on the run
            date (:func:`~shield_toolbox.config.get_rig_config_for_date`).

    Raises:
        ValueError: If the run has no upstream or no downstream Baratron.
    """
    if rig is None:
        rig = get_rig_config_for_date(_run_date(run))

    upstream_name, upstream_torr = _baratron_pressure(run, "upstream")
    downstream_name, downstream_torr = _baratron_pressure(run, "downstream")

    in_run = run_window_mask(run.voltage(upstream_name))
    if not in_run.any():
        raise ValueError(
            f"Run {run.run_id}: upstream gauge saturated to the final sample — "
            "no analysable window"
        )

    temperature_K, temperature_source = _sample_temperature(run, rig, in_run)

    time_in = run.time_s[in_run]
    plateau = stable_upstream_pressure(time_in, upstream_torr[in_run])
    fit_in = fit_downstream_rise(time_in, downstream_torr[in_run])

    # Expand the fit mask (defined on the in-run window) to the full trace.
    fit_used = np.zeros(len(run.time_s), dtype=bool)
    fit_used[np.nonzero(in_run)[0][fit_in.used]] = True
    fit = DownstreamFit(
        slope_torr_per_s=fit_in.slope_torr_per_s,
        intercept_torr=fit_in.intercept_torr,
        used=fit_used,
    )

    permeability = permeability_takaishi_sensui(
        slope_torr_per_s=fit.slope_torr_per_s,
        temperature_K=temperature_K,
        sample_thickness_m=sample.thickness_m,
        downstream_pressure_torr=float(downstream_torr[in_run][-1]),
        upstream_pressure_torr=plateau.average_torr,
        rig=rig,
    )

    timeseries = _build_timeseries(
        run, upstream_torr, downstream_torr, temperature_K, in_run, fit_used
    )

    return ProcessedRun(
        run_id=run.run_id,
        sample=sample,
        rig=rig,
        timeseries=timeseries,
        upstream_plateau=plateau,
        downstream_fit=fit,
        sample_temperature_K=temperature_K,
        temperature_source=temperature_source,
        permeability=permeability,
        furnace_setpoint=run.furnace_setpoint,
        valve_times_s=run.valve_times_s,
    )


def _run_date(run: PermeationRun) -> date:
    start = run.start_time
    if start is not None:
        return start.date()
    raw = run.metadata.get("run_info", {}).get("date")
    if raw:
        return date.fromisoformat(raw)
    raise ValueError(f"Run {run.run_id} has no date in its metadata")


def _baratron_pressure(run: PermeationRun, location: str) -> tuple[str, np.ndarray]:
    """Calibrated pressure trace (Torr) of the Baratron at ``location``."""
    for gauge in run.metadata.get("gauges", []):
        if (
            gauge.get("gauge_location") == location
            and "Baratron" in gauge.get("type", "")
            and gauge.get("name") in run.gauge_voltages
        ):
            calibration = Baratron626D(full_scale_torr=float(gauge["full_scale_torr"]))
            return gauge["name"], calibration.to_torr(run.voltage(gauge["name"]))
    raise ValueError(
        f"Run {run.run_id} has no {location} Baratron with recorded data; "
        f"gauges: {sorted(run.gauge_voltages)}"
    )


def _sample_temperature(
    run: PermeationRun, rig: RigConfig, in_run: np.ndarray
) -> tuple[float, str]:
    """Mean thermocouple temperature over the run window, else setpoint + offset.

    Thermocouple conversion currently omits cold-junction compensation
    (matching the legacy analysis); pressure-only runs fall back to
    ``furnace_setpoint + furnace_setpoint_offset_K``.
    """
    if run.thermocouple_mv:
        mv = next(iter(run.thermocouple_mv.values()))
        kelvin = np.asarray(rig.thermocouple.to_kelvin(mv[in_run]), dtype=float)
        return float(np.mean(kelvin)), "thermocouple"
    if run.furnace_setpoint is None:
        raise ValueError(
            f"Run {run.run_id} has neither thermocouple data nor a furnace "
            "setpoint — sample temperature unknown"
        )
    return run.furnace_setpoint + rig.furnace_setpoint_offset_K, (
        "furnace_setpoint_offset"
    )


def _build_timeseries(
    run: PermeationRun,
    upstream_torr: np.ndarray,
    downstream_torr: np.ndarray,
    temperature_K: float,
    in_run: np.ndarray,
    fit_used: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"timestamp": run.timestamps, "time_s": run.time_s}
        | {f"{name}_voltage_V": trace for name, trace in run.gauge_voltages.items()}
    )
    frame["upstream_torr"] = upstream_torr
    frame["upstream_err_torr"] = pressure_reading_error_torr(upstream_torr)
    frame["downstream_torr"] = downstream_torr
    frame["downstream_err_torr"] = pressure_reading_error_torr(downstream_torr)

    if run.thermocouple_mv:
        # No cold-junction compensation (matches legacy).
        mv = next(iter(run.thermocouple_mv.values()))
        frame["temperature_K"] = np.asarray(
            TypeKThermocouple().to_kelvin(mv), dtype=float
        )
    else:
        frame["temperature_K"] = np.nan

    frame["in_run"] = in_run
    frame["fit_used"] = fit_used
    return frame
