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
    apparent_permeability_vs_time,
    diffusivity_from_time_lag,
    downstream_baseline_torr,
    fit_downstream_rise,
    permeability_takaishi_sensui,
    run_window_mask,
    solubility_from_permeability,
    stable_upstream_pressure,
    time_lag_from_fit,
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
    """The sample mounted for a run.

    Recorded by the DAS in metadata v1.4 (``sample_substrate``,
    ``sample_coating``, ``sample_coating_layers``) and backfilled into the
    stored runs; supplied by hand only for runs whose metadata predates the
    backfill.
    """

    substrate: str
    coating: str = "uncoated"
    thickness_m: float = 0.00088
    sample_id: str | None = None
    coating_layers: tuple[dict, ...] = ()
    """Coating layers as ``{"material": ..., "thickness_nm": ...}`` dicts,
    ordered as named on the sample; empty for an uncoated sample."""

    @classmethod
    def from_metadata(cls, metadata: dict) -> SampleInfo | None:
        """Build the sample description from a run's metadata.

        Reads the v1.4 sample fields, falling back to the substrate-only
        ``material`` (v1.0) / ``sample_material`` (v1.3) names. Returns None
        if the metadata carries no sample description at all.
        """
        run_info = metadata.get("run_info", {})
        substrate = run_info.get(
            "sample_substrate",
            run_info.get("material", run_info.get("sample_material")),
        )
        if substrate is None:
            return None
        defaults = cls(substrate="")
        return cls(
            substrate=substrate,
            coating=run_info.get("sample_coating", defaults.coating),
            thickness_m=run_info.get("sample_thickness", defaults.thickness_m),
            coating_layers=tuple(run_info.get("sample_coating_layers", ())),
        )


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
    time_lag_s: float | None
    """Permeation time lag τ, or None when no valid lag was extractable
    (e.g. the steady-state fit extrapolates below the baseline)."""
    diffusivity_m2_per_s: float | None
    """D = e²/(6τ); None whenever the time lag is."""
    solubility: UFloat | None
    """S = Φ/D, H/(m³·Pa^0.5); None whenever the time lag is."""
    permeation_start_s: float
    """When upstream pressure was applied to the sample (time-lag zero)."""
    permeation_start_source: str
    """``"v3_open_time"`` (loading-valve event) or ``"window_start"``."""
    downstream_baseline_torr: float
    """Pre-breakthrough downstream pressure used for the time-lag intercept."""
    furnace_setpoint: float | None
    valve_times_s: dict[str, float]

    def result_dict(self) -> dict:
        """The scalar results and provenance, as written to ``result.json``."""
        window = self.timeseries["in_run"].to_numpy()
        time_s = self.timeseries["time_s"].to_numpy()
        sample = asdict(self.sample)
        sample["coating_layers"] = list(sample["coating_layers"])
        return {
            "run_id": self.run_id,
            "sample": sample,
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
                "time_lag": {
                    "time_lag_s": self.time_lag_s,
                    "permeation_start_s": self.permeation_start_s,
                    "permeation_start_source": self.permeation_start_source,
                    "baseline_torr": self.downstream_baseline_torr,
                },
                "diffusivity": {
                    "value": self.diffusivity_m2_per_s,
                    "units": "m^2/s",
                },
                "solubility": {
                    "nominal": None
                    if self.solubility is None
                    else self.solubility.nominal_value,
                    "std_dev": None
                    if self.solubility is None
                    else self.solubility.std_dev,
                    "units": "H/(m^3·Pa^0.5)",
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
    sample: SampleInfo | None = None,
    rig: RigConfig | None = None,
) -> ProcessedRun:
    """Process a loaded run into a :class:`ProcessedRun`.

    Args:
        run: The loaded raw run.
        sample: Substrate / coating / thickness mounted for this run;
            defaults to the sample description recorded in the run's
            metadata (:meth:`SampleInfo.from_metadata`).
        rig: Rig configuration; defaults to the one in service on the run
            date (:func:`~shield_toolbox.config.get_rig_config_for_date`).

    Raises:
        ValueError: If the run has no upstream or no downstream Baratron,
            or if ``sample`` is omitted and the metadata records no sample
            description.
    """
    if sample is None:
        sample = SampleInfo.from_metadata(run.metadata)
        if sample is None:
            raise ValueError(
                f"Run {run.run_id}: metadata has no sample description — "
                "pass sample=SampleInfo(...) explicitly"
            )
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

    # Time-lag method: τ from the steady-state fit's baseline crossing,
    # counted from the loading-valve opening; then D = e²/6τ and S = Φ/D.
    if "v3_open_time" in run.valve_times_s:
        permeation_start_s = run.valve_times_s["v3_open_time"]
        permeation_start_source = "v3_open_time"
    else:
        permeation_start_s = float(time_in[0])
        permeation_start_source = "window_start"
    baseline_torr = downstream_baseline_torr(
        run.time_s, downstream_torr, permeation_start_s
    )
    if fit.slope_torr_per_s > 0:
        time_lag_s = time_lag_from_fit(fit, baseline_torr, permeation_start_s)
    else:
        time_lag_s = 0.0  # no rising signal — handled as degenerate below
    if time_lag_s > 0:
        diffusivity = diffusivity_from_time_lag(time_lag_s, sample.thickness_m)
        solubility = solubility_from_permeability(permeability, diffusivity)
    else:
        # Degenerate geometry (e.g. baseline above the fit at the start
        # time) — report no lag rather than a nonsense diffusivity.
        time_lag_s = None
        diffusivity = None
        solubility = None

    # Instantaneous apparent permeability over the run window (NaN outside).
    permeability_t = np.full(len(run.time_s), np.nan)
    permeability_t[in_run] = apparent_permeability_vs_time(
        time_in,
        downstream_torr[in_run],
        upstream_pressure_torr=plateau.average_torr,
        temperature_K=temperature_K,
        sample_thickness_m=sample.thickness_m,
        rig=rig,
    )

    timeseries = _build_timeseries(
        run,
        upstream_torr,
        downstream_torr,
        temperature_K,
        in_run,
        fit_used,
        permeability_t,
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
        time_lag_s=time_lag_s,
        diffusivity_m2_per_s=diffusivity,
        solubility=solubility,
        permeation_start_s=permeation_start_s,
        permeation_start_source=permeation_start_source,
        downstream_baseline_torr=baseline_torr,
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
    permeability_t: np.ndarray,
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

    frame["apparent_permeability"] = permeability_t
    frame["in_run"] = in_run
    frame["fit_used"] = fit_used
    return frame
