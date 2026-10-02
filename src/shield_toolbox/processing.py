"""End-to-end processing of a recorded run into a stored :class:`ProcessedRun`.

``process_run`` takes a loaded :class:`~shield_toolbox.io.run.PermeationRun`
plus the sample description, converts raw voltages to pressures and
temperature, restricts analysis to the valid run window (both Baratrons off
their saturation caps), runs the background-subtracted time-lag method
(:mod:`shield_toolbox.analysis.time_lag`) for permeability, diffusivity and
solubility, and returns a :class:`ProcessedRun` that can be written to disk
as::

    <output_dir>/<substrate>/<coating>/<run_id>/
    ├── timeseries.parquet   full-resolution processed time series
    └── result.json          sample info, provenance, and scalar results

The parquet file keeps every raw voltage column alongside the processed
pressures so a stored run is re-analysable without the raw data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from uncertainties import UFloat

from shield_toolbox import __version__
from shield_toolbox.analysis import (
    BackgroundFit,
    LeakRateFit,
    NoiseRecording,
    SteadyStateFit,
    diffusivity_from_time_lag,
    downstream_window_mask,
    find_noise_recording,
    fit_background,
    fit_leak_rate,
    fit_steady_state,
    initial_time,
    leak_molar_rate_mol_per_s,
    permeability_takaishi_sensui,
    run_window_mask,
    solubility_from_permeability,
)
from shield_toolbox.analysis.time_lag import (
    NOISE_MARGIN_FRACTION,
    NOISE_START_S,
    ONSET_SIGMA,
    PRESSURISED_TORR,
    SS_START_TAUS,
)
from shield_toolbox.config import RigConfig, get_rig_config_for_date
from shield_toolbox.constants import TORR_TO_PA, ZERO_CELSIUS_K
from shield_toolbox.gauges import (
    TYPE_K_MAX_MV,
    TYPE_K_MIN_MV,
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

        Reads the v1.4 sample fields plus the v1.5 ``sample_id``, falling
        back to the substrate-only ``material`` (v1.0) / ``sample_material``
        (v1.3) names. Returns None if the metadata carries no sample
        description at all.
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
            sample_id=run_info.get("sample_id"),
            coating_layers=tuple(run_info.get("sample_coating_layers", ())),
        )


@dataclass(frozen=True)
class TimeLagSettings:
    """Settings of the background-subtracted time-lag method.

    Every keyword of :func:`process_run` beyond ``run``/``sample``/``rig``
    sets one of these; they are stored in ``result.json``.
    """

    upstream_pressure_torr: float | None = None
    """Upstream pressure used in Φ, Torr. None: the measured mean over the
    steady-state window."""
    analysis_hours: float | None = 30.0
    """Only the first this many hours after ``t_init`` are analysed. None:
    the whole run."""
    steady_state_start_taus: float = SS_START_TAUS
    """The steady-state window starts at this many τ_L after ``t_init``."""
    steady_state_end_s: float | None = None
    """End of the steady-state window, s after ``t_init``. None: the last
    usable sample."""
    steady_state_start_s: float | None = None
    """Fixed start of the steady-state window, s after ``t_init``; overrides
    ``steady_state_start_taus`` (no τ_L iteration)."""
    noise_start_s: float = NOISE_START_S
    """The noise recording starts this long after ``t_init``."""
    noise_margin_fraction: float = NOISE_MARGIN_FRACTION
    """The noise recording ends this fraction of the pre-rise time (t_init to
    the detected onset) before the onset."""
    onset_sigma: float = ONSET_SIGMA
    """Rise-onset detection threshold, in standard errors."""
    noise_end_s: float | None = None
    """Manual end of the noise recording, s after ``t_init``; overrides the
    detected one."""
    background_slope_pa_per_s: float | None = None
    """Override of the fitted background rate ``b`` (e.g. 0, or ±1σ, for
    sensitivity checks). None: the fitted rate."""
    downstream_max_torr: float = 0.95
    """Downstream readings at or above this are excluded (95 % of the 1 Torr
    Baratron's full scale, before it saturates)."""


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
    unknown or the thermocouple reading is outside the Type K range),
    ``in_run``, ``time_since_init_s``, ``downstream_pa``,
    ``downstream_filtered_pa`` (background removed), ``noise_recording``,
    ``usable`` and ``fit_used`` (the steady-state window)."""
    settings: TimeLagSettings
    initial_time_s: float
    """``t_init``: when the upstream pressure stepped up (time-lag zero)."""
    noise: NoiseRecording
    background: BackgroundFit
    """Background line fitted to the noise recording, in Pa and Pa/s."""
    background_slope_pa_per_s: float
    """Background rate actually subtracted: the fitted ``b`` unless
    overridden in the settings."""
    steady_state: SteadyStateFit
    """Steady-state fit on the filtered signal, in Pa/s."""
    sample_temperature_K: float
    """Mean sample temperature over the steady-state window."""
    temperature_source: str
    """``"thermocouple"`` or ``"furnace_setpoint_offset"``."""
    upstream_pressure_torr: float
    """Upstream pressure used in Φ."""
    upstream_pressure_measured_torr: float
    """Mean measured upstream pressure over the steady-state window."""
    downstream_pressure_torr: float
    """Last raw downstream pressure in the steady-state window (where the
    Takaishi–Sensui correction is evaluated)."""
    permeability: UFloat
    """H/(m·s·Pa^0.5), with propagated uncertainty."""
    time_lag_s: float | None
    """τ_L, or None when the steady-state line crosses zero before
    ``t_init``."""
    diffusivity_m2_per_s: float | None
    """D = e²/(6τ_L); None whenever the time lag is."""
    solubility: UFloat | None
    """S = Φ/D, H/(m³·Pa^0.5); None whenever the time lag is."""
    furnace_setpoint: float | None
    valve_times_s: dict[str, float]

    @property
    def steady_state_window_s(self) -> tuple[float, float]:
        """Start and end of the steady-state window, s after ``t_init``."""
        t_rel = self.timeseries["time_since_init_s"].to_numpy()[self.steady_state.used]
        return float(t_rel[0]), float(t_rel[-1])

    def refit(self, **settings) -> ProcessedRun:
        """Re-run the analysis on the stored time series with some settings
        changed, e.g. ``refit(steady_state_start_taus=2)`` or
        ``refit(background_slope_pa_per_s=0.0)``."""
        return _analyse(
            run_id=self.run_id,
            sample=self.sample,
            rig=self.rig,
            timeseries=self.timeseries,
            settings=replace(self.settings, **settings),
            furnace_setpoint=self.furnace_setpoint,
            valve_times_s=self.valve_times_s,
        )

    def result_dict(self) -> dict:
        """The scalar results and provenance, as written to ``result.json``."""
        window = self.timeseries["in_run"].to_numpy()
        time_s = self.timeseries["time_s"].to_numpy()
        sample = asdict(self.sample)
        sample["coating_layers"] = list(sample["coating_layers"])
        ss_start_s, ss_end_s = self.steady_state_window_s
        return {
            "run_id": self.run_id,
            "run_type": "permeation_exp",
            "sample": sample,
            "provenance": {
                "rig_version": self.rig.version,
                "toolbox_version": __version__,
                "processed_utc": datetime.now(UTC).isoformat(),
                "method": "background-subtracted time lag",
                "settings": asdict(self.settings),
            },
            "run_info": {
                "furnace_setpoint": self.furnace_setpoint,
                "valve_times_s": self.valve_times_s,
                "duration_s": float(time_s[-1]),
                "n_samples": int(len(time_s)),
            },
            "window": {
                "rule": (
                    "Baratrons saturate at ~10.12 V raw; in-run = after the last "
                    "upstream reading >= 10 V, up to the first downstream "
                    "reading >= 10 V"
                ),
                "in_run_start_s": float(time_s[window][0]) if window.any() else None,
                "n_in_run": int(window.sum()),
            },
            "temperature": {
                "sample_temperature_K": self.sample_temperature_K,
                "source": self.temperature_source,
            },
            "results": {
                "initial_time_s": self.initial_time_s,
                "noise_recording": {
                    "start_s": self.noise.start_s,
                    "end_s": self.noise.end_s,
                    "onset_s": self.noise.onset_s,
                    "onset_from": self.noise.onset_from,
                    "noise_sd_pa": self.noise.noise_sd,
                    "n_samples": int(self.noise.used.sum()),
                },
                "background": {
                    "level_pa": _ufloat_dict(self.background.level),
                    "slope_pa_per_s": _ufloat_dict(self.background.slope),
                    "subtracted_slope_pa_per_s": self.background_slope_pa_per_s,
                },
                "steady_state": {
                    "slope_pa_per_s": self.steady_state.slope,
                    "converged": self.steady_state.converged,
                    "window_start_s": ss_start_s,
                    "window_end_s": ss_end_s,
                    "n_samples": int(self.steady_state.used.sum()),
                },
                "upstream_pressure_torr": self.upstream_pressure_torr,
                "upstream_pressure_measured_torr": self.upstream_pressure_measured_torr,
                "downstream_pressure_torr": self.downstream_pressure_torr,
                "permeability": {
                    **_ufloat_dict(self.permeability),
                    "units": "H/(m·s·Pa^0.5)",
                },
                "time_lag": {"time_lag_s": self.time_lag_s},
                "diffusivity": {
                    "value": self.diffusivity_m2_per_s,
                    "units": "m^2/s",
                },
                "solubility": {
                    **_ufloat_dict(self.solubility),
                    "units": "H/(m^3·Pa^0.5)",
                },
            },
        }

    def output_dir(self, base_dir: str | Path) -> Path:
        """``<base>/<substrate>/<coating>/<run_id>``, names made path-safe."""
        return _sample_output_dir(base_dir, self.sample, self.run_id)

    def write(self, base_dir: str | Path) -> Path:
        """Write ``timeseries.parquet`` + ``result.json``; returns the run dir."""
        run_dir = self.output_dir(base_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.timeseries.to_parquet(run_dir / TIMESERIES_FILENAME, index=False)
        with open(run_dir / RESULT_FILENAME, "w") as f:
            json.dump(self.result_dict(), f, indent=2)
        return run_dir


@dataclass(frozen=True)
class LeakTestResult:
    """A processed leak test: the background leak rate of the sealed assembly.

    Produced by :func:`process_leak_test` from a ``run_type="leak_test"``
    run — recorded with the sample installed and sealed, upstream
    unpressurized, and the downstream volume isolated at a setpoint inside
    the 1 Torr Baratron's range. A standalone diagnostic: it is not applied
    to any permeation run.
    """

    run_id: str
    sample: SampleInfo
    rig: RigConfig
    timeseries: pd.DataFrame
    """Columns: ``timestamp``, ``time_s``, one ``<gauge>_voltage_V`` per
    gauge, ``downstream_torr``/``downstream_err_torr``, ``fit_used``."""
    fit: LeakRateFit
    measurement_start_s: float
    """Start of the fitted window on the run's time axis."""
    measurement_start_source: str
    """``"downstream_isolated_time"`` (spacebar event) or ``"trace_start"``."""
    downstream_setpoint_torr: float | None
    """The setpoint recorded by the DAS, if any."""
    furnace_setpoint: float | None

    @property
    def rate_torr_per_s(self) -> float:
        """The background leak rate, in Torr/s."""
        return self.fit.rate_torr_per_s

    def result_dict(self) -> dict:
        """The scalar results and provenance, as written to ``result.json``."""
        time_s = self.timeseries["time_s"].to_numpy()
        sample = asdict(self.sample)
        sample["coating_layers"] = list(sample["coating_layers"])
        molar_rate = leak_molar_rate_mol_per_s(self.fit.rate_torr_per_s, self.rig)
        return {
            "run_id": self.run_id,
            "run_type": "leak_test",
            "sample": sample,
            "provenance": {
                "rig_version": self.rig.version,
                "toolbox_version": __version__,
                "processed_utc": datetime.now(UTC).isoformat(),
            },
            "run_info": {
                "furnace_setpoint": self.furnace_setpoint,
                "downstream_setpoint_torr": self.downstream_setpoint_torr,
                "duration_s": float(time_s[-1]),
                "n_samples": int(len(time_s)),
            },
            "results": {
                "leak_rate_torr_per_s": self.fit.rate_torr_per_s,
                "leak_molar_rate": {
                    "nominal": molar_rate.nominal_value,
                    "std_dev": molar_rate.std_dev,
                    "units": "mol/s",
                },
                "intercept_torr": self.fit.intercept_torr,
                "mean_pressure_torr": self.fit.mean_pressure_torr,
                "r_squared": self.fit.r_squared,
                "n_fit_samples": int(self.fit.used.sum()),
                "measurement_start_s": self.measurement_start_s,
                "measurement_start_source": self.measurement_start_source,
            },
        }

    def output_dir(self, base_dir: str | Path) -> Path:
        """``<base>/<substrate>/<coating>/<run_id>``, names made path-safe."""
        return _sample_output_dir(base_dir, self.sample, self.run_id)

    def write(self, base_dir: str | Path) -> Path:
        """Write ``timeseries.parquet`` + ``result.json``; returns the run dir."""
        run_dir = self.output_dir(base_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.timeseries.to_parquet(run_dir / TIMESERIES_FILENAME, index=False)
        with open(run_dir / RESULT_FILENAME, "w") as f:
            json.dump(self.result_dict(), f, indent=2)
        return run_dir


def process_leak_test(
    run: PermeationRun,
    sample: SampleInfo | None = None,
    rig: RigConfig | None = None,
) -> LeakTestResult:
    """Process a leak-test run into the sample's background leak rate.

    Fits a straight line to the downstream Baratron pressure from the
    ``downstream_isolated_time`` spacebar event (start of the trace when the
    event was not recorded) to the end of the run. The permeation-analysis
    machinery (run window, reliable-band fit) is deliberately not used: a
    leak test has no upstream charge and may sit below the permeation fit's
    reliability floor.

    Args:
        run: The loaded leak-test run.
        sample: Sample mounted during the test; defaults to the metadata's
            sample description.
        rig: Rig configuration; defaults to the one in service on the run
            date.

    Raises:
        ValueError: If the metadata says the run is not a leak test, if the
            run has no downstream Baratron, or if ``sample`` is omitted and
            the metadata records no sample description.
    """
    run_type = run.metadata.get("run_info", {}).get("run_type")
    if run_type is not None and run_type != "leak_test":
        raise ValueError(
            f"Run {run.run_id} is a {run_type!r} run, not a leak test — "
            "use process_run for permeation runs"
        )
    if sample is None:
        sample = SampleInfo.from_metadata(run.metadata)
        if sample is None:
            raise ValueError(
                f"Run {run.run_id}: metadata has no sample description — "
                "pass sample=SampleInfo(...) explicitly"
            )
    if rig is None:
        rig = get_rig_config_for_date(_run_date(run))

    _, downstream_torr = _baratron_pressure(run, "downstream")

    if "downstream_isolated_time" in run.valve_times_s:
        start_s = run.valve_times_s["downstream_isolated_time"]
        start_source = "downstream_isolated_time"
    else:
        start_s = float(run.time_s[0])
        start_source = "trace_start"
    fit = fit_leak_rate(run.time_s, downstream_torr, start_s=start_s)

    timeseries = pd.DataFrame(
        {"timestamp": run.timestamps, "time_s": run.time_s}
        | {f"{name}_voltage_V": trace for name, trace in run.gauge_voltages.items()}
    )
    timeseries["downstream_torr"] = downstream_torr
    timeseries["downstream_err_torr"] = pressure_reading_error_torr(downstream_torr)
    timeseries["fit_used"] = fit.used

    setpoint = run.metadata.get("run_info", {}).get("downstream_setpoint_torr")
    return LeakTestResult(
        run_id=run.run_id,
        sample=sample,
        rig=rig,
        timeseries=timeseries,
        fit=fit,
        measurement_start_s=start_s,
        measurement_start_source=start_source,
        downstream_setpoint_torr=None if setpoint is None else float(setpoint),
        furnace_setpoint=run.furnace_setpoint,
    )


def process_run(
    run: PermeationRun,
    sample: SampleInfo | None = None,
    rig: RigConfig | None = None,
    **settings,
) -> ProcessedRun:
    """Process a loaded run with the background-subtracted time-lag method.

    Converts the gauge voltages to pressures and temperature, finds
    ``t_init`` and the noise recording, fits and subtracts the background
    line, fits the steady-state line from 3 τ_L (iterated), and derives
    Φ (Takaishi–Sensui), D = e²/6τ_L and S = Φ/D. See
    :mod:`shield_toolbox.analysis.time_lag` for the method.

    Args:
        run: The loaded raw run.
        sample: Substrate / coating / thickness mounted for this run;
            defaults to the sample description recorded in the run's
            metadata (:meth:`SampleInfo.from_metadata`).
        rig: Rig configuration; defaults to the one in service on the run
            date (:func:`~shield_toolbox.config.get_rig_config_for_date`).
        **settings: Any :class:`TimeLagSettings` field, e.g.
            ``analysis_hours=None`` or ``steady_state_start_taus=2``.

    Raises:
        ValueError: If the run has no upstream or no downstream Baratron,
            if ``sample`` is omitted and the metadata records no sample
            description, if the run never leaves the gauges' saturation
            window, or if no rise onset is found.
    """
    time_lag_settings = TimeLagSettings(**settings)
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

    # In-run: after the upstream fill comes off its saturation cap, and before
    # the downstream gauge first saturates (after which the downstream reading,
    # and on the rebuilt rig the thermocouple, are no longer valid).
    in_run = run_window_mask(run.voltage(upstream_name)) & downstream_window_mask(
        run.voltage(downstream_name)
    )
    if not in_run.any():
        raise ValueError(
            f"Run {run.run_id}: no analysable window (upstream saturated to the "
            "final sample, or downstream saturated before the upstream came off its cap)"
        )

    timeseries = _build_timeseries(run, upstream_torr, downstream_torr, in_run)
    return _analyse(
        run_id=run.run_id,
        sample=sample,
        rig=rig,
        timeseries=timeseries,
        settings=time_lag_settings,
        furnace_setpoint=run.furnace_setpoint,
        valve_times_s=run.valve_times_s,
    )


def _analyse(
    run_id: str,
    sample: SampleInfo,
    rig: RigConfig,
    timeseries: pd.DataFrame,
    settings: TimeLagSettings,
    furnace_setpoint: float | None,
    valve_times_s: dict[str, float],
) -> ProcessedRun:
    """Steps 1–4 and the derived properties, on a built time series."""
    ts = timeseries.copy()
    time_s = ts["time_s"].to_numpy()
    upstream_torr = ts["upstream_torr"].to_numpy()
    downstream_torr = ts["downstream_torr"].to_numpy()
    downstream_pa = downstream_torr * TORR_TO_PA

    # Steps 1–2: t_init from the upstream step; noise recording before the rise.
    t_init = initial_time(time_s, upstream_torr)
    t_rel = time_s - t_init
    noise = find_noise_recording(
        t_rel,
        downstream_pa,
        start_s=settings.noise_start_s,
        margin_fraction=settings.noise_margin_fraction,
        n_sigma=settings.onset_sigma,
        end_s=settings.noise_end_s,
    )

    # Step 3: background line through the noise recording.
    background = fit_background(t_rel, downstream_pa, noise.used)
    slope = (
        background.slope.nominal_value
        if settings.background_slope_pa_per_s is None
        else settings.background_slope_pa_per_s
    )

    # Step 4: steady-state line on the filtered signal.
    filtered = downstream_pa - (background.level.nominal_value + slope * t_rel)
    usable = (
        ts["in_run"].to_numpy()
        & (t_rel > 0)
        & (upstream_torr > PRESSURISED_TORR)  # also drops recording dropouts
        & (downstream_torr < settings.downstream_max_torr)
    )
    if settings.analysis_hours is not None:
        usable &= t_rel <= settings.analysis_hours * 3600
    steady = fit_steady_state(
        t_rel,
        filtered,
        usable,
        start_taus=settings.steady_state_start_taus,
        end_s=settings.steady_state_end_s,
        start_s=settings.steady_state_start_s,
    )

    temperature_K, temperature_source = _window_temperature(
        ts["temperature_K"].to_numpy()[steady.used], rig, furnace_setpoint, run_id
    )
    upstream_measured = float(np.mean(upstream_torr[steady.used]))
    upstream_used = (
        upstream_measured
        if settings.upstream_pressure_torr is None
        else float(settings.upstream_pressure_torr)
    )
    downstream_last = float(downstream_torr[steady.used][-1])
    permeability = permeability_takaishi_sensui(
        slope_torr_per_s=steady.slope / TORR_TO_PA,
        temperature_K=temperature_K,
        sample_thickness_m=sample.thickness_m,
        downstream_pressure_torr=downstream_last,
        upstream_pressure_torr=upstream_used,
        rig=rig,
    )
    if steady.time_lag_s > 0:
        time_lag_s = steady.time_lag_s
        diffusivity = diffusivity_from_time_lag(time_lag_s, sample.thickness_m)
        solubility = solubility_from_permeability(permeability, diffusivity)
    else:
        # The steady-state line crosses zero before t_init — no valid lag.
        time_lag_s = diffusivity = solubility = None

    ts["time_since_init_s"] = t_rel
    ts["downstream_pa"] = downstream_pa
    ts["downstream_filtered_pa"] = filtered
    ts["noise_recording"] = noise.used
    ts["usable"] = usable
    ts["fit_used"] = steady.used

    return ProcessedRun(
        run_id=run_id,
        sample=sample,
        rig=rig,
        timeseries=ts,
        settings=settings,
        initial_time_s=t_init,
        noise=noise,
        background=background,
        background_slope_pa_per_s=float(slope),
        steady_state=steady,
        sample_temperature_K=temperature_K,
        temperature_source=temperature_source,
        upstream_pressure_torr=upstream_used,
        upstream_pressure_measured_torr=upstream_measured,
        downstream_pressure_torr=downstream_last,
        permeability=permeability,
        time_lag_s=time_lag_s,
        diffusivity_m2_per_s=diffusivity,
        solubility=solubility,
        furnace_setpoint=furnace_setpoint,
        valve_times_s=valve_times_s,
    )


def _sample_output_dir(base_dir: str | Path, sample: SampleInfo, run_id: str) -> Path:
    """``<base>/<substrate>/<coating>/<run_id>``, names made path-safe."""

    def safe(name: str) -> str:
        return name.strip().replace("/", "-").replace(" ", "_")

    return Path(base_dir) / safe(sample.substrate) / safe(sample.coating) / safe(run_id)


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


def _window_temperature(
    temperature_K: np.ndarray,
    rig: RigConfig,
    furnace_setpoint: float | None,
    run_id: str,
) -> tuple[float, str]:
    """Mean thermocouple temperature over the window, else setpoint + offset.

    Thermocouple conversion omits cold-junction compensation (matching the
    legacy analysis). Runs without valid thermocouple readings fall back to
    the furnace setpoint (recorded in °C) converted to kelvin plus
    ``furnace_setpoint_offset_K``.
    """
    if np.isfinite(temperature_K).any():
        return float(np.nanmean(temperature_K)), "thermocouple"
    if furnace_setpoint is None:
        raise ValueError(
            f"Run {run_id} has neither thermocouple data nor a furnace "
            "setpoint — sample temperature unknown"
        )
    return furnace_setpoint + ZERO_CELSIUS_K + rig.furnace_setpoint_offset_K, (
        "furnace_setpoint_offset"
    )


def _ufloat_dict(value: UFloat | None) -> dict:
    if value is None:
        return {"nominal": None, "std_dev": None}
    return {"nominal": value.nominal_value, "std_dev": value.std_dev}


def _build_timeseries(
    run: PermeationRun,
    upstream_torr: np.ndarray,
    downstream_torr: np.ndarray,
    in_run: np.ndarray,
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
        # No cold-junction compensation (matches legacy). Readings outside the
        # Type K range (e.g. once the downstream gauge saturates on the rebuilt
        # rig) are left as NaN instead of raising.
        mv = np.asarray(next(iter(run.thermocouple_mv.values())), dtype=float)
        valid = np.isfinite(mv) & (mv >= TYPE_K_MIN_MV) & (mv <= TYPE_K_MAX_MV)
        temperature = np.full(len(mv), np.nan)
        temperature[valid] = TypeKThermocouple().to_kelvin(mv[valid])
        frame["temperature_K"] = temperature
    else:
        frame["temperature_K"] = np.nan

    frame["in_run"] = in_run
    return frame
