"""Plots for processed runs, one per step of the background-subtracted
time-lag method, plus run overlays.

Every function accepts and returns a matplotlib ``Axes`` and never calls
``plt.show()`` or ``savefig`` — display and saving are the caller's job.
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from shield_toolbox.analysis.time_lag import PRESSURISED_TORR
from shield_toolbox.processing import LegacyProcessedRun, ProcessedRun

UP_COLOR = "tab:orange"
DOWN_COLOR = "0.35"
BG_COLOR = "tab:green"
FIT_COLOR = "tab:blue"
NOISE_COLOR = "tab:cyan"


def _get_ax(ax: Axes | None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    return ax


def _upstream_normalised(processed: ProcessedRun) -> np.ndarray:
    upstream = processed.timeseries["upstream_torr"].to_numpy()
    return upstream / np.median(upstream[upstream > PRESSURISED_TORR])


def plot_initial_time(
    processed: ProcessedRun, t_max_s: float | None = None, ax: Axes | None = None
) -> Axes:
    """Steps 1–2: downstream pressure and normalised upstream around ``t_init``.

    The noise recording is shaded and the detected rise onset marked. The
    downstream is shown relative to the start of the noise recording.

    Args:
        processed: The processed run.
        t_max_s: Last time shown, s after ``t_init``. Default: one hour past
            the onset, in minutes; longer spans are drawn in hours.
        ax: Axes to draw on.
    """
    ax = _get_ax(ax)
    ts = processed.timeseries
    noise = processed.noise
    t_rel = ts["time_since_init_s"].to_numpy()
    down_pa = ts["downstream_pa"].to_numpy()
    if t_max_s is None:
        t_max_s = (noise.end_s if noise.onset_s is None else noise.onset_s) + 3600
    unit, scale = ("min", 60) if t_max_s <= 6 * 3600 else ("h", 3600)

    level = down_pa[noise.used][:60].mean()
    shown = t_rel <= t_max_s
    rel = down_pa[shown] - level
    ax.plot(
        t_rel[shown] / scale, rel, ":", color=DOWN_COLOR, lw=1.2, label="P$_{down}$"
    )
    top = max(np.percentile(rel, 99.5), 0.05)
    ax.plot(
        t_rel[shown] / scale,
        _upstream_normalised(processed)[shown] * top,
        color=UP_COLOR,
        lw=1.5,
        label="upstream (normalised)",
    )
    ax.axvspan(
        noise.start_s / scale,
        noise.end_s / scale,
        color=NOISE_COLOR,
        alpha=0.2,
        label="noise recording",
    )
    if noise.onset_s is not None:
        ax.axvline(
            noise.onset_s / scale,
            color="tab:red",
            lw=0.8,
            ls="--",
            label="detected rise onset",
        )
    ax.set_xlabel(f"time since t$_{{init}}$ ({unit})")
    ax.set_ylabel("P$_{down}$ (Pa, relative)")
    ax.set_title(
        f"{processed.run_id}: steps 1–2 "
        f"(t$_{{init}}$ = {processed.initial_time_s:.0f} s into the recording)"
    )
    ax.legend(fontsize=8, loc="upper left")
    return ax


def plot_background(processed: ProcessedRun, ax: Axes | None = None) -> Axes:
    """Step 3: the noise recording and its background line.

    Both are drawn relative to the fit at the start of the noise recording.
    """
    ax = _get_ax(ax)
    ts = processed.timeseries
    used = processed.noise.used
    x = ts["time_since_init_s"].to_numpy()[used]
    y = ts["downstream_pa"].to_numpy()[used]
    a = processed.background.level.nominal_value
    b = processed.background.slope.nominal_value
    x0 = x - x[0]
    ax.plot(
        x0,
        y - (a + b * x[0]),
        "+",
        ms=3,
        color="0.55",
        alpha=0.25,
        label="noise recording",
    )
    ax.plot(x0, b * x0, color=BG_COLOR, lw=2, label=f"linear fit {b:.2e} Pa/s")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("P$_{down}$ − fit at start (Pa)")
    title = f"{processed.run_id}"
    if processed.furnace_setpoint is not None:
        title += f" ({processed.furnace_setpoint:g} °C)"
    ax.set_title(f"{title}: step 3")
    ax.legend(fontsize=8)
    return ax


def plot_steady_state(processed: ProcessedRun, ax: Axes | None = None) -> Axes:
    """Step 4: the filtered downstream, the steady-state line and τ_L."""
    ax = _get_ax(ax)
    ts = processed.timeseries
    t_rel = ts["time_since_init_s"].to_numpy()
    filtered = ts["downstream_filtered_pa"].to_numpy()
    usable = ts["usable"].to_numpy()
    steady = processed.steady_state
    show = usable | (t_rel <= 0)
    hours = t_rel / 3600

    ax.plot(
        hours[show],
        filtered[show],
        ":",
        color=DOWN_COLOR,
        lw=1.5,
        label="filtered P$_{down}$",
    )
    top = filtered[usable].max()
    up_norm = np.where(t_rel < 0, 0, _upstream_normalised(processed))
    ax.plot(
        hours[show],
        up_norm[show] * top,
        color=UP_COLOR,
        lw=1.5,
        label="upstream (normalised)",
    )
    line_t = np.array([steady.time_lag_s, t_rel[usable][-1]])
    ax.plot(
        line_t / 3600,
        steady.evaluate(line_t),
        color=FIT_COLOR,
        lw=1.5,
        label=f"S∞ = {steady.slope:.2e} Pa/s",
    )
    start_s, end_s = processed.steady_state_window_s
    ax.axvspan(
        start_s / 3600,
        end_s / 3600,
        color=FIT_COLOR,
        alpha=0.07,
        label="steady-state window",
    )
    ax.annotate(
        "",
        xy=(0, -0.04 * top),
        xytext=(steady.time_lag_s / 3600, -0.04 * top),
        arrowprops=dict(arrowstyle="<->", color=FIT_COLOR),
    )
    ax.text(
        0,
        -0.10 * top,
        f"τ$_L$ = {steady.time_lag_s / 3600:.2f} h",
        color=FIT_COLOR,
        ha="left",
        va="top",
    )
    ax.set_ylim(-0.16 * top, 1.08 * top)
    ax.set_xlabel("time since t$_{init}$ (h)")
    ax.set_ylabel("P$_{down}$ (Pa), background removed")
    ax.set_title(f"{processed.run_id}: step 4")
    ax.legend(fontsize=8, loc="upper left")
    return ax


def plot_residuals(processed: ProcessedRun, ax: Axes | None = None) -> Axes:
    """Filtered downstream minus the steady-state line, over the usable span."""
    ax = _get_ax(ax)
    ts = processed.timeseries
    t_rel = ts["time_since_init_s"].to_numpy()
    usable = ts["usable"].to_numpy()
    resid = ts["downstream_filtered_pa"].to_numpy() - processed.steady_state.evaluate(
        t_rel
    )
    hours = t_rel / 3600
    ax.plot(hours[usable], resid[usable], color=DOWN_COLOR, lw=0.8)
    ax.axhline(0, color=FIT_COLOR, lw=1)
    start_s, end_s = processed.steady_state_window_s
    ax.axvspan(start_s / 3600, end_s / 3600, color=FIT_COLOR, alpha=0.07)
    ax.set_xlabel("time since t$_{init}$ (h)")
    ax.set_ylabel("data − steady-state line (Pa)")
    ax.set_title("residuals")
    return ax


def plot_filtered_rises(
    processed_runs: Iterable[ProcessedRun],
    ax: Axes | None = None,
    colors: dict[str, object] | None = None,
) -> Axes:
    """Several runs' filtered rises overlaid, each with its steady-state
    line and τ_L marked.

    Args:
        processed_runs: The runs to overlay.
        ax: Axes to draw on.
        colors: Optional colour per run ID; defaults to the tab10 cycle.
    """
    ax = _get_ax(ax)
    for i, processed in enumerate(processed_runs):
        color = (colors or {}).get(processed.run_id, plt.cm.tab10(i % 10))
        ts = processed.timeseries
        t_rel = ts["time_since_init_s"].to_numpy()
        usable = ts["usable"].to_numpy()
        steady = processed.steady_state
        ax.plot(
            t_rel[usable] / 3600,
            ts["downstream_filtered_pa"].to_numpy()[usable],
            ":",
            color=color,
            lw=1.8,
            label=processed.run_id,
        )
        line_t = np.array([steady.time_lag_s, t_rel[usable][-1]])
        ax.plot(
            line_t / 3600,
            steady.evaluate(line_t),
            color=color,
            lw=1,
            label=(
                f"S∞ {steady.slope:.2e} Pa/s, τ$_L$ {steady.time_lag_s / 3600:.2f} h"
            ),
        )
        ax.plot(steady.time_lag_s / 3600, 0, "o", color=color)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlim(0, None)
    ax.set_xlabel("time since t$_{init}$ (h)")
    ax.set_ylabel("P$_{down}$ (Pa), background removed")
    ax.legend(fontsize=8)
    return ax


def plot_temperature(processed: ProcessedRun, ax: Axes | None = None) -> Axes:
    """Sample temperature vs time since ``t_init``.

    The steady-state window is shaded; the dashed line is the analysis
    temperature (its mean). Runs without thermocouple data get a flat line
    at the fallback value (furnace setpoint + rig offset), labeled as such.
    """
    ax = _get_ax(ax)
    ts = processed.timeseries
    hours = ts["time_since_init_s"].to_numpy() / 3600
    temperature = ts["temperature_K"].to_numpy()

    if np.isfinite(temperature).any():
        ax.plot(hours, temperature, color=DOWN_COLOR, lw=1, label="thermocouple")
    else:
        ax.set_xlim(hours[0], hours[-1])
    start_s, end_s = processed.steady_state_window_s
    ax.axvspan(
        start_s / 3600,
        end_s / 3600,
        color=FIT_COLOR,
        alpha=0.07,
        label="steady-state window",
    )
    source = (
        "" if processed.temperature_source == "thermocouple" else ", setpoint-derived"
    )
    ax.axhline(
        processed.sample_temperature_K,
        color="tab:red",
        ls="--",
        lw=1,
        label=f"analysis T {processed.sample_temperature_K:.1f} K{source}",
    )
    ax.set_xlabel("time since t$_{init}$ (h)")
    ax.set_ylabel("Sample temperature (K)")
    ax.set_title(f"{processed.run_id} — temperature")
    ax.legend(fontsize=8)
    return ax


def plot_run_overview(processed: ProcessedRun, axes=None):
    """2×2 overview: steps 1–2, step 3, step 4, and the step-4 residuals.

    ``axes`` is any indexable of four ``Axes`` (e.g. a flattened 2×2 grid);
    created if not given. Returns the four axes as a flat array.
    """
    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    axes = np.asarray(axes).ravel()
    plot_initial_time(processed, ax=axes[0])
    plot_background(processed, ax=axes[1])
    plot_steady_state(processed, ax=axes[2])
    plot_residuals(processed, ax=axes[3])
    return axes


def plot_legacy_run(processed: LegacyProcessedRun, axes=None):
    """Legacy method: upstream with its mean, and the downstream rise with
    the tail asymptote extended to its P = 0 crossing (τ).

    ``axes`` is any indexable of two ``Axes``; created if not given. Returns
    the two axes as a flat array.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax_up, ax_down = np.asarray(axes).ravel()
    ts = processed.timeseries
    hours = (ts["time_s"].to_numpy() - ts["time_s"].iloc[0]) / 3600

    ax_up.plot(hours, ts["upstream_torr"], color=UP_COLOR, lw=1, label="upstream")
    ax_up.axhline(
        processed.upstream_pressure_torr,
        color="tab:red",
        ls="--",
        lw=1,
        label=f"mean of last 75 %: {processed.upstream_pressure_torr:.1f} Torr",
    )
    ax_up.set_xlabel("time since start of recording (h)")
    ax_up.set_ylabel("Upstream pressure (Torr)")
    ax_up.set_title(f"{processed.run_id} — upstream")
    ax_up.legend(fontsize=8)

    fit = processed.fit
    ax_down.plot(
        hours, ts["downstream_torr"], ":", color=DOWN_COLOR, lw=1.5, label="P$_{down}$"
    )
    used = ts["fit_used"].to_numpy()
    ax_down.plot(
        hours[used],
        ts["downstream_torr"].to_numpy()[used],
        color=FIT_COLOR,
        lw=2,
        alpha=0.4,
        label="fitted (last 25 % of 0.05–0.95 Torr)",
    )
    line_s = np.array([fit.time_lag_s, hours[used][-1] * 3600])
    ax_down.plot(
        line_s / 3600,
        fit.evaluate(line_s),
        color=FIT_COLOR,
        lw=1.5,
        label=f"asymptote {fit.slope_torr_per_s:.2e} Torr/s",
    )
    ax_down.plot(fit.time_lag_s / 3600, 0, "o", color=FIT_COLOR)
    ax_down.text(
        fit.time_lag_s / 3600,
        0,
        f"  τ = {fit.time_lag_s / 3600:.2f} h",
        color=FIT_COLOR,
        va="bottom",
    )
    ax_down.set_ylim(bottom=0)
    ax_down.set_xlabel("time since start of recording (h)")
    ax_down.set_ylabel("Downstream pressure (Torr)")
    ax_down.set_title(f"{processed.run_id} — legacy time lag")
    ax_down.legend(fontsize=8)
    return np.array([ax_up, ax_down])
