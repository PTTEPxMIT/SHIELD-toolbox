"""Plots for a single processed run: upstream and downstream pressure with
the extracted plateau / rise fit overlaid.

Every function accepts and returns a matplotlib ``Axes`` and never calls
``plt.show()`` or ``savefig`` — display and saving are the caller's job.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from shield_toolbox.processing import ProcessedRun


def _get_ax(ax: Axes | None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    return ax


def plot_upstream(processed: ProcessedRun, ax: Axes | None = None) -> Axes:
    """Upstream pressure vs time with the stable-plateau average marked.

    The pre-window region (upstream gauge still on its saturation cap) is
    shaded; the dashed line is the plateau average used for the
    permeability's sqrt(P_up).
    """
    ax = _get_ax(ax)
    ts = processed.timeseries
    time_s = ts["time_s"].to_numpy()
    ax.plot(time_s, ts["upstream_torr"], color="tab:blue", lw=1, label="upstream")

    in_run = ts["in_run"].to_numpy()
    if not in_run.all():
        window_start = time_s[in_run][0]
        ax.axvspan(time_s[0], window_start, color="0.85", label="gauge saturated")

    plateau = processed.upstream_plateau.average_torr
    ax.axhline(
        plateau,
        color="tab:red",
        ls="--",
        lw=1,
        label=f"plateau {plateau:.1f} Torr",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Upstream pressure (Torr)")
    ax.set_title(f"{processed.run_id} — upstream")
    ax.legend()
    return ax


def plot_downstream(processed: ProcessedRun, ax: Axes | None = None) -> Axes:
    """Downstream pressure vs time with the weighted linear rise fit.

    The fitted line is drawn over the samples that entered the fit (reliable
    gauge range only); its slope is the permeation signal.
    """
    ax = _get_ax(ax)
    ts = processed.timeseries
    time_s = ts["time_s"].to_numpy()
    ax.plot(time_s, ts["downstream_torr"], color="tab:blue", lw=1, label="downstream")

    fit = processed.downstream_fit
    fit_time = time_s[ts["fit_used"].to_numpy()]
    if len(fit_time):
        fit_span = np.array([fit_time[0], fit_time[-1]])
        ax.plot(
            fit_span,
            fit.evaluate(fit_span),
            color="tab:red",
            ls="--",
            lw=1.5,
            label=f"fit {fit.slope_torr_per_s:.2e} Torr/s",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Downstream pressure (Torr)")
    ax.set_title(f"{processed.run_id} — downstream")
    ax.legend()
    return ax


def plot_temperature(processed: ProcessedRun, ax: Axes | None = None) -> Axes:
    """Sample temperature vs time.

    Runs without thermocouple data get a flat line at the fallback value
    (furnace setpoint + rig offset), labeled as such.
    """
    ax = _get_ax(ax)
    ts = processed.timeseries
    time_s = ts["time_s"].to_numpy()
    temperature = ts["temperature_K"].to_numpy()

    if np.isfinite(temperature).any():
        ax.plot(time_s, temperature, color="tab:blue", lw=1, label="thermocouple")
    else:
        ax.axhline(
            processed.sample_temperature_K,
            color="tab:blue",
            lw=1.5,
            label=f"setpoint-derived {processed.sample_temperature_K:.0f} K",
        )
        ax.set_xlim(time_s[0], time_s[-1])
    ax.axhline(
        processed.sample_temperature_K,
        color="tab:red",
        ls="--",
        lw=1,
        label=f"analysis T {processed.sample_temperature_K:.1f} K",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sample temperature (K)")
    ax.set_title(f"{processed.run_id} — temperature")
    ax.legend()
    return ax


def plot_permeability(processed: ProcessedRun, ax: Axes | None = None) -> Axes:
    """Instantaneous apparent permeability vs time, log scale.

    The trace is the Savitzky–Golay-smoothed downstream dP/dt converted to
    permeability (no thermal-transpiration correction); the dashed line is
    the steady-state Takaishi–Sensui value from the linear fit. Non-positive
    samples (gauge noise before breakthrough) are masked on the log axis.
    """
    ax = _get_ax(ax)
    ts = processed.timeseries
    time_s = ts["time_s"].to_numpy()
    phi = ts["apparent_permeability"].to_numpy()

    shown = np.isfinite(phi) & (phi > 0)
    # Stop at the end of the reliable downstream range — once the 1 Torr
    # gauge saturates, dP/dt (and so the apparent permeability) is meaningless.
    fit_used = ts["fit_used"].to_numpy()
    if fit_used.any():
        shown &= np.arange(len(phi)) <= np.nonzero(fit_used)[0][-1]
    ax.plot(
        time_s[shown],
        phi[shown],
        color="tab:blue",
        lw=1,
        label="apparent (smoothed dP/dt)",
    )
    steady = processed.permeability.nominal_value
    if steady > 0:
        ax.axhline(
            steady,
            color="tab:red",
            ls="--",
            lw=1,
            label=f"steady-state {steady:.2e}",
        )
        # Floor the log axis so pre-breakthrough gauge noise doesn't dominate.
        ax.set_ylim(bottom=steady * 1e-4)
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Permeability (H/(m·s·Pa$^{0.5}$))")
    ax.set_title(f"{processed.run_id} — permeability")
    ax.legend()
    return ax


def plot_run_overview(processed: ProcessedRun, axes=None):
    """2×2 overview: upstream, downstream, temperature, permeability.

    ``axes`` is any indexable of four ``Axes`` (e.g. a flattened 2×2 grid);
    created if not given. Returns the four axes as a flat array.
    """
    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = np.asarray(axes).ravel()
    plot_upstream(processed, ax=axes[0])
    plot_downstream(processed, ax=axes[1])
    plot_temperature(processed, ax=axes[2])
    plot_permeability(processed, ax=axes[3])
    return axes
