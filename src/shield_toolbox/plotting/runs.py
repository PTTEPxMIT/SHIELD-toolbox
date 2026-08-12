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


def plot_run_overview(
    processed: ProcessedRun, axes: tuple[Axes, Axes] | None = None
) -> tuple[Axes, Axes]:
    """Upstream and downstream panels side by side for one processed run."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_upstream(processed, ax=axes[0])
    plot_downstream(processed, ax=axes[1])
    return axes
