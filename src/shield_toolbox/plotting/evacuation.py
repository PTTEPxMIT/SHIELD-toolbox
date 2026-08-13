"""Pump-down trace plot with the fitted decay model.

Same contract as the other plot modules: accepts/returns an ``Axes``, never
calls ``plt.show()`` or ``savefig``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from shield_toolbox.analysis import EvacuationFit


def plot_evacuation(
    time_s: npt.ArrayLike,
    pressure_torr: npt.ArrayLike,
    fit: EvacuationFit | None = None,
    target_torr: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Measured pump-down pressure vs time, log scale, with the fitted model.

    Args:
        time_s: Time axis in seconds.
        pressure_torr: Measured pressure in Torr.
        fit: Optional :func:`~shield_toolbox.analysis.fit_evacuation` result,
            drawn as a smooth curve.
        target_torr: Optional target pressure; drawn as a horizontal line
            and, when a fit is given, marked with the predicted time to
            reach it.
        ax: Axes to draw on; created if not given.
    """
    if ax is None:
        _, ax = plt.subplots()
    time_arr = np.asarray(time_s, dtype=float)
    ax.semilogy(
        time_arr / 60.0,
        np.asarray(pressure_torr, dtype=float),
        "o",
        ms=3,
        color="tab:blue",
        label="measured",
    )

    if fit is not None:
        smooth_s = np.linspace(time_arr.min(), time_arr.max(), 500)
        ax.semilogy(
            smooth_s / 60.0,
            fit.pressure_at(smooth_s),
            color="tab:red",
            lw=1.5,
            label=f"fit (1/B = {1 / fit.rate_per_s / 60:.1f} min)",
        )

    if target_torr is not None:
        label = f"target {target_torr:.1e} Torr"
        if fit is not None:
            try:
                label += f" @ {fit.time_to_reach(target_torr) / 60:.0f} min"
            except ValueError:
                label += " (unreachable)"
        ax.axhline(target_torr, color="0.4", ls=":", lw=1, label=label)

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Pressure (Torr)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()
    return ax
