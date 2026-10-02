"""Matplotlib plotting helpers. Functions accept/return ``ax`` and never call
``plt.show()`` or ``savefig``."""

from shield_toolbox.plotting.campaign import plot_arrhenius
from shield_toolbox.plotting.evacuation import plot_evacuation
from shield_toolbox.plotting.furnace import plot_furnace_log
from shield_toolbox.plotting.runs import (
    plot_background,
    plot_filtered_rises,
    plot_initial_time,
    plot_legacy_run,
    plot_residuals,
    plot_run_overview,
    plot_steady_state,
    plot_temperature,
)

__all__ = [
    "plot_arrhenius",
    "plot_background",
    "plot_evacuation",
    "plot_filtered_rises",
    "plot_furnace_log",
    "plot_initial_time",
    "plot_legacy_run",
    "plot_residuals",
    "plot_run_overview",
    "plot_steady_state",
    "plot_temperature",
]
