"""Process one or more recorded SHIELD runs with the background-subtracted
time-lag method.

Each run is given as a SHIELD-Data run ID (fetched via ``shield_data``) or a
local run directory. For each run the script finds ``t_init`` (upstream
step), the noise recording before the downstream rise, fits and subtracts
the background line, fits the steady-state line from 3 τ_L, and derives
permeability (Takaishi–Sensui), diffusivity and solubility. It writes the
processed artifact (``timeseries.parquet`` + ``result.json``) under
``<output>/<substrate>/<coating>/<run_id>/`` and draws the four-step overview.

Example::

    uv run python scripts/process_run.py \\
        26.09.25_run_1_17h59 26.09.28_run_1_18h50 --upstream-torr 500 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from shield_toolbox import fetch_run, load_run, process_run
from shield_toolbox.plotting import plot_run_overview
from shield_toolbox.processing import SampleInfo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "runs", nargs="+", help="SHIELD-Data run IDs or local run directories"
    )
    parser.add_argument(
        "--upstream-torr",
        type=float,
        default=None,
        help="Upstream pressure used in Φ (default: measured mean over the "
        "steady-state window)",
    )
    parser.add_argument(
        "--analysis-hours",
        type=float,
        default=30.0,
        help="Analyse only this many hours after t_init (default: 30; 0 = all)",
    )
    parser.add_argument(
        "--substrate",
        default=None,
        help="Override the sample substrate (default: from the run metadata)",
    )
    parser.add_argument("--coating", default="uncoated", help="With --substrate")
    parser.add_argument(
        "--thickness-mm", type=float, default=0.88, help="With --substrate"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed_runs"),
        help="Base directory for processed artifacts (default: processed_runs/)",
    )
    parser.add_argument(
        "--save-plots",
        type=Path,
        default=None,
        metavar="DIR",
        help="Also save each run's overview figure as a PNG in DIR",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show the figures interactively"
    )
    args = parser.parse_args()

    sample = None
    if args.substrate is not None:
        sample = SampleInfo(
            substrate=args.substrate,
            coating=args.coating,
            thickness_m=args.thickness_mm * 1e-3,
        )

    for run_ref in args.runs:
        run = load_run(run_ref) if Path(run_ref).is_dir() else fetch_run(run_ref)
        processed = process_run(
            run,
            sample,
            upstream_pressure_torr=args.upstream_torr,
            analysis_hours=args.analysis_hours or None,
        )
        out_dir = processed.write(args.output)

        noise = processed.noise
        start_s, end_s = processed.steady_state_window_s
        print(f"{run.run_id}:")
        print(f"  t_init          : {processed.initial_time_s:.0f} s")
        onset = "none" if noise.onset_s is None else f"{noise.onset_s / 60:.1f} min"
        print(
            f"  noise recording : {noise.start_s / 60:g}–{noise.end_s / 60:.3g} min "
            f"(onset {onset}, {noise.onset_from})"
        )
        print(f"  background b    : {processed.background.slope:.2uP} Pa/s")
        print(
            f"  S∞              : {processed.steady_state.slope:.3e} Pa/s "
            f"(window {start_s / 3600:.1f}–{end_s / 3600:.1f} h"
            + ("" if processed.steady_state.converged else ", τ_L not converged")
            + ")"
        )
        print(
            f"  temperature     : {processed.sample_temperature_K:.1f} K "
            f"({processed.temperature_source})"
        )
        print(
            f"  P_up            : {processed.upstream_pressure_torr:.1f} Torr used, "
            f"{processed.upstream_pressure_measured_torr:.1f} Torr measured"
        )
        print(f"  permeability    : {processed.permeability:.2uP} H/(m·s·Pa^0.5)")
        if processed.time_lag_s is not None:
            print(f"  τ_L             : {processed.time_lag_s / 3600:.2f} h")
            print(f"  diffusivity     : {processed.diffusivity_m2_per_s:.2e} m²/s")
            print(f"  solubility      : {processed.solubility:.2uP} H/(m³·Pa^0.5)")
        print(f"  written to      : {out_dir}")

        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
        plot_run_overview(processed, axes=axes)
        fig.tight_layout()
        if args.save_plots:
            args.save_plots.mkdir(parents=True, exist_ok=True)
            fig_path = args.save_plots / f"{processed.run_id}.png"
            fig.savefig(fig_path, dpi=150)
            print(f"  figure          : {fig_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
