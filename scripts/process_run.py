"""Process one or more recorded SHIELD runs end to end.

For each run directory this script loads the raw data, processes it
(pressure calibration, run-window trimming, upstream plateau, downstream
rise fit, Takaishi–Sensui permeability), writes the processed artifact
(``timeseries.parquet`` + ``result.json``) under
``<output>/<substrate>/<coating>/<run_id>/``, and produces an overview
figure of upstream/downstream pressure with the fits.

Example::

    uv run python scripts/process_run.py \\
        ../SHIELD-Data/run_data/25.10.06_run_1_10h41 \\
        --substrate 316L --coating uncoated --thickness-mm 0.88 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from shield_toolbox import load_run
from shield_toolbox.plotting import plot_run_overview
from shield_toolbox.processing import SampleInfo, process_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run directories")
    parser.add_argument("--substrate", required=True, help="Substrate material")
    parser.add_argument("--coating", default="uncoated", help="Coating material")
    parser.add_argument(
        "--thickness-mm", type=float, default=0.88, help="Sample thickness (mm)"
    )
    parser.add_argument("--sample-id", default=None, help="Optional sample ID")
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
        help="Also save the overview figure of each run as a PNG in DIR",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show the figures interactively"
    )
    args = parser.parse_args()

    sample = SampleInfo(
        substrate=args.substrate,
        coating=args.coating,
        thickness_m=args.thickness_mm * 1e-3,
        sample_id=args.sample_id,
    )

    for run_dir in args.run_dirs:
        run = load_run(run_dir)
        processed = process_run(run, sample)
        out_dir = processed.write(args.output)

        results = processed.result_dict()["results"]
        perm = processed.permeability
        print(f"{run.run_id}:")
        print(
            f"  temperature    : {processed.sample_temperature_K:.1f} K "
            f"({processed.temperature_source})"
        )
        print(f"  P_up plateau   : {results['upstream_pressure_torr']:.2f} Torr")
        print(
            f"  dP_down/dt     : {results['downstream_rise_torr_per_s']:.3e} Torr/s "
            f"({results['n_fit_samples']} samples in fit)"
        )
        print(f"  permeability   : {perm:.2e} H/(m·s·Pa^0.5)")
        print(f"  written to     : {out_dir}")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        plot_run_overview(processed, axes=axes)
        fig.tight_layout()
        if args.save_plots:
            args.save_plots.mkdir(parents=True, exist_ok=True)
            fig_path = args.save_plots / f"{processed.run_id}.png"
            fig.savefig(fig_path, dpi=150)
            print(f"  figure         : {fig_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
