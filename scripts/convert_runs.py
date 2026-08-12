"""Convert old-generation SHIELD run directories to the canonical format.

One-time migration: after conversion every run holds ``shield_data.csv`` +
``run_metadata.json`` (the current DAS recorder layout) and loads with
``shield_toolbox.load_run``. Original CSVs are left untouched; converting in
place adds the canonical CSV next to them and upgrades the metadata file.

Examples::

    # In place (adds shield_data.csv inside each run directory)
    uv run python scripts/convert_runs.py ../SHIELD-Data/run_data/*

    # Into a separate tree, originals completely untouched
    uv run python scripts/convert_runs.py ../SHIELD-Data/run_data/* \\
        --dest converted_runs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shield_toolbox import convert_run, load_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run directories")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination base directory (default: convert in place)",
    )
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        if not run_dir.is_dir():
            print(f"skipped {run_dir} (not a directory)")
            continue
        dest = None if args.dest is None else args.dest / run_dir.name
        out = convert_run(run_dir, dest)
        run = load_run(out)  # verify the converted run loads
        print(
            f"converted {run_dir.name}: {len(run.time_s)} samples, "
            f"gauges {sorted(run.gauge_voltages)}"
            + (f" -> {out}" if dest is not None else " (in place)")
        )


if __name__ == "__main__":
    main()
