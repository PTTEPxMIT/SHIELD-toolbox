"""Fit and plot the Arrhenius trend of a processed campaign.

Aggregates every ``result.json`` under a processed-runs directory (as written
by ``process_run(...).write()`` / ``scripts/process_run.py``), fits
ln(property) vs 1/T, and prints the activation energy and pre-exponential.

Example::

    uv run python scripts/arrhenius.py processed_runs \\
        --substrate "316L steel" --coating none --quantity permeability --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from shield_toolbox import arrhenius, load_results
from shield_toolbox.campaign import ARRHENIUS_QUANTITIES
from shield_toolbox.plotting import plot_arrhenius


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "results_dir",
        type=Path,
        nargs="?",
        default=Path("processed_runs"),
        help="Processed-runs base directory (default: processed_runs/)",
    )
    parser.add_argument("--substrate", default=None, help="Filter by substrate")
    parser.add_argument("--coating", default=None, help="Filter by coating")
    parser.add_argument(
        "--quantity",
        choices=ARRHENIUS_QUANTITIES,
        default="permeability",
        help="Property to fit (default: permeability)",
    )
    parser.add_argument(
        "--save", type=Path, default=None, metavar="PNG", help="Save the figure"
    )
    parser.add_argument(
        "--show", action="store_true", help="Show the figure interactively"
    )
    args = parser.parse_args()

    results = load_results(
        args.results_dir, substrate=args.substrate, coating=args.coating
    )
    fit = arrhenius(results, quantity=args.quantity)

    print(f"{len(results)} runs, {args.quantity}:")
    print(f"  E_a             : {fit.activation_energy_J_per_mol / 1000:.1f} kJ/mol")
    print(f"  pre-exponential : {fit.pre_exponential:.3e}")

    _, ax = plt.subplots(figsize=(7, 5))
    plot_arrhenius(results, fit=fit, quantity=args.quantity, ax=ax)
    plt.tight_layout()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=150)
        print(f"  figure          : {args.save}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
