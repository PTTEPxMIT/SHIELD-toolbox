"""Fetch stored runs from SHIELD-Data by run ID.

Bridges the toolbox to the ``shield_data`` package: ``fetch_run("25.10.06_run_1_10h41")``
downloads (and caches) the run's ``measurements.parquet`` + metadata from the
SHIELD-Data GitHub repo and returns the same :class:`~shield_toolbox.io.run.PermeationRun`
that :func:`~shield_toolbox.io.loader.load_run` builds from a local directory,
so everything downstream (``process_run``, plotting) is identical.

``shield_data`` is an optional dependency — install the sibling checkout with
``uv pip install -e ../SHIELD-Data`` (see the README's setup section).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from shield_toolbox.io.loader import run_from_frame
from shield_toolbox.io.run import PermeationRun


def fetch_run(run_id: str) -> PermeationRun:
    """Fetch one stored run from SHIELD-Data and load it as a :class:`PermeationRun`.

    Args:
        run_id: Stored run identifier, e.g. ``"25.10.06_run_1_10h41"`` —
            browse them with ``shield_data.catalogue()``.

    Returns:
        The loaded run. ``PermeationRun.path`` is the bare run ID (fetched
        runs live in the ``shield_data`` per-user cache, not a project
        directory).

    Raises:
        ImportError: If the ``shield_data`` package is not installed.
    """
    sd = _shield_data()
    frame = sd.load(run_id)
    # sd.load appends a string run_id column; run_from_frame ignores it.
    metadata = sd.load_metadata(run_id)
    return run_from_frame(frame, metadata, path=Path(run_id), source=run_id)


def find_leak_test_id(catalogue: pd.DataFrame, run: PermeationRun) -> str | None:
    """Run ID of the most recent leak test recorded before ``run`` on the
    same sample, or None when there is none.

    The pairing convention for background-leak correction: leak-test runs
    (``run_type == "leak_test"``) carrying the same ``sample_id`` as ``run``,
    started before it — the latest wins. Re-running a leak test (e.g. after
    re-sealing the sample) therefore automatically supersedes the old one
    for all later permeation runs. Typical use::

        leak_id = find_leak_test_id(sd.catalogue(), run)
        leak = process_leak_test(fetch_run(leak_id)) if leak_id else None
        processed = process_run(run, leak=leak)

    Args:
        catalogue: The SHIELD-Data run catalogue (``shield_data.catalogue()``).
        run: The permeation run to find a leak test for.

    Raises:
        ValueError: If ``run`` has no ``start_time`` in its metadata (the
            leak tests cannot be ordered against it).

    Warns:
        UserWarning: When ``run`` has no ``sample_id`` and pairing falls back
            to matching substrate + coating, which cannot distinguish two
            physical specimens of the same type.
    """
    if "run_type" not in catalogue.columns:
        return None
    leak_tests = catalogue[catalogue["run_type"] == "leak_test"]
    if leak_tests.empty:
        return None

    run_info = run.metadata.get("run_info", {})
    sample_id = run_info.get("sample_id")
    if sample_id is not None and "sample_id" in leak_tests.columns:
        leak_tests = leak_tests[leak_tests["sample_id"] == sample_id]
    else:
        substrate = run_info.get(
            "sample_substrate",
            run_info.get("material", run_info.get("sample_material")),
        )
        coating = run_info.get("sample_coating", "uncoated")
        warnings.warn(
            f"Run {run.run_id} has no sample_id — pairing its leak test by "
            f"substrate={substrate!r} + coating={coating!r}, which cannot "
            "distinguish two physical specimens of the same type",
            stacklevel=2,
        )
        leak_tests = leak_tests[
            (leak_tests["substrate"] == substrate)
            & (leak_tests["coating"].fillna("uncoated") == coating)
        ]
    if leak_tests.empty:
        return None

    run_start = run.start_time
    if run_start is None:
        raise ValueError(
            f"Run {run.run_id} has no start_time in its metadata — cannot "
            "order leak tests against it"
        )
    starts = pd.to_datetime(leak_tests["start_time"])
    prior = starts[starts < pd.Timestamp(run_start)]
    if prior.empty:
        return None
    return str(leak_tests.loc[prior.idxmax(), "run_id"])


def _shield_data():
    try:
        import shield_data
    except ImportError as error:
        raise ImportError(
            "fetch_run needs the shield_data package (the SHIELD-Data sibling "
            "repo). Install it into this environment with "
            "`uv pip install -e ../SHIELD-Data` from the toolbox root, or use "
            "load_run(<run directory>) for local data."
        ) from error
    return shield_data
