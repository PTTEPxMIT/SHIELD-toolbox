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

from pathlib import Path

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
