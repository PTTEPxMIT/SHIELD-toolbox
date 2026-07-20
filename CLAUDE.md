# SHIELD-toolbox

`shield_toolbox` is the analysis package for the SHIELD hydrogen gas-driven
permeation rig. It processes recorded runs with the time-lag method to extract
permeability, diffusivity, and solubility of fusion-relevant materials and
coatings.

One of three sibling repos, cloned side-by-side, deliberately separate
(no monorepo, no submodules):

- `SHIELD_DAS` (`shield_das`) — **records** rig data
- `SHIELD-Data` (`shield_data`) — **stores & serves** runs (`sd.load` / `sd.catalogue`)
- `SHIELD-toolbox` (this repo, `shield_toolbox`) — **processes** the data

## Commands

- Setup: `uv venv --python 3.13 .venv && uv sync`, then sibling editable
  installs: `uv pip install -e ../SHIELD-Data -e ../SHIELD_DAS`
  (siblings are NOT pyproject dependencies — they aren't on PyPI)
- After initial setup use `uv sync --inexact` — a plain `uv sync` prunes the
  venv to the lockfile and uninstalls the sibling editable installs
- Test: `uv run pytest` (single test: `uv run pytest tests/test_x.py::test_name`)
- Lint/format: `uv run ruff check --fix . && uv run ruff format .`
- Pre-commit: `uv run pre-commit run --all-files`

## Git workflow (strict)

- `main` is protected — NEVER commit to main. Branch, push,
  `gh pr create --fill`, then squash-merge (`gh pr merge --squash --delete-branch`).
- Branch names: `setup/*`, `feat/*`, `fix/*`. One reviewable feature per PR.

## Architecture

- src/ layout, package `shield_toolbox` (installed as `shield-toolbox`).
- `config.py` — versioned rig configurations as frozen dataclasses
  (`SHIELD_V1`, `SHIELD_V2`, `get_rig_config`, `get_rig_config_for_date`).
  The rig is being upgraded: downstream volume and sample area WILL change.
  NEVER hard-code rig constants anywhere else — always go through the config.
- `constants.py` — universal physical constants only (R, k_B, N_A, unit
  conversions), nothing rig-specific.
- `gauges.py` — voltage→pressure/temperature calibrations (Baratron 626D,
  WGM701, Type-K thermocouple).
- `io/` — loads the 3 raw-data format generations behind one
  `load_run(path) → PermeationRun`; `PermeationRun` is the central container
  and holds no analysis logic.
- `analysis/` — pure physics functions: no file I/O, no plotting.
- `plotting/` — matplotlib only; every function accepts/returns `ax` and never
  calls `plt.show()` or `savefig` (saving is the caller's job).
- Uncertainties propagate via the `uncertainties` package (`ufloat`) in
  derived quantities.

## Gotchas

- `import shield_das` CRASHES on macOS (its `keyboard` dependency fails under
  CoreFoundation). `shield_data` works fine on macOS.
- Raw run data lives OUTSIDE the repo (`~/Documents/PTTEP/Results/MM.DD/run_N_HHhMM/`)
  and is gitignored — tests use small synthetic fixtures in `tests/fixtures/`,
  never real runs.
- Data format generations: newest runs have a combined `shield_data.csv`;
  older runs have `pressure_gauge_data.csv` + `thermocouple_data.csv`; the
  oldest have no thermocouple file (temperature falls back to
  `furnace_setpoint − 18 K` — that −18 K offset applies to old runs only).
- CVM211 gauge calibration is intentionally `NotImplementedError`.
- Notebook outputs are stripped by the nbstripout pre-commit hook.

## Physics cheat-sheet (context — don't re-derive)

- Time lag τ from the x-intercept of the downstream pressure tail asymptote;
  diffusivity D = e²/(6τ) for sample thickness e.
- Steady-state flux J = (dP/dt)·V/(R·T·A)·N_A (V = downstream volume,
  A = sample area).
- Permeability Φ = J·e/√P_up (thermal-transpiration corrected variants:
  Takaishi–Sensui and free-molecular-flow).
- Solubility S = Φ/D. Arrhenius fits: ln(property) vs 1/T.
