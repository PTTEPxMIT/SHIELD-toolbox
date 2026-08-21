# SHIELD-toolbox

`shield_toolbox` — the analysis package for the **SHIELD** hydrogen gas-driven
permeation rig. It processes recorded runs with the time-lag method to extract
**permeability, diffusivity, and solubility** of materials and coatings
relevant to fusion engineering.

This repo is one of three that make up the SHIELD software stack. They are kept
**related but separate** — independent repos, no monorepo or submodules:

| Repo | Package | Role in the data flow |
|------|---------|-----------------------|
| [`SHIELD_DAS`](https://github.com/PTTEPxMIT/SHIELD_DAS) | `shield_das` | **Records** rig data (LabJack + live Dash UI) |
| [`SHIELD-Data`](https://github.com/PTTEPxMIT/SHIELD-Data) | `shield_data` | **Stores & serves** runs via `sd.load` / `sd.catalogue` |
| **SHIELD-toolbox** (this repo) | `shield_toolbox` | **Processes** the served data (analysis package + notebooks) |

**Data flow:** DAS records → Data stores/serves → toolbox processes.

## Quick start

From run ID to material properties in a few lines (after the
[setup](#local-development-setup) below):

```python
import shield_data as sd
from shield_toolbox import fetch_run, process_run
from shield_toolbox.plotting import plot_run_overview

sd.catalogue()                                  # what runs exist?
p = process_run(fetch_run("25.10.06_run_1_10h41"))
p.permeability                                  # Φ  (2.69±0.53)e+12 H/(m·s·Pa^0.5)
p.diffusivity_m2_per_s                          # D  1.22e-10 m²/s (time-lag method)
p.solubility                                    # S = Φ/D  (2.20±0.44)e+22 H/(m³·Pa^0.5)
plot_run_overview(p)                            # 2×2 diagnostic figure
p.write("processed_runs")                       # store the processed artifact
```

Each step is documented in the sections below; once several runs are
processed, [fit their temperature dependence](#campaign-analysis-arrhenius-fits-across-runs).
For an executable walkthrough of the whole path — catalogue → fetch →
process → overview figure → Arrhenius fit — open
[`notebooks/example_run_analysis.ipynb`](notebooks/example_run_analysis.ipynb).

## Layout

```
src/shield_toolbox/   the installable analysis package
tests/                pytest suite (synthetic fixtures only — no real run data)
scripts/              standalone processing scripts
notebooks/            example / exploratory analysis notebooks
assets/               figures, schematics, and other static files
```

## Local development setup

Clone the three repos side-by-side under one parent directory, then create an
isolated Python 3.13 environment for the toolbox that links to the local sibling
clones (so their edits are picked up with no publish step):

```bash
# from the toolbox repo root, with SHIELD-Data and SHIELD_DAS cloned alongside it
uv venv --python 3.13 .venv
uv sync                # installs shield_toolbox (editable) + dev tools
uv pip install -e ../SHIELD-Data -e ../SHIELD_DAS
source .venv/bin/activate
pre-commit install     # ruff + nbstripout hooks
```

> **Note:** a plain `uv sync` makes the venv match the lockfile *exactly*, which
> uninstalls the sibling editable installs. After the first setup, use
> `uv sync --inexact` (or re-run the `uv pip install -e ...` line after syncing).

Quick check that data access works:

```python
import shield_data as sd
print(sd.catalogue()[["run_id", "date", "furnace_setpoint"]])
```

> **macOS note:** `import shield_das` currently fails on macOS because its
> `data_recorder` module imports the `keyboard` package at load time, which
> crashes under CoreFoundation. `shield_data` is unaffected. Processing work
> that only needs the served data is fine on macOS.

## Loading run data

Every analysis starts from a `PermeationRun` — the raw recorded run (timestamps,
gauge voltages, valve events, metadata). There are two ways to get one:

**Fetch a stored run by ID** (the normal route — no manual downloads):

```python
from shield_toolbox import fetch_run

run = fetch_run("25.10.06_run_1_10h41")
```

`fetch_run` pulls the run from SHIELD-Data via the `shield_data` package
(sha256-verified, cached per-user, so each run is downloaded once). Browsing
the stored runs (`sd.catalogue()`), filtering them, and everything else about
raw data access is `shield_data`'s job — see the
[SHIELD-Data README](https://github.com/PTTEPxMIT/SHIELD-Data#quick-start)
for that.

**Load a local run directory**:

```python
from shield_toolbox import load_run

run = load_run("../SHIELD-Data/run_data/25.10.06_run_1_10h41")  # stored layout
run = load_run("results/25.10.06/run_1_10h41")                  # fresh rig output
```

`load_run` accepts both on-disk layouts — `measurements.parquet` as stored in
SHIELD-Data, and `shield_data.csv` as written by the DAS on the rig — paired
with their `run_metadata.json`. Old-generation directories
(`pressure_gauge_data.csv` [+ `thermocouple_data.csv`]) need a one-time
upgrade first:

```python
from shield_toolbox import convert_run

convert_run("old_run_dir")                    # in place
# or in bulk: uv run python scripts/convert_runs.py <dirs...> --dest converted_runs
```

However it was loaded, the resulting `PermeationRun` is identical, so
everything downstream (`process_run`, plotting) behaves the same.

## Processing a run: Φ, τ, D, S

`process_run` turns a loaded run into a `ProcessedRun`: it calibrates the
Baratron voltages to pressures, restricts analysis to the valid run window
(after the upstream gauge comes off its saturation cap), averages the stable
upstream plateau, fits the steady-state downstream rise, and extracts the
transport properties:

- **Permeability Φ** from the rise slope (Takaishi–Sensui
  thermal-transpiration corrected, uncertainty propagated), H/(m·s·Pa^0.5)
- **Time lag τ** from where the steady-state fit extrapolates back to the
  pre-breakthrough baseline, measured from the loading-valve opening
  (`v3_open_time`)
- **Diffusivity D = e²/(6τ)**, m²/s
- **Solubility S = Φ/D**, H/(m³·Pa^0.5)

```python
from shield_toolbox import fetch_run, process_run
from shield_toolbox.plotting import plot_run_overview

processed = process_run(fetch_run("25.10.06_run_1_10h41"))
print(processed.permeability)         # (2.69+/-0.53)e+12
print(processed.time_lag_s)           # 1055.9
print(processed.diffusivity_m2_per_s) # 1.22e-10
print(processed.solubility)           # (2.20+/-0.44)e+22

processed.write("processed_runs")     # <base>/<substrate>/<coating>/<run_id>/
plot_run_overview(processed)          # 2×2: upstream, downstream, T, Φ(t)
```

The sample description (substrate/coating/thickness) comes from the run
metadata automatically; the rig constants come from the versioned rig config
for the run date. `write()` stores `timeseries.parquet` (full processed time
series) and `result.json` (all scalar results + provenance). Runs where no
valid time lag exists (e.g. the fit extrapolates below the baseline) store
`null` for τ/D/S rather than a nonsense number.

Command-line equivalent for one or many runs:

```bash
uv run python scripts/process_run.py ../SHIELD-Data/run_data/25.10.06_run_1_10h41 --show
```

## Background-leak correction (leak tests)

A **leak test** (`run_type="leak_test"` in the DAS) is a short run recorded
with the sample installed and sealed, the upstream side unpressurized, and
the downstream volume isolated at a setpoint inside the 1 Torr Baratron's
range. Its downstream dP/dt is background — seal leakage plus outgassing,
not permeation — and can be subtracted from later permeation runs on the
same sample:

```python
import shield_data as sd
from shield_toolbox import fetch_run, find_leak_test_id, process_leak_test, process_run

run = fetch_run("26.09.01_run_1_10h00")

# Pair the run with its sample's most recent prior leak test (by sample_id).
leak_id = find_leak_test_id(sd.catalogue(), run)
leak = process_leak_test(fetch_run(leak_id)) if leak_id else None
print(leak.rate_torr_per_s)           # e.g. 2.1e-06 (Torr/s)

processed = process_run(run, leak=leak)   # opt-in: omit leak for uncorrected
```

The correction subtracts the leak accumulated since permeation start from
the downstream trace before fitting, so slope, permeability, time lag, and
the Φ(t) trace stay mutually consistent; the applied rate and source
leak-test run ID are stored in `result.json` (`results.leak`) and surface as
`leak_rate_torr_per_s` / `leak_test_run_id` columns in `load_results`.
Re-running a leak test (e.g. after re-sealing) automatically supersedes the
old one for all later runs. `LeakTestResult.write()` stores leak tests in
the same `<substrate>/<coating>/<run_id>/` tree; `load_results` skips them.

## Campaign analysis: Arrhenius fits across runs

Once several runs of the same sample are processed, aggregate them and fit
the temperature dependence of any extracted property:

```python
from shield_toolbox import arrhenius, load_results
from shield_toolbox.plotting import plot_arrhenius

results = load_results("processed_runs", substrate="316L steel", coating="none")
fit = arrhenius(results, quantity="permeability")   # or "diffusivity" / "solubility"
print(fit.activation_energy_J_per_mol / 1000)        # kJ/mol
print(fit.pre_exponential)

plot_arrhenius(results, fit=fit)                     # log(Φ) vs 1000/T, Ea in legend
```

`load_results` walks the `processed_runs/` tree back into one row-per-run
DataFrame (temperature, Φ, τ, D, S, with uncertainties); `arrhenius` runs an
uncertainty-weighted fit of ln(property) vs 1/T. Don't mix substrates or
coatings in one fit — filter first. CLI version:

```bash
uv run python scripts/arrhenius.py processed_runs --substrate "316L steel" --show
```

## Rig utilities: furnace logs & pump-down prediction

**Furnace-controller logs.** The Eurotherm furnace controller exports its own
logs (`LOG*.csv` / `TCCOMP*.csv`) independently of the DAS. Load them to
check heating/cooling behaviour, or to calibrate the sample-vs-furnace
temperature offset (the source of the `furnace_setpoint_offset_K = −18 K`
fallback used for old runs without a sample thermocouple):

```python
from shield_toolbox import load_furnace_log
from shield_toolbox.analysis import furnace_temperature_offset
from shield_toolbox.plotting import plot_furnace_log

furnace = load_furnace_log("Data/TCCOMP410292025182243.csv")
plot_furnace_log(furnace)                      # measured PV vs working setpoint

# With a simultaneous sample-thermocouple trace (°C, same clock):
offset = furnace_temperature_offset(sample_temp_c, furnace["furnace_temperature_C"].iloc[-1])
plot_furnace_log(furnace, sample_time_s=t_s, sample_temperature_c=sample_temp_c)
```

The offset is signed: negative means the sample runs cooler than the furnace.

**Evacuation (pump-down) prediction.** Fit a measured pressure-decay trace to
`p(t) = A·exp(−B·(t+C)) + D` and predict how long reaching a target vacuum
takes — including for a scaled-up volume (the time constant V/q grows
linearly with volume):

```python
from shield_toolbox.analysis import fit_evacuation
from shield_toolbox.plotting import plot_evacuation

fit = fit_evacuation(time_s, pressure_torr)     # times in seconds
fit.time_to_reach(3e-6)                         # seconds to 3e-6 Torr
fit.for_volume_ratio(100).time_to_reach(3e-6)   # same pump, 100× the volume
plot_evacuation(time_s, pressure_torr, fit=fit, target_torr=3e-6)
```

## Sample description (substrate + coating)

Since run-metadata v1.4 the DAS records what was mounted on the rig, and the
stored runs in SHIELD-Data have been backfilled, so every run's
`run_metadata.json` carries three fields in `run_info`:

- `sample_substrate` — substrate material, spelled out in full
  (`"carbon steel"`, `"316L steel"`, ...)
- `sample_coating_layers` — the coating as an ordered list of layers, each
  `{"material": ..., "thickness_nm": ...}` with materials spelled out in
  full (`"tungsten"`, `"silicon carbide"`, `"chromium"`, `"alumina"`);
  empty for an uncoated sample. Multi-layer stacks are simply multiple
  entries, e.g. 200nm tungsten + 50nm chromium is two layers.
- `sample_coating` — human-readable summary derived from the layers
  (`"800nm tungsten"`, `"none"` for uncoated)

The toolbox consumes these automatically: `process_run(run)` builds its
`SampleInfo` from the run's metadata via `SampleInfo.from_metadata`, which
also accepts the legacy substrate-only names (`material` in v1.0,
`sample_material` in v1.3 — no coating information). Passing
`sample=SampleInfo(...)` by hand overrides the metadata and remains the only
option for runs whose metadata predates the backfill; `process_run` raises
if the sample is neither recorded nor supplied.

`SampleInfo` carries the same structure (`substrate`, `coating`,
`coating_layers`, `thickness_m`), the processed-run layout on disk keys off
it (`<output_dir>/<substrate>/<coating>/<run_id>/`), and `result.json`
records the full description — including the per-layer breakdown — under
`sample`.

The per-run assignment table for the backfilled historical runs (which
sample was mounted when, and how it was inferred) lives in the
[SHIELD-Data README](https://github.com/PTTEPxMIT/SHIELD-Data#backfilled-sample-assignments-2026-08-11).

## Contributing

All changes go through pull requests — `main` is protected (a PR is required to
merge). Never commit directly to `main`:

```bash
git checkout main && git pull
git checkout -b <feature-branch>
# work, commit
git push -u origin <feature-branch>
gh pr create --fill
# review, then: gh pr merge --squash --delete-branch
```
