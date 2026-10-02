# SHIELD-toolbox

`shield_toolbox` — the analysis package for the **SHIELD** hydrogen gas-driven
permeation rig. It processes recorded runs with the background-subtracted
time-lag method to extract
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

sd.catalogue()  # what runs exist?
p = process_run(fetch_run("26.09.25_run_1_17h59"))
p.permeability  # Φ  (9.5±1.1)e+11 H/(m·s·Pa^0.5)
p.diffusivity_m2_per_s  # D  2.47e-11 m²/s (time-lag method)
p.solubility  # S = Φ/D  (3.84±0.45)e+22 H/(m³·Pa^0.5)
plot_run_overview(p)  # 2×2: the four steps of the method
p.write("processed_runs")  # store the processed artifact
```

Each step is documented in the sections below; once several runs are
processed, [fit their temperature dependence](#campaign-analysis-arrhenius-fits-across-runs).
For an executable walkthrough of the whole path — catalogue → fetch →
process → step-by-step figures → sensitivity checks → Arrhenius fit — open
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

`process_run` turns a loaded run into a `ProcessedRun` with the
background-subtracted time-lag method. It calibrates the Baratron voltages to
pressures, restricts analysis to the valid run window (both Baratrons off
their saturation caps), and then:

1. **Noise recording.** Before hydrogen has crossed the sample, the sealed
   downstream volume sees only the background (seal leakage plus outgassing).
   The noise recording runs from 1 min after `t_init` to a quarter of the
   pre-rise time before the detected onset of the downstream rise (i.e. to
   0.75 × onset; the early flux builds gradually). The onset is the first
   10 min block more than 5σ above the background line fitted before it; a
   1 min block search takes over when the rise starts within minutes (high
   temperature). The search covers the whole run; if no onset is detected,
   the noise recording is the first 30 min after it starts.
2. **Initial time.** `t_init` is the first sample where the upstream passes
   half its plateau.
3. **Background fit.** A straight line `a + b·(t − t_init)` through the noise
   recording.
4. **Time lag on filtered data.** The background line is subtracted, so the
   filtered signal starts at 0 at `t_init`. The steady-state line
   `S∞·(t − t_init − τ_L)` is fitted from 3 τ_L (iterated) to the end of
   usable data: the first 30 h after `t_init`, downstream below 0.95 Torr,
   upstream pressurised. If the next 3 τ_L start would lie past the data,
   the iteration keeps its last fit (`steady_state.converged` is False).

`process_run` always returns a fit and never decides a run is unusable;
judging whether the rise is at steady state is up to you (the step plots
and `converged` flag help).

From the fit:

- **Permeability Φ** from `S∞` (Takaishi–Sensui thermal-transpiration
  corrected at the last downstream pressure in the window, mean sample
  temperature over the window, upstream Baratron mean over the window minus
  its pre-start bias — the reading before the step — just as the downstream
  background is removed; uncertainty propagated), H/(m·s·Pa^0.5)
- **Time lag τ_L**, the steady-state line's zero crossing after `t_init`
- **Diffusivity D = e²/(6τ_L)**, m²/s
- **Solubility S = Φ/D**, H/(m³·Pa^0.5)

```python
from shield_toolbox import fetch_run, process_run
from shield_toolbox.plotting import plot_run_overview

processed = process_run(fetch_run("26.09.25_run_1_17h59"))
print(processed.initial_time_s)  # 63.8 (s into the recording)
print(processed.noise.end_s / 60)  # 21.0 (noise recording ends, min after t_init)
print(processed.background.slope)  # (1.602+/-0.013)e-04 Pa/s
print(processed.steady_state.slope)  # 3.12e-03 Pa/s
print(processed.time_lag_s / 3600)  # 1.80 h
print(processed.permeability)  # (9.5+/-1.1)e+11, measured upstream Baratron P_up

processed.write("processed_runs")  # <base>/<substrate>/<coating>/<run_id>/
plot_run_overview(processed)  # 2×2: steps 1–2, step 3, step 4, residuals
```

Every setting of the method is a keyword of `process_run` (the fields of
`TimeLagSettings`), stored in `result.json`:

| Keyword | Default | Meaning |
|---------|---------|---------|
| `upstream_pressure_torr` | measured | P_up in Φ: the upstream Baratron mean over the steady-state window minus its pre-start bias (median of the last 60 s before the step); a number overrides it |
| `analysis_hours` | 30 | analyse the first N h after `t_init` (None = whole run) |
| `steady_state_start_taus` | 3 | steady-state window starts at N·τ_L |
| `steady_state_start_s` | — | fixed steady-state window start, s after `t_init` (no iteration) |
| `steady_state_end_s` | last usable | steady-state window end, s after `t_init` |
| `noise_start_s` | 60 | noise recording start, s after `t_init` |
| `noise_margin_fraction` | 0.25 | noise recording ends this fraction of the pre-rise time before the onset |
| `onset_sigma` | 5 | onset detection threshold (standard errors) |
| `noise_end_s` | detected | manual end of the noise recording |
| `background_slope_pa_per_s` | fitted | override `b` (sensitivity checks) |
| `downstream_max_torr` | 0.95 | exclude downstream readings at or above this |

`processed.refit(**settings)` re-runs the analysis on the stored time series
with some settings changed — the example notebook uses it for its window and
background-slope sensitivity tables. The plots in `shield_toolbox.plotting`
draw each step (`plot_initial_time`, `plot_background`, `plot_steady_state`,
`plot_residuals`) and overlay runs (`plot_filtered_rises`).

The sample description (substrate/coating/thickness) comes from the run
metadata automatically; the rig constants come from the versioned rig config
for the run date. `write()` stores `timeseries.parquet` (full processed time
series, including the filtered signal and the noise-recording and
steady-state masks) and `result.json` (all scalar results, settings and
provenance). If the steady-state line crosses zero before `t_init`, τ/D/S are
stored as `null` rather than a nonsense number.

Command-line equivalent for one or many runs (run IDs or local run
directories):

```bash
uv run python scripts/process_run.py 26.09.25_run_1_17h59 26.09.28_run_1_18h50 \
    --save-plots figures
```

## Leak tests

A **leak test** (`run_type="leak_test"` in the DAS) is a short run recorded
with the sample installed and sealed, the upstream side unpressurized, and
the downstream volume isolated at a setpoint inside the 1 Torr Baratron's
range. Its downstream dP/dt is the background of the sealed assembly — seal
leakage plus outgassing. It is a standalone diagnostic: it is not applied to
any permeation run.

```python
from shield_toolbox import fetch_run, process_leak_test

leak = process_leak_test(fetch_run("26.09.21_run_1_15h25"))
print(leak.rate_torr_per_s)  # e.g. 4.2e-07 (Torr/s)
```

`LeakTestResult.write()` stores leak tests in the same
`<substrate>/<coating>/<run_id>/` tree; `load_results` skips them.

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
