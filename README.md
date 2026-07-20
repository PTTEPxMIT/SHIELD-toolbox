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
