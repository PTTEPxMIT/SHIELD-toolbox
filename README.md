# SHIELD-toolbox

Functions, scripts, and notebooks for processing data produced by the **SHIELD**
gas-driven permeation rig.

This repo is one of three that make up the SHIELD software stack. They are kept
**related but separate** — independent repos, no monorepo or submodules:

| Repo | Package | Role in the data flow |
|------|---------|-----------------------|
| [`SHIELD_DAS`](https://github.com/PTTEPxMIT/SHIELD_DAS) | `shield_das` | **Records** rig data (LabJack + live Dash UI) |
| [`SHIELD-Data`](https://github.com/PTTEPxMIT/SHIELD-Data) | `shield_data` | **Stores & serves** runs via `sd.load` / `sd.catalogue` |
| **SHIELD-toolbox** (this repo) | — | **Processes** the served data (scripts + notebooks) |

**Data flow:** DAS records → Data stores/serves → toolbox processes.

## Layout

```
scripts/     standalone processing scripts
notebooks/   exploratory / analysis notebooks
assets/      figures, schematics, and other static files
```

## Local development setup

Clone the three repos side-by-side under one parent directory, then create an
isolated Python 3.13 environment for the toolbox that links to the local sibling
clones (so their edits are picked up with no publish step):

```bash
# from the toolbox repo root, with SHIELD-Data and SHIELD_DAS cloned alongside it
uv venv --python 3.13 .venv
uv pip install --python .venv -e ../SHIELD-Data -e ../SHIELD_DAS
uv pip install --python .venv -r requirements-dev.txt
source .venv/bin/activate
```

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
