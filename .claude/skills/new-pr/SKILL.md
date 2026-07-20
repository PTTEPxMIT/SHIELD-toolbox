---
name: new-pr
description: Create a feature branch, commit the current work, and open a PR following this repo's workflow. Use when the user asks to open a PR, ship a feature, or push changes.
---

# Opening a PR in SHIELD-toolbox

`main` is protected — never commit to it directly. Every change lands via a
squash-merged PR.

## Steps

1. Make sure you start from fresh `main` (skip if already on a feature branch
   with the work committed):

   ```bash
   git checkout main && git pull
   git checkout -b <prefix>/<short-slug>
   ```

   Branch prefixes: `feat/` (new functionality), `fix/` (bug fixes),
   `setup/` (repo infrastructure). Keep the slug short and kebab-case.

2. Verify before committing:

   ```bash
   uv run ruff check . && uv run ruff format --check . && uv run pytest
   ```

3. Commit with a concise imperative message, then push and open the PR:

   ```bash
   git push -u origin <branch>
   gh pr create --fill
   ```

   For anything non-trivial, replace `--fill` with an explicit `--title` and
   `--body` following `.github/pull_request_template.md` (summary, test plan,
   breaking changes).

4. After approval/CI green, merge with:

   ```bash
   gh pr merge --squash --delete-branch
   ```

   Do not merge someone else's PR or a PR awaiting review without being asked.

## Notes

- One reviewable feature per PR — split unrelated changes.
- CI runs ruff + pytest; a red PR should not be merged.
