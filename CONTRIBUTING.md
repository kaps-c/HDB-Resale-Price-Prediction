# Working on this project

Use `python`, an isolated environment, and the documented dependency lock.
Read `docs/design.md` and `UBIQUITOUS_LANGUAGE.md` before changing behaviour.

Build one end-to-end slice at a time. Add a failing behavioural test, implement the
minimum change, then refactor. Test public interfaces rather than internal names.
Public functions use type annotations; docstrings explain contracts and decisions.

Run before committing:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

PEP 8 naming and imports apply, with an explicit 88-character line-length exception.
Keep imports and optional dependencies out of unrelated execution paths. Do not
commit datasets, model binaries, logs, or secrets. Notebook outputs should be
compact, useful, and reproducible; omit verbose training logs.
Before publishing notebook changes, run `python scripts/check_notebooks.py --full`
with the frozen snapshot and optional neural dependencies installed. Commit the
executed outputs, including data analysis and evaluation figures. The runner
rejects empty output cells; synthetic smoke runs do not replace published results.

Use Conventional Commits: `feat(scope): ...`, `fix(scope): ...`, `test(scope): ...`,
`refactor(scope): ...`, `docs(scope): ...`, or `chore(scope): ...`.
Use a PowerShell here-string for commit messages:

```powershell
$message = @'
feat(models): add ridge baseline
'@
$message | git commit --file -
```

Install optional local hooks with `python -m pre_commit install`. CI runs the same
lint and test checks plus small notebook smoke runs. Full experiments remain local.
