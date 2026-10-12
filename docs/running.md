# Running and reproducing the experiments

Run all commands from the repository root. Published scores use the frozen snapshot.

## Run locally

Verified with Python 3.13.5 on Windows. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation -e .
python scripts/restore_snapshot.py
python -m hdb_resale train --data data/raw/resale.csv --model median --output artifacts/median
python -m hdb_resale train --data data/raw/resale.csv --model ridge --output artifacts/ridge
python -m hdb_resale evaluate --data data/raw/resale.csv --bundle artifacts/median
python -m hdb_resale evaluate --data data/raw/resale.csv --bundle artifacts/ridge
python -m hdb_resale compare artifacts/median artifacts/ridge
python -m hdb_resale predict --bundle artifacts/ridge --input examples/flats.csv --output artifacts/predictions.csv
```

Add the boosted-tree experiment and refresh the comparison:

```powershell
python -m hdb_resale train --data data/raw/resale.csv --model trees --output artifacts/trees
python -m hdb_resale evaluate --data data/raw/resale.csv --bundle artifacts/trees
python scripts/export_reports.py artifacts/median artifacts/ridge artifacts/trees
```

Optional neural experiments (install the separate CPU dependency lock first):

```powershell
python -m pip install -r requirements-neural.txt
python -m hdb_resale train --data data/raw/resale.csv --model mlp --output artifacts/mlp --epochs 20 --threads 2 --device cpu
python -m hdb_resale train --data data/raw/resale.csv --model lstm --output artifacts/lstm --epochs 5 --threads 2 --device cpu
python -m hdb_resale evaluate --data data/raw/resale.csv --bundle artifacts/mlp
python -m hdb_resale evaluate --data data/raw/resale.csv --bundle artifacts/lstm
python scripts/export_reports.py artifacts/median artifacts/ridge artifacts/trees artifacts/mlp artifacts/lstm
python scripts/check_notebooks.py --full
```

The two neural training commands can run in separate terminals concurrently:
each uses two CPU threads and an independent output directory. Run at most two
experiments concurrently and only one on CUDA. A compatible CUDA PyTorch build
can use `--device cuda`; CUDA was not tested in this case study. The locked
environment uses CPU PyTorch. Runtime comparisons depend on hardware and contention.
The default notebook smoke check skips optional neural notebooks when PyTorch is
absent; CI additionally checks them in a separate CPU neural job.

On Linux/macOS, activate with `source .venv/bin/activate`. Use a new output
directory when retraining: existing runs are never overwritten. The example
input contains synthetic flats to demonstrate the interface, not benchmark rows.
Open notebooks with the `.venv` interpreter in your notebook editor. The development
lock includes the kernel and execution tools, not a Jupyter web server.

The snapshot restoration script reads the exact CSV from the original Git commit
and verifies its SHA-256. It requires full Git history containing `e45aff1`.
If using a source ZIP or shallow clone, obtain the documented snapshot separately
or use `tests/fixtures/transactions.csv` for a synthetic smoke run. Do not substitute
a newer download and claim the same results. See [data notes](../data/README.md).

## Methodology

- Numeric: transaction month, floor area, lease commencement year, remaining lease
  in months, and lower/upper storey bounds.
- Categorical: town, flat type, street, and flat model. Block is excluded initially.
- Ridge: scaled numeric inputs, one-hot categories, `alpha=1`, `solver="lsqr"`.
- Trees: 200 boosting iterations, 31 leaves, learning rate 0.1, L2 penalty 1;
  native categories with training-fitted rare-category grouping. Internal random
  validation is disabled; this fixed configuration was not tuned on the test set.
- MLP: 128/64 hidden units, ReLU, dropout 0.1, Adam at 0.001, batch size 1024.
  Maximum 20 epochs and patience 5; selected epoch 9, stopped after epoch 14.
- LSTM: four layers, 64 hidden units, dropout 0.2, five-epoch budget. Ten feature
  positions act as timesteps; this is not transaction-history forecasting. Unlike
  the MLP, it retains ordinal categorical inputs to test the original compact
  architecture. Both neural models standardise targets using training prices and
  select checkpoints by validation MAE. Weights are separate CPU-portable files.
- Preprocessing is fitted only on training rows. Unknown categories are supported;
  malformed required fields raise errors instead of fabricated defaults.
- Identical recorded rows remain: no transaction identifier establishes duplication.
- Model bundles store configuration, preprocessing, feature order, versions, seed,
  dataset fingerprint, split dates, and validation metrics. Load only trusted bundles.
- Subgroup MAE/RMSE/R² and counts are saved for month, town, flat type, and storey.

## Checks

```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python scripts/check_notebooks.py
```

CI is configured for checks and synthetic notebook execution on Windows and Linux;
optional neural checks run on Linux. Local verification used Windows CPU.
Full-data training is local. See [contributing](../CONTRIBUTING.md) for test-first development,
PEP 8 conventions, and Conventional Commits.


## Publishing notebooks

Run `python scripts/check_notebooks.py --full` before committing notebooks.
This executes all six against the real snapshot, reuses completed bundles, and
saves output cells. Full publication requires PyTorch; the synthetic smoke check
may skip optional neural notebooks. Every code cell must produce output.
Figures and analysis tables are visible in the saved notebooks.
