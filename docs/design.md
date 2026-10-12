# Design and delivery contract

## Problem

Estimate an individual HDB transaction's resale price from its recorded attributes
and month. This is a historical ML engineering case study, not a current valuation
service or a forecast of market-wide prices.

## Architecture

CSV -> validation -> chronological split -> fitted preprocessing and model ->
model bundle -> predictions and evaluation. CLI commands and notebooks call the
same package. No notebook imports, Colab mounts, implicit working-directory data
paths inside library functions, or shared checkpoint filenames.

The first complete slice is median/Ridge training, persistence, prediction, and
evaluation. Boosted trees follow. Optional neural models follow only after a usable
baseline portfolio exists. Each slice includes behavioural tests and documentation.

## Evaluation contract

- Train: January 2017 through December 2023.
- Validation: January through December 2024.
- Test: January through March 2025; March may be incomplete.
- Fit all learned transformations on training rows only.
- Model selection uses validation MAE in SGD. Test results never select parameters.
- Report MAE, RMSE, R-squared, row counts, runtime, and subgroup errors.
- The original work already examined 2025: describe this as a corrected historical
  holdout, not a pristine prospective evaluation.
- Preserve identical records: there is no unique transaction identifier proving
  that repeated attributes are duplicate transactions.

## Original defects

The original code fits preprocessing before splitting; uses inconsistent feature
orders at inference; treats feature columns as LSTM timesteps; selects checkpoints
by training loss while labelling them validation loss; sometimes admits future
transactions through a fallback; guesses missing encodings/scaling; and repeats
prediction code in another notebook. Its results are not a trustworthy benchmark.

## Boundaries

Use a small Python package, argparse, scikit-learn, matplotlib, and optional PyTorch.
Do not introduce a web service, database, tracking server, model registry, generic
plugin architecture, or distributed training. Keep large data and models local.
Save lightweight results and figures for review without running training.
