# Frozen data snapshot

The user-supplied CSV contains 202,399 transactions from January 2017 through
March 2025. Its original filename was
`ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv`.
Original retrieval date and March completeness are unknown. The repository does
not claim a verified original download URL or grant a new licence to this data.

`snapshot.json` records the exact SHA-256 and coverage. The two original CSVs are
byte-identical. The XLSX is archived, not treated as a verified equivalent.
There are 298 repeated full rows; these are retained because the dataset lacks
unique transaction IDs.

Run `python scripts/restore_snapshot.py` from a full clone to restore and verify
`data/raw/resale.csv` from commit `e45aff1`. Existing files with another fingerprint
are rejected. The original data remains in Git history; ignoring it now does not
remove it from previous commits.

For tests and quick smoke runs, `tests/fixtures/transactions.csv` contains clearly
synthetic transactions spanning all three splits. Do not report fixture metrics as
real-world model performance.
