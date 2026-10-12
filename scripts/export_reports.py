"""Export compact measured results and a figure without model binaries."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from hdb_resale.experiment import compare_runs
from hdb_resale.plots import plot_comparison


def main() -> None:
    """Export only explicitly selected, comparable experiment directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    table = compare_runs(args.runs)
    if table["model"].duplicated().any():
        raise ValueError("Export one selected run per model")
    for run in args.runs:
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        destination = args.output / metadata["model"]
        destination.mkdir(parents=True, exist_ok=True)
        for filename in [
            "metadata.json",
            "test.json",
            "validation_groups.csv",
            "test_groups.csv",
            "history.csv",
        ]:
            source = run / filename
            if source.exists():
                shutil.copyfile(source, destination / filename)
    table.to_csv(args.output / "comparison.csv", index=False)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)
    figure = plot_comparison(table)
    figure.savefig(figures / "comparison.png", dpi=160)
    plt.close(figure)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
