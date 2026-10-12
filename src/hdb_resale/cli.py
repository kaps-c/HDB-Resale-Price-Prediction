"""Command-line access to the same functions used in the notebooks."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from hdb_resale.experiment import (
    compare_runs,
    evaluate_experiment,
    load_bundle,
    predict,
    train_experiment,
)


def main(argv: list[str] | None = None) -> int:
    """Run one explicit operation and return a shell exit status."""
    parser = argparse.ArgumentParser(description="Reproducible HDB resale experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "train", help="Train and validate without touching test labels"
    )
    train.add_argument("--data", type=Path, required=True)
    train.add_argument(
        "--model",
        choices=["median", "ridge", "trees", "mlp", "lstm"],
        required=True,
    )
    train.add_argument("--output", type=Path, required=True, help="New run directory")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--epochs", type=int, default=20, help="Neural epoch ceiling")
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluate = commands.add_parser("evaluate", help="Evaluate a frozen model")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    inference = commands.add_parser(
        "predict", help="Predict prices for an attribute-only CSV"
    )
    inference.add_argument("--bundle", type=Path, required=True)
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser(
        "compare", help="Compare runs, ordered by validation MAE"
    )
    compare.add_argument("runs", nargs="+", type=Path)
    compare.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "train":
            result = train_experiment(
                args.data,
                args.model,
                args.output,
                seed=args.seed,
                threads=args.threads,
                epochs=args.epochs,
                device=args.device,
            )
        elif args.command == "evaluate":
            result = evaluate_experiment(args.bundle, args.data, split=args.split)
        elif args.command == "predict":
            frame = pd.read_csv(args.input)
            values = predict(load_bundle(args.bundle), frame)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="") as stream:
                pd.DataFrame({"predicted_price_sgd": values}).to_csv(
                    stream, index=False
                )
            result = {"rows": len(values), "output": str(args.output)}
        else:
            table = compare_runs(args.runs)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8", newline="") as stream:
                    table.to_csv(stream, index=False)
            print(table.to_csv(index=False), end="")
            return 0
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
