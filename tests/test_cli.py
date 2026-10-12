import json
import subprocess
import sys

import pandas as pd
from conftest import transaction_frame


def test_cli_train_evaluate_predict_and_compare(tmp_path):
    source = tmp_path / "data.csv"
    transaction_frame().to_csv(source, index=False)
    run = tmp_path / "run"

    def command(*args):
        return subprocess.run(
            [sys.executable, "-m", "hdb_resale", *map(str, args)],
            capture_output=True,
            text=True,
            check=True,
        )

    result = command("train", "--data", source, "--model", "ridge", "--output", run)
    assert json.loads(result.stdout)["model"] == "ridge"
    result = command("evaluate", "--data", source, "--bundle", run)
    assert json.loads(result.stdout)["count"] == 24
    sample = tmp_path / "sample.csv"
    transaction_frame().tail(3).drop(columns="resale_price").to_csv(sample, index=False)
    predictions = tmp_path / "predictions.csv"
    command("predict", "--bundle", run, "--input", sample, "--output", predictions)
    assert len(pd.read_csv(predictions)) == 3
    result = command("compare", run)
    assert "test_mae_sgd" in result.stdout
    collision = subprocess.run(
        [
            sys.executable,
            "-m",
            "hdb_resale",
            "train",
            "--data",
            str(source),
            "--model",
            "ridge",
            "--output",
            str(run),
        ],
        capture_output=True,
        text=True,
    )
    assert collision.returncode != 0
    assert "Error:" in collision.stderr
