"""Train, persist, evaluate, and predict through one shared interface."""

import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from hdb_resale.data import FEATURES, TARGET, load_data, split_data, validate_data
from hdb_resale.evaluation import regression_metrics, subgroup_metrics
from hdb_resale.models import build_model


def file_fingerprint(path: str | Path) -> str:
    """Identify the exact input snapshot."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Write portable metadata with no NaN or Infinity values."""
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def train_experiment(
    data_path: str | Path,
    model_name: str,
    output_dir: str | Path,
    *,
    seed: int = 42,
    threads: int = 4,
    epochs: int = 20,
    device: str = "auto",
) -> dict:
    """Fit on training rows and report validation only; never overwrite a run."""
    if threads < 1:
        raise ValueError("threads must be positive")
    splits = split_data(load_data(data_path))
    model = build_model(model_name, seed, epochs=epochs, device=device, threads=threads)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    train, validation = splits["train"], splits["validation"]
    started = time.perf_counter()
    with threadpool_limits(limits=threads):
        if model_name in {"mlp", "lstm"}:
            x_train = model[:-1].fit_transform(train.drop(columns=TARGET))
            x_validation = model[:-1].transform(validation.drop(columns=TARGET))
            model.named_steps["model"].fit(
                x_train,
                train[TARGET].to_numpy(),
                validation_data=(x_validation, validation[TARGET].to_numpy()),
            )
        else:
            model.fit(train.drop(columns=TARGET), train[TARGET])
        fit_seconds = time.perf_counter() - started
        predicted = np.asarray(model.predict(validation.drop(columns=TARGET)))
    metadata = {
        "format_version": 1,
        "model": model_name,
        "seed": seed,
        "data_sha256": file_fingerprint(data_path),
        "features": FEATURES,
        "split_dates": {
            "train_end": "2023-12",
            "validation_end": "2024-12",
            "test_end": "2025-03",
        },
        "splits": {key: len(value) for key, value in splits.items()},
        "configuration": {
            key: value
            for key, value in model.named_steps["model"].get_params().items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "python": platform.python_version(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "joblib"]
        },
        "fit_seconds": fit_seconds,
        "threads": threads,
        "validation": regression_metrics(validation[TARGET], predicted),
    }
    bundle = {"model": model, "metadata": metadata}
    if model_name in {"mlp", "lstm"}:
        import torch

        estimator = model.named_steps["model"]
        metadata["dependencies"]["torch"] = torch.__version__
        metadata["training"] = {
            "epochs_completed": len(estimator.history_),
            "best_epoch": estimator.best_epoch_,
            "device": estimator.device_used_,
        }
        pd.DataFrame(estimator.history_).to_csv(output / "history.csv", index=False)
        torch.save(estimator.weights_, output / "weights.pt")
        estimator.weights_ = None
    joblib.dump(bundle, output / "model.joblib")
    write_json(output / "metadata.json", metadata)
    subgroup_metrics(validation, predicted).to_csv(
        output / "validation_groups.csv", index=False
    )
    return metadata


def load_bundle(path: str | Path) -> dict:
    """Load a trusted local bundle. Joblib files must not come from strangers."""
    path = Path(path)
    bundle = joblib.load(path / "model.joblib" if path.is_dir() else path)
    if bundle.get("metadata", {}).get("format_version") != 1:
        raise ValueError("Unsupported model bundle format")
    if bundle["metadata"]["features"] != FEATURES:
        raise ValueError("Model feature contract does not match this package")
    if bundle["metadata"]["model"] in {"mlp", "lstm"}:
        import torch

        directory = path if path.is_dir() else path.parent
        bundle["model"].named_steps["model"].weights_ = torch.load(
            directory / "weights.pt",
            map_location="cpu",
            weights_only=True,
        )
    return bundle


def predict(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Predict SGD prices in input order; labels are neither required nor used."""
    features = validate_data(frame, require_target=False).drop(
        columns=TARGET, errors="ignore"
    )
    with threadpool_limits(limits=4):
        values = np.asarray(bundle["model"].predict(features), dtype=float).reshape(-1)
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise ValueError("Model returned invalid predictions")
    return values


def evaluate_experiment(
    bundle_dir: str | Path,
    data_path: str | Path,
    *,
    split: str = "test",
) -> dict:
    """Evaluate a frozen model on its original snapshot and save compact results."""
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test")
    bundle = load_bundle(bundle_dir)
    if file_fingerprint(data_path) != bundle["metadata"]["data_sha256"]:
        raise ValueError("Dataset fingerprint differs from the training snapshot")
    frame = split_data(load_data(data_path))[split]
    started = time.perf_counter()
    predicted = predict(bundle, frame)
    report = {
        "model": bundle["metadata"]["model"],
        "split": split,
        "data_sha256": bundle["metadata"]["data_sha256"],
        **regression_metrics(frame[TARGET], predicted),
        "predict_seconds": time.perf_counter() - started,
    }
    output = Path(bundle_dir)
    write_json(output / f"{split}.json", report)
    subgroup_metrics(frame, predicted).to_csv(
        output / f"{split}_groups.csv", index=False
    )
    predictions = frame[["month", "town", "flat_type", "storey_range", TARGET]].copy()
    predictions["predicted_price"] = predicted
    predictions.to_csv(output / f"{split}_predictions.csv", index=False)
    return report


def compare_runs(run_dirs: list[str | Path]) -> pd.DataFrame:
    """Compare runs on identical data; rank by validation MAE, never test MAE."""
    rows = []
    fingerprints = set()
    for directory in run_dirs:
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        fingerprints.add(metadata["data_sha256"])
        row = {
            "model": metadata["model"],
            "validation_mae_sgd": metadata["validation"]["mae_sgd"],
            "fit_seconds": metadata["fit_seconds"],
        }
        test_path = directory / "test.json"
        if test_path.exists():
            test = json.loads(test_path.read_text(encoding="utf-8"))
            row.update(
                {
                    f"test_{key}": test[key]
                    for key in ["count", "mae_sgd", "rmse_sgd", "r2"]
                }
            )
        rows.append(row)
    if not rows or len(fingerprints) != 1:
        raise ValueError("Comparison requires runs using one identical dataset")
    return pd.DataFrame(rows).sort_values("validation_mae_sgd").reset_index(drop=True)
