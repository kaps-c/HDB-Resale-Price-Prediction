"""Metrics in SGD and subgroup summaries with sample counts."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute errors; R-squared is undefined for singleton or constant groups."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim != 1 or actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Expected nonempty, matching one-dimensional price arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Prices and predictions must be finite")
    return {
        "count": int(actual.size),
        "mae_sgd": float(mean_absolute_error(actual, predicted)),
        "rmse_sgd": float(root_mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted))
        if len(np.unique(actual)) > 1
        else None,
    }


def subgroup_metrics(frame: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    """Summarise errors by month, town, flat type, and storey range."""
    data = frame.copy()
    data["predicted_price"] = predicted
    data["month"] = pd.to_datetime(data["month"]).dt.strftime("%Y-%m")
    records = []
    for dimension in ["month", "town", "flat_type", "storey_range"]:
        for group, subset in data.groupby(dimension, observed=True):
            records.append(
                {
                    "dimension": dimension,
                    "group": group,
                    **regression_metrics(
                        subset["resale_price"], subset["predicted_price"]
                    ),
                }
            )
    return pd.DataFrame(records)
