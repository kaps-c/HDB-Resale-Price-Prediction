"""Validate raw transactions and construct deterministic, target-free features."""

from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL = ["town", "flat_type", "street_name", "flat_model"]
NUMERIC = [
    "month_index",
    "floor_area_sqm",
    "lease_commence_date",
    "lease_months",
    "storey_lower",
    "storey_upper",
]
FEATURES = NUMERIC + CATEGORICAL
REQUIRED = [
    "month",
    "town",
    "flat_type",
    "street_name",
    "storey_range",
    "floor_area_sqm",
    "flat_model",
    "lease_commence_date",
    "remaining_lease",
]
TARGET = "resale_price"


def validate_data(frame: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    """Return a validated copy; fail rather than silently dropping invalid rows."""
    required = REQUIRED + ([TARGET] if require_target else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Input must contain at least one transaction")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported")
    result = frame.copy(deep=True)
    for column in required:
        if result[column].isna().any():
            raise ValueError(f"{column}: missing values are not supported")
    if not pd.api.types.is_datetime64_any_dtype(result["month"]):
        if not result["month"].astype(str).str.fullmatch(r"\d{4}-\d{2}").all():
            raise ValueError("month: expected YYYY-MM")
    try:
        result["month"] = pd.to_datetime(result["month"], format="%Y-%m")
    except (ValueError, TypeError) as error:
        raise ValueError("month: expected a valid YYYY-MM date") from error
    if result["month"].dt.tz is not None or (result["month"].dt.day != 1).any():
        raise ValueError("month: expected timezone-free month-start dates")
    for column in CATEGORICAL:
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError(f"{column}: empty categories are not supported")
    numeric = ["floor_area_sqm", "lease_commence_date"]
    if require_target:
        numeric.append(TARGET)
    for column in numeric:
        values = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{column}: expected finite positive numbers")
        result[column] = values
    lease_year = result["lease_commence_date"]
    if (
        (lease_year % 1 != 0)
        | (lease_year < 1900)
        | (lease_year > result["month"].dt.year)
    ).any():
        raise ValueError(
            "lease_commence_date: expected a year no later than transaction"
        )
    lease = (
        result["remaining_lease"]
        .astype(str)
        .str.extract(r"^(\d+) years?(?: (\d{1,2}) months?)?$")
    )
    years = pd.to_numeric(lease[0], errors="coerce")
    months = pd.to_numeric(lease[1].fillna("0"), errors="coerce")
    total = years * 12 + months
    if years.isna().any() or ((months > 11) | (total <= 0) | (total > 1188)).any():
        raise ValueError("remaining_lease: expected years and optional months (0-11)")
    storeys = result["storey_range"].astype(str).str.extract(r"^(\d{2}) TO (\d{2})$")
    lower = pd.to_numeric(storeys[0], errors="coerce")
    upper = pd.to_numeric(storeys[1], errors="coerce")
    if lower.isna().any() or ((lower < 1) | (upper < lower)).any():
        raise ValueError("storey_range: expected ascending bounds such as 01 TO 03")
    return result


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a local CSV without discarding repeated transaction rows."""
    return validate_data(pd.read_csv(path, dtype={"block": str}))


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features in fixed order without consulting the target column."""
    data = validate_data(frame, require_target=False)
    result = data[CATEGORICAL + ["floor_area_sqm", "lease_commence_date"]].copy()
    result["month_index"] = (
        (data["month"].dt.year - 2000) * 12 + data["month"].dt.month - 1
    )
    lease = (
        data["remaining_lease"]
        .astype(str)
        .str.extract(r"^(\d+) years?(?: (\d{1,2}) months?)?$")
    )
    result["lease_months"] = lease[0].astype(int) * 12 + lease[1].fillna(0).astype(int)
    bounds = data["storey_range"].str.extract(r"^(\d{2}) TO (\d{2})$").astype(int)
    result["storey_lower"] = bounds[0]
    result["storey_upper"] = bounds[1]
    return result[FEATURES]


def split_data(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fixed historical splits; reject rows outside the frozen snapshot period."""
    data = validate_data(frame)
    dates = data["month"]
    if ((dates < "2017-01-01") | (dates >= "2025-04-01")).any():
        raise ValueError(
            "month: experiments require the 2017-01 through 2025-03 snapshot"
        )
    splits = {
        "train": data.loc[dates < "2024-01-01"].copy(),
        "validation": data.loc[(dates >= "2024-01-01") & (dates < "2025-01-01")].copy(),
        "test": data.loc[dates >= "2025-01-01"].copy(),
    }
    if any(part.empty for part in splits.values()):
        raise ValueError(
            "Each chronological split must contain at least one transaction"
        )
    return splits
