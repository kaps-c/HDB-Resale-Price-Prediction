import pandas as pd
import pytest

from hdb_resale.data import FEATURES, make_features, split_data, validate_data


def test_features_parse_hdb_fields_without_mutation():
    row = pd.DataFrame(
        [
            {
                "month": "2024-01",
                "town": "GEYLANG",
                "flat_type": "3 ROOM",
                "street_name": "CIRCUIT RD",
                "storey_range": "01 TO 03",
                "floor_area_sqm": 56,
                "flat_model": "Standard",
                "lease_commence_date": 1969,
                "remaining_lease": "43 years 01 month",
                "resale_price": 358000,
            }
        ]
    )
    original = row.copy(deep=True)
    features = make_features(row)
    assert list(features.columns) == FEATURES
    assert features.loc[0, "lease_months"] == 517
    assert features.loc[0, "storey_lower"] == 1
    assert features.loc[0, "storey_upper"] == 3
    assert "resale_price" not in features
    pd.testing.assert_frame_equal(row, original)


def test_chronological_splits_preserve_rows_and_exclude_future():
    from conftest import transaction_frame

    frame = transaction_frame()
    splits = split_data(frame)
    assert len(splits["train"]) + len(splits["validation"]) + len(
        splits["test"]
    ) == len(frame)
    assert splits["train"]["month"].max() < splits["validation"]["month"].min()
    assert splits["validation"]["month"].max() < splits["test"]["month"].min()
    assert splits["train"]["month"].max().year == 2023


@pytest.mark.parametrize(
    "column,value",
    [
        ("month", "2024-13"),
        ("floor_area_sqm", -1),
        ("remaining_lease", "43 years 15 months"),
        ("storey_range", "09 TO 03"),
        ("town", ""),
        ("resale_price", float("inf")),
    ],
)
def test_invalid_inputs_are_rejected(column, value):
    from conftest import transaction_frame

    frame = transaction_frame().astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        validate_data(frame)
