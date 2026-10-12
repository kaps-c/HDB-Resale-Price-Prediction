import numpy as np
import pandas as pd
from conftest import transaction_frame

from hdb_resale.experiment import load_bundle, predict, train_experiment


def test_baseline_roundtrip_is_target_free_and_order_independent(tmp_path):
    data = transaction_frame()
    csv = tmp_path / "input.csv"
    data.to_csv(csv, index=False)
    output = tmp_path / "ridge"
    report = train_experiment(csv, "ridge", output)
    bundle = load_bundle(output)
    sample = data.tail(4).drop(columns="resale_price")
    expected = predict(bundle, sample)
    actual = predict(load_bundle(output), sample[sample.columns[::-1]])
    np.testing.assert_allclose(actual, expected)
    assert np.isfinite(actual).all()
    assert report["splits"]["train"] == 48
    assert report["validation"]["count"] == 24
    assert "test" not in report
    assert (output / "metadata.json").exists()


def test_median_and_preprocessing_only_learn_training_rows(tmp_path):
    data = transaction_frame()
    later = data["month"] >= "2024-01"
    data.loc[later, "floor_area_sqm"] = 9999
    data.loc[later, "town"] = "UNSEEN TOWN"
    data.loc[later, "resale_price"] = 9999999
    csv = tmp_path / "input.csv"
    data.to_csv(csv, index=False)
    train_experiment(csv, "median", tmp_path / "median")
    sample = data.tail(2).drop(columns="resale_price")
    values = predict(load_bundle(tmp_path / "median"), sample)
    np.testing.assert_allclose(values, data.loc[~later, "resale_price"].median())
    train_experiment(csv, "ridge", tmp_path / "ridge")
    ridge = load_bundle(tmp_path / "ridge")
    # The extreme future values must not affect predictions on training inputs.
    original = transaction_frame()
    original.to_csv(csv, index=False)
    train_experiment(csv, "ridge", tmp_path / "original")
    pd.testing.assert_series_equal(
        pd.Series(predict(ridge, original.head())),
        pd.Series(predict(load_bundle(tmp_path / "original"), original.head())),
    )
    assert np.isfinite(predict(ridge, sample)).all()
