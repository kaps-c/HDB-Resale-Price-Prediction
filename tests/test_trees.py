import numpy as np
from conftest import transaction_frame

from hdb_resale.experiment import load_bundle, predict, train_experiment


def test_trees_support_unseen_and_high_cardinality_categories(tmp_path):
    frame = transaction_frame()
    # Exceed the tree estimator's native category limit in training data.
    train = frame.loc[frame["month"] < "2024-01"].iloc[[0] * 300].copy()
    train["street_name"] = [f"STREET {i}" for i in range(300)]
    train["floor_area_sqm"] = np.arange(300) + 30
    train["resale_price"] = train["floor_area_sqm"] * 3000
    import pandas as pd

    frame = pd.concat(
        [train, frame.loc[frame["month"] >= "2024-01"]], ignore_index=True
    )
    source = tmp_path / "input.csv"
    frame.to_csv(source, index=False)
    report = train_experiment(source, "trees", tmp_path / "trees")
    sample = frame.tail(2).drop(columns="resale_price").copy()
    sample["street_name"] = "NEW STREET"
    values = predict(load_bundle(tmp_path / "trees"), sample)
    assert len(values) == 2
    assert np.isfinite(values).all()
    assert report["validation"]["count"] == 24
