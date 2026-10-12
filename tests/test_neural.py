import numpy as np
import pandas as pd
import pytest
from conftest import transaction_frame

from hdb_resale.experiment import load_bundle, predict, train_experiment

pytest.importorskip("torch")


@pytest.mark.parametrize("name", ["mlp", "lstm"])
def test_neural_checkpoint_is_portable_and_target_free(tmp_path, name):
    frame = transaction_frame()
    source = tmp_path / "data.csv"
    frame.to_csv(source, index=False)
    output = tmp_path / name
    report = train_experiment(source, name, output, epochs=2, threads=1, device="cpu")
    assert (output / "weights.pt").exists()
    assert report["training"]["epochs_completed"] == 2
    history = pd.read_csv(output / "history.csv")
    assert report["validation"]["mae_sgd"] == pytest.approx(
        history["validation_mae_sgd"].min(),
        rel=1e-5,
    )
    sample = frame.tail(3).drop(columns="resale_price")
    before = predict(load_bundle(output), sample)
    after = predict(load_bundle(output), sample[sample.columns[::-1]])
    np.testing.assert_allclose(before, after, rtol=1e-6)
    assert np.isfinite(after).all()
    assert len(after) == 3
