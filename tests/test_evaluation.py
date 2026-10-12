import numpy as np
import pytest

from hdb_resale.evaluation import regression_metrics


def test_metrics_use_dollars_and_handle_undefined_r_squared():
    result = regression_metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert result["mae_sgd"] == 15.0
    assert result["rmse_sgd"] == pytest.approx(np.sqrt(250))
    assert regression_metrics(np.array([100.0]), np.array([110.0]))["r2"] is None
    with pytest.raises(ValueError, match="finite"):
        regression_metrics(np.array([100.0]), np.array([np.nan]))
