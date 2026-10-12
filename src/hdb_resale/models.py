"""Small model factories; preprocessing is fitted inside each pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from hdb_resale.data import CATEGORICAL, NUMERIC, make_features


def build_model(name: str, seed: int = 42) -> Pipeline:
    """Construct an unfitted model with the same raw-input contract."""
    features = FunctionTransformer(make_features)
    if name == "median":
        return Pipeline(
            [("features", features), ("model", DummyRegressor(strategy="median"))]
        )
    if name != "ridge":
        raise ValueError(f"Unknown model: {name}")
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0, solver="lsqr", random_state=seed)),
        ]
    )
