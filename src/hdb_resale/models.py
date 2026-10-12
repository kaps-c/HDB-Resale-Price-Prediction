"""Small model factories; preprocessing is fitted inside each pipeline."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from hdb_resale.data import CATEGORICAL, NUMERIC, make_features


def build_model(
    name: str,
    seed: int = 42,
    *,
    epochs: int = 20,
    device: str = "auto",
    threads: int = 4,
) -> Pipeline:
    """Construct an unfitted model with the same raw-input contract."""
    features = FunctionTransformer(make_features)
    if name == "median":
        return Pipeline(
            [("features", features), ("model", DummyRegressor(strategy="median"))]
        )
    if name in {"mlp", "lstm"}:
        from hdb_resale.neural import NeuralRegressor

        categorical = (
            OneHotEncoder(handle_unknown="ignore", max_categories=256)
            if name == "mlp"
            else Pipeline(
                [
                    (
                        "ordinal",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                    ("scale", StandardScaler()),
                ]
            )
        )
        preprocessor = ColumnTransformer(
            [
                ("numeric", StandardScaler(), NUMERIC),
                ("categorical", categorical, CATEGORICAL),
            ]
        )
        return Pipeline(
            [
                ("features", features),
                ("preprocessor", preprocessor),
                (
                    "model",
                    NeuralRegressor(
                        architecture=name,
                        seed=seed,
                        epochs=epochs,
                        device=device,
                        threads=threads,
                    ),
                ),
            ]
        )
    if name == "trees":
        preprocessor = ColumnTransformer(
            [
                ("numeric", "passthrough", NUMERIC),
                (
                    "categorical",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=np.nan,
                        max_categories=254,
                    ),
                    CATEGORICAL,
                ),
            ]
        )
        return Pipeline(
            [
                ("features", features),
                ("preprocessor", preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=200,
                        learning_rate=0.1,
                        max_leaf_nodes=31,
                        l2_regularization=1.0,
                        early_stopping=False,
                        random_state=seed,
                        categorical_features=[False] * len(NUMERIC)
                        + [True] * len(CATEGORICAL),
                    ),
                ),
            ]
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
