"""Small, reusable plots; callers control display and file output."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


def plot_monthly_prices(frame: pd.DataFrame) -> Figure:
    """Plot observed monthly medians and mark the evaluation boundaries."""
    data = frame.copy()
    data["month"] = pd.to_datetime(data["month"])
    monthly = data.groupby("month")["resale_price"].median()
    figure, axis = plt.subplots(figsize=(9, 4), layout="constrained")
    axis.plot(monthly.index, monthly.values, color="#176b87", linewidth=2)
    for date, label in [("2024-01-01", "Validation"), ("2025-01-01", "Test")]:
        axis.axvline(pd.Timestamp(date), color="#9b4444", linestyle="--", alpha=0.7)
        axis.text(pd.Timestamp(date), monthly.max(), label, rotation=90, va="top")
    axis.set(
        title="Observed HDB resale prices",
        xlabel="Transaction month",
        ylabel="Median price (SGD)",
    )
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.spines[["top", "right"]].set_visible(False)
    return figure


def plot_comparison(table: pd.DataFrame) -> Figure:
    """Show validation and test errors without ranking on the test set."""
    columns = ["validation_mae_sgd"]
    if "test_mae_sgd" in table:
        columns.append("test_mae_sgd")
    figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    table.set_index("model")[columns].plot.barh(ax=axis, color=["#176b87", "#d49a38"])
    axis.set(
        title="Price prediction error", xlabel="Mean absolute error (SGD)", ylabel=""
    )
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.legend(["Validation · 2024", "Test · 2025"][: len(columns)], frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    return figure


def plot_data_profile(frame: pd.DataFrame) -> Figure:
    """Show price spread and transaction mix before interpreting model errors."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].hist(frame["resale_price"], bins=40, color="#176b87")
    axes[0].set(
        title="Observed price distribution", xlabel="Price (SGD)", ylabel="Count"
    )
    axes[0].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[0].xaxis.set_major_locator(MaxNLocator(4))
    frame["flat_type"].value_counts().sort_values().plot.barh(
        ax=axes[1], color="#176b87"
    )
    axes[1].set(title="Transaction mix", xlabel="Count", ylabel="Flat type")
    axes[1].xaxis.set_major_locator(MaxNLocator(4))
    axes[1].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    return figure


def plot_diagnostics(predictions: pd.DataFrame, groups: pd.DataFrame) -> Figure:
    """Plot all held-out errors, with subgroup counts and a signed residual."""
    actual = predictions["resale_price"]
    predicted = predictions["predicted_price"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    axes[0, 0].hexbin(actual, predicted, gridsize=35, mincnt=1, cmap="Blues")
    bounds = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[0, 0].plot(bounds, bounds, color="#9b4444", linestyle="--")
    axes[0, 0].set(
        title="Prediction versus observation",
        xlabel="Actual price (SGD)",
        ylabel="Predicted price (SGD)",
    )
    axes[0, 1].hexbin(actual, predicted - actual, gridsize=35, mincnt=1, cmap="Blues")
    axes[0, 1].axhline(0, color="#9b4444", linestyle="--")
    axes[0, 1].set(
        title="Residuals: negative means underprediction",
        xlabel="Actual price (SGD)",
        ylabel="Predicted minus actual (SGD)",
    )
    for axis, dimension in zip(axes[1], ["month", "flat_type"], strict=True):
        subset = groups.loc[groups["dimension"] == dimension].sort_values("group")
        labels = [
            f"{row.group} (n={row.count:,})" for row in subset.itertuples(index=False)
        ]
        axis.barh(labels, subset["mae_sgd"], color="#176b87")
        axis.set(title=f"Test error by {dimension}", xlabel="MAE (SGD)")
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.xaxis.set_major_locator(MaxNLocator(4))
    for axis in axes[0]:
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    return figure


def plot_learning_curve(history: pd.DataFrame) -> Figure:
    """Keep standardized training loss separate from dollar validation error."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 3), layout="constrained")
    axes[0].plot(history["epoch"], history["train_mse_scaled"], color="#176b87")
    axes[0].set(title="Training", xlabel="Epoch", ylabel="MSE (standardized target)")
    axes[1].plot(history["epoch"], history["validation_mae_sgd"], color="#176b87")
    best = history.loc[history["validation_mae_sgd"].idxmin(), "epoch"]
    axes[1].axvline(
        best, color="#9b4444", linestyle="--", label=f"Selected epoch {best:g}"
    )
    axes[1].set(title="Validation", xlabel="Epoch", ylabel="MAE (SGD)")
    axes[1].legend(frameon=False)
    return figure
