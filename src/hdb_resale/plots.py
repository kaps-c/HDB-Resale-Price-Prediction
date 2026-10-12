"""Small, reusable plots; callers control display and file output."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter


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
