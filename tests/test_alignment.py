import pandas as pd
import numpy as np

from src.backtest import backtest_momentum


def test_next_period_alignment():
    # 3 assets, 5 months
    dates = pd.date_range("2020-01-31", periods=5, freq="M")
    momentum = pd.DataFrame(
        [[1, 2, 3], [1, 2, 3], [3, 2, 1], [3, 1, 2], [2, 3, 1]],
        index=dates,
        columns=["A", "B", "C"],
    )
    prices = pd.DataFrame(
        [[100, 100, 100], [110, 100, 90], [121, 95, 99.0], [120, 100, 110], [126, 104, 121]],
        index=dates,
        columns=["A", "B", "C"],
    )
    returns = prices.pct_change().dropna()

    df = backtest_momentum(momentum, returns, cost_per_trade=0.0, top_frac=1/3)
    # First tradable date is when we have next-period return
    assert df.index.min() >= returns.index.min()
    # Cumulative should be positive given choosing top each month in this toy example
    assert df["Cumulative"].iloc[-1] > 0
