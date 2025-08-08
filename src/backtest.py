import pandas as pd
from typing import Optional, Dict, Set

try:
    from .factors import zscore
except ImportError:  # fallback when running as script
    from factors import zscore


def backtest_momentum(
    momentum: pd.DataFrame,
    returns: pd.DataFrame,
    cost_per_trade: float = 0.0,
    top_frac: float = 0.2,
) -> pd.DataFrame:
    """
    Long-only momentum: pick top `top_frac` by momentum each month.
    Uses returns_next alignment and realistic turnover costs.
    """
    n = max(1, int(len(momentum.columns) * top_frac))
    returns_next = returns.shift(-1)

    portfolio_returns = []
    prev_holdings: Set[str] = set()

    for date in momentum.index:
        if date not in returns_next.index:
            continue
        row = momentum.loc[date].dropna()
        if row.empty:
            continue
        top = row.nlargest(n).index
        nxt = date
        gross = returns_next.loc[nxt, top].mean(skipna=True)

        # Turnover: entries + exits vs previous holdings
        holdings = set(top)
        entries = len(holdings - prev_holdings)
        exits = len(prev_holdings - holdings)
        turnover = entries + exits
        prev_holdings = holdings

        net = gross - cost_per_trade * turnover
        portfolio_returns.append((nxt, net))

    df = pd.DataFrame(portfolio_returns, columns=["Date", "Return"]).set_index("Date")
    df = df.dropna()
    df["Cumulative"] = (1 + df["Return"]).cumprod()
    return df


def backtest_multifactor(
    momentum: pd.DataFrame,
    value: pd.Series,
    quality: pd.Series,
    returns: pd.DataFrame,
    cost_per_trade: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
    frac: float = 0.2,
) -> pd.DataFrame:
    """
    Long-short multifactor: top/bottom `frac` by weighted z-score of
    momentum (cross-section), value (static x-section), quality (static x-section).
    Uses returns_next alignment and realistic turnover.
    """
    if weights is None:
        weights = {"momentum": 0.4, "value": 0.3, "quality": 0.3}

    n = max(1, int(len(momentum.columns) * frac))
    returns_next = returns.shift(-1)

    # Precompute static factor z-scores once
    val_z = zscore(value)
    qual_z = zscore(quality)

    mf_returns = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()

    for date in momentum.index:
        if date not in returns_next.index:
            continue
        mom_row = momentum.loc[date].dropna()
        if mom_row.empty:
            continue
        mom_z = zscore(mom_row)

        # Align static z-scores to current universe
        v = val_z.reindex(mom_row.index)
        q = qual_z.reindex(mom_row.index)
        comp = weights["momentum"] * mom_z + weights["value"] * v + weights["quality"] * q

        longs = comp.nlargest(n).index
        shorts = comp.nsmallest(n).index

        gross_long = returns_next.loc[date, longs].mean(skipna=True)
        gross_short = returns_next.loc[date, shorts].mean(skipna=True)

        # Turnover: changes in long and short books
        longs_set, shorts_set = set(longs), set(shorts)
        turnover = len(longs_set - prev_longs) + len(prev_longs - longs_set) \
                 + len(shorts_set - prev_shorts) + len(prev_shorts - shorts_set)
        prev_longs, prev_shorts = longs_set, shorts_set

        net = (gross_long - gross_short) - cost_per_trade * turnover
        mf_returns.append((date, net))

    df = pd.DataFrame(mf_returns, columns=["Date", "Return"]).set_index("Date")
    df = df.dropna()
    df["Cumulative"] = (1 + df["Return"]).cumprod()
    return df