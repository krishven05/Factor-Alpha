import numpy as np
import pandas as pd


def _is_valid_series(s: pd.Series) -> bool:
    return isinstance(s, pd.Series) and len(s.dropna()) > 0


def cagr(cum: pd.Series):
    if not _is_valid_series(cum) or len(cum) < 2:
        return float("nan")
    try:
        years = (cum.index[-1] - cum.index[0]).days / 365.25
        if years <= 0:
            return float("nan")
        final = cum.iloc[-1]
        if final <= 0:
            return float("nan")
        return final ** (1 / years) - 1
    except Exception:
        return float("nan")


def sharpe(returns: pd.Series, freq: int = 12):
    if not _is_valid_series(returns):
        return float("nan")
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return float("nan")
    return (mu / sigma) * np.sqrt(freq)


def max_drawdown(cum: pd.Series):
    if not _is_valid_series(cum):
        return float("nan")
    try:
        dd = (cum / cum.cummax() - 1).min()
        return float(dd)
    except Exception:
        return float("nan")


def print_metrics(df: pd.DataFrame, name: str):
    if df is None or df.empty:
        print(f"{name:20s} | No data")
        return
    c = cagr(df.get("Cumulative", pd.Series(dtype=float)))
    s = sharpe(df.get("Return", pd.Series(dtype=float)))
    m = max_drawdown(df.get("Cumulative", pd.Series(dtype=float)))
    c_str = f"{c:6.2%}" if np.isfinite(c) else "   N/A"
    s_str = f"{s:5.2f}" if np.isfinite(s) else "  N/A"
    m_str = f"{m:6.2%}" if np.isfinite(m) else "   N/A"
    print(f"{name:20s} | CAGR: {c_str} | Sharpe: {s_str} | MaxDD: {m_str}")
