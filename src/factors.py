import numpy as np
import pandas as pd


def zscore(series: pd.Series, clip: float = 5.0) -> pd.Series:
    """
    Standardize a series to mean=0, std=1.
    Handles zero/NaN std and clips extreme z-scores.
    """
    s = series.astype(float).copy()
    mu = s.mean()
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        z = pd.Series(0.0, index=s.index)
    else:
        z = (s - mu) / sd
    if clip is not None:
        z = z.clip(lower=-clip, upper=clip)
    return z


def compute_momentum(prices: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """
    Compute percentage change over the past `window` periods (months).
    Assumes `prices` is monthly close prices.
    """
    return prices.pct_change(periods=window, fill_method=None).dropna(how="all")


def compute_value(funds: pd.DataFrame) -> pd.Series:
    """
    Value signal = inverse P/E. Invalid/non-positive PE becomes NaN.
    """
    pe = funds["PE"].astype(float)
    pe[~np.isfinite(pe) | (pe <= 0)] = np.nan
    val = 1.0 / pe
    return val


def compute_quality(funds: pd.DataFrame) -> pd.Series:
    """
    Quality signal = ROE directly (sanitized).
    """
    q = funds.get(
        "Quality",
        funds.get("ROE", pd.Series(index=funds.index, dtype=float)),
    ).astype(float)
    q[~np.isfinite(q)] = np.nan
    return q