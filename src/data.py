import pandas as pd
import yfinance as yf
import requests
import time
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional


CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(name: str) -> Path:
    return CACHE_DIR / name


def get_sp500_tickers(use_cache: bool = True, refresh: bool = False) -> List[str]:
    """
    Get current S&P 500 tickers from Wikipedia (cached).
    Falls back to cached file on failure.
    """
    cache_fp = _cache_path("sp500_tickers.parquet")
    if use_cache and cache_fp.exists() and not refresh:
        try:
            return pd.read_parquet(cache_fp)["Symbol"].tolist()
        except Exception:
            pass

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FactorAlpha/1.0)"}
    html = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            html = resp.text
            break
        except Exception:
            time.sleep(1.0 * (attempt + 1))

    if html is None:
        # Fallback to cache if available
        if cache_fp.exists():
            return pd.read_parquet(cache_fp)["Symbol"].tolist()
        raise RuntimeError("Failed to fetch S&P 500 tickers and no cache present")

    tables = pd.read_html(io.StringIO(html), header=0)
    table = tables[0]
    tickers = table.Symbol.str.replace(r"\.", "-", regex=True)
    # Save cache
    try:
        pd.DataFrame({"Symbol": tickers}).to_parquet(cache_fp)
    except Exception:
        pass
    return tickers.tolist()


def download_prices(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
    batch_size: int = 100,
    auto_adjust: bool = True,
    progress: bool = False,
    threads: bool = False,
    retries: int = 3,
    sleep: float = 1.0,
) -> pd.DataFrame:
    """
    Download prices in batches. Supports monthly interval to avoid resampling.
    Returns a wide DataFrame of Close prices (tickers as columns).
    """
    df_list = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                data = yf.download(
                    batch,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    progress=progress,
                    threads=threads,
                )
                # yf returns multiindex columns when multiple tickers
                if isinstance(data.columns, pd.MultiIndex):
                    close = data.get("Close")
                else:
                    close = data
                if close is None:
                    raise RuntimeError("No 'Close' data returned")
                df_list.append(close)
                break
            except Exception as e:
                last_err = e
                time.sleep(sleep * (attempt + 1))
        else:
            raise RuntimeError(f"Failed to download batch {batch}: {last_err}")

    if not df_list:
        raise RuntimeError("No price data downloaded")

    prices = pd.concat(df_list, axis=1)

    # Ensure expected columns order and uniqueness
    # Flatten potential multiindex columns and keep only requested tickers
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(-1)
    # Keep unique columns
    prices = prices.loc[:, ~prices.columns.duplicated()]
    # Reindex to requested tickers where available
    common = [t for t in tickers if t in prices.columns]
    prices = prices.reindex(columns=common)

    # Sort by date and ensure DatetimeIndex
    prices = prices.sort_index()
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)

    # Drop columns that are entirely NaN
    prices = prices.dropna(axis=1, how="all")

    return prices


def fetch_fundamentals(
    tickers: List[str],
    use_cache: bool = True,
    refresh: bool = False,
    max_workers: int = 10,
    retries: int = 2,
    sleep: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch trailing P/E and ROE for each ticker using yfinance.
    Cached to parquet by ticker set. Basic sanitization applied.
    """
    import hashlib

    thash = hashlib.md5(
        ",".join(sorted(tickers)).encode()
    ).hexdigest()[:8]
    cache_fp = _cache_path(f"fundamentals_{thash}.parquet")

    if use_cache and cache_fp.exists() and not refresh:
        try:
            return pd.read_parquet(cache_fp)
        except Exception:
            pass

    def fetch_one(t: str) -> Dict[str, Optional[float]]:
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                info = yf.Ticker(t).info
                return {
                    "PE": info.get("trailingPE", None),
                    "ROE": info.get("returnOnEquity", None),
                }
            except Exception as e:
                last_err = e
                time.sleep(sleep * (attempt + 1))
        # final failure
        return {"PE": None, "ROE": None}

    rows = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_one, t): t for t in tickers}
        for f in as_completed(futures):
            t = futures[f]
            try:
                rows[t] = f.result()
            except Exception:
                rows[t] = {"PE": None, "ROE": None}

    funds = pd.DataFrame.from_dict(rows, orient="index")

    # Sanitize fundamentals
    funds.replace([pd.NA, float("inf"), float("-inf")], pd.NA, inplace=True)
    # Invalid PE values (<=0) are not usable for 1/PE
    funds.loc[funds["PE"] <= 0, "PE"] = pd.NA
    # Compute Value and Quality
    funds["Value"] = 1.0 / funds["PE"]
    funds["Quality"] = funds["ROE"]

    # Optional winsorization/clipping
    for col in ["Value", "Quality"]:
        if col in funds.columns:
            funds[col] = funds[col].astype(float)
            funds[col] = funds[col].clip(lower=funds[col].quantile(0.01), upper=funds[col].quantile(0.99))

    # Drop rows missing both signals
    funds = funds.dropna(subset=["Value", "Quality"], how="all")

    # Cache
    try:
        funds.to_parquet(cache_fp)
    except Exception:
        pass

    return funds