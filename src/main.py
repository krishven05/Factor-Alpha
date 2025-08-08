"""project complete"""
 
import argparse
import hashlib
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from backtest import backtest_momentum, backtest_multifactor
from data import download_prices, fetch_fundamentals, get_sp500_tickers
from factors import compute_momentum, compute_quality, compute_value
from metrics import print_metrics


# -----------------------------
# Utilities
# -----------------------------

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _tickers_hash(tickers):
    s = ",".join(sorted(tickers))
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _load_config(config_path: Path | None) -> dict:
    if not config_path or not config_path.exists():
        return {}
    with config_path.open("r") as f:
        return yaml.safe_load(f) or {}


def _load_or_download_prices(tickers, start, end, prefer_monthly=True, use_cache=True):
    """Load prices from cache if available; else download and cache.
    If prefer_monthly=True, try to request monthly data upstream (if supported by download_prices).
    """
    cache_dir = Path(__file__).resolve().parents[1] / "data" / "cache"
    _ensure_dir(cache_dir)

    thash = _tickers_hash(tickers)
    cache_base = cache_dir / f"prices_{start}_{end}_{thash}"

    # Try parquet first, else pickle
    parquet_fp = cache_base.with_suffix(".parquet")
    pkl_fp = cache_base.with_suffix(".pkl")

    if use_cache and parquet_fp.exists():
        try:
            return pd.read_parquet(parquet_fp)
        except Exception:
            pass
    if use_cache and pkl_fp.exists():
        try:
            return pd.read_pickle(pkl_fp)
        except Exception:
            pass

    # Download fresh
    try:
        if prefer_monthly:
            try:
                prices = download_prices(tickers, start, end, interval="1mo")  # if supported
            except TypeError:
                prices = download_prices(tickers, start, end)
        else:
            prices = download_prices(tickers, start, end)
    except Exception as e:
        raise RuntimeError(f"Failed to download prices: {e}") from e

    # Cache
    try:
        prices.to_parquet(parquet_fp)
    except Exception:
        prices.to_pickle(pkl_fp)

    return prices


def analyze_ticker(ticker, monthly, momentum, funds, show_plot=False):
    ticker = ticker.upper()
    if ticker not in monthly.columns:
        print(f"❌ {ticker} not in universe")
        return

    print(f"\n📊 Analysis for {ticker}")
    print("=" * 50)

    # Fundamentals
    val = funds.loc[ticker, "Value"] if ticker in funds.index else np.nan
    qual = funds.loc[ticker, "Quality"] if ticker in funds.index else np.nan
    print(f"Value (1/PE): {val if pd.notna(val) else 'N/A'}")
    print(f"Quality (ROE): {qual if pd.notna(qual) else 'N/A'}")

    # Momentum snapshot
    if ticker in momentum.columns:
        m = momentum[ticker].dropna()
        if len(m) > 0:
            recent = m.tail(5)
            print(f"Current 6m momentum: {recent.iloc[-1]:.2%}")
            print(f"Avg last 5: {recent.mean():.2%} | Vol: {recent.std():.2%}")
    else:
        print("No momentum available")

    # Price performance
    px = monthly[ticker].dropna()
    if len(px) >= 2:
        total_ret = px.iloc[-1] / px.iloc[0] - 1
        months = len(px)
        ann = (px.iloc[-1] / px.iloc[0]) ** (12 / months) - 1
        print(f"Total return: {total_ret:.2%} | Annualized: {ann:.2%} | Months: {months}")

    if show_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(px.index, px.values, label=f"{ticker} Price", lw=2)
        ax1.set_title(f"{ticker} - Price & Momentum")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        if ticker in momentum.columns:
            mm = momentum[ticker].dropna()
            ax2.plot(mm.index, mm.values, color="orange", lw=2, label="6m Momentum")
            ax2.axhline(0, color="red", ls="--", alpha=0.6)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        plt.tight_layout()
        plt.show()


# -----------------------------
# Main
# -----------------------------

def run(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("factor-alpha")

    # Config
    repo_root = Path(__file__).resolve().parents[1]
    cfg = _load_config(repo_root / "factor_alpha.yaml")

    # Optional: non-interactive backend when not showing plots
    if not args.show and (args.save_figs or args.no_plot):
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            plt.switch_backend("Agg")
        except Exception:
            pass

    # 1) Load tickers and fundamentals
    if cfg.get("universe", "sp500") == "sp500":
        tickers = get_sp500_tickers()
    else:
        tickers = get_sp500_tickers()  # future: other universes

    limit = args.limit if args.limit else int(cfg.get("limit", 0) or 0)
    if limit and limit > 0:
        tickers = tickers[:limit]

    funds = fetch_fundamentals(tickers)
    funds.replace([np.inf, -np.inf], np.nan, inplace=True)
    funds.dropna(subset=["Value", "Quality"], inplace=True)
    tickers = [t for t in tickers if t in funds.index]
    log.info("Universe size: %d", len(tickers))

    # 2) Prices (use cache and monthly where possible)
    start = args.start or cfg.get("start", "2015-01-01")
    end = args.end or cfg.get("end", "2024-12-31")
    prices = _load_or_download_prices(
        tickers,
        start,
        end,
        prefer_monthly=True,
        use_cache=not args.no_cache,
    )

    # 3) Monthly series and factors
    monthly = prices.resample("ME").last()
    mom_window = int(args.mom_window or cfg.get("mom_window", 6))
    momentum = compute_momentum(monthly, window=mom_window)
    returns = monthly.pct_change(fill_method=None).dropna()

    # Precompute static factors once
    value = compute_value(funds)
    quality = compute_quality(funds)

    # 4) Backtests
    cost = float(args.cost if args.cost is not None else cfg.get("cost_per_trade", 0.0005))
    weights = cfg.get("weights", {"momentum": 0.4, "value": 0.3, "quality": 0.3})
    mom_df = backtest_momentum(momentum, returns, cost_per_trade=cost)
    mf_df = backtest_multifactor(
        momentum,
        value,
        quality,
        returns,
        cost_per_trade=cost,
        weights=weights,
    )

    # 4b) Benchmark (SPY) and excess returns
    spy_prices = download_prices(["SPY"], start, end, interval="1mo")
    spy = spy_prices.resample("ME").last()["SPY"].dropna()
    spy_ret = spy.pct_change().dropna()
    spy_curve = (1 + spy_ret).cumprod()

    # 5) Combined plot (optional)
    if not args.no_plot:
        if args.save_figs and not args.show:
            results_dir = Path(__file__).resolve().parents[1] / "results"
            _ensure_dir(results_dir)

        # Rebase all curves to 1 at the first common date
        common_idx = mom_df.index.intersection(mf_df.index).intersection(spy_curve.index)
        common_idx = common_idx.sort_values()
        if len(common_idx) == 0:
            raise RuntimeError("No overlapping dates between strategies and SPY")
        mom_norm = (
            mom_df.loc[common_idx, "Cumulative"] / mom_df.loc[common_idx, "Cumulative"].iloc[0]
        )
        mf_norm = (
            mf_df.loc[common_idx, "Cumulative"] / mf_df.loc[common_idx, "Cumulative"].iloc[0]
        )
        spy_norm = spy_curve.loc[common_idx] / spy_curve.loc[common_idx].iloc[0]

        plt.figure(figsize=(12, 6))
        plt.plot(common_idx, mom_norm, label="Momentum Only", lw=2.2)
        plt.plot(common_idx, mf_norm, label="Multi-Factor LS", lw=2.2)
        plt.plot(common_idx, spy_norm, label="SPY", lw=1.8, alpha=0.9)
        plt.title("Equity Curves vs SPY (Rebased)")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        if args.save_figs:
            out_fp = results_dir / "equity_curves.png"
            plt.tight_layout()
            plt.savefig(out_fp, dpi=200)
            print(f"💾 Saved combined plot to {out_fp}")
        if args.show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

    # 6) Metrics
    print_metrics(mom_df, "Momentum Only")
    print_metrics(mf_df, "Multi-Factor LS")
    # Excess vs SPY
    common_idx_mom = mom_df.index.intersection(spy_ret.index)
    mom_excess = mom_df.loc[common_idx_mom, "Return"] - spy_ret.loc[common_idx_mom]
    common_idx_mf = mf_df.index.intersection(spy_ret.index)
    mf_excess = mf_df.loc[common_idx_mf, "Return"] - spy_ret.loc[common_idx_mf]
    if len(mom_excess) > 0:
        print_metrics(
            pd.DataFrame({"Return": mom_excess, "Cumulative": (1 + mom_excess).cumprod()}),
            "Momentum Excess",
        )
    if len(mf_excess) > 0:
        print_metrics(
            pd.DataFrame({"Return": mf_excess, "Cumulative": (1 + mf_excess).cumprod()}),
            "Multi-Factor Excess",
        )

    # 7) Optional: ticker analysis
    if args.ticker:
        analyze_ticker(args.ticker, monthly, momentum, funds, show_plot=args.show)
    elif args.interactive:
        print("\nEnter tickers to analyze (or 'q' to quit):")
        while True:
            try:
                line = input(">> ").strip()
                if not line:
                    continue
                if line.lower() in {"q", "quit", "exit"}:
                    break
                analyze_ticker(line, monthly, momentum, funds, show_plot=args.show)
            except KeyboardInterrupt:
                print("\nExiting…")
                break

    return 0


def main():
    parser = argparse.ArgumentParser(description="Factor Alpha - Momentum & Multifactor")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of tickers for faster runs"
    )
    parser.add_argument(
        "--mom-window", type=int, default=None, help="Momentum lookback in months"
    )
    parser.add_argument("--cost", type=float, default=None, help="Cost per trade")
    parser.add_argument("--no-cache", action="store_true", help="Disable price caching")

    # Plotting options
    parser.add_argument("--no-plot", action="store_true", help="Do not create plots")
    parser.add_argument("--save-figs", action="store_true", help="Save plots to results/")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")

    # Ticker analysis
    parser.add_argument("--ticker", help="Analyze a specific ticker symbol")
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive ticker lookup mode"
    )

    args = parser.parse_args()

    if args.no_plot:
        args.show = False

    return run(args)


if __name__ == "__main__":
    sys.exit(main())