# Factor Alpha

Momentum & Multifactor backtesting toolkit on the S&P 500 universe.

## Features
- Monthly data loading with caching
- Momentum and long/short multi-factor portfolios (momentum, value, quality)
- Turnover-based trading costs
- Combined equity curves and SPY benchmark
- CLI flags and YAML config
- Robust metrics with guardrails

## Quickstart

Create venv and install:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Run a quick backtest (no plots):

```bash
python src/main.py --start 2018-01-01 --end 2024-12-31 --limit 50 --no-plot
```

Save plots without showing windows:

```bash
python src/main.py --save-figs --no-plot
```

Analyze a ticker:

```bash
python src/main.py --ticker AAPL --show
```

## Config
Edit `factor_alpha.yaml` to set defaults (dates, costs, weights, universe size). CLI flags override the config.

## Development
- Formatting: black
- Linting: ruff
- Types: mypy
- Tests: pytest

Run checks locally:

```bash
pre-commit install
pre-commit run --all-files
pytest -q
```

## CI
GitHub Actions runs linting and tests on PRs.

## License
MIT (add a LICENSE file as needed).
