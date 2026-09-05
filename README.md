# factor-research

Decile backtests + Information Coefficient (IC) analysis for equity
factors, on a curated universe of 50 large-cap US tickers (or the full
S&P 500 via yfinance).

Implements the canonical quantitative workflow:

1. **Pull prices** (yfinance) for a universe of tickers.
2. **Compute a factor** at month-end (momentum, value, quality, low-vol, size).
3. **Form decile portfolios** at each rebalance date, equal-weight.
4. **Compute monthly IC** = Spearman(factor, next-month return).
5. **Report** top-minus-bottom spread Sharpe, IC mean/std/IR/hit-rate, and
   cumulative-return + IC charts.

## Factors

| name        | definition                                                    | fundamentals needed |
|-------------|---------------------------------------------------------------|---------------------|
| `momentum`  | 12-month return skipping the most recent month (Jegadeesh-Titman) | no                |
| `value`     | Earnings yield (1 / trailing P/E) — book/price proxy           | yes                 |
| `quality`   | Composite z-score of ROE and gross margin                       | yes                 |
| `low_vol`   | Negative trailing 252-day realized vol                          | no                  |
| `size`      | Negative log market-cap                                         | yes (auto-fetched)  |

## Install

```bash
git clone https://github.com/JoshRiang/factor-research.git
cd factor-research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Momentum backtest from 2010-01-01 over the curated 50-ticker universe
python -m factor --factor momentum --start 2010-01-01

# Value (earnings yield) — auto-fetches yfinance fundamentals
python -m factor --factor value --start 2012-01-01

# Quintiles instead of deciles
python -m factor --factor low_vol --start 2015-01-01 --n-deciles 5

# Full S&P 500 (slower; relies on Wikipedia scrape)
python -m factor --factor quality --universe sp500 --start 2015-01-01

# CSV + JSON only, no charts
python -m factor --factor size --no-charts --start 2014-01-01
```

Outputs land in `output/<factor>/`:

```
output/momentum/
├── decile_returns.csv           # period returns per decile
├── ic_series.csv                # monthly IC time-series
├── ic_summary.csv               # per-year IC mean/std/IR/hit-rate
├── top_minus_bottom.csv         # long-top / short-bottom period returns
├── summary.json                 # headline stats
├── cumulative_deciles.png       # chart: cumulative returns by decile
├── ic_timeseries.png            # chart: monthly IC bars + 12m rolling mean
└── top_minus_bottom.png         # chart: cumulative top-minus-bottom spread
```

## Library use

```python
import pandas as pd
from factor import (
    load_universe, fetch_prices, fetch_fundamentals,
    compute_factor, decile_returns, information_coefficient,
    top_minus_bottom_spread, annualized_sharpe,
)

tickers = load_universe("default")
prices  = fetch_prices(tickers, start="2010-01-01")
funds   = fetch_fundamentals(prices.columns.tolist())

factor = compute_factor("momentum", prices)
month_ends = pd.DatetimeIndex([prices.index[prices.index.to_period('M') == m][-1]
                               for m in prices.index.to_period('M').unique()])

dec_rets = decile_returns(factor, prices, month_ends)
spread   = top_minus_bottom_spread(dec_rets)
print("Top-minus-bottom Sharpe:", annualized_sharpe(spread))

ic = information_coefficient(factor, prices, month_ends)
print("IC mean:", ic.mean(), "IC IR:", ic.mean() / ic.std())
```

## Tests

```bash
pytest tests/ -v
```

Tests are fully **offline** — they fabricate synthetic price/fundamental
data so the decile + IC pipeline is fully covered without hitting the
network.

## Methodology notes

* **Universe**: default = curated 50 large-cap US tickers spanning
  sectors.  Edit `factor/data.py::DEFAULT_UNIVERSE` to customise.
* **Rebalance frequency**: monthly.  The first business day ≥ month-end
  for each calendar month is the rebalance timestamp.
* **Deciles**: formed by cross-sectional rank of the factor at each
  rebalance; missing values dropped.
* **Returns**: equal-weight within decile, simple returns, no
  transaction costs, no survivorship-bias correction (point-in-time
  fundamentals, but yfinance's fundamentals snapshot has look-ahead).
* **IC**: Spearman rank correlation of factor_t vs
  (1 + ret_{t→t+1}) product of daily returns in the next month.
  Reported by calendar year and overall.

## Known limitations

* `value` and `quality` rely on yfinance's *current* fundamentals
  (`.info`) — these change slowly enough that treating them as a
  stationary signal across the backtest is a common approximation in
  toy/backtest research, but it is *not* point-in-time.  For a
  production-quality PIT pipeline you'd want a fundamentals vendor
  (Compustat, Bloomberg, FundamentalsIQ).
* No risk-adjustment (Fama-French, Black-Jensen-Scholes) — pure long/short
  spread Sharpe only.
* Universe fixed at start (no delisting handling).

## License

MIT.