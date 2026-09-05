"""Command-line entry point: ``python -m factor``.

Examples
--------
Run a momentum decile backtest from 2010-01-01 over the default universe
and save outputs to ``output/momentum/``::

    python -m factor --factor momentum --start 2010-01-01 \
        --output-dir output/momentum

Use the full S&P 500 universe (slower)::

    python -m factor --factor value --universe sp500 --start 2015-01-01

Skip charts (CSV only)::

    python -m factor --factor momentum --no-charts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import (
    compute_factor,
    fetch_fundamentals,
    fetch_prices,
    information_coefficient,
    ic_summary,
    decile_returns,
    top_minus_bottom_spread,
    annualized_sharpe,
    cumulative_returns,
    load_universe,
)
from .visualize import (
    plot_cumulative_decile_returns,
    plot_ic_timeseries,
    plot_top_minus_bottom,
)


def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return one timestamp per calendar month (the last available date <= month-end)."""
    if idx.empty:
        return idx
    months = idx.to_period("M").unique().sort_values()
    return pd.DatetimeIndex([idx[idx.to_period("M") == m][-1] for m in months])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m factor")
    p.add_argument(
        "--factor",
        required=True,
        choices=["momentum", "value", "quality", "low_vol", "size"],
        help="Which factor to backtest",
    )
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    p.add_argument(
        "--universe",
        default="default",
        help='Universe specifier: "default" (curated 50), "sp500", or path to .txt',
    )
    p.add_argument(
        "--n-deciles",
        type=int,
        default=10,
        help="Number of buckets (10 = deciles)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV + chart outputs.  Defaults to output/<factor>/",
    )
    p.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip writing PNG charts (CSV/JSON only)",
    )
    p.add_argument(
        "--cache-dir",
        default=".cache/yfinance",
        help="Directory to cache yfinance downloads",
    )
    p.add_argument(
        "--quiet-yfinance",
        action="store_true",
        help="Suppress yfinance progress/info chatter",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet_yfinance else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("factor")

    tickers = load_universe(args.universe)
    log.info("Universe: %d tickers (%s)", len(tickers), args.universe)

    output_dir = Path(args.output_dir or f"output/{args.factor}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Prices ----------------------------------------------------------
    prices = fetch_prices(tickers, start=args.start, end=args.end, cache_dir=cache_dir)
    prices = prices.dropna(how="all").ffill().dropna(how="all")
    log.info("Prices: %s × %s (%s → %s)", *prices.shape, prices.index[0].date(), prices.index[-1].date())

    # --- 2. Fundamentals (if needed) ---------------------------------------
    needs_fund = args.factor in {"value", "quality", "size"}
    fundamentals = None
    if needs_fund:
        log.info("Fetching fundamentals (yfinance .info)...")
        fundamentals = fetch_fundamentals(prices.columns.tolist(), cache_dir=cache_dir)
        log.info("Fundamentals: %d tickers", len(fundamentals))

    # --- 3. Factor ----------------------------------------------------------
    factor_wide = compute_factor(args.factor, prices, fundamentals=fundamentals)
    # Reindex factor to prices.index (compute_factor already does this)
    factor_wide = factor_wide.reindex(prices.index)

    # --- 4. Decile backtest -------------------------------------------------
    rebalance_dates = _month_ends(prices.index)
    dec_rets = decile_returns(factor_wide, prices, rebalance_dates, n_deciles=args.n_deciles)

    # --- 5. IC --------------------------------------------------------------
    ic_series = information_coefficient(factor_wide, prices, rebalance_dates)
    ic_stats = ic_summary(ic_series)

    # --- 6. Top-minus-bottom spread ----------------------------------------
    spread = top_minus_bottom_spread(dec_rets)
    spread_sharpe = annualized_sharpe(spread)
    # Also per-decile Sharpe for the headline bucket
    top_col = dec_rets.columns[-1]
    bot_col = dec_rets.columns[0]
    top_sharpe = annualized_sharpe(dec_rets[top_col])
    bot_sharpe = annualized_sharpe(dec_rets[bot_col])

    # --- 7. Persist ---------------------------------------------------------
    dec_rets.to_csv(output_dir / "decile_returns.csv", index_label="date")
    ic_series.to_csv(output_dir / "ic_series.csv", index_label="date")
    ic_stats.to_csv(output_dir / "ic_summary.csv", index=False)
    spread.to_frame("spread").to_csv(output_dir / "top_minus_bottom.csv", index_label="date")

    summary = {
        "factor": args.factor,
        "start": args.start,
        "end": args.end or "today",
        "universe_size": int(len(tickers)),
        "actual_tickers": int(prices.shape[1]),
        "n_periods": int(len(dec_rets)),
        "top_minus_bottom": {
            "annualized_return": float(spread.dropna().mean() * 12),
            "annualized_vol": float(spread.dropna().std(ddof=1) * (12 ** 0.5)),
            "sharpe": float(spread_sharpe),
        },
        "top_decile": {"sharpe": float(top_sharpe)},
        "bottom_decile": {"sharpe": float(bot_sharpe)},
        "ic": {
            "mean": float(ic_stats.loc[ic_stats["year"] == "ALL", "ic_mean"].iloc[0])
            if not ic_stats.empty
            else float("nan"),
            "std": float(ic_stats.loc[ic_stats["year"] == "ALL", "ic_std"].iloc[0])
            if not ic_stats.empty
            else float("nan"),
            "ir": float(ic_stats.loc[ic_stats["year"] == "ALL", "ic_ir"].iloc[0])
            if not ic_stats.empty
            else float("nan"),
            "hit_rate": float(ic_stats.loc[ic_stats["year"] == "ALL", "hit_rate"].iloc[0])
            if not ic_stats.empty
            else float("nan"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # --- 8. Charts ----------------------------------------------------------
    if not args.no_charts:
        try:
            plot_cumulative_decile_returns(
                dec_rets,
                title=f"{args.factor.upper()}: Cumulative Decile Returns",
                save_path=output_dir / "cumulative_deciles.png",
            )
            plot_ic_timeseries(
                ic_series,
                title=f"{args.factor.upper()}: Information Coefficient",
                save_path=output_dir / "ic_timeseries.png",
            )
            plot_top_minus_bottom(
                spread,
                title=f"{args.factor.upper()}: Top Minus Bottom Decile",
                save_path=output_dir / "top_minus_bottom.png",
            )
            plt_close_all()
        except Exception as exc:  # noqa: BLE001
            log.warning("Chart generation failed: %s", exc)

    # --- 9. Console report --------------------------------------------------
    print("\n" + "=" * 70)
    print(f" Factor: {args.factor}  |  Universe: {args.universe} ({len(tickers)} tickers)")
    print(f" Period: {args.start} → {args.end or 'today'}  |  Deciles: {args.n_deciles}")
    print("=" * 70)
    print(f" Decile period returns ({dec_rets.shape[0]} periods)")
    print("-" * 70)
    summary_table = pd.DataFrame(
        {
            "mean": dec_rets.mean(),
            "std": dec_rets.std(ddof=1),
            "sharpe": [annualized_sharpe(dec_rets[c]) for c in dec_rets.columns],
        }
    )
    print(summary_table.round(4).to_string())
    print("-" * 70)
    print(
        f" Top-minus-bottom spread: ann.ret={summary['top_minus_bottom']['annualized_return']:.3f}  "
        f"ann.vol={summary['top_minus_bottom']['annualized_vol']:.3f}  "
        f"Sharpe={spread_sharpe:.3f}"
    )
    print(
        f" IC: mean={summary['ic']['mean']:.4f}  std={summary['ic']['std']:.4f}  "
        f"IR={summary['ic']['ir']:.3f}  hit-rate={summary['ic']['hit_rate']:.2%}"
    )
    print(f" Outputs written to: {output_dir}")
    print("=" * 70 + "\n")

    return 0


def plt_close_all() -> None:
    import matplotlib.pyplot as plt

    plt.close("all")


if __name__ == "__main__":
    sys.exit(main())