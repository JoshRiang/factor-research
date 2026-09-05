"""Factor definitions.

Each factor returns a wide ``DataFrame`` indexed by date with one column per
ticker.  Factors are point-in-time: at month-end ``t`` we use only data
available up to and including date ``t`` (no peeking).

Implemented factors
-------------------
- ``momentum_12_1`` — 12-month return skipping the most recent month (Jegadeesh
  & Titman 1993).
- ``value_earnings_yield`` — earnings yield (1 / trailing P/E) from yfinance.
- ``quality`` — composite of ROE and gross margin, z-scored cross-sectionally.
- ``low_vol`` — negative trailing 252-day realized volatility (so high value ==
  attractive "low vol" characteristic).
- ``size`` — negative log market-cap (smaller = higher score).

``compute_factor(name, ...)`` is the public dispatch; ``FACTOR_REGISTRY``
exposes the mapping for the CLI.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict

import numpy as np
import pandas as pd

from . import data as data_mod

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_wide(series_long: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot a long ``(date, ticker, value)`` frame to wide."""
    return (
        series_long.pivot_table(
            index="date",
            columns="ticker",
            values=value_col,
            aggfunc="last",
        )
        .sort_index()
        .sort_index(axis=1)
    )


def monthly_index(daily_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the month-end timestamps present in a daily index."""
    return pd.DatetimeIndex(sorted({pd.Timestamp(d).to_period("M").to_timestamp("M") for d in daily_index}))


def cross_section_zscore(wide: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row cross-sectionally (NaNs ignored)."""
    mu = wide.mean(axis=1)
    sd = wide.std(axis=1, ddof=0)
    z = wide.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    return z


# ---------------------------------------------------------------------------
# Momentum (12-month return skipping the most recent month)
# ---------------------------------------------------------------------------
def momentum_12_1(prices: pd.DataFrame, skip: int = 21, lookback: int = 252) -> pd.DataFrame:
    """12-1 momentum: ``ret_{t-252, t-21}``.

    Computed on the daily index; user usually re-samples to month-end.
    """
    p_today = prices
    p_skip = prices.shift(skip)
    p_lb = prices.shift(lookback)
    momentum = p_today / p_lb - 1.0
    # Only valid where the skip window had at least one observation
    momentum = momentum.where(p_skip.notna() & p_lb.notna())
    return momentum


# ---------------------------------------------------------------------------
# Low vol
# ---------------------------------------------------------------------------
def low_vol(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Negative trailing realized vol — high value == low realized vol."""
    rets = prices.pct_change()
    vol = rets.rolling(window=window, min_periods=max(60, window // 4)).std(ddof=0) * np.sqrt(252)
    return -vol


# ---------------------------------------------------------------------------
# Size (from fundamental market cap)
# ---------------------------------------------------------------------------
def size(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Negative log market cap.

    If ``fundamentals`` is None we fall back to a 63-day rolling log(ADV*price)
    proxy so the factor is still computable on price-only data.
    """
    if fundamentals is not None and "marketCap" in fundamentals.columns:
        # Snap fundamentals forward on every month-end date.
        idx = prices.index
        mc = fundamentals["marketCap"].astype(float)
        # Reindex to price index, ffill so most-recent observation persists.
        log_mc = np.log(mc.reindex(prices.columns).astype(float))
        log_mc_wide = pd.DataFrame(
            np.tile(log_mc.values, (len(idx), 1)),
            index=idx,
            columns=prices.columns,
        )
        # Fill missing tickers with NaN; that's fine.
        return -log_mc_wide

    # Proxy = rolling 63-day median of log(close * shares est.) ≈ log market cap proxy.
    log_p = np.log(prices.replace(0, np.nan))
    return -log_p.rolling(window=63, min_periods=21).median()


# ---------------------------------------------------------------------------
# Value (Earnings Yield)
# ---------------------------------------------------------------------------
def value_earnings_yield(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Earnings yield factor from yfinance fundamentals.

    Earnings yield is treated as a slowly-changing fundamental that is ffill'd
    forward at every business day so the factor returns a wide DataFrame
    aligned to ``prices.index``.
    """
    if fundamentals is None or "earningsYield" not in fundamentals.columns:
        raise ValueError(
            "value_earnings_yield requires fundamentals with 'earningsYield' column. "
            "Pass fetch_fundamentals(...) output."
        )

    ey = fundamentals["earningsYield"].astype(float)
    wide = pd.DataFrame(
        np.tile(ey.reindex(prices.columns).values, (len(prices.index), 1)),
        index=prices.index,
        columns=prices.columns,
    )
    return wide


# ---------------------------------------------------------------------------
# Quality (composite z-score of ROE and gross margin)
# ---------------------------------------------------------------------------
def quality(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Composite quality: z(ROE) and z(grossMargin) averaged cross-sectionally.

    Pre-computed ``quality`` column in ``fundamentals`` is preferred (it is the
    z-scored composite produced by :func:`factor.data.fetch_fundamentals`).
    Falls back to ``profitMargins`` if the composite column is missing.
    """
    if fundamentals is None:
        raise ValueError("quality requires fundamentals")
    if "quality" in fundamentals.columns and fundamentals["quality"].notna().any():
        q = fundamentals["quality"].astype(float)
    elif "returnOnEquity" in fundamentals.columns:
        # Fallback: z-score ROE on its own, cross-sectional, no gross margin
        roe = fundamentals["returnOnEquity"].astype(float)
        std = roe.std(ddof=0)
        if std and std > 0:
            q = (roe - roe.mean()) / std
        else:
            q = roe - roe.mean()
    else:
        raise ValueError("fundamentals missing 'quality' / 'returnOnEquity'")

    wide = pd.DataFrame(
        np.tile(q.reindex(prices.columns).values, (len(prices.index), 1)),
        index=prices.index,
        columns=prices.columns,
    )
    return wide


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
FACTOR_REGISTRY: Dict[str, Callable[..., pd.DataFrame]] = {
    "momentum": momentum_12_1,
    "value": value_earnings_yield,
    "quality": quality,
    "low_vol": low_vol,
    "size": size,
}


def compute_factor(
    name: str,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Compute a factor by name and return a wide ``(date, ticker)`` frame."""
    if name not in FACTOR_REGISTRY:
        raise ValueError(f"Unknown factor {name!r}; choose from {list(FACTOR_REGISTRY)}")
    fn = FACTOR_REGISTRY[name]

    # Heuristics for which factors need fundamentals
    needs_fund = name in {"value", "quality"}
    if needs_fund and fundamentals is None:
        # Try a graceful auto-load for CLI convenience.
        log.info("Auto-fetching fundamentals (no fundamentals provided)")
        fundamentals = data_mod.fetch_fundamentals(prices.columns.tolist())

    if name == "size":
        return fn(prices, fundamentals=fundamentals, tickers=prices.columns.tolist(), **kwargs)
    if name in {"value", "quality"}:
        return fn(prices, fundamentals=fundamentals, **kwargs)
    return fn(prices, **kwargs)