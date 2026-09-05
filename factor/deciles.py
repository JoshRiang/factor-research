"""Decile portfolio formation.

At each rebalance date (default: month-end), split the cross-section into
``n`` equal-sized groups based on factor value, then compute equal-weight
portfolio returns through the next rebalance date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def form_deciles(
    factor_wide: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Assign each ticker to a decile at each rebalance date.

    Parameters
    ----------
    factor_wide : DataFrame
        Wide (date × ticker) factor scores.
    rebalance_dates : DatetimeIndex
        Subset of ``factor_wide.index`` at which to form groups.  We
        forward-fill group assignments so the output is a full
        (date × ticker) DataFrame with integer decile labels (1..n_deciles,
        or NaN if the ticker has no factor value at the time).
    n_deciles : int
        Number of buckets.  10 = deciles, 5 = quintiles, etc.

    Returns
    -------
    DataFrame
        Integer decile assignment, same shape as ``factor_wide``.
    """
    factor_wide = factor_wide.reindex(rebalance_dates)
    # Cross-sectional rank using average ties → percentile → bucket
    rank_pct = factor_wide.rank(axis=1, pct=True, na_option="keep")
    # Quantile bucketing via searchsorted on [0, 1/n, 2/n, ..., 1]
    edges = np.linspace(0.0, 1.0, n_deciles + 1)
    # searchsorted on each row (vectorized via apply)
    def _row_bucket(row: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=row.index)
        mask = row.notna()
        if mask.any():
            vals = row[mask].values
            # Assign bucket index 0..n_deciles-1 via np.searchsorted
            # but cap at n_deciles - 1.
            b = np.minimum(np.searchsorted(edges, vals, side="right") - 1, n_deciles - 1)
            out.loc[mask] = b + 1
        return out

    bucketed = rank_pct.apply(_row_bucket, axis=1)
    bucketed = bucketed.reindex(factor_wide.index.union(factor_wide.index)) if False else bucketed

    # Forward-fill across the full daily index so subsequent returns can
    # be assigned to a held bucket.  Use the rebalance_dates index first.
    full_index = factor_wide.index
    # If our factor_wide was sub-selected to rebalance_dates only, reindex
    # back to a representative daily index using the original caller.
    bucketed = bucketed.reindex(full_index).ffill()
    return bucketed.astype(float)


def decile_returns(
    factor_wide: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Compute equal-weight return of each decile per rebalance period.

    Returns a ``DataFrame`` indexed by rebalance date with columns
    ``"D1"`` (worst factor) … ``"D{n_deciles}"`` (best factor).  Within a
    period, returns are compounded from daily simple returns of the
    equal-weight basket of that decile.
    """
    # Align factor to prices index
    factor_aligned = factor_wide.reindex(prices.index)
    buckets = form_deciles(factor_aligned, rebalance_dates, n_deciles=n_deciles)

    daily_rets = prices.pct_change()
    rebal = pd.DatetimeIndex(sorted(set(rebalance_dates).intersection(prices.index)))
    if len(rebal) < 2:
        raise ValueError("Need at least 2 rebalance dates")

    out = pd.DataFrame(index=rebal[:-1], columns=[f"D{i+1}" for i in range(n_deciles)], dtype=float)
    for i in range(len(rebal) - 1):
        t0, t1 = rebal[i], rebal[i + 1]
        window = daily_rets.loc[t0:t1].iloc[1:]  # exclude t0 (we buy at t0 close)
        if window.empty:
            continue
        # Membership as of t0
        members = buckets.loc[t0]
        for d in range(1, n_deciles + 1):
            tickers_in_d = members.index[members == d].tolist()
            valid = [t for t in tickers_in_d if t in window.columns and window[t].notna().any()]
            if not valid:
                out.iloc[i, d - 1] = np.nan
                continue
            period_ret = (1.0 + window[valid]).prod().mean() - 1.0  # equal-weight
            out.iloc[i, d - 1] = period_ret
    return out.astype(float)


def top_minus_bottom_spread(deciles_returns: pd.DataFrame) -> pd.Series:
    """Long top decile, short bottom decile (D{n} - D1) per period."""
    last = deciles_returns.columns[-1]
    first = deciles_returns.columns[0]
    return deciles_returns[last] - deciles_returns[first]


def cumulative_returns(period_returns: pd.DataFrame) -> pd.DataFrame:
    """Compound (1 + r) starting at 1."""
    return (1.0 + period_returns.fillna(0.0)).cumprod()


def annualized_sharpe(period_returns: pd.Series, periods_per_year: float = 12.0) -> float:
    """Annualized Sharpe ratio of a periodic return series.

    Default ``periods_per_year=12`` because the factor backtest is monthly.
    """
    r = period_returns.dropna()
    if r.empty:
        return float("nan")
    mu = r.mean() * periods_per_year
    sd = r.std(ddof=1) * np.sqrt(periods_per_year)
    # Treat near-epsilon std as zero — avoids spurious Sharpe blowups.
    if not np.isfinite(sd) or abs(sd) < 1e-12:
        return float("nan")
    return float(mu / sd)