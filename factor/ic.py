"""Information Coefficient (IC): Spearman rank correlation between the factor
and the *next-period* return, computed per rebalance period.

The IC time-series is then summarised by calendar year, plus the standard
``IC mean / std / t-stat / IC>0 ratio`` statistics that appear in factor
research.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def information_coefficient(
    factor_wide: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Spearman IC of factor vs next 1-period return, per rebalance date.

    Each period uses factor values as of ``t`` and the simple return from
    ``t`` (exclusive) to ``t+1`` (inclusive), giving a 1-month-ahead IC for
    monthly rebalance frequency.
    """
    factor_aligned = factor_wide.reindex(prices.index)
    daily_rets = prices.pct_change()
    rebal = pd.DatetimeIndex(sorted(set(rebalance_dates).intersection(prices.index)))

    ics: dict[pd.Timestamp, float] = {}
    for i in range(len(rebal) - 1):
        t0, t1 = rebal[i], rebal[i + 1]
        f_t = factor_aligned.loc[t0]
        window = daily_rets.loc[t0:t1].iloc[1:]
        if window.empty:
            continue
        next_ret = (1.0 + window).prod() - 1.0
        df = pd.concat([f_t, next_ret], axis=1, keys=["f", "r"]).dropna()
        if len(df) < 5:
            continue
        rho, _ = spearmanr(df["f"].values, df["r"].values)
        ics[t0] = float(rho) if rho is not None and not np.isnan(rho) else np.nan

    return pd.Series(ics, name="IC").sort_index()


def ic_summary(ic_series: pd.Series) -> pd.DataFrame:
    """Per-year + overall summary of an IC series.

    Columns: ``year``, ``n``, ``ic_mean``, ``ic_std``, ``ic_ir`` (IC mean / IC std),
    ``hit_rate`` (fraction of IC > 0), ``t_stat`` (IC mean / (IC std / sqrt(n))).
    """
    if ic_series.empty:
        return pd.DataFrame(
            columns=["year", "n", "ic_mean", "ic_std", "ic_ir", "hit_rate", "t_stat"]
        )

    s = ic_series.dropna()
    df = pd.DataFrame({"year": s.index.year, "ic": s.values})

    def _agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        mu = g["ic"].mean()
        sd = g["ic"].std(ddof=1)
        ir = mu / sd if sd and not np.isnan(sd) and sd != 0 else np.nan
        hit = (g["ic"] > 0).mean()
        t = mu / (sd / np.sqrt(n)) if sd and not np.isnan(sd) and sd != 0 else np.nan
        return pd.Series(
            {"n": n, "ic_mean": mu, "ic_std": sd, "ic_ir": ir, "hit_rate": hit, "t_stat": t}
        )

    by_year = df.groupby("year").apply(_agg, include_groups=False).reset_index()
    overall = _agg(df)
    overall_row = pd.DataFrame(
        [
            {
                "year": "ALL",
                "n": overall["n"],
                "ic_mean": overall["ic_mean"],
                "ic_std": overall["ic_std"],
                "ic_ir": overall["ic_ir"],
                "hit_rate": overall["hit_rate"],
                "t_stat": overall["t_stat"],
            }
        ]
    )
    return pd.concat([by_year, overall_row], ignore_index=True)