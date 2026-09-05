"""Matplotlib visualisations for factor research."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-friendly

import matplotlib.pyplot as plt
import pandas as pd

from .deciles import cumulative_returns


def plot_cumulative_decile_returns(
    decile_returns_df: pd.DataFrame,
    title: str = "Cumulative Returns by Decile",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot one line per decile of cumulative returns.

    ``decile_returns_df`` is the wide output of :func:`factor.deciles.decile_returns`.
    """
    cum = cumulative_returns(decile_returns_df)
    fig, ax = plt.subplots(figsize=(11, 6))
    for col in cum.columns:
        ax.plot(cum.index, cum[col], label=col, linewidth=1.4)
    ax.set_title(title)
    ax.set_ylabel("Cumulative growth of $1")
    ax.set_xlabel("Rebalance date")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig


def plot_ic_timeseries(
    ic_series: pd.Series,
    title: str = "Information Coefficient (Spearman)",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot IC time-series as bars (green > 0, red < 0) with rolling mean."""
    s = ic_series.dropna()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in s.values]
    ax.bar(s.index, s.values, color=colors, width=20)
    if len(s) >= 12:
        roll = s.rolling(12, min_periods=6).mean()
        ax.plot(roll.index, roll.values, color="black", linewidth=1.6, label="12-period rolling mean")
        ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_title(title)
    ax.set_ylabel("Spearman IC")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig


def plot_top_minus_bottom(
    spread: pd.Series,
    title: str = "Top minus Bottom Decile (cumulative)",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    cum = (1.0 + spread.fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cum.index, cum.values, color="#1f77b4", linewidth=1.8)
    ax.axhline(1.0, color="black", linewidth=0.7)
    ax.set_title(title)
    ax.set_ylabel("Cumulative growth of $1")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig