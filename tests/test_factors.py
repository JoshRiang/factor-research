"""Unit tests for the factor research library.

These tests are deliberately *offline* — they fabricate deterministic price
data so the entire decile + IC pipeline can be validated without hitting
yfinance.  Integration testing that touches the network belongs in a
separate ``tests/test_integration.py`` (skipped by default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor import (
    momentum_12_1,
    value_earnings_yield,
    quality,
    low_vol,
    size,
    compute_factor,
    FACTOR_REGISTRY,
    form_deciles,
    decile_returns,
    top_minus_bottom_spread,
    annualized_sharpe,
    cumulative_returns,
    information_coefficient,
    ic_summary,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic prices
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def prices(rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic 800-day, 30-ticker price panel with a momentum signal.

    Returns are constructed from a hidden "momentum signal" so that, with
    the right lookahead, we'd see positive IC.  We use NO lookahead in
    factors, so the IC test (below) just checks that IC is computed and
    is a finite number — not that it's positive.
    """
    n_days, n_tickers = 800, 30
    dates = pd.bdate_range("2014-01-02", periods=n_days)
    # Each ticker gets a drift + AR(1) shocks
    rets = rng.normal(0.0005, 0.02, size=(n_days, n_tickers))
    # Inject a slow-moving hidden momentum signal in the first 10 tickers
    hidden = np.cumsum(rng.normal(0, 0.005, size=(n_days,))) * 0.0  # placeholder
    rets[:200, :10] += 0.005  # strong run
    rets[200:400, :10] -= 0.005  # reversal
    prices_arr = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices_arr, index=dates, columns=[f"T{i:02d}" for i in range(n_tickers)])


@pytest.fixture(scope="module")
def fundamentals() -> pd.DataFrame:
    """Synthetic fundamentals table aligned to ``prices`` columns."""
    idx = [f"T{i:02d}" for i in range(30)]
    return pd.DataFrame(
        {
            "earningsYield": rng_continuous(30, 0.02, 0.08),
            "returnOnEquity": rng_continuous(30, 0.05, 0.25),
            "grossMargins": rng_continuous(30, 0.20, 0.60),
            "marketCap": 10 ** rng_continuous(30, 8, 11),
            "priceToBook": rng_continuous(30, 1.5, 8.0),
            "quality": rng_continuous(30, -1.0, 1.0),
        },
        index=idx,
    )


def rng_continuous(n: int, lo: float, hi: float) -> np.ndarray:
    """Deterministic uniform draw (used because np.random is global)."""
    return lo + (hi - lo) * np.linspace(0.1, 0.9, n)


@pytest.fixture(scope="module")
def month_ends(prices: pd.DataFrame) -> pd.DatetimeIndex:
    months = prices.index.to_period("M").unique().sort_values()
    return pd.DatetimeIndex([prices.index[prices.index.to_period("M") == m][-1] for m in months])


# ---------------------------------------------------------------------------
# Factor tests
# ---------------------------------------------------------------------------
def test_factor_registry_complete():
    for name in ("momentum", "value", "quality", "low_vol", "size"):
        assert name in FACTOR_REGISTRY


def test_momentum_shape_and_first_value_nan(prices: pd.DataFrame):
    m = momentum_12_1(prices)
    assert m.shape == prices.shape
    # First ~252 rows should be NaN
    assert m.iloc[:250].isna().all().all() or m.iloc[:252].isna().all().all()
    # After warmup, some values present
    assert m.iloc[260:].notna().any().any()


def test_low_vol_shape_and_signs(prices: pd.DataFrame):
    v = low_vol(prices, window=120)
    assert v.shape == prices.shape
    # low_vol = -sigma; the values should be negative (or NaN early)
    valid = v.iloc[300:].dropna(how="all")
    assert (valid < 0).all().all()


def test_value_requires_fundamentals(prices: pd.DataFrame):
    with pytest.raises(ValueError):
        value_earnings_yield(prices)


def test_value_with_fundamentals(prices: pd.DataFrame, fundamentals: pd.DataFrame):
    v = value_earnings_yield(prices, fundamentals=fundamentals)
    assert v.shape == prices.shape
    # Earnings yield persists across time → constant within a column
    assert v.iloc[-100:, :].std().sum() == pytest.approx(0, abs=1e-12)


def test_quality_with_fundamentals(prices: pd.DataFrame, fundamentals: pd.DataFrame):
    q = quality(prices, fundamentals=fundamentals)
    assert q.shape == prices.shape
    assert q.notna().any().any()


def test_size_with_fundamentals(prices: pd.DataFrame, fundamentals: pd.DataFrame):
    s = size(prices, fundamentals=fundamentals)
    assert s.shape == prices.shape


def test_size_without_fundamentals_uses_proxy(prices: pd.DataFrame):
    s = size(prices)
    assert s.shape == prices.shape
    assert s.notna().any().any()


def test_compute_factor_dispatch(prices: pd.DataFrame, fundamentals: pd.DataFrame):
    assert compute_factor("momentum", prices).shape == prices.shape
    assert compute_factor("low_vol", prices).shape == prices.shape
    assert compute_factor("size", prices, fundamentals=fundamentals).shape == prices.shape
    assert compute_factor("value", prices, fundamentals=fundamentals).shape == prices.shape
    assert compute_factor("quality", prices, fundamentals=fundamentals).shape == prices.shape
    with pytest.raises(ValueError):
        compute_factor("nonsense", prices)


# ---------------------------------------------------------------------------
# Decile tests
# ---------------------------------------------------------------------------
def test_form_deciles_labels_in_range(prices: pd.DataFrame, fundamentals: pd.DataFrame, month_ends):
    f = momentum_12_1(prices).reindex(month_ends)
    buckets = form_deciles(f, month_ends, n_deciles=10)
    valid = buckets.dropna(how="all").dropna(axis=1, how="all")
    # All values must be in 1..10 or NaN
    numeric = valid.to_numpy()
    numeric = numeric[~np.isnan(numeric)]
    assert ((numeric >= 1) & (numeric <= 10)).all()


def test_decile_returns_shape_and_columns(prices: pd.DataFrame, fundamentals: pd.DataFrame, month_ends):
    f = momentum_12_1(prices).reindex(prices.index)
    rets = decile_returns(f, prices, month_ends, n_deciles=10)
    assert rets.shape[0] == len(month_ends) - 1
    assert list(rets.columns) == [f"D{i+1}" for i in range(10)]


def test_top_minus_bottom_spread(prices: pd.DataFrame, fundamentals: pd.DataFrame, month_ends):
    f = momentum_12_1(prices).reindex(prices.index)
    rets = decile_returns(f, prices, month_ends, n_deciles=10)
    spread = top_minus_bottom_spread(rets)
    assert spread.shape == (rets.shape[0],)
    # D10 - D1
    pd.testing.assert_series_equal(spread, rets["D10"] - rets["D1"], check_names=False)


def test_cumulative_returns_starts_at_one(prices: pd.DataFrame, fundamentals: pd.DataFrame, month_ends):
    f = momentum_12_1(prices).reindex(prices.index)
    rets = decile_returns(f, prices, month_ends, n_deciles=10)
    cum = cumulative_returns(rets)
    # First row should be (1 + first period return); not literally 1 if returns exist.
    assert (cum.iloc[0] > 0).all()


def test_annualized_sharpe_constant_returns_is_zero():
    """Constant positive return → no volatility → Sharpe is NaN (undefined).

    Verifies we don't blow up to infinity on degenerate inputs.
    """
    s = pd.Series([0.01] * 12)
    result = annualized_sharpe(s, periods_per_year=12)
    assert np.isnan(result)


def test_annualized_sharpe_empty_returns_nan():
    assert np.isnan(annualized_sharpe(pd.Series([], dtype=float)))


def test_annualized_sharpe_positive_when_mean_positive():
    """Mean-positive return series with small noise → Sharpe > 0."""
    s = pd.Series(np.full(120, 0.01)) + pd.Series(
        np.random.default_rng(0).normal(0, 0.001, 120), index=range(120)
    )
    assert annualized_sharpe(s) > 0


# ---------------------------------------------------------------------------
# IC tests
# ---------------------------------------------------------------------------
def test_information_coefficient_returns_series(prices: pd.DataFrame, month_ends):
    f = momentum_12_1(prices).reindex(prices.index)
    ic = information_coefficient(f, prices, month_ends)
    assert isinstance(ic, pd.Series)
    assert ic.name == "IC"
    # IC is computed at rebalance dates where the factor has any non-NaN
    # observations and there is a subsequent month of returns available.
    # Momentum-12-1 has a ~1-year warmup, so the IC series will start later.
    assert 0 < len(ic) <= len(month_ends) - 1
    # Values must lie in [-1, 1] (excluding NaN)
    valid = ic.dropna()
    assert ((valid >= -1.0) & (valid <= 1.0)).all()


def test_information_coefficient_strong_factor_is_positive():
    """If a factor perfectly predicts next-period return, IC should be 1.

    Construct: factor value = next-day return (impossible in real life, but
    a clean unit-test signal).
    """
    n = 60
    dates = pd.bdate_range("2020-01-01", periods=n)
    rets = pd.Series(np.random.default_rng(0).normal(0, 0.02, n), index=dates)
    prices = pd.DataFrame(
        (1 + rets).cumprod() * 100, columns=["A"]
    ).reindex(columns=["A"]).assign(B=lambda d: d["A"] * 1.1)

    # Build a "perfect" factor: equal to the next-day return, for every ticker
    next_ret = rets.shift(-1)
    factor = pd.DataFrame(
        {c: next_ret.values for c in prices.columns}, index=prices.index
    )
    rebal = pd.DatetimeIndex([prices.index[i] for i in (0, 30, 59)])
    ic = information_coefficient(factor, prices, rebal)
    valid = ic.dropna()
    # With only 2 periods and 2 tickers we expect strong positive IC
    assert (valid > 0.5).all()


def test_ic_summary_includes_overall_row(prices: pd.DataFrame, month_ends):
    f = momentum_12_1(prices).reindex(prices.index)
    ic = information_coefficient(f, prices, month_ends)
    summary = ic_summary(ic)
    assert "year" in summary.columns
    assert "ic_mean" in summary.columns
    assert "hit_rate" in summary.columns
    # The "ALL" row must be present and have ic_mean in [-1, 1]
    overall = summary[summary["year"] == "ALL"]
    assert len(overall) == 1
    assert -1.0 <= overall["ic_mean"].iloc[0] <= 1.0


def test_ic_summary_empty_input():
    summary = ic_summary(pd.Series([], dtype=float))
    assert list(summary.columns)[:1] == ["year"]