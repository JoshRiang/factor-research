"""Factor Research: decile backtests and information coefficient analysis."""

from .factors import (
    momentum_12_1,
    value_earnings_yield,
    quality,
    low_vol,
    size,
    compute_factor,
    FACTOR_REGISTRY,
)
from .deciles import (
    form_deciles,
    decile_returns,
    top_minus_bottom_spread,
    cumulative_returns,
    annualized_sharpe,
)
from .ic import information_coefficient, ic_summary
from .data import load_universe, DEFAULT_UNIVERSE, fetch_prices, fetch_fundamentals

__all__ = [
    # factors
    "momentum_12_1",
    "value_earnings_yield",
    "quality",
    "low_vol",
    "size",
    "compute_factor",
    "FACTOR_REGISTRY",
    # deciles
    "form_deciles",
    "decile_returns",
    "top_minus_bottom_spread",
    "cumulative_returns",
    "annualized_sharpe",
    # IC
    "information_coefficient",
    "ic_summary",
    # data
    "load_universe",
    "DEFAULT_UNIVERSE",
    "fetch_prices",
    "fetch_fundamentals",
]

__version__ = "0.1.0"