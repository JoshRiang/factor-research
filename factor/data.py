"""Universe and market data loaders.

Pulls end-of-day prices (yfinance) and fundamentals (yfinance ``.info`` /
quarterly financials) into tidy long-format DataFrames indexed by
``(date, ticker)``.  A curated 50-large-cap US universe is the default;
``load_universe("sp500")`` can be used to attempt the full S&P 500 list
(yfinance is slow / flaky for that, so we fall back to the curated list
on failure).
"""

from __future__ import annotations

import io as _io
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curated universe — 50 large-cap US tickers spanning sectors.  Chosen for
# data availability going back to ~2010 and reasonable trading liquidity.
# ---------------------------------------------------------------------------
DEFAULT_UNIVERSE: list[str] = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "ADBE", "CSCO",
    "INTC", "AMD", "QCOM", "TXN", "IBM",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "USB", "PNC",
    # Healthcare
    "JNJ", "PFE", "MRK", "ABBV", "LLY", "UNH", "TMO", "ABT", "MDT", "BMY",
    # Consumer
    "AMZN", "TSLA", "HD", "MCD", "NKE", "PG", "KO", "PEP", "WMT", "COST",
    # Industrials / Energy
    "BA", "GE", "CAT", "XOM", "CVX", "COP",
]


def load_universe(name: str = "default", cache_dir: str | os.PathLike | None = None) -> list[str]:
    """Return a list of tickers.

    Parameters
    ----------
    name : {"default", "sp500", path-to-txt-file}
        ``"default"`` returns the curated 50-ticker list.
        ``"sp500"`` attempts to download the Wikipedia S&P 500 table (slow,
        often blocked).  Falls back to the curated list on any failure.
        A path to a newline-delimited .txt file returns that list.
    cache_dir : optional directory for caching the sp500 list.
    """
    if name == "default":
        return list(DEFAULT_UNIVERSE)

    if name == "sp500":
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            syms = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
            return syms
        except Exception as exc:  # noqa: BLE001
            log.warning("Falling back to default universe (sp500 fetch failed: %s)", exc)
            return list(DEFAULT_UNIVERSE)

    path = Path(name)
    if path.exists() and path.is_file():
        return [t.strip() for t in path.read_text().splitlines() if t.strip()]

    raise ValueError(f"Unknown universe specifier: {name!r}")


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------
def fetch_prices(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    cache_dir: str | os.PathLike | None = None,
    retries: int = 3,
    sleep: float = 0.5,
) -> pd.DataFrame:
    """Download adjusted close prices for ``tickers``.

    Returns a ``DataFrame`` indexed by ``Date`` with one column per ticker
    (adjusted close).  Missing columns are dropped with a warning.
    """
    import yfinance as yf  # local import — optional dep

    tickers = list(tickers)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    end_str = end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    frames: dict[str, pd.Series] = {}
    missing: list[str] = []
    for tkr in tickers:
        if cache_dir is not None:
            cached = cache_dir / f"{tkr}_{start}_{end_str}.csv"
            if cached.exists():
                try:
                    s = pd.read_csv(cached, parse_dates=["Date"]).set_index("Date")["Close"]
                    s.name = tkr
                    frames[tkr] = s
                    continue
                except Exception:
                    pass

        last_exc: Exception | None = None
        s: pd.Series | None = None
        for attempt in range(retries):
            try:
                df = yf.download(
                    tkr,
                    start=str(start),
                    end=str(end_str),
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                if df.empty or "Close" not in df.columns:
                    last_exc = RuntimeError("empty download")
                else:
                    s = df["Close"].copy()
                    s.name = tkr
                    break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(sleep * (attempt + 1))
        if s is None:
            missing.append(tkr)
            log.warning("Failed to download %s: %s", tkr, last_exc)
            continue
        frames[tkr] = s
        if cache_dir is not None:
            try:
                out = frames[tkr].to_frame(name="Close").reset_index()
                out.to_csv(cached, index=False)
            except Exception:
                pass

    if not frames:
        raise RuntimeError("No price data downloaded for any ticker")

    prices = pd.concat(frames.values(), axis=1, join="outer")
    prices.columns = list(frames.keys())
    prices = prices.sort_index()
    if missing:
        log.info("Missing tickers (skipped): %s", missing)

    return prices


# ---------------------------------------------------------------------------
# Fundamentals — earnings yield (value proxy) and quality metrics
# ---------------------------------------------------------------------------
def _safe_get_info(ticker: str, retries: int = 3, sleep: float = 0.3) -> dict:
    import yfinance as yf

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            obj = yf.Ticker(ticker)
            info = obj.info or {}
            if info:
                return info
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(sleep * (attempt + 1))
    log.debug("Info fetch failed for %s: %s", ticker, last_exc)
    return {}


def fetch_fundamentals(
    tickers: Iterable[str],
    cache_dir: str | os.PathLike | None = None,
) -> pd.DataFrame:
    """Fetch point-in-time fundamentals from yfinance ``.info``.

    Returns a ``DataFrame`` indexed by ticker with columns:

    ``trailingPE``, ``forwardPE``, ``priceToBook``, ``marketCap``,
    ``grossMargins``, ``operatingMargins``, ``profitMargins``,
    ``returnOnEquity``, ``returnOnAssets``, ``beta``,
    ``earningsYield`` (1 / trailingPE),
    ``bookToPrice`` (1 / priceToBook when priceToBook > 0),
    ``size`` (log marketCap),
    ``quality`` (mean z-score of ROE and grossMargin across the universe).
    """
    tickers = list(tickers)
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "fundamentals_snapshot.csv"
        if cache_path.exists() and cache_path.stat().st_mtime > (time.time() - 7 * 86400):
            try:
                cached = pd.read_csv(cache_path).set_index("ticker")
                # Only return cached tickers
                cached = cached.loc[[t for t in cached.index if t in set(tickers)]]
                if len(cached) == len(tickers):
                    return cached
            except Exception:
                pass

    rows: list[dict] = []
    for tkr in tickers:
        info = _safe_get_info(tkr)
        rows.append(
            {
                "ticker": tkr,
                "trailingPE": info.get("trailingPE"),
                "priceToBook": info.get("priceToBook"),
                "marketCap": info.get("marketCap"),
                "grossMargins": info.get("grossMargins"),
                "operatingMargins": info.get("operatingMargins"),
                "profitMargins": info.get("profitMargins"),
                "returnOnEquity": info.get("returnOnEquity"),
                "returnOnAssets": info.get("returnOnAssets"),
                "beta": info.get("beta"),
            }
        )
        time.sleep(0.05)  # gentle rate limit

    df = pd.DataFrame(rows).set_index("ticker")
    df["earningsYield"] = 1.0 / df["trailingPE"].replace({0: np.nan})
    df["bookToPrice"] = 1.0 / df["priceToBook"].replace({0: np.nan})

    # Size — log market cap (lower size value = smaller-cap)
    df["size"] = np.log(df["marketCap"].astype(float))

    # Quality — z-score of ROE and grossMargin across universe, then average.
    # Higher quality == more attractive.
    def _zscore(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        std = s.std(ddof=0)
        if not std or np.isnan(std) or std == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / std

    z_roe = _zscore(df["returnOnEquity"])
    z_gm = _zscore(df["grossMargins"])
    df["quality"] = (z_roe.fillna(0) + z_gm.fillna(0)) / 2.0

    if cache_path is not None:
        try:
            df.reset_index().to_csv(cache_path, index=False)
        except Exception:
            pass

    return df


def realized_vol_252(returns: pd.DataFrame) -> pd.Series:
    """Trailing 252-day annualized realized vol per ticker (used by ``low_vol``)."""
    return returns.rolling(window=252, min_periods=126).std(ddof=0) * np.sqrt(252)