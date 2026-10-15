"""Fetch historical and latest OHLCV candles from a public exchange API via ccxt.

No API key is required or used here -- OHLCV market data is public. This module
never places orders and never touches an account.
"""
from __future__ import annotations

import time

import ccxt
import pandas as pd


def _make_exchange(exchange_id: str) -> ccxt.Exchange:
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": True})


def fetch_ohlcv_history(exchange_id: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch up to `limit` historical candles, paging backwards as needed."""
    exchange = _make_exchange(exchange_id)
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

    all_rows: list[list] = []
    since = exchange.milliseconds() - limit * timeframe_ms
    while len(all_rows) < limit:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts == since:
            break
        since = last_ts + timeframe_ms
        if len(batch) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.tail(limit).reset_index(drop=True)


def fetch_latest_candles(exchange_id: str, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    """Fetch the most recent `limit` candles -- used by the paper-trading loop."""
    exchange = _make_exchange(exchange_id)
    batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(batch, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df
