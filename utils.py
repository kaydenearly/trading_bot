from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown.min()


def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    if returns.std() == 0 or returns.empty:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def periods_per_year(timeframe: str) -> float:
    """Approximate number of candles per year for a ccxt timeframe string."""
    unit = timeframe[-1]
    n = int(timeframe[:-1])
    minutes_per_unit = {"m": 1, "h": 60, "d": 60 * 24, "w": 60 * 24 * 7}[unit]
    minutes = n * minutes_per_unit
    return (365 * 24 * 60) / minutes


def timeframe_to_hours(timeframe: str) -> float:
    unit = timeframe[-1]
    n = int(timeframe[:-1])
    hours_per_unit = {"m": 1 / 60, "h": 1, "d": 24, "w": 24 * 7}[unit]
    return n * hours_per_unit


def sortino_ratio(returns: pd.Series, periods_per_year: float, mar: float = 0.0) -> float:
    """Like Sharpe, but only penalizes downside deviation below `mar` (minimum
    acceptable return, default 0 per period)."""
    if returns.empty:
        return 0.0
    downside = returns[returns < mar] - mar
    downside_std = np.sqrt((downside**2).mean()) if len(downside) else 0.0
    if downside_std == 0:
        return 0.0
    return float((returns.mean() - mar) / downside_std * np.sqrt(periods_per_year))


def cagr(equity: pd.Series, periods_per_year: float) -> float:
    """Compound annual growth rate from a normalized equity curve (equity[0] > 0)."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / periods_per_year
    if years <= 0 or total_return <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def exposure_pct(position: pd.Series) -> float:
    """Fraction of bars with a nonzero position (time spent in the market)."""
    if position.empty:
        return 0.0
    return float((position != 0).mean())


def trade_stats(trades: pd.DataFrame) -> dict:
    """Round-trip-trade statistics. `trades` needs a `return_pct` column (and
    optionally `holding_bars`). Returns zeros/NaN-safe defaults when empty."""
    if trades.empty:
        return {
            "n_trades": 0, "win_rate": 0.0, "profit_factor": float("nan"),
            "avg_win_pct": 0.0, "avg_loss_pct": 0.0, "expectancy_pct": 0.0,
            "max_consecutive_losses": 0, "avg_holding_bars": 0.0,
        }

    wins = trades[trades["return_pct"] > 0]["return_pct"]
    losses = trades[trades["return_pct"] <= 0]["return_pct"]
    win_rate = len(wins) / len(trades)
    gross_profit = wins.sum()
    gross_loss = -losses.sum()
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else float("nan")

    is_loss = (trades["return_pct"] <= 0).astype(int)
    streak = 0
    max_streak = 0
    for v in is_loss:
        streak = streak + 1 if v else 0
        max_streak = max(max_streak, streak)

    return {
        "n_trades": len(trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_win_pct": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_pct": float(losses.mean()) if len(losses) else 0.0,
        "expectancy_pct": float(trades["return_pct"].mean()),
        "max_consecutive_losses": int(max_streak),
        "avg_holding_bars": float(trades["holding_bars"].mean()) if "holding_bars" in trades else 0.0,
    }
