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
