"""Shared train/backtest logic for a single symbol.

Used by backtest.py (one symbol, verbose + plots), universe.py (screens many
symbols quickly), and retrain.yml (retrains the chosen symbols' models).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data_feed import fetch_ohlcv_history
from features import make_dataset
from model import predict_proba_up, train_model
from utils import max_drawdown, periods_per_year, sharpe_ratio


def symbol_slug(symbol: str) -> str:
    """'BTC/USD' -> 'BTC_USD', safe for use as a filename / JSON key-free path."""
    return symbol.replace("/", "_")


def simulate(test_df: pd.DataFrame, proba_up: pd.Series, cfg: dict) -> pd.DataFrame:
    """Hysteresis long/flat strategy with fees + slippage, bar-by-bar."""
    buy_th = cfg["buy_threshold"]
    sell_th = cfg["sell_threshold"]
    fee_rate = cfg["fee_rate"]
    slippage = cfg["slippage_bps"] / 10_000

    position = 0.0  # 0 = flat, 1 = fully long
    cash = 1.0      # normalized starting equity = 1.0
    units = 0.0
    equity_curve = []
    positions = []

    closes = test_df["close"].values
    for i, p_up in enumerate(proba_up.values):
        price = closes[i]
        if p_up >= buy_th:
            desired = 1.0
        elif p_up <= sell_th:
            desired = 0.0
        else:
            desired = position

        if desired != position:
            fill_price = price * (1 + slippage) if desired > position else price * (1 - slippage)
            if desired == 1.0:
                units = cash / fill_price
                fee = units * fill_price * fee_rate
                cash = -fee
            else:
                proceeds = units * fill_price
                fee = proceeds * fee_rate
                cash = proceeds - fee
                units = 0.0
            position = desired

        equity_curve.append(cash + units * price)
        positions.append(position)

    out = test_df[["timestamp", "close"]].copy()
    out["equity"] = equity_curve
    out["position"] = positions
    out["strategy_return"] = out["equity"].pct_change().fillna(0)
    out["buy_hold_equity"] = out["close"] / out["close"].iloc[0]
    return out


@dataclass
class EvalResult:
    symbol: str
    model: object
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    result: pd.DataFrame          # simulate() output
    strat_total_return: float
    bh_total_return: float
    strat_sharpe: float
    strat_max_drawdown: float
    n_trades: int
    n_candles: int


def train_and_evaluate(symbol: str, cfg: dict, log=None) -> EvalResult | None:
    """Fetch history for `symbol`, train on the first train_fraction, backtest
    the rest. Returns None if there isn't enough usable history."""
    try:
        raw = fetch_ohlcv_history(cfg["exchange"], symbol, cfg["timeframe"], cfg["history_limit"])
    except Exception as e:
        if log:
            log.warning("Skipping %s: failed to fetch history (%s)", symbol, e)
        return None

    dataset = make_dataset(raw, cfg["label_horizon"], cfg["label_threshold"])
    min_rows = cfg.get("universe", {}).get("min_candles", 200)
    if len(dataset) < min_rows:
        if log:
            log.info("Skipping %s: only %d usable candles (need >= %d)", symbol, len(dataset), min_rows)
        return None

    split = int(len(dataset) * cfg["train_fraction"])
    train_df = dataset.iloc[:split].reset_index(drop=True)
    test_df = dataset.iloc[split:].reset_index(drop=True)
    if len(train_df) < 50 or len(test_df) < 20:
        if log:
            log.info("Skipping %s: train/test split too small (%d/%d)", symbol, len(train_df), len(test_df))
        return None

    model = train_model(train_df)
    proba_up = predict_proba_up(model, test_df)
    result = simulate(test_df, proba_up, cfg)

    ppy = periods_per_year(cfg["timeframe"])
    strat_sharpe = sharpe_ratio(result["strategy_return"], ppy)
    strat_dd = max_drawdown(result["equity"])
    strat_total_return = result["equity"].iloc[-1] - 1.0
    bh_total_return = result["buy_hold_equity"].iloc[-1] - 1.0
    n_trades = int((result["position"].diff().fillna(0) != 0).sum())

    return EvalResult(
        symbol=symbol, model=model, train_df=train_df, test_df=test_df, result=result,
        strat_total_return=strat_total_return, bh_total_return=bh_total_return,
        strat_sharpe=strat_sharpe, strat_max_drawdown=strat_dd, n_trades=n_trades,
        n_candles=len(dataset),
    )
