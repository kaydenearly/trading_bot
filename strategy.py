"""Shared train/backtest logic for a single symbol.

Used by backtest.py (one symbol, verbose + plots), universe.py (screens many
symbols quickly), walk_forward.py (many folds per symbol), and retrain.yml
(retrains the chosen symbols' models).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data_feed import fetch_ohlcv_history
from features import make_dataset
from model import predict_proba_up, train_model
from utils import cagr, exposure_pct, max_drawdown, periods_per_year, sharpe_ratio, sortino_ratio, trade_stats


def symbol_slug(symbol: str) -> str:
    """'BTC/USD' -> 'BTC_USD', safe for use as a filename / JSON key-free path."""
    return symbol.replace("/", "_")


def fetch_dataset(symbol: str, cfg: dict) -> pd.DataFrame:
    """Fetch full history for `symbol` and build the feature/label dataset once.
    Reused across many walk-forward folds instead of re-fetching per fold."""
    raw = fetch_ohlcv_history(cfg["exchange"], symbol, cfg["timeframe"], cfg["history_limit"])
    return make_dataset(raw, cfg["label_horizon"], cfg["label_threshold"])


def simulate(test_df: pd.DataFrame, proba_up: pd.Series, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hysteresis long/flat strategy with fees + slippage, bar-by-bar.

    Returns (result_df, trades_df). trades_df has one row per completed
    round-trip (entry -> exit) with return_pct and holding_bars, used for
    win rate / profit factor / expectancy / consecutive-loss stats.
    """
    buy_th = cfg["buy_threshold"]
    sell_th = cfg["sell_threshold"]
    fee_rate = cfg["fee_rate"]
    slippage = cfg["slippage_bps"] / 10_000

    position = 0.0  # 0 = flat, 1 = fully long
    cash = 1.0      # normalized starting equity = 1.0
    units = 0.0
    equity_curve = []
    positions = []
    trades = []
    entry_idx = None
    entry_equity = None
    entry_price = None

    closes = test_df["close"].values
    timestamps = test_df["timestamp"].values
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
                entry_idx, entry_equity, entry_price = i, cash + units * fill_price, fill_price
            else:
                proceeds = units * fill_price
                fee = proceeds * fee_rate
                cash = proceeds - fee
                units = 0.0
                exit_equity = cash
                if entry_idx is not None:
                    trades.append({
                        "entry_time": timestamps[entry_idx], "exit_time": timestamps[i],
                        "entry_price": entry_price, "exit_price": fill_price,
                        "return_pct": exit_equity / entry_equity - 1.0,
                        "holding_bars": i - entry_idx,
                    })
                entry_idx = entry_equity = entry_price = None
            position = desired

        equity_curve.append(cash + units * price)
        positions.append(position)

    out = test_df[["timestamp", "close"]].copy()
    out["equity"] = equity_curve
    out["position"] = positions
    out["strategy_return"] = out["equity"].pct_change().fillna(0)
    out["buy_hold_equity"] = out["close"] / out["close"].iloc[0]
    trades_df = pd.DataFrame(trades)
    return out, trades_df


@dataclass
class EvalResult:
    symbol: str
    model: object
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    result: pd.DataFrame          # simulate() bar-by-bar output
    trades: pd.DataFrame          # simulate() round-trip trades
    strat_total_return: float
    bh_total_return: float
    strat_sharpe: float
    strat_sortino: float
    strat_cagr: float
    strat_max_drawdown: float
    strat_exposure: float
    n_trades: int
    n_candles: int
    stats: dict                   # trade_stats() dict: win_rate, profit_factor, expectancy, ...


def fit_and_test(symbol: str, train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: dict) -> EvalResult | None:
    """Train on train_df, evaluate on test_df. The reusable core of a single
    walk-forward fold (or the one split train_and_evaluate uses)."""
    if len(train_df) < 50 or len(test_df) < 20:
        return None

    model = train_model(train_df)
    proba_up = predict_proba_up(model, test_df)
    result, trades = simulate(test_df, proba_up, cfg)

    ppy = periods_per_year(cfg["timeframe"])
    strat_sharpe = sharpe_ratio(result["strategy_return"], ppy)
    strat_sortino = sortino_ratio(result["strategy_return"], ppy)
    strat_cagr = cagr(result["equity"], ppy)
    strat_dd = max_drawdown(result["equity"])
    strat_exposure = exposure_pct(result["position"])
    strat_total_return = result["equity"].iloc[-1] - 1.0
    bh_total_return = result["buy_hold_equity"].iloc[-1] - 1.0
    n_trades = int((result["position"].diff().fillna(0) != 0).sum())
    stats = trade_stats(trades)

    return EvalResult(
        symbol=symbol, model=model, train_df=train_df, test_df=test_df, result=result, trades=trades,
        strat_total_return=strat_total_return, bh_total_return=bh_total_return,
        strat_sharpe=strat_sharpe, strat_sortino=strat_sortino, strat_cagr=strat_cagr,
        strat_max_drawdown=strat_dd, strat_exposure=strat_exposure, n_trades=n_trades,
        n_candles=len(train_df) + len(test_df), stats=stats,
    )


def train_and_evaluate(symbol: str, cfg: dict, log=None) -> EvalResult | None:
    """Fetch history for `symbol`, train on the first train_fraction, backtest
    the rest (a single 70/30-style split). Returns None if there isn't enough
    usable history. For proper multi-fold validation see walk_forward.py."""
    try:
        dataset = fetch_dataset(symbol, cfg)
    except Exception as e:
        if log:
            log.warning("Skipping %s: failed to fetch history (%s)", symbol, e)
        return None

    min_rows = cfg.get("universe", {}).get("min_candles", 200)
    if len(dataset) < min_rows:
        if log:
            log.info("Skipping %s: only %d usable candles (need >= %d)", symbol, len(dataset), min_rows)
        return None

    split = int(len(dataset) * cfg["train_fraction"])
    train_df = dataset.iloc[:split].reset_index(drop=True)
    test_df = dataset.iloc[split:].reset_index(drop=True)

    ev = fit_and_test(symbol, train_df, test_df, cfg)
    if ev is None and log:
        log.info("Skipping %s: train/test split too small (%d/%d)", symbol, len(train_df), len(test_df))
    return ev
