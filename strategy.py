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
    """Long/flat strategy with ATR-based risk management, fees + slippage,
    bar-by-bar, using OHLC so stop/target fills are checked intrabar.

    Position sizing is risk-based: a trade that hits its stop should cost
    about risk_per_trade_pct of equity, regardless of the symbol's
    volatility (sized inversely to ATR), capped at 100% of equity (no
    leverage). The stop only ratchets favorably and only repositions once
    it would improve by more than the noise threshold -- a trailing stop
    tied to a moving reference that's allowed to loosen can fail to ever
    trigger, which is exactly the kind of bug this guards against.

    Exit priority each bar: hard stop (loss protection, always honored) ->
    take-profit target -> model-signal flip, but the signal only closes a
    position that's currently profitable (it doesn't override the stop).

    Returns (result_df, trades_df). trades_df's return_pct is the P&L on
    capital actually committed to that trade (not blended with idle cash),
    so win rate / profit factor stay meaningful regardless of position size.
    """
    buy_th = cfg["buy_threshold"]
    sell_th = cfg["sell_threshold"]
    fee_rate = cfg["fee_rate"]
    slippage = cfg["slippage_bps"] / 10_000
    risk_cfg = cfg["risk"]
    risk_per_trade = risk_cfg["risk_per_trade_pct"]
    stop_atr_mult = risk_cfg["stop_atr_mult"]
    target_atr_mult = risk_cfg["target_atr_mult"]
    trail_threshold_mult = risk_cfg["stop_trail_threshold_atr_mult"]

    position = 0.0  # 0 = flat, 1 = long
    cash = 1.0       # normalized starting equity = 1.0
    units = 0.0
    entry_price = stop_price = target_price = stop_distance = None
    capital_committed = None
    entry_idx = None
    equity_curve = []
    positions = []
    trades = []

    closes = test_df["close"].values
    highs = test_df["high"].values
    lows = test_df["low"].values
    atrs = test_df["atr"].values
    timestamps = test_df["timestamp"].values

    def record_trade(exit_idx, fill_price):
        nonlocal entry_idx, entry_price, capital_committed
        proceeds = units * fill_price
        fee = proceeds * fee_rate
        capital_returned = proceeds - fee
        trades.append({
            "entry_time": timestamps[entry_idx], "exit_time": timestamps[exit_idx],
            "entry_price": entry_price, "exit_price": fill_price,
            "return_pct": capital_returned / capital_committed - 1.0,
            "holding_bars": exit_idx - entry_idx,
        })
        return capital_returned

    for i, p_up in enumerate(proba_up.values):
        close, high, low, atr = closes[i], highs[i], lows[i], atrs[i]

        if position == 1.0:
            if low <= stop_price:
                fill_price = stop_price * (1 - slippage)
                cash += record_trade(i, fill_price)
                units, position = 0.0, 0.0
            elif high >= target_price:
                fill_price = target_price * (1 - slippage)
                cash += record_trade(i, fill_price)
                units, position = 0.0, 0.0
            elif p_up <= sell_th and close > entry_price:
                fill_price = close * (1 - slippage)
                cash += record_trade(i, fill_price)
                units, position = 0.0, 0.0
            else:
                candidate_stop = close - stop_distance
                if candidate_stop > stop_price + trail_threshold_mult * atr:
                    stop_price = candidate_stop
        elif p_up >= buy_th and atr > 0:
            equity_before = cash
            fill_price = close * (1 + slippage)
            stop_distance = stop_atr_mult * atr
            risk_amount = equity_before * risk_per_trade
            raw_units = risk_amount / stop_distance
            max_affordable_units = equity_before / fill_price
            units = min(raw_units, max_affordable_units)
            cost = units * fill_price
            fee = cost * fee_rate
            cash -= cost + fee
            capital_committed = cost + fee
            entry_price = fill_price
            stop_price = entry_price - stop_distance
            target_price = entry_price + target_atr_mult * atr
            entry_idx = i
            position = 1.0

        equity_curve.append(cash + units * close)
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
