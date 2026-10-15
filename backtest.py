"""Walk-forward backtest: train on the first `train_fraction` of history,
evaluate the strategy (with simulated fees + slippage) on the untouched
remainder, and compare against buy-and-hold.

Usage:
    python backtest.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_feed import fetch_ohlcv_history
from features import make_dataset
from model import predict_proba_up, save_model, train_model
from utils import load_config, max_drawdown, periods_per_year, setup_logging, sharpe_ratio

log = setup_logging("backtest")


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
                cash = -fee  # cash fully deployed into units, minus fee
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


def main():
    cfg = load_config()

    log.info("Fetching %d candles of %s %s from %s...",
              cfg["history_limit"], cfg["symbol"], cfg["timeframe"], cfg["exchange"])
    raw = fetch_ohlcv_history(cfg["exchange"], cfg["symbol"], cfg["timeframe"], cfg["history_limit"])

    dataset = make_dataset(raw, cfg["label_horizon"], cfg["label_threshold"])
    split = int(len(dataset) * cfg["train_fraction"])
    train_df = dataset.iloc[:split].reset_index(drop=True)
    test_df = dataset.iloc[split:].reset_index(drop=True)
    log.info("Train rows: %d (%s -> %s)", len(train_df), train_df["timestamp"].iloc[0], train_df["timestamp"].iloc[-1])
    log.info("Test rows:  %d (%s -> %s)", len(test_df), test_df["timestamp"].iloc[0], test_df["timestamp"].iloc[-1])

    model = train_model(train_df)
    save_model(model, cfg["model_file"])

    proba_up = predict_proba_up(model, test_df)
    result = simulate(test_df, proba_up, cfg)

    ppy = periods_per_year(cfg["timeframe"])
    strat_sharpe = sharpe_ratio(result["strategy_return"], ppy)
    strat_dd = max_drawdown(result["equity"])
    strat_total_return = result["equity"].iloc[-1] - 1.0
    bh_total_return = result["buy_hold_equity"].iloc[-1] - 1.0
    n_trades = (result["position"].diff().fillna(0) != 0).sum()

    log.info("=" * 60)
    log.info("Strategy total return:   %+.2f%%", 100 * strat_total_return)
    log.info("Buy & hold total return: %+.2f%%", 100 * bh_total_return)
    log.info("Strategy Sharpe (ann.):  %.2f", strat_sharpe)
    log.info("Strategy max drawdown:   %.2f%%", 100 * strat_dd)
    log.info("Number of trades:        %d", n_trades)
    log.info("=" * 60)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result["timestamp"], result["equity"], label="ML strategy")
    ax.plot(result["timestamp"], result["buy_hold_equity"], label="Buy & hold", alpha=0.7)
    ax.set_title(f"Backtest: {cfg['symbol']} {cfg['timeframe']} (test period, after fees+slippage)")
    ax.set_ylabel("Equity (normalized)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("backtest_result.png", dpi=150)
    log.info("Saved equity curve plot to backtest_result.png")

    result.to_csv("backtest_result.csv", index=False)
    log.info("Saved detailed results to backtest_result.csv")
    log.info("Model retrained on train split and saved to %s (used by paper_trade.py)", cfg["model_file"])


if __name__ == "__main__":
    main()
