"""Walk-forward backtest for every symbol in `symbols_file`.

For each symbol: train on the first `train_fraction` of history, evaluate on
the untouched remainder (fees + slippage applied), and save that model to
`models_dir/<SYMBOL>.joblib` for paper_trade.py to use. Also builds an
equal-weighted portfolio curve across all symbols for comparison.

Usage:
    python backtest.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from model import save_model
from strategy import symbol_slug, train_and_evaluate
from utils import load_config, setup_logging

log = setup_logging("backtest")


def main():
    cfg = load_config()
    with open(cfg["symbols_file"]) as f:
        symbols = json.load(f)

    os.makedirs(cfg["models_dir"], exist_ok=True)

    rows = []
    curves = {}  # symbol -> normalized equity Series (index = position in test set)
    bh_curves = {}

    for symbol in symbols:
        log.info("[%s] Backtesting...", symbol)
        ev = train_and_evaluate(symbol, cfg, log=log)
        if ev is None:
            continue

        model_path = f"{cfg['models_dir']}/{symbol_slug(symbol)}.joblib"
        save_model(ev.model, model_path)

        rows.append({
            "symbol": symbol,
            "total_return_pct": 100 * ev.strat_total_return,
            "buy_hold_return_pct": 100 * ev.bh_total_return,
            "sharpe": ev.strat_sharpe,
            "max_drawdown_pct": 100 * ev.strat_max_drawdown,
            "n_trades": ev.n_trades,
            "test_rows": len(ev.test_df),
        })
        curves[symbol] = ev.result["equity"].reset_index(drop=True)
        bh_curves[symbol] = ev.result["buy_hold_equity"].reset_index(drop=True)

        ev.result.to_csv(f"backtest_{symbol_slug(symbol)}.csv", index=False)

    if not rows:
        log.error("No symbol produced a usable backtest. Nothing to report.")
        return

    board = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    log.info("=" * 70)
    log.info("\n%s", board.to_string(index=False))
    log.info("=" * 70)
    board.to_csv("backtest_result.csv", index=False)
    log.info("Saved per-symbol summary to backtest_result.csv")

    # Equal-weighted portfolio curve: align by position (each symbol's test
    # set may start/end on slightly different candles), truncate to the
    # shortest, average the normalized equity curves.
    min_len = min(len(s) for s in curves.values())
    port_strategy = sum(s.iloc[:min_len].values for s in curves.values()) / len(curves)
    port_bh = sum(s.iloc[:min_len].values for s in bh_curves.values()) / len(bh_curves)

    fig, ax = plt.subplots(figsize=(10, 5))
    for symbol, s in curves.items():
        ax.plot(s.iloc[:min_len].values, alpha=0.35, linewidth=1, label=f"{symbol} (strategy)")
    ax.plot(port_strategy, color="black", linewidth=2.5, label="Portfolio (equal-weight, strategy)")
    ax.plot(port_bh, color="black", linewidth=2.5, linestyle="--", label="Portfolio (equal-weight, buy & hold)")
    ax.set_title(f"Backtest: {len(curves)} symbols, {cfg['timeframe']} (test period, after fees+slippage)")
    ax.set_xlabel("Candles into test period")
    ax.set_ylabel("Equity (normalized, 1.0 = start)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig("backtest_result.png", dpi=150)
    log.info("Saved combined equity curve plot to backtest_result.png")

    port_total_return = port_strategy[-1] - 1.0
    port_bh_return = port_bh[-1] - 1.0
    log.info("Equal-weight portfolio total return: %+.2f%% (buy & hold: %+.2f%%)",
              100 * port_total_return, 100 * port_bh_return)


if __name__ == "__main__":
    main()
