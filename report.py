"""Summarize and plot the paper trading account's performance so far.

Reads `state_file` from config.yaml (the same file paper_trade.py writes to)
and produces a console summary plus `paper_report.png` showing the equity
curve against a buy-and-hold baseline over the same period.

Usage:
    python report.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paper_broker import load_state
from utils import load_config, setup_logging

log = setup_logging("report")


def main():
    cfg = load_config()
    state = load_state(cfg["state_file"], cfg["starting_cash"])

    print("=" * 60)
    print(f"Starting cash:     ${cfg['starting_cash']:,.2f}")
    print(f"Cash:              ${state.cash:,.2f}")
    print(f"Position:          {state.position:.6f} units @ last price ${state.last_price:,.2f}")
    current_equity = state.cash + state.position * state.last_price
    print(f"Current equity:    ${current_equity:,.2f}")
    total_return = current_equity / cfg["starting_cash"] - 1.0
    print(f"Total return:      {100 * total_return:+.2f}%")
    print(f"Last candle seen:  {state.last_candle_ts}")
    print(f"Number of trades:  {len(state.trade_log)}")
    print("=" * 60)

    if state.trade_log:
        trades = pd.DataFrame(state.trade_log)
        print("\nLast 10 trades:")
        print(trades.tail(10)[["candle_ts", "side", "units", "fill_price", "fee"]].to_string(index=False))

    if not state.equity_log:
        log.info("No equity history yet -- run paper_trade.py (or let the scheduled workflow run) at least once.")
        return

    eq = pd.DataFrame(state.equity_log)
    eq["time"] = pd.to_datetime(eq["time"])
    eq["buy_hold_equity"] = cfg["starting_cash"] * (eq["price"] / eq["price"].iloc[0])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq["time"], eq["equity"], label="Paper trading strategy")
    ax.plot(eq["time"], eq["buy_hold_equity"], label="Buy & hold", alpha=0.7)
    ax.set_title(f"Paper trading performance: {cfg['symbol']} {cfg['timeframe']}")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig("paper_report.png", dpi=150)
    log.info("Saved paper_report.png (%d equity snapshots, %s -> %s)",
              len(eq), eq["time"].iloc[0], eq["time"].iloc[-1])


if __name__ == "__main__":
    main()
