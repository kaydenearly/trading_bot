"""Summarize and plot the paper trading portfolio's performance so far.

Reads `state_file` from config.yaml (the same file paper_trade.py writes to,
one sleeve per symbol) and produces a console summary plus `paper_report.png`
showing the combined equity curve against a buy-and-hold baseline.

Usage:
    python report.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paper_broker import equity, load_portfolio
from utils import load_config, setup_logging

log = setup_logging("report")


def main():
    cfg = load_config()
    with open(cfg["symbols_file"]) as f:
        active_symbols = json.load(f)

    portfolio = load_portfolio(cfg["state_file"], active_symbols, cfg["starting_cash"])

    print("=" * 70)
    print(f"Starting cash (total):  ${cfg['starting_cash']:,.2f}")
    total_equity = 0.0
    total_trades = 0
    rows = []
    for symbol, state in portfolio.items():
        sym_equity = equity(state, state.last_price)
        total_equity += sym_equity
        total_trades += len(state.trade_log)
        status = "active" if symbol in active_symbols else "inactive (dropped from universe)"
        rows.append({
            "symbol": symbol,
            "status": status,
            "cash": state.cash,
            "position": state.position,
            "last_price": state.last_price,
            "equity": sym_equity,
            "n_trades": len(state.trade_log),
        })
    print(f"Current total equity:   ${total_equity:,.2f}")
    print(f"Total return:           {100 * (total_equity / cfg['starting_cash'] - 1.0):+.2f}%")
    print(f"Total trades:           {total_trades}")
    print("=" * 70)
    print(pd.DataFrame(rows).to_string(index=False))

    frames = []
    for symbol, state in portfolio.items():
        if not state.equity_log:
            continue
        df = pd.DataFrame(state.equity_log)[["time", "equity", "price"]]
        df["time"] = pd.to_datetime(df["time"])
        df = df.rename(columns={"equity": f"eq__{symbol}", "price": f"px__{symbol}"})
        frames.append(df.set_index("time"))

    if not frames:
        log.info("No equity history yet -- run paper_trade.py (or let the scheduled workflow run) at least once.")
        return

    merged = pd.concat(frames, axis=1).sort_index().ffill().dropna(how="any")
    if merged.empty:
        log.info("Not enough overlapping equity history across symbols yet to plot a combined curve.")
        return

    eq_cols = [c for c in merged.columns if c.startswith("eq__")]
    px_cols = [c for c in merged.columns if c.startswith("px__")]
    merged["portfolio_equity"] = merged[eq_cols].sum(axis=1)

    n_symbols = len(px_cols)
    per_symbol_start_cash = cfg["starting_cash"] / n_symbols
    bh_equity = sum(
        per_symbol_start_cash * (merged[c] / merged[c].iloc[0]) for c in px_cols
    )
    merged["buy_hold_equity"] = bh_equity

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(merged.index, merged["portfolio_equity"], label="Paper trading portfolio")
    ax.plot(merged.index, merged["buy_hold_equity"], label="Buy & hold (equal-weight)", alpha=0.7)
    ax.set_title(f"Paper trading performance: {n_symbols} symbols, {cfg['timeframe']}")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig("paper_report.png", dpi=150)
    log.info("Saved paper_report.png (%d equity snapshots, %s -> %s)",
              len(merged), merged.index[0], merged.index[-1])


if __name__ == "__main__":
    main()
