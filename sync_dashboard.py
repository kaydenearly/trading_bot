"""Build the JSON files the live dashboard artifact gets synced from.

Reads the current paper trading state (symbols.json + paper_state.json) and
fetches a fresh window of recent candles per symbol, writing one file per
artifact DB document under --out-dir. A separate step (outside this script,
run from a Claude session) pushes these files into the artifact's `db` via
the Artifact tool's write_db action -- this script only prepares the data.

Usage:
    python sync_dashboard.py --out-dir /tmp/dashboard_sync
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import risk
from data_feed import fetch_latest_candles
from paper_broker import equity, load_portfolio
from strategy import symbol_slug
from utils import load_config, setup_logging

log = setup_logging("sync_dashboard")

CANDLE_LIMIT = 150
TRADE_LOG_LIMIT = 30
EQUITY_LOG_LIMIT = 300


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/tmp/dashboard_sync")
    args = parser.parse_args()

    cfg = load_config()
    with open(cfg["symbols_file"]) as f:
        active_symbols = json.load(f)

    portfolio = load_portfolio(cfg["state_file"], active_symbols, cfg["starting_cash"])

    symbols_dir = os.path.join(args.out_dir, "symbols")
    os.makedirs(symbols_dir, exist_ok=True)

    total_equity = 0.0
    total_trades = 0
    total_realized_pnl = 0.0
    total_unrealized_pnl = 0.0
    symbol_summaries = []

    for symbol, state in portfolio.items():
        slug = symbol_slug(symbol)
        is_active = symbol in active_symbols
        sym_equity = equity(state, state.last_price)
        total_equity += sym_equity
        total_trades += len(state.trade_log)

        realized_pnl = sum(t.get("pnl_dollars", 0.0) for t in state.trade_log if t["side"] == "sell")
        unrealized_pnl = (
            state.position * state.last_price - state.capital_committed
            if state.position > 0 and state.capital_committed else 0.0
        )
        total_realized_pnl += realized_pnl
        total_unrealized_pnl += unrealized_pnl

        candles = []
        if is_active:
            try:
                raw = fetch_latest_candles(cfg["exchange"], symbol, cfg["timeframe"], limit=CANDLE_LIMIT)
                candles = [
                    {
                        "t": row.timestamp.isoformat(),
                        "o": float(row.open), "h": float(row.high),
                        "l": float(row.low), "c": float(row.close),
                        "v": float(row.volume),
                    }
                    for row in raw.itertuples()
                ]
            except Exception as e:
                log.warning("%s: failed to fetch candles (%s)", symbol, e)

        doc = {
            "symbol": symbol,
            "status": "active" if is_active else "inactive",
            "cash": state.cash,
            "position": state.position,
            "last_price": state.last_price,
            "equity": sym_equity,
            "n_trades": len(state.trade_log),
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "entry_price": state.entry_price,
            "stop_price": state.stop_price,
            "target_price": state.target_price,
            "candles": candles,
            "trades": state.trade_log[-TRADE_LOG_LIMIT:],
        }
        with open(os.path.join(symbols_dir, f"{slug}.json"), "w") as f:
            json.dump(doc, f)

        symbol_summaries.append({
            "symbol": symbol, "slug": slug, "status": doc["status"],
            "equity": sym_equity, "n_trades": len(state.trade_log),
            "last_price": state.last_price,
            "realized_pnl": realized_pnl, "unrealized_pnl": unrealized_pnl,
        })

    risk_state = risk.load_risk_state(cfg["risk_state_file"])
    risk_summary = {
        "shutdown": risk_state.shutdown,
        "shutdown_reason": risk_state.shutdown_reason,
        "cooldown_remaining": risk_state.cooldown_remaining,
        "consecutive_losses": risk_state.consecutive_losses,
        "peak_equity": risk_state.peak_equity,
        "drawdown_pct": 100 * (total_equity / risk_state.peak_equity - 1.0) if risk_state.peak_equity > 0 else 0.0,
        "day_start_equity": risk_state.day_start_equity,
        "day_pnl_pct": 100 * (total_equity / risk_state.day_start_equity - 1.0) if risk_state.day_start_equity > 0 else 0.0,
        "week_start_equity": risk_state.week_start_equity,
        "week_pnl_pct": 100 * (total_equity / risk_state.week_start_equity - 1.0) if risk_state.week_start_equity > 0 else 0.0,
    }

    validation = []
    if os.path.exists("walk_forward_summary.csv"):
        import csv
        with open("walk_forward_summary.csv") as f:
            validation = list(csv.DictReader(f))

    meta = {
        "exchange": cfg["exchange"],
        "timeframe": cfg["timeframe"],
        "starting_cash": cfg["starting_cash"],
        "total_equity": total_equity,
        "total_return_pct": 100 * (total_equity / cfg["starting_cash"] - 1.0),
        "total_trades": total_trades,
        "total_realized_pnl": total_realized_pnl,
        "total_unrealized_pnl": total_unrealized_pnl,
        "active_symbols": active_symbols,
        "symbols": symbol_summaries,
        "risk": risk_summary,
        "validation": validation,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    # Combined portfolio equity curve: merge each sleeve's equity_log by
    # timestamp (forward-filled), same method as report.py.
    import pandas as pd

    frames = []
    for symbol, state in portfolio.items():
        if not state.equity_log:
            continue
        df = pd.DataFrame(state.equity_log)[["time", "equity", "price"]]
        df["time"] = pd.to_datetime(df["time"])
        df = df.rename(columns={"equity": f"eq__{symbol}", "price": f"px__{symbol}"})
        frames.append(df.set_index("time"))

    points = []
    if frames:
        merged = pd.concat(frames, axis=1, sort=False).sort_index().ffill().dropna(how="any")
        eq_cols = [c for c in merged.columns if c.startswith("eq__")]
        px_cols = [c for c in merged.columns if c.startswith("px__")]
        if not merged.empty:
            merged["portfolio_equity"] = merged[eq_cols].sum(axis=1)
            per_symbol_start_cash = cfg["starting_cash"] / len(px_cols)
            merged["buy_hold_equity"] = sum(
                per_symbol_start_cash * (merged[c] / merged[c].iloc[0]) for c in px_cols
            )
            tail = merged.tail(EQUITY_LOG_LIMIT)
            points = [
                {
                    "t": ts.isoformat(),
                    "equity": float(row["portfolio_equity"]),
                    "buy_hold_equity": float(row["buy_hold_equity"]),
                }
                for ts, row in tail.iterrows()
            ]

    with open(os.path.join(args.out_dir, "equity_history.json"), "w") as f:
        json.dump({"points": points}, f)

    log.info("Wrote dashboard sync data to %s (%d symbols, %d equity points)",
              args.out_dir, len(portfolio), len(points))


if __name__ == "__main__":
    main()
