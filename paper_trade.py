"""Run the trained model live against a simulated paper broker.

This only reads PUBLIC market data (no API keys, no exchange account) and
simulates fills locally in `state_file`. No real order is ever placed and no
real funds are ever at risk.

Usage:
    python paper_trade.py            # loops forever, checks for a new candle every poll_seconds
    python paper_trade.py --once     # does a single check-and-trade cycle, then exits
"""
from __future__ import annotations

import sys
import time

from data_feed import fetch_latest_candles
from features import FEATURE_COLUMNS, build_features
from model import load_model, predict_proba_up
from paper_broker import equity, execute_target_position, load_state, save_state
from utils import load_config, setup_logging

log = setup_logging("paper_trade")


def run_once(cfg: dict) -> None:
    model = load_model(cfg["model_file"])
    state = load_state(cfg["state_file"], cfg["starting_cash"])

    raw = fetch_latest_candles(cfg["exchange"], cfg["symbol"], cfg["timeframe"], limit=200)
    feat = build_features(raw).dropna().reset_index(drop=True)
    if feat.empty:
        log.warning("Not enough data yet to compute features; skipping this cycle.")
        return

    latest = feat.iloc[[-1]]
    latest_ts = str(latest["timestamp"].iloc[0])

    if latest_ts == state.last_candle_ts:
        log.info("No new closed candle yet (last processed: %s). Skipping.", latest_ts)
        return

    proba_up = predict_proba_up(model, latest).iloc[0]
    price = float(latest["close"].iloc[0])

    if proba_up >= cfg["buy_threshold"]:
        desired = 1.0
    elif proba_up <= cfg["sell_threshold"]:
        desired = 0.0
    else:
        desired = 1.0 if state.position * price > 1e-6 else 0.0  # hold current stance

    before_equity = equity(state, price)
    state = execute_target_position(
        state, desired, price,
        fee_rate=cfg["fee_rate"], slippage_bps=cfg["slippage_bps"], candle_ts=latest_ts,
    )
    after_equity = equity(state, price)
    save_state(cfg["state_file"], state)

    log.info(
        "candle=%s price=%.2f P(up)=%.3f target=%s cash=%.2f position=%.6f equity=%.2f (was %.2f)",
        latest_ts, price, proba_up, "LONG" if desired == 1.0 else "FLAT",
        state.cash, state.position, after_equity, before_equity,
    )


def main():
    cfg = load_config()
    once = "--once" in sys.argv

    log.info("Paper trading %s %s on simulated %s exchange feed (no real funds, no API keys).",
              cfg["symbol"], cfg["timeframe"], cfg["exchange"])

    if once:
        run_once(cfg)
        return

    while True:
        try:
            run_once(cfg)
        except Exception:
            log.exception("Error during paper trading cycle; will retry next poll.")
        time.sleep(cfg["poll_seconds"])


if __name__ == "__main__":
    main()
