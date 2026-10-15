"""Run the trained per-symbol models live against a simulated paper broker.

This only reads PUBLIC market data (no API keys, no exchange account) and
simulates fills locally in `state_file`. No real order is ever placed and no
real funds are ever at risk. Trades every symbol in `symbols_file` (produced
by universe.py), each with its own equal slice of starting_cash.

Usage:
    python paper_trade.py            # loops forever, checks for a new candle every poll_seconds
    python paper_trade.py --once     # does a single check-and-trade cycle (all symbols), then exits
"""
from __future__ import annotations

import json
import sys
import time

from data_feed import fetch_latest_candles
from features import build_features
from model import load_model, predict_proba_up
from paper_broker import equity, execute_target_position, load_portfolio, portfolio_equity, save_portfolio
from strategy import symbol_slug
from utils import load_config, setup_logging

log = setup_logging("paper_trade")


def load_symbols(cfg: dict) -> list[str]:
    with open(cfg["symbols_file"]) as f:
        return json.load(f)


def trade_one_symbol(symbol: str, cfg: dict, state) -> None:
    model_path = f"{cfg['models_dir']}/{symbol_slug(symbol)}.joblib"
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        log.warning("No trained model for %s at %s yet; run train.py. Skipping.", symbol, model_path)
        return

    raw = fetch_latest_candles(cfg["exchange"], symbol, cfg["timeframe"], limit=200)
    feat = build_features(raw).dropna().reset_index(drop=True)
    if feat.empty:
        log.warning("%s: not enough data yet to compute features; skipping.", symbol)
        return

    latest = feat.iloc[[-1]]
    latest_ts = str(latest["timestamp"].iloc[0])

    if latest_ts == state.last_candle_ts:
        log.info("%s: no new closed candle yet (last processed: %s). Skipping.", symbol, latest_ts)
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
    execute_target_position(
        state, desired, price,
        fee_rate=cfg["fee_rate"], slippage_bps=cfg["slippage_bps"], candle_ts=latest_ts,
    )
    after_equity = equity(state, price)

    log.info(
        "%s candle=%s price=%.2f P(up)=%.3f target=%s cash=%.2f position=%.6f equity=%.2f (was %.2f)",
        symbol, latest_ts, price, proba_up, "LONG" if desired == 1.0 else "FLAT",
        state.cash, state.position, after_equity, before_equity,
    )


def run_once(cfg: dict) -> None:
    symbols = load_symbols(cfg)
    portfolio = load_portfolio(cfg["state_file"], symbols, cfg["starting_cash"])

    for symbol in symbols:
        try:
            trade_one_symbol(symbol, cfg, portfolio[symbol])
        except Exception:
            log.exception("%s: error during this cycle; leaving its sleeve unchanged.", symbol)

    save_portfolio(cfg["state_file"], portfolio)
    log.info("Portfolio equity across %d symbols: $%.2f", len(symbols), portfolio_equity(portfolio))


def main():
    cfg = load_config()
    once = "--once" in sys.argv

    symbols = load_symbols(cfg)
    log.info("Paper trading %s on simulated %s exchange feed (no real funds, no API keys): %s",
              cfg["timeframe"], cfg["exchange"], ", ".join(symbols))

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
