"""Run the trained per-symbol models live against a simulated paper broker.

This only reads PUBLIC market data (no API keys, no exchange account) and
simulates fills locally in `state_file`. No real order is ever placed and no
real funds are ever at risk. Trades every symbol in `symbols_file` (produced
by universe.py), each with its own equal slice of starting_cash.

Uses the same ATR-based risk management as strategy.simulate() (via
paper_broker.evaluate_bar) plus portfolio-wide controls (risk.py): max
concurrent positions, max exposure, max daily/weekly loss, a drawdown kill
switch, and a cooldown after consecutive losses. Those controls only ever
block a NEW entry -- an open position's stop/target/signal exit always runs.

Usage:
    python paper_trade.py            # loops forever, checks for a new candle every poll_seconds
    python paper_trade.py --once     # does a single check-and-trade cycle (all symbols), then exits
"""
from __future__ import annotations

import json
import sys
import time

import risk
from data_feed import fetch_latest_candles
from features import build_features
from model import load_model, predict_proba_up
from paper_broker import equity, evaluate_bar, load_portfolio, portfolio_equity, record_equity, save_portfolio
from strategy import symbol_slug
from utils import load_config, setup_logging

log = setup_logging("paper_trade")


def load_symbols(cfg: dict) -> list[str]:
    with open(cfg["symbols_file"]) as f:
        return json.load(f)


def trade_one_symbol(symbol: str, cfg: dict, state, portfolio: dict, risk_state: risk.RiskState, total_equity: float) -> None:
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
    close = float(latest["close"].iloc[0])
    high = float(latest["high"].iloc[0])
    low = float(latest["low"].iloc[0])
    atr = float(latest["atr"].iloc[0])

    # The portfolio-wide risk gate only applies to opening a brand new
    # position -- an existing position's stop/target/signal exit in
    # evaluate_bar always runs regardless of these limits.
    opening_new = state.position == 0 and proba_up >= cfg["buy_threshold"] and atr > 0
    if opening_new:
        risk_amount = equity(state, close) * cfg["risk"]["risk_per_trade_pct"]
        stop_distance = cfg["risk"]["stop_atr_mult"] * atr
        est_units = min(risk_amount / stop_distance, equity(state, close) / close)
        allowed, reason = risk.can_open_new_position(
            risk_state, total_equity, portfolio, new_position_value=est_units * close, cfg=cfg,
        )
        if not allowed:
            log.info("%s: entry signal (P(up)=%.3f) blocked by risk control: %s", symbol, proba_up, reason)
            state.last_price = close
            state.last_candle_ts = latest_ts
            record_equity(state, close, latest_ts)
            return

    before_equity = equity(state, close)
    state, outcome = evaluate_bar(state, close, high, low, atr, proba_up, latest_ts, cfg)
    after_equity = equity(state, close)

    if outcome is not None:
        risk.record_trade_outcome(risk_state, outcome, cfg)

    log.info(
        "%s candle=%s price=%.2f P(up)=%.3f position=%.6f stop=%s target=%s cash=%.2f equity=%.2f (was %.2f)%s",
        symbol, latest_ts, close, proba_up, state.position,
        f"{state.stop_price:.4f}" if state.stop_price else "-",
        f"{state.target_price:.4f}" if state.target_price else "-",
        state.cash, after_equity, before_equity,
        f" [{outcome.upper()}]" if outcome else "",
    )


def run_once(cfg: dict) -> None:
    symbols = load_symbols(cfg)
    portfolio = load_portfolio(cfg["state_file"], symbols, cfg["starting_cash"])
    risk_state = risk.load_risk_state(cfg["risk_state_file"])

    total_equity = portfolio_equity(portfolio)
    risk.update_on_cycle(risk_state, total_equity, cfg)
    if risk_state.shutdown:
        log.warning("PORTFOLIO SHUTDOWN ACTIVE: %s -- no new entries will open; existing positions still managed.",
                     risk_state.shutdown_reason)

    for symbol in symbols:
        try:
            trade_one_symbol(symbol, cfg, portfolio[symbol], portfolio, risk_state, total_equity)
        except Exception:
            log.exception("%s: error during this cycle; leaving its sleeve unchanged.", symbol)

    save_portfolio(cfg["state_file"], portfolio)
    risk.save_risk_state(cfg["risk_state_file"], risk_state)
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
