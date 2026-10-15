"""Local webhook receiver for TradingView alerts.

TradingView's "Webhook URL" alert feature POSTs a JSON body to a URL you
control when a Pine Script alert condition fires. This server receives that
POST, checks a shared secret, and feeds the resulting buy/sell signal into
the same simulated paper_broker used by paper_trade.py.

This never places a real order and never touches a real exchange account --
it only updates a local JSON ledger (`webhook_state_file`).

TradingView alert "Message" field, set the content type to JSON and use:

    {
      "secret": "<the webhook_secret from config.yaml>",
      "symbol": "{{ticker}}",
      "action": "buy",
      "price": {{close}}
    }

(Make a second alert/condition with "action": "sell" for the exit signal.)

TradingView needs a PUBLIC url to reach this server -- it cannot reach
127.0.0.1 on your machine directly. Run this server, then expose it with a
tunnel, e.g.:

    ngrok http 8787

...and use the printed https://*.ngrok-free.app/webhook URL as the alert's
Webhook URL. Keep this process and the tunnel running for alerts to arrive.

Usage:
    python webhook_server.py
"""
from __future__ import annotations

from flask import Flask, jsonify, request

from paper_broker import equity, execute_target_position, load_state, save_state
from utils import load_config, setup_logging

log = setup_logging("webhook_server")
cfg = load_config()
app = Flask(__name__)

ACTION_TO_TARGET = {
    "buy": 1.0,
    "long": 1.0,
    "sell": 0.0,
    "flat": 0.0,
    "exit": 0.0,
    "close": 0.0,
}


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/webhook")
def webhook():
    payload = request.get_json(silent=True)
    if payload is None:
        log.warning("Rejected webhook: body is not valid JSON")
        return jsonify({"error": "expected JSON body"}), 400

    if payload.get("secret") != cfg["webhook_secret"]:
        log.warning("Rejected webhook: bad or missing secret from %s", request.remote_addr)
        return jsonify({"error": "unauthorized"}), 401

    action = str(payload.get("action", "")).lower()
    if action not in ACTION_TO_TARGET:
        log.warning("Rejected webhook: unknown action %r", action)
        return jsonify({"error": f"unknown action {action!r}, expected one of {sorted(ACTION_TO_TARGET)}"}), 400

    try:
        price = float(payload["price"])
        if price <= 0:
            raise ValueError("price must be positive")
    except (KeyError, TypeError, ValueError) as e:
        log.warning("Rejected webhook: bad price %r (%s)", payload.get("price"), e)
        return jsonify({"error": "missing or invalid numeric 'price'"}), 400

    symbol = payload.get("symbol", "unknown")
    desired = ACTION_TO_TARGET[action]

    state = load_state(cfg["webhook_state_file"], cfg["starting_cash"])
    before_equity = equity(state, price)
    state = execute_target_position(
        state, desired, price,
        fee_rate=cfg["fee_rate"], slippage_bps=cfg["slippage_bps"],
        candle_ts=f"tv-alert:{symbol}",
    )
    after_equity = equity(state, price)
    save_state(cfg["webhook_state_file"], state)

    log.info(
        "TradingView alert symbol=%s action=%s price=%.2f -> target=%s cash=%.2f position=%.6f equity=%.2f (was %.2f)",
        symbol, action, price, "LONG" if desired == 1.0 else "FLAT",
        state.cash, state.position, after_equity, before_equity,
    )

    return jsonify({
        "status": "ok",
        "action": action,
        "target_position": desired,
        "cash": state.cash,
        "position": state.position,
        "equity": after_equity,
    })


def main():
    log.info("TradingView webhook receiver listening on http://%s:%d/webhook", cfg["webhook_host"], cfg["webhook_port"])
    log.info("This simulates a paper account in %s -- no real orders, no real exchange account.", cfg["webhook_state_file"])
    if cfg["webhook_secret"].startswith("change-me"):
        log.warning("webhook_secret in config.yaml still looks like a placeholder -- set a real random value.")
    app.run(host=cfg["webhook_host"], port=cfg["webhook_port"])


if __name__ == "__main__":
    main()
