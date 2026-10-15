"""Screen many crypto pairs and pick the ones worth actually paper-trading.

1. List all active USD spot pairs on the exchange (public data, no keys).
2. Keep the most liquid `scan_pool_size` of them by 24h quote volume.
3. Backtest each candidate with the same walk-forward method as backtest.py.
4. Rank by test-period Sharpe ratio and take the top `select_top_n`.
5. Save the chosen symbols to `symbols_file` (default symbols.json) -- this
   is what train.py / paper_trade.py / report.py read to know the universe.

Usage:
    python universe.py
"""
from __future__ import annotations

import json

import ccxt
import pandas as pd

from strategy import train_and_evaluate
from utils import load_config, setup_logging

log = setup_logging("universe")


# Stablecoins and fiat currencies: near-zero volatility against USD by
# design, so they're not real trading opportunities -- exclude them even
# when they're among the most liquid pairs (they usually are).
NON_TRADEABLE_BASES = {
    "USDT", "USDC", "USDG", "DAI", "PYUSD", "TUSD", "USDP", "GUSD", "FDUSD", "EURT",
    "EUR", "GBP", "JPY", "CHF", "AUD", "CAD",
}


def list_candidates(exchange_id: str, quote: str, pool_size: int) -> list[str]:
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    markets = exchange.load_markets()

    usd_spot = [
        m["symbol"] for m in markets.values()
        if m.get("spot") and m.get("active") and m.get("quote") == quote
        and m.get("base") not in NON_TRADEABLE_BASES
    ]
    log.info("Found %d active tradeable %s spot pairs on %s (stablecoins/fiat excluded)", len(usd_spot), quote, exchange_id)

    tickers = exchange.fetch_tickers(usd_spot)
    volumes = []
    for sym in usd_spot:
        t = tickers.get(sym, {})
        qv = t.get("quoteVolume")
        if qv is None and t.get("baseVolume") and t.get("last"):
            qv = t["baseVolume"] * t["last"]
        volumes.append((sym, qv or 0))

    volumes.sort(key=lambda x: x[1], reverse=True)
    top = [sym for sym, _ in volumes[:pool_size]]
    log.info("Top %d most liquid %s pairs: %s", len(top), quote, ", ".join(top))
    return top


def main():
    cfg = load_config()
    uni_cfg = cfg["universe"]

    candidates = list_candidates(cfg["exchange"], uni_cfg["quote"], uni_cfg["scan_pool_size"])

    rows = []
    for symbol in candidates:
        log.info("Backtesting %s...", symbol)
        ev = train_and_evaluate(symbol, cfg, log=log)
        if ev is None:
            continue
        rows.append({
            "symbol": symbol,
            "sharpe": ev.strat_sharpe,
            "total_return_pct": 100 * ev.strat_total_return,
            "buy_hold_return_pct": 100 * ev.bh_total_return,
            "max_drawdown_pct": 100 * ev.strat_max_drawdown,
            "n_trades": ev.n_trades,
            "n_candles": ev.n_candles,
        })

    if not rows:
        log.error("No candidate symbol had enough history to evaluate. Nothing selected.")
        return

    board = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 120)
    log.info("\n%s", board.to_string(index=False))

    top_n = uni_cfg["select_top_n"]
    selected = board.head(top_n)["symbol"].tolist()
    log.info("Selected top %d by backtest Sharpe ratio: %s", len(selected), selected)

    with open(cfg["symbols_file"], "w") as f:
        json.dump(selected, f, indent=2)
    log.info("Saved selection to %s", cfg["symbols_file"])

    board.to_csv("universe_scan.csv", index=False)
    log.info("Saved full scan results to universe_scan.csv")


if __name__ == "__main__":
    main()
