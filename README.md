# ML Crypto Trading Bot (paper trading only)

A self-contained pipeline: fetch public market data → engineer features →
train a gradient-boosting classifier to predict short-term direction →
backtest with realistic costs → run live against a **simulated** paper
broker. No API keys, no real exchange account, and no real money are ever
involved — `paper_broker.py` is a local bookkeeping simulation only.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python train.py       # fetch history, engineer features, train + save model.joblib
python backtest.py    # walk-forward backtest on held-out data, saves backtest_result.png/.csv
python paper_trade.py            # loop: checks for a new closed candle every poll_seconds, trades the simulated account
python paper_trade.py --once     # single check-and-trade cycle, then exit (good for cron/testing)
```

State for the paper account lives in `paper_state.json` (cash, position,
trade log) — delete it to reset to `starting_cash`.

## How it works

- **`data_feed.py`** — pulls OHLCV candles via `ccxt` from a public exchange
  endpoint (default: Kraken, since Binance blocks requests from the US).
  No authentication of any kind.
- **`features.py`** — returns over multiple windows, rolling volatility, RSI,
  MACD, Bollinger %B, SMA ratios, volume z-score. Label = 1 if price rises
  more than `label_threshold` over the next `label_horizon` candles.
- **`model.py`** — `HistGradientBoostingClassifier` (scikit-learn, no extra
  native deps) predicts P(price goes up).
- **`backtest.py`** — trains on the first `train_fraction` of history only,
  evaluates on the untouched remainder (no lookahead), applies `fee_rate`
  and `slippage_bps` on every simulated fill, and plots strategy vs.
  buy-and-hold equity curves.
- **`paper_broker.py`** — the "exchange" for paper trading: a JSON file
  tracking cash/position, filled at the public price plus the same
  fee/slippage model as the backtest.

## Tuning

Everything lives in `config.yaml`: symbol, timeframe, label horizon/threshold,
buy/sell probability thresholds, fee/slippage assumptions, starting cash,
poll interval.

## Honest caveats

- This is a simple baseline (one model, one symbol, long/flat only, hourly
  bars). Backtest results on a few weeks of data are **not** statistically
  reliable — treat this as a scaffold to iterate on, not a finished edge.
- Kraken's public OHLC endpoint only returns roughly the most recent 720
  candles regardless of `history_limit`. For more history, lower the
  timeframe (e.g. `15m`) or fetch from an exchange with deeper paginated
  history.
- Going from paper trading to **real money** is a separate, deliberate step
  that requires you to generate your own exchange API keys and wire up
  order placement yourself — do that only after you've validated the
  strategy extensively and understand the risk. I won't enter credentials
  or place real trades on your behalf.
