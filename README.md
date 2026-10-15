# ML Crypto Trading Bot (paper trading only)

### 📊 [Live dashboard — Sleeve Desk](https://claude.ai/code/artifact/45335d48-eb5d-498f-bb00-e57fbe667d37)

Candlestick charts with trade markers, portfolio equity vs. buy-and-hold, and
a live trade feed. Updates itself automatically every hour — works from any
browser, including on your phone. No login or setup needed to view it.

A self-contained pipeline: screen many crypto pairs for liquidity and
backtested performance → pick the best ones → train a gradient-boosting
classifier per symbol → backtest with realistic costs → run live against a
**simulated** paper broker, equal-weighted across the selected symbols. No
API keys, no real exchange account, and no real money are ever involved —
`paper_broker.py` is a local bookkeeping simulation only.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python universe.py    # screen the most liquid USD pairs, backtest each, pick the top N -> symbols.json
python train.py       # train + save models/<SYMBOL>.joblib for each symbol in symbols.json
python backtest.py    # walk-forward backtest for each symbol, saves backtest_result.png/.csv (also retrains + saves each model)
python paper_trade.py            # loop: checks for a new closed candle every poll_seconds, trades every selected symbol
python paper_trade.py --once     # single check-and-trade cycle (all symbols), then exit (good for cron/testing)
python report.py                 # prints a portfolio summary + saves paper_report.png (combined equity curve vs. buy & hold)
python walk_forward.py BTC/USD   # proper multi-fold out-of-sample validation for one or more symbols (see below)
python walk_forward.py --all     # same, for every symbol currently in symbols.json
```

Run `universe.py` first -- it creates `symbols.json`, which every other
script reads to know what to trade. Re-run it any time to re-pick the
universe (the weekly workflow does this automatically).

State for the paper portfolio lives in `paper_state.json`: one "sleeve" per
symbol (cash, position, trade log, equity history), each starting with an
equal share of `starting_cash`. A symbol dropped from the universe on a
later rescan keeps its sleeve (frozen, no new trades) rather than being
silently liquidated.

## Running automatically (GitHub Actions)

Two scheduled workflows in `.github/workflows/` run this for you without any
server of your own:

- **`paper_trade.yml`** — every hour, runs one paper-trading cycle across
  all selected symbols and regenerates `paper_report.png`, committing both
  back to the repo.
- **`retrain.yml`** — weekly, **re-screens the universe** (`universe.py`)
  and retrains each selected symbol's model (`backtest.py`), committing the
  updated `symbols.json`, `models/`, and backtest results.

Both also have `workflow_dispatch` enabled, so you can trigger either one
on demand from the repo's **Actions** tab, or with:

```bash
gh workflow run paper_trade.yml
gh workflow run retrain.yml
```

## Seeing results

- **In the repo**: open `paper_report.png` (combined equity curve, updated
  hourly) or `backtest_result.png` (per-symbol + portfolio backtest,
  updated weekly) in GitHub's file browser — both render inline. Open
  `universe_scan.csv` to see every candidate symbol's backtest metrics, not
  just the ones that got selected.
- **Locally**: `git pull` to get the latest committed state, then
  `python report.py` for a text summary (per-symbol and total equity,
  return %, trade counts) plus a freshly rendered `paper_report.png`.

## How it works

- **`universe.py`** — lists active USD spot pairs on the exchange, keeps the
  `scan_pool_size` most liquid by 24h volume (excluding stablecoins/fiat
  pairs, which have no real volatility to trade), backtests each one, and
  saves the top `select_top_n` by backtest Sharpe ratio to `symbols.json`.
- **`data_feed.py`** — pulls OHLCV candles via `ccxt` from a public exchange
  endpoint (default: Bitstamp — see "Honest caveats" for why, instead of
  Kraken or Binance). No authentication of any kind.
- **`features.py`** — returns over multiple windows, rolling volatility, RSI,
  MACD, Bollinger %B, SMA ratios, volume z-score. Label = 1 if price rises
  more than `label_threshold` over the next `label_horizon` candles.
- **`strategy.py`** — shared per-symbol train/backtest logic used by both
  `universe.py` (screening) and `backtest.py` (final evaluation + model
  saved for live trading).
- **`model.py`** — `HistGradientBoostingClassifier` (scikit-learn, no extra
  native deps) predicts P(price goes up), one model per symbol.
- **`backtest.py`** — for each symbol: trains on the first `train_fraction`
  of history, evaluates on the untouched remainder (no lookahead), applies
  `fee_rate` and `slippage_bps` on every simulated fill. Also builds an
  equal-weighted portfolio curve across all selected symbols.
- **`paper_broker.py`** — the "exchange" for paper trading: a JSON file with
  one sleeve per symbol, filled at the public price plus the same
  fee/slippage model as the backtest. `evaluate_bar()` runs the same
  ATR stop/target/sizing logic as `strategy.simulate()`, so live paper
  trading and the backtest don't diverge.
- **`risk.py`** — portfolio-wide controls that gate *new* entries only (an
  open position's stop/target always runs regardless): max concurrent
  positions, max total exposure, max daily/weekly loss, a max-drawdown
  kill switch, and a cooldown after consecutive losses. State persists in
  `risk_state.json`.
- **`walk_forward.py`** — rolling-origin walk-forward validation: many
  independent train/test folds across a symbol's full history (expanding
  training window, non-overlapping test windows), instead of the one split
  `backtest.py` uses. Reports Sharpe/Sortino/CAGR/drawdown/win
  rate/profit factor/expectancy/max consecutive losses per fold *and*
  aggregated (mean, worst-fold, % of folds profitable), plus a chained
  out-of-sample equity curve. This is the tool to actually trust for a
  robustness judgment — see "Honest caveats" below for why.

## Walk-forward results so far (be skeptical of backtests)

Running `walk_forward.py` on BTC/USD, ETH/USD, LINK/USD, and SOL/USD over a
full year of real Bitstamp data (18 independent out-of-sample folds each)
shows the current ML classifier **does not have a real edge** on any of
them: mean Sharpe ratios of roughly -3 to -4.5, only 11-28% of folds
profitable, and the chained out-of-sample equity curve underperforms simple
buy-and-hold on every symbol tested. This is a very different picture than
the single 30-day backtest suggested (which showed some symbols with
Sharpe ratios above 9) — that number was almost certainly noise from a tiny,
cherry-picked test window, exactly the kind of result walk-forward
validation exists to catch.

## Risk management

Added on top of the (still unvalidated) ML signal:

- **Position sizing is risk-based**, not "100% in or flat": each trade sizes
  so that hitting its stop costs about `risk_per_trade_pct` of the sleeve's
  equity, inversely proportional to the symbol's volatility (ATR) — capped
  at 100% of equity (no leverage). A volatile symbol gets a smaller position
  than a calm one for the same dollar risk.
- **ATR stop-loss and take-profit**, checked intrabar (against each candle's
  high/low, not just its close). The stop only ever ratchets favorably and
  only repositions once it would improve by more than a noise threshold — a
  trailing stop that's allowed to loosen can fail to ever trigger.
- **A model-signal exit only closes a position that's currently
  profitable** — it doesn't override the hard stop, which is the actual
  loss-limiting mechanism.
- **Portfolio-wide controls** (`risk.py`, state in `risk_state.json`): max
  concurrent positions, max total exposure, max daily/weekly loss, a
  max-drawdown kill switch, and a cooldown after consecutive losses. These
  only ever block a *new* entry — an open position's exit always runs.

**This bounds the damage, it does not fix the signal.** Re-running
walk-forward with risk management on shows max drawdown per fold is now
reliably capped (worst fold: -11% vs. previously unbounded), but the
negative mean Sharpe is essentially unchanged — risk management controls
*how much* a bad signal can lose, it doesn't make a bad signal profitable.
Improving or replacing the signal itself is separate, still-open work.

## Tuning

Everything lives in `config.yaml`: universe screening (`scan_pool_size`,
`select_top_n`), timeframe, label horizon/threshold, buy/sell probability
thresholds, risk management (`risk_per_trade_pct`, ATR stop/target
multiples, portfolio-wide caps), fee/slippage assumptions, total starting
cash, poll interval.

## Honest caveats

- This is still a simple baseline (one model per symbol, long/flat only,
  hourly bars, equal-weight allocation across symbols). Walk-forward
  validation (see above) shows the underlying ML signal does not currently
  have a real edge — risk management (see above) now bounds how much that
  costs, but doesn't fix it. Treat this as a scaffold to iterate on, not a
  working strategy.
- `backtest.py`/`universe.py` still use a single train/test split, which is
  known to be unreliable (see above) — `walk_forward.py` exists precisely
  because that single split can make a symbol look good by chance. Universe
  *selection* hasn't been switched over to walk-forward metrics yet; that's
  next, along with a correlation check so "more symbols" means real
  diversification rather than five correlated bets on the same move.
- `webhook_server.py` (the parked TradingView-webhook path) still uses the
  older `execute_target_position` (simple 0%/100% rebalance, no ATR
  stop/target) rather than `evaluate_bar()` — it's a separate, currently
  unused code path, not wired into the risk-managed live loop.
- We switched from Kraken to Bitstamp because Kraken's public OHLC endpoint
  silently caps at ~720 candles (~30 days at 1h) no matter what
  `history_limit` requests — it ignores `since` pagination entirely past
  that point. Bitstamp honors real pagination (BTC/USD hourly data is
  available back to 2012) with comparable USD-pair breadth. Binance is not
  an option at all here since it blocks API requests from the US.
- No correlation awareness in portfolio construction yet — "top N by
  backtest Sharpe" could pick several correlated bets on the same
  underlying move rather than real diversification. Next priority.
- Going from paper trading to **real money** is a separate, deliberate step
  that requires you to generate your own exchange API keys and wire up
  order placement yourself — do that only after you've validated the
  strategy extensively and understand the risk. I won't enter credentials
  or place real trades on your behalf.
