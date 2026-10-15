"""Rolling-origin walk-forward validation: many independent train/test folds
across a symbol's full history, instead of one lucky (or unlucky) split.

Fold 1: train on [0:T0),            test on [T0:T0+W)
Fold 2: train on [0:T0+W),          test on [T0+W:T0+2W)
Fold 3: train on [0:T0+2W),         test on [T0+2W:T0+3W)
...

Each fold's test window is strictly after its own training data -- no fold
ever trains on data from its own future, and later folds get more training
history (expanding window) but are always evaluated on unseen data. The
aggregate statistics (mean, std, worst-fold) across folds are what "robust"
actually means here, as opposed to a single split that might just be a lucky
draw.

Usage:
    python walk_forward.py BTC/USD
    python walk_forward.py BTC/USD ETH/USD        # multiple symbols
    python walk_forward.py --all                   # every symbol in symbols.json
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategy import fetch_dataset, fit_and_test, symbol_slug
from utils import periods_per_year, setup_logging, load_config

log = setup_logging("walk_forward")


def make_folds(n_rows: int, initial_train: int, test_window: int, step: int):
    """Yield (train_start, train_end, test_start, test_end) index tuples."""
    train_end = initial_train
    while train_end + test_window <= n_rows:
        yield 0, train_end, train_end, train_end + test_window
        train_end += step


def run_walk_forward(symbol: str, cfg: dict, log=None) -> dict | None:
    wf_cfg = cfg["walk_forward"]
    try:
        dataset = fetch_dataset(symbol, cfg)
    except Exception as e:
        if log:
            log.warning("%s: failed to fetch history (%s)", symbol, e)
        return None

    folds = list(make_folds(len(dataset), wf_cfg["initial_train_candles"],
                             wf_cfg["test_window_candles"], wf_cfg["step_candles"]))
    if len(folds) < wf_cfg["min_folds"]:
        if log:
            log.info("%s: only %d folds possible (need >= %d) from %d candles -- skipping",
                      symbol, len(folds), wf_cfg["min_folds"], len(dataset))
        return None

    fold_results = []
    stitched_equity = [1.0]   # chained OOS equity: each fold continues from the previous fold's ending level
    stitched_bh = [1.0]
    stitched_timestamps = [dataset["timestamp"].iloc[folds[0][2]]]
    all_trades = []

    for fold_i, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        train_df = dataset.iloc[tr_start:tr_end].reset_index(drop=True)
        test_df = dataset.iloc[te_start:te_end].reset_index(drop=True)
        ev = fit_and_test(symbol, train_df, test_df, cfg)
        if ev is None:
            continue

        fold_results.append({
            "fold": fold_i + 1,
            "train_candles": len(train_df),
            "test_start": test_df["timestamp"].iloc[0],
            "test_end": test_df["timestamp"].iloc[-1],
            "total_return_pct": 100 * ev.strat_total_return,
            "bh_return_pct": 100 * ev.bh_total_return,
            "sharpe": ev.strat_sharpe,
            "sortino": ev.strat_sortino,
            "cagr_pct": 100 * ev.strat_cagr,
            "max_drawdown_pct": 100 * ev.strat_max_drawdown,
            "exposure_pct": 100 * ev.strat_exposure,
            "n_trades": ev.n_trades,
            "win_rate_pct": 100 * ev.stats["win_rate"],
            "profit_factor": ev.stats["profit_factor"],
            "expectancy_pct": 100 * ev.stats["expectancy_pct"],
            "max_consecutive_losses": ev.stats["max_consecutive_losses"],
            "avg_holding_bars": ev.stats["avg_holding_bars"],
        })

        # stitch: scale this fold's normalized equity curve to continue from
        # where the stitched curve currently ends
        base = stitched_equity[-1]
        bh_base = stitched_bh[-1]
        stitched_equity.extend((base * ev.result["equity"]).tolist())
        stitched_bh.extend((bh_base * ev.result["buy_hold_equity"]).tolist())
        stitched_timestamps.extend(test_df["timestamp"].tolist())

        if not ev.trades.empty:
            t = ev.trades.copy()
            t["fold"] = fold_i + 1
            all_trades.append(t)

    if not fold_results:
        if log:
            log.info("%s: no fold produced a usable result", symbol)
        return None

    board = pd.DataFrame(fold_results)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    summary = {
        "symbol": symbol,
        "n_folds": len(board),
        "sharpe_mean": board["sharpe"].mean(), "sharpe_std": board["sharpe"].std(), "sharpe_worst": board["sharpe"].min(),
        "sortino_mean": board["sortino"].mean(),
        "total_return_mean_pct": board["total_return_pct"].mean(),
        "cagr_mean_pct": board["cagr_pct"].mean(),
        "max_drawdown_worst_pct": board["max_drawdown_pct"].min(),
        "win_rate_mean_pct": board["win_rate_pct"].mean(),
        "profit_factor_mean": board["profit_factor"].replace([np.inf, -np.inf], np.nan).mean(),
        "pct_folds_profitable": 100 * (board["total_return_pct"] > 0).mean(),
        "stitched_total_return_pct": 100 * (stitched_equity[-1] - 1.0),
        "stitched_bh_return_pct": 100 * (stitched_bh[-1] - 1.0),
    }

    return {
        "symbol": symbol, "folds": board, "trades": trades_df, "summary": summary,
        "stitched_equity": pd.Series(stitched_equity[1:]),
        "stitched_bh": pd.Series(stitched_bh[1:]),
        "stitched_timestamps": pd.Series(stitched_timestamps[1:]),
    }


def plot_stitched(wf: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wf["stitched_timestamps"], wf["stitched_equity"], label="Walk-forward OOS strategy")
    ax.plot(wf["stitched_timestamps"], wf["stitched_bh"], label="Buy & hold", alpha=0.7)
    for _, row in wf["folds"].iterrows():
        ax.axvline(row["test_start"], color="gray", alpha=0.15, linewidth=1)
    ax.set_title(f"Walk-forward OOS equity: {wf['symbol']} ({wf['summary']['n_folds']} folds, gray lines = fold boundaries)")
    ax.set_ylabel("Equity (normalized, chained across folds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main():
    cfg = load_config()
    args = sys.argv[1:]
    if args == ["--all"]:
        with open(cfg["symbols_file"]) as f:
            symbols = json.load(f)
    elif args:
        symbols = args
    else:
        log.error("Usage: python walk_forward.py SYMBOL [SYMBOL ...] | --all")
        return

    summaries = []
    for symbol in symbols:
        log.info("Running walk-forward for %s...", symbol)
        wf = run_walk_forward(symbol, cfg, log=log)
        if wf is None:
            continue
        summaries.append(wf["summary"])

        slug = symbol_slug(symbol)
        wf["folds"].to_csv(f"walk_forward_{slug}_folds.csv", index=False)
        if not wf["trades"].empty:
            wf["trades"].to_csv(f"walk_forward_{slug}_trades.csv", index=False)
        plot_stitched(wf, f"walk_forward_{slug}.png")

        log.info("[%s] %d folds | Sharpe mean=%.2f worst=%.2f | stitched OOS return=%+.1f%% (buy&hold %+.1f%%) | %.0f%% of folds profitable",
                  symbol, wf["summary"]["n_folds"], wf["summary"]["sharpe_mean"], wf["summary"]["sharpe_worst"],
                  wf["summary"]["stitched_total_return_pct"], wf["summary"]["stitched_bh_return_pct"],
                  wf["summary"]["pct_folds_profitable"])

    if summaries:
        board = pd.DataFrame(summaries).sort_values("sharpe_mean", ascending=False)
        pd.set_option("display.width", 140)
        log.info("\n%s", board.to_string(index=False))
        board.to_csv("walk_forward_summary.csv", index=False)
        log.info("Saved walk_forward_summary.csv")


if __name__ == "__main__":
    main()
