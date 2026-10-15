"""Train (or retrain) a model for every symbol in `symbols_file`.

Fits each model on the first `train_fraction` of that symbol's history
(same split backtest.py evaluates against) and saves it to
`models_dir/<SYMBOL>.joblib`.

Usage:
    python train.py
"""
from __future__ import annotations

import json
import os

from data_feed import fetch_ohlcv_history
from features import make_dataset
from model import save_model, train_model
from strategy import symbol_slug
from utils import load_config, setup_logging

log = setup_logging("train")


def main():
    cfg = load_config()
    with open(cfg["symbols_file"]) as f:
        symbols = json.load(f)

    os.makedirs(cfg["models_dir"], exist_ok=True)

    for symbol in symbols:
        log.info("[%s] Fetching %d candles of %s from %s...",
                  symbol, cfg["history_limit"], cfg["timeframe"], cfg["exchange"])
        raw = fetch_ohlcv_history(cfg["exchange"], symbol, cfg["timeframe"], cfg["history_limit"])

        dataset = make_dataset(raw, cfg["label_horizon"], cfg["label_threshold"])
        split = int(len(dataset) * cfg["train_fraction"])
        train_df = dataset.iloc[:split]
        if len(train_df) < 50:
            log.warning("[%s] Only %d training rows, skipping.", symbol, len(train_df))
            continue

        model = train_model(train_df)
        model_path = f"{cfg['models_dir']}/{symbol_slug(symbol)}.joblib"
        save_model(model, model_path)
        log.info("[%s] Trained on %d rows, label up=%.1f%%, saved to %s",
                  symbol, len(train_df), 100 * train_df["label"].mean(), model_path)

    log.info("Done. Next: run `python backtest.py` to evaluate these models on held-out data.")


if __name__ == "__main__":
    main()
