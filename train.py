"""Fetch historical data, build features/labels, train the model, save it.

Usage:
    python train.py
"""
from __future__ import annotations

from data_feed import fetch_ohlcv_history
from features import make_dataset
from model import save_model, train_model
from utils import load_config, setup_logging

log = setup_logging("train")


def main():
    cfg = load_config()

    log.info("Fetching %d candles of %s %s from %s...",
              cfg["history_limit"], cfg["symbol"], cfg["timeframe"], cfg["exchange"])
    raw = fetch_ohlcv_history(cfg["exchange"], cfg["symbol"], cfg["timeframe"], cfg["history_limit"])
    log.info("Fetched %d candles (%s -> %s)", len(raw), raw["timestamp"].iloc[0], raw["timestamp"].iloc[-1])

    dataset = make_dataset(raw, cfg["label_horizon"], cfg["label_threshold"])
    log.info("Built dataset with %d usable rows after feature warm-up", len(dataset))

    split = int(len(dataset) * cfg["train_fraction"])
    train_df = dataset.iloc[:split]
    log.info("Training on %d rows (%s -> %s)", len(train_df),
              train_df["timestamp"].iloc[0], train_df["timestamp"].iloc[-1])
    log.info("Label balance in training set: up=%.1f%%",
              100 * train_df["label"].mean())

    model = train_model(train_df)
    save_model(model, cfg["model_file"])
    log.info("Saved model to %s", cfg["model_file"])
    log.info("Next: run `python backtest.py` to evaluate it on held-out data.")


if __name__ == "__main__":
    main()
