"""Feature engineering and label construction for the ML trading signal.

All indicators are computed from OHLCV alone with pandas/numpy (no ta-lib
dependency). Every feature at row i only uses information available up to and
including candle i, so there is no lookahead leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "vol_12", "vol_24",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_pct_b",
    "sma_ratio_8_24", "sma_ratio_24_96",
    "volume_z_24",
    "hl_range",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal, macd - macd_signal


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given raw OHLCV columns, return df with feature columns appended."""
    out = df.copy()
    close = out["close"]

    for n in (1, 3, 6, 12, 24):
        out[f"ret_{n}"] = close.pct_change(n)

    log_ret = np.log(close / close.shift(1))
    out["vol_12"] = log_ret.rolling(12).std()
    out["vol_24"] = log_ret.rolling(24).std()

    out["rsi_14"] = _rsi(close, 14)

    macd, macd_signal, macd_hist = _macd(close)
    out["macd"] = macd / close
    out["macd_signal"] = macd_signal / close
    out["macd_hist"] = macd_hist / close

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    out["bb_pct_b"] = (close - lower) / (upper - lower).replace(0, np.nan)

    out["sma_ratio_8_24"] = close.rolling(8).mean() / close.rolling(24).mean()
    out["sma_ratio_24_96"] = close.rolling(24).mean() / close.rolling(96).mean()

    vol = out["volume"]
    vol_mean = vol.rolling(24).mean()
    vol_std = vol.rolling(24).std()
    out["volume_z_24"] = (vol - vol_mean) / vol_std.replace(0, np.nan)

    out["hl_range"] = (out["high"] - out["low"]) / close

    return out


def build_labels(df: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    """1 if forward return over `horizon` bars exceeds `threshold`, else 0.

    NaN for the final `horizon` rows, where the forward return is unknown.
    """
    forward_ret = df["close"].shift(-horizon) / df["close"] - 1.0
    label = (forward_ret > threshold).astype(float)
    label[forward_ret.isna()] = np.nan
    return label


def make_dataset(df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    """Build features + label and drop rows with any NaN (warm-up / tail)."""
    feat = build_features(df)
    feat["label"] = build_labels(df, horizon, threshold)
    cols = ["timestamp"] + FEATURE_COLUMNS + ["label", "close"]
    return feat[cols].dropna().reset_index(drop=True)
