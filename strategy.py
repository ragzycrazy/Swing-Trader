"""
Swing-trading signal logic for a ~2-week hold.

A BUY signal fires only when ALL filters agree — this is what makes it
"high-confidence" rather than a loose single-indicator trigger. None of
this guarantees a winning trade; it narrows the field to setups that have
historically had better odds, and every signal ships with a stop-loss.

Tune the thresholds below to make the filter stricter or looser.
"""
import pandas as pd
import numpy as np


# --- Tunable parameters -----------------------------------------------
TREND_SMA_FAST = 20
TREND_SMA_SLOW = 50
RSI_PERIOD = 14
RSI_MIN = 45
RSI_MAX = 65
VOLUME_LOOKBACK = 20
VOLUME_MULTIPLIER = 1.5
MIN_RISK_REWARD = 2.0
HOLD_DAYS = 10  # ~2 trading weeks
# ------------------------------------------------------------------------


def candles_to_df(candles: list) -> pd.DataFrame:
    """Convert Angel One candle list [[ts, o, h, l, c, v], ...] to a DataFrame."""
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma_fast"] = df["close"].rolling(TREND_SMA_FAST).mean()
    df["sma_slow"] = df["close"].rolling(TREND_SMA_SLOW).mean()
    df["vol_avg"] = df["volume"].rolling(VOLUME_LOOKBACK).mean()

    # RSI(14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Recent swing low (20-bar) for stop placement
    df["swing_low_20"] = df["low"].rolling(20).min()
    return df


def evaluate_signal(df: pd.DataFrame) -> dict | None:
    """
    Look at the most recent bar and decide if it's a high-confidence buy.
    Returns a signal dict, or None if the setup doesn't qualify.
    """
    if len(df) < TREND_SMA_SLOW + 5:
        return None

    row = df.iloc[-1]
    if any(pd.isna(row[c]) for c in ["sma_fast", "sma_slow", "rsi", "swing_low_20"]):
        return None

    trend_ok = row["close"] > row["sma_slow"] and row["sma_fast"] > row["sma_slow"]
    momentum_ok = RSI_MIN <= row["rsi"] <= RSI_MAX
    volume_ok = row["volume"] >= VOLUME_MULTIPLIER * row["vol_avg"]

    if not (trend_ok and momentum_ok and volume_ok):
        return None

    entry = float(row["close"])
    stop = float(row["swing_low_20"])
    risk = entry - stop
    if risk <= 0:
        return None

    target = entry + MIN_RISK_REWARD * risk
    risk_reward = (target - entry) / risk

    if risk_reward < MIN_RISK_REWARD:
        return None

    return {
        "entry_price": round(entry, 2),
        "stop_loss": round(stop, 2),
        "target_price": round(target, 2),
        "risk_reward": round(risk_reward, 2),
        "rsi": round(float(row["rsi"]), 1),
        "hold_days": HOLD_DAYS,
        "timestamp": str(row["timestamp"]),
    }
