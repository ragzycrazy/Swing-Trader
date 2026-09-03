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
TREND_SLOPE_LOOKBACK = 5       # sma_fast must be rising over this many bars
RSI_PERIOD = 14
RSI_MIN = 45
RSI_MAX = 65
VOLUME_LOOKBACK = 20
VOLUME_MULTIPLIER = 1.5
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 1.5      # stop = entry - 1.5x ATR
MIN_RISK_REWARD = 1.5          # target = entry + 1.5x the stop distance
HOLD_DAYS = 10                 # ~2 trading weeks
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
    df["sma_fast_prior"] = df["sma_fast"].shift(TREND_SLOPE_LOOKBACK)
    df["vol_avg"] = df["volume"].rolling(VOLUME_LOOKBACK).mean()

    # RSI(14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR(14) — average true range, used to size the stop/target to how
    # much this specific stock typically moves, instead of a fixed lookback
    prior_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prior_close).abs(),
        (df["low"] - prior_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(ATR_PERIOD).mean()
    return df


DEFAULT_PARAMS = {
    "rsi_min": RSI_MIN,
    "rsi_max": RSI_MAX,
    "volume_multiplier": VOLUME_MULTIPLIER,
    "atr_stop_multiplier": ATR_STOP_MULTIPLIER,
    "min_risk_reward": MIN_RISK_REWARD,
    "hold_days": HOLD_DAYS,
}


def evaluate_signal(df: pd.DataFrame, params: dict = None) -> dict | None:
    """
    Look at the most recent bar and decide if it's a high-confidence buy.
    Returns a signal dict, or None if the setup doesn't qualify.

    The returned entry_price/stop_loss/target_price are REFERENCE levels
    based on the signal bar's own close — useful for live display ("here's
    roughly what today's setup looks like"), but not what the backtester
    actually uses to simulate a trade. See backtester.py: it re-anchors
    entry to the NEXT bar's open, since you can't act on a bar's close
    before that bar has finished — computing a "buy at today's close"
    signal using today's own close is look-ahead bias.

    `params` optionally overrides any of DEFAULT_PARAMS — used by the
    optimizer to test many threshold combinations without editing this
    file. Indicator windows (SMA/RSI/ATR periods) stay fixed; only the
    entry thresholds are tunable, to keep the search space sane.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    if len(df) < TREND_SMA_SLOW + ATR_PERIOD:
        return None

    row = df.iloc[-1]
    if any(pd.isna(row[c]) for c in ["sma_fast", "sma_slow", "sma_fast_prior", "rsi", "atr"]):
        return None

    trend_ok = (
        row["close"] > row["sma_slow"]
        and row["sma_fast"] > row["sma_slow"]
        and row["sma_fast"] > row["sma_fast_prior"]  # 20-day average is rising, not flat
    )
    momentum_ok = p["rsi_min"] <= row["rsi"] <= p["rsi_max"]
    volume_ok = row["volume"] >= p["volume_multiplier"] * row["vol_avg"]

    if not (trend_ok and momentum_ok and volume_ok):
        return None

    entry = float(row["close"])
    atr = float(row["atr"])
    if atr <= 0:
        return None

    stop = entry - p["atr_stop_multiplier"] * atr
    risk = entry - stop
    target = entry + p["min_risk_reward"] * risk
    risk_reward = (target - entry) / risk

    return {
        "entry_price": round(entry, 2),
        "stop_loss": round(stop, 2),
        "target_price": round(target, 2),
        "risk_reward": round(risk_reward, 2),
        "rsi": round(float(row["rsi"]), 1),
        "atr": round(atr, 2),
        "hold_days": p["hold_days"],
        "timestamp": str(row["timestamp"]),
    }
