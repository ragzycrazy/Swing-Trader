"""
Backtests the strategy in strategy.py against historical daily candles.

For every bar where evaluate_signal() would have fired, simulates entering
at the NEXT bar's open (not the signal bar's own close — see the note in
strategy.py on why) and holding for HOLD_DAYS trading days, or exiting
early if target/stop is hit. Reports aggregate stats net of an estimated
transaction cost, so the numbers reflect what you'd actually keep, not
just raw price movement.
"""
import pandas as pd
from strategy import add_indicators, evaluate_signal, DEFAULT_PARAMS

# Estimated round-trip cost for an NSE equity delivery trade: STT (0.1% on
# both buy and sell legs), exchange transaction charges, SEBI fees, stamp
# duty, and GST on those charges. Assumes zero brokerage (standard for
# delivery trades on most discount brokers, including Angel One's base
# plan). Does NOT include DP charges (~₹20 flat per sell, charged by your
# depository) since that's a fixed rupee amount, not a percentage, and
# this backtest doesn't track position size in rupees. For a real trade,
# knock a little more off for that. Adjust this constant if your actual
# costs differ.
ROUND_TRIP_COST_PCT = 0.25


def run_backtest(candles: list, params: dict = None) -> dict:
    df = add_indicators_from_candles(candles)
    return run_backtest_on_df(df, params)


def run_backtest_on_df(df: pd.DataFrame, params: dict = None) -> dict:
    """
    Same simulation as run_backtest, but takes an already-indicator-computed
    DataFrame. Lets the optimizer compute indicators once per stock and
    reuse them across hundreds of parameter combinations, instead of
    recomputing SMA/RSI/ATR from scratch every time — those don't change
    between combinations, only the entry thresholds do.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    trades = []

    i = 0
    while i < len(df):
        window = df.iloc[: i + 1]
        signal = evaluate_signal(window, p)
        if signal is None:
            i += 1
            continue

        sig_idx = i
        entry_idx = sig_idx + 1
        if entry_idx >= len(df):
            break  # signal fired on the last available bar — no next day to enter on

        # Re-anchor entry to the next bar's open, and recompute stop/target
        # from THAT price (not the signal bar's close) — this is the
        # actual price you'd have been able to act on.
        atr = signal["atr"]
        entry_price = float(df.iloc[entry_idx]["open"])
        stop = entry_price - p["atr_stop_multiplier"] * atr
        risk = entry_price - stop
        target = entry_price + p["min_risk_reward"] * risk

        exit_idx = min(entry_idx + p["hold_days"], len(df) - 1)
        outcome = "time_exit"
        exit_price = float(df.iloc[exit_idx]["close"])

        # Walk forward day-by-day (including entry day itself, in case of
        # a gap straight through stop or target) to check for early exits
        for j in range(entry_idx, exit_idx + 1):
            bar = df.iloc[j]
            if bar["low"] <= stop:
                exit_price = stop
                outcome = "stopped_out"
                exit_idx = j
                break
            if bar["high"] >= target:
                exit_price = target
                outcome = "target_hit"
                exit_idx = j
                break

        gross_return_pct = (exit_price - entry_price) / entry_price * 100
        net_return_pct = gross_return_pct - ROUND_TRIP_COST_PCT
        hold_days_actual = exit_idx - entry_idx

        trades.append({
            "signal_date": str(df.iloc[sig_idx]["timestamp"]),
            "entry_date": str(df.iloc[entry_idx]["timestamp"]),
            "exit_date": str(df.iloc[exit_idx]["timestamp"]),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "stop_loss": round(stop, 2),
            "target_price": round(target, 2),
            "outcome": outcome,
            "gross_return_pct": round(gross_return_pct, 2),
            "return_pct": round(net_return_pct, 2),  # net of estimated costs — win/loss and averages use this
            "hold_days": hold_days_actual,
        })

        # Skip ahead past this trade's exit — you can't open a second
        # position in the same stock while the first is still open, so the
        # scan for the next signal resumes only once this one has closed.
        i = exit_idx + 1

    return summarize(trades)


def add_indicators_from_candles(candles: list) -> pd.DataFrame:
    from strategy import candles_to_df
    df = candles_to_df(candles)
    return add_indicators(df)


def summarize(trades: list) -> dict:
    if not trades:
        return {
            "total_trades": 0,
            "win_rate_pct": None,
            "avg_return_pct": None,
            "avg_hold_days": None,
            "max_drawdown_pct": None,
            "trades": [],
        }

    df = pd.DataFrame(trades)
    wins = df[df["return_pct"] > 0]
    win_rate = len(wins) / len(df) * 100

    # Simple running equity curve (assumes trades don't overlap capital)
    equity = (1 + df["return_pct"] / 100).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100
    max_dd = drawdown.min()

    return {
        "total_trades": len(df),
        "win_rate_pct": round(float(win_rate), 1),
        "avg_return_pct": round(float(df["return_pct"].mean()), 2),
        "avg_gross_return_pct": round(float(df["gross_return_pct"].mean()), 2),
        "avg_hold_days": round(float(df["hold_days"].mean()), 1),
        "max_drawdown_pct": round(float(max_dd), 2),
        "trades": trades,
    }
