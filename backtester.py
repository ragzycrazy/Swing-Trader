"""
Backtests the strategy in strategy.py against historical daily candles.

For every bar where evaluate_signal() would have fired, simulates holding
the position for HOLD_DAYS trading days (or exiting early if target/stop is
hit), then reports aggregate stats. This is what tells you whether
"high-confidence" is actually earning its name for a given stock/timeframe
before you trust it live.
"""
import pandas as pd
from strategy import add_indicators, evaluate_signal, HOLD_DAYS


def run_backtest(candles: list) -> dict:
    df = add_indicators_from_candles(candles)
    trades = []

    for i in range(len(df)):
        window = df.iloc[: i + 1]
        signal = evaluate_signal(window)
        if signal is None:
            continue

        entry_idx = i
        entry_price = signal["entry_price"]
        stop = signal["stop_loss"]
        target = signal["target_price"]

        exit_idx = min(entry_idx + HOLD_DAYS, len(df) - 1)
        outcome = "time_exit"
        exit_price = float(df.iloc[exit_idx]["close"])

        # Walk forward day-by-day to check for early stop/target hits
        for j in range(entry_idx + 1, exit_idx + 1):
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

        return_pct = (exit_price - entry_price) / entry_price * 100
        hold_days_actual = exit_idx - entry_idx

        trades.append({
            "entry_date": str(df.iloc[entry_idx]["timestamp"]),
            "exit_date": str(df.iloc[exit_idx]["timestamp"]),
            "entry_price": entry_price,
            "exit_price": round(exit_price, 2),
            "stop_loss": stop,
            "target_price": target,
            "outcome": outcome,
            "return_pct": round(return_pct, 2),
            "hold_days": hold_days_actual,
        })

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
        "win_rate_pct": round(win_rate, 1),
        "avg_return_pct": round(df["return_pct"].mean(), 2),
        "avg_hold_days": round(df["hold_days"].mean(), 1),
        "max_drawdown_pct": round(float(max_dd), 2),
        "trades": trades,
    }
