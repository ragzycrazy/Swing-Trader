"""
Searches for better entry-threshold parameters using walk-forward
validation, rather than just picking whatever performed best on the full
history (which would be overfitting — finding numbers that happen to fit
past noise, not a real edge).

Method:
  1. Split the stock's history chronologically: first ~70% = TRAIN,
     last ~30% = TEST. TEST is never touched during the search.
  2. Try every parameter combination in the grid on TRAIN, score each,
     and discard any with too few trades to be meaningful (a single
     lucky trade isn't evidence of anything).
  3. Take the top candidates from TRAIN, then re-evaluate ONLY those on
     TEST — data the search never saw.
  4. Return combos ranked by TEST performance, with TRAIN performance
     also shown for comparison. A combo that looks great on TRAIN but
     falls apart on TEST is a classic overfitting sign — including both
     numbers makes that visible instead of hiding it.

This doesn't guarantee future results either — no backtest can — but it's
a meaningfully more honest check than tuning against a single dataset.
"""
import pandas as pd
from backtester import run_backtest_on_df, summarize
from strategy import add_indicators, candles_to_df, evaluate_signal, DEFAULT_PARAMS

# Search grid — kept intentionally modest so this finishes in a reasonable
# time per request. RSI is expressed as paired (min, max) bands rather than
# independent min/max values — that halves the grid by construction (no
# invalid min>=max combos) and every band still means something sensible.
RSI_BANDS = [(40, 60), (45, 65), (50, 70)]
PARAM_GRID = {
    "rsi_band": RSI_BANDS,
    "volume_multiplier": [1.3, 1.5, 1.8],
    "atr_stop_multiplier": [1.0, 1.5, 2.0],
    "min_risk_reward": [1.2, 1.5, 2.0],
    "hold_days": [7, 10, 15],
}

MIN_TRADES_TRAIN = 5   # discard combos with fewer trades than this on TRAIN
MIN_TRADES_TEST = 2    # and fewer than this on TEST
TOP_N_FROM_TRAIN = 15  # how many TRAIN candidates advance to TEST evaluation
TOP_N_RESULTS = 3       # how many final combos to return


def _param_combinations():
    import itertools
    keys = list(PARAM_GRID.keys())
    for values in itertools.product(*(PARAM_GRID[k] for k in keys)):
        combo = dict(zip(keys, values))
        rsi_min, rsi_max = combo.pop("rsi_band")
        combo["rsi_min"] = rsi_min
        combo["rsi_max"] = rsi_max
        yield combo


def _score(result: dict) -> float:
    """Combines win rate and average return into one ranking number.
    Weighted toward avg_return since that's what actually matters for P&L,
    but win rate still counts — a strategy with 90% losses and one huge
    win is not what most swing traders can stomach holding through."""
    if not result["total_trades"]:
        return float("-inf")
    return (result["avg_return_pct"] or 0) * 0.7 + (result["win_rate_pct"] or 0) * 0.3


def optimize(candles: list) -> dict:
    df = add_indicators(candles_to_df(candles))

    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    if len(train_df) < 100 or len(test_df) < 40:
        return {
            "error": "Not enough history for a reliable train/test split. "
                     "Try requesting more days (700+) for this stock.",
        }

    # Step 1: evaluate every combination on TRAIN
    train_results = []
    for combo in _param_combinations():
        result = run_backtest_on_df(train_df, combo)
        if result["total_trades"] < MIN_TRADES_TRAIN:
            continue
        train_results.append((combo, result, _score(result)))

    if not train_results:
        return {
            "error": "No parameter combination produced enough trades on "
                     "the training window. This stock may be too quiet for "
                     "this strategy shape, or needs a longer history.",
        }

    train_results.sort(key=lambda r: r[2], reverse=True)
    top_candidates = train_results[:TOP_N_FROM_TRAIN]

    # Step 2: re-evaluate only the TRAIN winners on TEST (data they never saw)
    validated = []
    for combo, train_result, _ in top_candidates:
        test_result = run_backtest_on_df(test_df, combo)
        if test_result["total_trades"] < MIN_TRADES_TEST:
            continue
        validated.append({
            "params": combo,
            "train": {
                "trades": train_result["total_trades"],
                "win_rate_pct": train_result["win_rate_pct"],
                "avg_return_pct": train_result["avg_return_pct"],
            },
            "test": {
                "trades": test_result["total_trades"],
                "win_rate_pct": test_result["win_rate_pct"],
                "avg_return_pct": test_result["avg_return_pct"],
            },
            "test_score": _score(test_result),
        })

    if not validated:
        return {
            "error": "Several parameter combinations looked promising on "
                     "the training window, but none held up with enough "
                     "trades on the more recent unseen window. That itself "
                     "is a useful (if disappointing) result — it suggests "
                     "this stock's recent behavior doesn't match its past "
                     "closely enough for this strategy shape to trust yet.",
        }

    validated.sort(key=lambda r: r["test_score"], reverse=True)
    top = validated[:TOP_N_RESULTS]
    for r in top:
        del r["test_score"]

    return {
        "train_window_bars": len(train_df),
        "test_window_bars": len(test_df),
        "default_params_for_comparison": DEFAULT_PARAMS,
        "top_results": top,
    }
