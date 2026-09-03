"""
FastAPI service tying together AngelClient, strategy, and backtester.

Endpoints:
  GET  /health
  POST /backtest         { symbol_token, exchange, trading_symbol, days }
  GET  /signals/live      { watchlist }  -> checks each symbol, returns any that qualify
  POST /watchlist         add/remove symbols to scan live

Run: uvicorn main:app --host 0.0.0.0 --port 8000
"""
import asyncio
from datetime import datetime, time
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from angel_client import AngelClient
from strategy import candles_to_df, add_indicators, evaluate_signal
from backtester import run_backtest
from optimizer import optimize
from instrument_lookup import search_symbols
import push

app = FastAPI(title="Swing Trader Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your dashboard's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

_client: AngelClient | None = None
WATCHLIST: dict[str, dict] = {}  # trading_symbol -> {symbol_token, exchange}


def get_client() -> AngelClient:
    global _client
    if _client is None:
        _client = AngelClient()
        _client.login()
    return _client


IST = ZoneInfo("Asia/Kolkata")


def market_is_open() -> bool:
    now_ist = datetime.now(IST)
    # NSE cash market hours, IST — checked explicitly in IST regardless of
    # which region the server itself is physically deployed in
    return time(9, 15) <= now_ist.time() <= time(15, 30) and now_ist.weekday() < 5


def _drop_incomplete_today_candle(df):
    """
    During market hours, the API's "today" candle is still forming — its
    close is just whatever the price happens to be right now, not a real
    confirmed close. Evaluating signals against a half-finished candle
    means the signal can appear, disappear, and change entry/stop/target
    levels every few minutes as the price moves, which isn't a stable
    basis for a decision. Dropping it means every signal is based on the
    last CONFIRMED close (yesterday's, if it's currently market hours),
    so it stays consistent for the rest of the session.
    """
    if len(df) == 0:
        return df
    today_ist = datetime.now(IST).date()
    if df.iloc[-1]["timestamp"].date() == today_ist:
        return df.iloc[:-1].reset_index(drop=True)
    return df


class BacktestRequest(BaseModel):
    symbol_token: str
    exchange: str
    trading_symbol: str
    days: int = 500


class WatchlistItem(BaseModel):
    trading_symbol: str
    symbol_token: str
    exchange: str


class PushSubscription(BaseModel):
    endpoint: str
    keys: dict


class BatchSymbol(BaseModel):
    trading_symbol: str
    symbol_token: str
    exchange: str


class BatchBacktestRequest(BaseModel):
    symbols: list[BatchSymbol] | None = None  # defaults to current watchlist if omitted
    days: int = 365


class OptimizeRequest(BaseModel):
    symbol_token: str
    exchange: str
    trading_symbol: str
    days: int = 700  # needs enough history for a meaningful train/test split


@app.get("/health")
def health():
    return {"status": "ok", "market_open": market_is_open()}


@app.post("/backtest")
def backtest(req: BacktestRequest):
    try:
        client = get_client()
        candles = client.get_historical_candles(
            req.symbol_token, req.exchange, interval="ONE_DAY", days=req.days
        )
        result = run_backtest(candles)
        result["trading_symbol"] = req.trading_symbol
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest/batch")
def backtest_batch(req: BatchBacktestRequest):
    """
    Runs the default-parameter strategy across several stocks at once
    (your whole watchlist, by default) over the same lookback window, and
    returns them sorted best-to-worst by average return. This is the
    fastest way to see which of your watchlist stocks this strategy shape
    actually suits, instead of testing one at a time.
    """
    if req.symbols:
        targets = [(s.trading_symbol, s.symbol_token, s.exchange) for s in req.symbols]
    else:
        targets = [(sym, meta["symbol_token"], meta["exchange"]) for sym, meta in WATCHLIST.items()]

    if not targets:
        raise HTTPException(status_code=400, detail="No symbols provided and watchlist is empty.")

    client = get_client()
    results = []
    for trading_symbol, symbol_token, exchange in targets:
        try:
            candles = client.get_historical_candles(
                symbol_token, exchange, interval="ONE_DAY", days=req.days
            )
            result = run_backtest(candles)
            result["trading_symbol"] = trading_symbol
            results.append(result)
        except Exception as e:
            results.append({"trading_symbol": trading_symbol, "error": str(e)})

    ranked = sorted(
        (r for r in results if "error" not in r and r["total_trades"] > 0),
        key=lambda r: r["avg_return_pct"],
        reverse=True,
    )
    no_trades = [r for r in results if "error" not in r and r["total_trades"] == 0]
    errors = [r for r in results if "error" in r]

    return {"days": req.days, "ranked": ranked, "no_signals": no_trades, "errors": errors}


@app.post("/backtest/optimize")
def backtest_optimize(req: OptimizeRequest):
    """
    Searches for better entry-threshold parameters for one stock using a
    train/test split (see optimizer.py) so the result isn't just numbers
    that happened to fit the past. Takes ~10-15 seconds — this is CPU-bound
    work, not something to poll frequently.
    """
    try:
        client = get_client()
        candles = client.get_historical_candles(
            req.symbol_token, req.exchange, interval="ONE_DAY", days=req.days
        )
        result = optimize(candles)
        result["trading_symbol"] = req.trading_symbol
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/watchlist")
def get_watchlist():
    return {"watchlist": list(WATCHLIST.keys())}


@app.post("/watchlist")
def add_to_watchlist(item: WatchlistItem):
    WATCHLIST[item.trading_symbol] = {
        "symbol_token": item.symbol_token,
        "exchange": item.exchange,
    }
    return {"watchlist": list(WATCHLIST.keys())}


@app.delete("/watchlist/{trading_symbol}")
def remove_from_watchlist(trading_symbol: str):
    WATCHLIST.pop(trading_symbol, None)
    return {"watchlist": list(WATCHLIST.keys())}


@app.get("/symbols/search")
def symbols_search(q: str, exchange: str = "NSE"):
    """Search tradable symbols by name, e.g. ?q=reliance — returns tokens
    ready to drop straight into the watchlist, no manual lookup needed."""
    try:
        return {"results": search_symbols(q, exchange)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/push/vapid-public-key")
def vapid_public_key():
    return {"key": push.VAPID_PUBLIC_KEY}


@app.post("/push/subscribe")
def push_subscribe(sub: PushSubscription):
    push.add_subscription(sub.model_dump())
    return {"status": "subscribed"}


@app.post("/push/unsubscribe")
def push_unsubscribe(sub: PushSubscription):
    push.remove_subscription(sub.endpoint)
    return {"status": "unsubscribed"}


@app.get("/signals/live")
def live_signals():
    """
    Scans every symbol on the watchlist and returns any that currently
    qualify as a high-confidence buy, based on the last CONFIRMED daily
    close (today's still-forming candle is excluded — see
    _drop_incomplete_today_candle). This means a signal reflects "as of
    yesterday's close, this stock qualifies" — stable throughout the
    session rather than flickering with every price tick. Also attaches
    the actual current live price for comparison, since the reference
    entry_price in the signal is based on that prior close, not this
    exact moment.
    """
    if not market_is_open():
        return {"market_open": False, "signals": []}

    client = get_client()
    signals = []
    for trading_symbol, meta in WATCHLIST.items():
        try:
            candles = client.get_historical_candles(
                meta["symbol_token"], meta["exchange"], interval="ONE_DAY", days=120
            )
            df = _drop_incomplete_today_candle(candles_to_df(candles))
            df = add_indicators(df)
            signal = evaluate_signal(df)
            if signal:
                signal["trading_symbol"] = trading_symbol
                try:
                    ltp_data = client.get_ltp(meta["symbol_token"], meta["exchange"], trading_symbol)
                    signal["current_price"] = ltp_data.get("ltp")
                except Exception:
                    signal["current_price"] = None  # LTP fetch failed — reference price still valid
                signals.append(signal)
        except Exception as e:
            # Don't let one bad symbol kill the whole scan
            signals.append({"trading_symbol": trading_symbol, "error": str(e)})

    return {"market_open": True, "signals": signals}


# --- Background scanner: pushes an alert the moment a NEW signal appears,
# so you get notified even if the dashboard isn't open. Polls every 5 min
# during market hours; dedups so you're not re-alerted for the same
# signal on every poll.
SCAN_INTERVAL_SECONDS = 300
_already_alerted: set[str] = set()


async def _background_scanner():
    while True:
        try:
            if market_is_open():
                result = live_signals()
                for s in result["signals"]:
                    if "error" in s:
                        continue
                    key = f"{s['trading_symbol']}:{s['timestamp']}"
                    if key not in _already_alerted:
                        _already_alerted.add(key)
                        push.notify_all(
                            title=f"Buy signal: {s['trading_symbol']}",
                            body=f"Entry ₹{s['entry_price']} · Target ₹{s['target_price']} · Stop ₹{s['stop_loss']}",
                        )
            else:
                _already_alerted.clear()  # reset dedup cache once market closes
        except Exception:
            pass  # never let the background loop die on a transient error
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


@app.on_event("startup")
async def start_background_scanner():
    asyncio.create_task(_background_scanner())
