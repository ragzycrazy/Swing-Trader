"""
Downloads and caches Angel One's instrument master (all tradable symbols +
their tokens), so the dashboard can search by name instead of you hunting
for tokens manually.

The file is large (~30MB, updated daily by Angel One) so we cache it on
disk and refresh once every 24h.
"""
import json
import time
import urllib.request
from pathlib import Path

MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
CACHE_PATH = Path(__file__).parent / "instrument_master_cache.json"
CACHE_TTL_SECONDS = 24 * 60 * 60

_cache: list | None = None
_cache_loaded_at: float = 0


def _download_master() -> list:
    with urllib.request.urlopen(MASTER_URL, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load() -> list:
    global _cache, _cache_loaded_at
    now = time.time()
    if _cache is not None and (now - _cache_loaded_at) < CACHE_TTL_SECONDS:
        return _cache

    if CACHE_PATH.exists() and (now - CACHE_PATH.stat().st_mtime) < CACHE_TTL_SECONDS:
        _cache = json.loads(CACHE_PATH.read_text())
        _cache_loaded_at = now
        return _cache

    data = _download_master()
    CACHE_PATH.write_text(json.dumps(data))
    _cache = data
    _cache_loaded_at = now
    return _cache


def search_symbols(query: str, exchange: str = "NSE", limit: int = 15) -> list[dict]:
    """
    Search by name or trading symbol, e.g. 'tcs' or 'reliance'.
    Only returns equity (EQ) series by default to keep results relevant
    for swing trading.
    """
    data = _load()
    q = query.strip().upper()
    if not q:
        return []

    results = []
    for row in data:
        if row.get("exch_seg") != exchange:
            continue
        name = row.get("name", "")
        symbol = row.get("symbol", "")
        if q in name.upper() or q in symbol.upper():
            if symbol.endswith("-EQ") or exchange != "NSE":
                results.append({
                    "trading_symbol": symbol,
                    "name": name,
                    "symbol_token": row.get("token"),
                    "exchange": row.get("exch_seg"),
                })
        if len(results) >= limit:
            break

    return results
