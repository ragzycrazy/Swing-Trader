"""
Thin wrapper around Angel One's SmartAPI (smartapi-python).

Handles login (with TOTP), historical candle fetch, and live LTP (last
traded price) fetch. Credentials are read from environment variables only —
never hardcode them.

Angel One SmartAPI docs: https://smartapi.angelbroking.com/docs

Note: the SmartApi package is imported lazily inside __init__ (not at the
top of this file) so that a problem with that dependency only breaks
Angel One-specific features, not the whole server.
"""
import os
from datetime import datetime, timedelta


class AngelClient:
    def __init__(self):
        try:
            import pyotp
            from SmartApi import SmartConnect
        except ImportError as e:
            raise RuntimeError(
                f"Angel One SDK failed to import ({e}). Check that "
                "smartapi-python and pyotp installed correctly."
            )

        self._pyotp = pyotp
        self.api_key = os.environ["ANGEL_API_KEY"]
        self.client_code = os.environ["ANGEL_CLIENT_CODE"]
        self.pin = os.environ["ANGEL_PIN"]
        self.totp_secret = os.environ["ANGEL_TOTP_SECRET"]
        self.smart = SmartConnect(api_key=self.api_key)
        self.session = None

    def login(self):
        """Authenticate using client code, PIN, and a freshly-generated TOTP."""
        totp = self._pyotp.TOTP(self.totp_secret).now()
        self.session = self.smart.generateSession(self.client_code, self.pin, totp)
        if not self.session.get("status"):
            raise RuntimeError(f"Angel One login failed: {self.session}")
        return self.session

    def get_historical_candles(self, symbol_token: str, exchange: str,
                                interval: str = "ONE_DAY", days: int = 300):
        """
        Fetch historical OHLCV candles.
        interval: ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, ONE_HOUR, ONE_DAY, etc.
        symbol_token: Angel One's numeric instrument token (look up via their
                      instrument master CSV: https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M"),
        }
        resp = self.smart.getCandleData(params)
        if not resp.get("status"):
            raise RuntimeError(f"Historical data fetch failed: {resp}")
        # Each candle: [timestamp, open, high, low, close, volume]
        return resp["data"]

    def get_ltp(self, symbol_token: str, exchange: str, trading_symbol: str):
        """Fetch last traded price for a live quote."""
        resp = self.smart.ltpData(exchange, trading_symbol, symbol_token)
        if not resp.get("status"):
            raise RuntimeError(f"LTP fetch failed: {resp}")
        return resp["data"]

    def place_order(self, trading_symbol: str, symbol_token: str, exchange: str,
                     transaction_type: str, quantity: int, order_type: str = "MARKET",
                     product_type: str = "DELIVERY", price: float = 0):
        """
        Place an order. transaction_type: BUY/SELL. Use with care — this
        sends a real order to your live Angel One account.
        """
        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": trading_symbol,
            "symboltoken": symbol_token,
            "transactiontype": transaction_type,
            "exchange": exchange,
            "ordertype": order_type,
            "producttype": product_type,
            "duration": "DAY",
            "price": str(price),
            "quantity": str(quantity),
        }
        return self.smart.placeOrder(order_params)
