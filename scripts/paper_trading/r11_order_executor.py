"""
R11 Order Executor — Binance Testnet REST API.

Credentials loaded from conf/local/credentials.yml:
  binance_testnet:
    api_key:    YOUR_KEY
    api_secret: YOUR_SECRET

Get testnet credentials at: https://testnet.binance.vision
Base URL:                    https://testnet.binance.vision
"""
from __future__ import annotations

import csv
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

ROOT         = Path(__file__).resolve().parents[2]
CREDS_FILE   = ROOT / "conf" / "local" / "secrets.yml"
STATE_DIR    = Path(__file__).parent / "state"
TRADE_LOG    = STATE_DIR / "trade_log.csv"
TESTNET_URL  = "https://testnet.binance.vision"
RECV_WINDOW  = 5000   # ms

_TRADE_LOG_FIELDS = [
    "timestamp", "date", "side", "symbol",
    "btc_qty", "fill_price", "usdt_value", "fees_usdt", "order_id", "note",
]


def _load_credentials() -> tuple[str, str]:
    if not CREDS_FILE.exists():
        raise FileNotFoundError(
            f"Credentials file not found: {CREDS_FILE}\n"
            "  Add to conf/local/secrets.yml:\n"
            "    api_keys:\n"
            "      binance_testnet:\n"
            "        api_key:    YOUR_KEY\n"
            "        api_secret: YOUR_SECRET"
        )
    with open(CREDS_FILE) as fh:
        creds = yaml.safe_load(fh)
    cfg = creds["binance_testnet"]
    return str(cfg["api_key"]), str(cfg["api_secret"])


def _sign(params: dict, secret: str) -> dict:
    query = "&".join(f"{k}={v}" for k, v in params.items())
    sig   = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    return {**params, "signature": sig}


def _headers(api_key: str) -> dict:
    return {"X-MBX-APIKEY": api_key, "Content-Type": "application/x-www-form-urlencoded"}


def _timestamp_ms() -> int:
    return int(time.time() * 1000)


def _ensure_trade_log() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not TRADE_LOG.exists():
        with open(TRADE_LOG, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=_TRADE_LOG_FIELDS).writeheader()


def _log_trade(row: dict) -> None:
    _ensure_trade_log()
    with open(TRADE_LOG, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_TRADE_LOG_FIELDS)
        writer.writerow({f: row.get(f, "") for f in _TRADE_LOG_FIELDS})


class OrderExecutor:
    """Interface to Binance Testnet for paper trading R11."""

    def __init__(self) -> None:
        self.api_key, self.secret = _load_credentials()
        _ensure_trade_log()
        # Connectivity check
        resp = requests.get(f"{TESTNET_URL}/api/v3/ping", timeout=10)
        resp.raise_for_status()
        print("[OrderExecutor] Connected to Binance Testnet ✓")

    def _get(self, path: str, params: dict | None = None, signed: bool = False) -> dict:
        p = params or {}
        if signed:
            p["timestamp"]  = _timestamp_ms()
            p["recvWindow"] = RECV_WINDOW
            p = _sign(p, self.secret)
        resp = requests.get(f"{TESTNET_URL}{path}", headers=_headers(self.api_key),
                            params=p, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, params: dict) -> dict:
        params["timestamp"]  = _timestamp_ms()
        params["recvWindow"] = RECV_WINDOW
        params = _sign(params, self.secret)
        resp = requests.post(f"{TESTNET_URL}{path}", headers=_headers(self.api_key),
                             params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_btc_price(self) -> float:
        """Current BTCUSDT mid price from Testnet."""
        data = self._get("/api/v3/ticker/price", {"symbol": "BTCUSDT"})
        return float(data["price"])

    def get_balances(self) -> dict[str, float]:
        """Return {"USDT": float, "BTC": float} free balances."""
        data = self._get("/api/v3/account", signed=True)
        result = {}
        for b in data.get("balances", []):
            if b["asset"] in ("USDT", "BTC"):
                result[b["asset"]] = float(b["free"])
        return result

    def get_portfolio_value(self) -> dict:
        """Compute total portfolio value in USDT."""
        balances  = self.get_balances()
        btc_price = self.get_btc_price()
        usdt      = balances.get("USDT", 0.0)
        btc       = balances.get("BTC",  0.0)
        btc_usdt  = btc * btc_price
        total     = usdt + btc_usdt
        return {
            "total_usdt":   total,
            "usdt_free":    usdt,
            "btc_held":     btc,
            "btc_usdt":     btc_usdt,
            "btc_price":    btc_price,
            "position_pct": btc_usdt / total if total > 0 else 0.0,
        }

    def _normalize_quantity(self, symbol: str, quantity: float) -> float:
        import math
        if not hasattr(self, "_exchange_info"):
            data = self._get("/api/v3/exchangeInfo")
            self._exchange_info = {s["symbol"]: s for s in data["symbols"]}
        symbol_info = self._exchange_info[symbol]
        lot_filter  = next(f for f in symbol_info["filters"] if f["filterType"] == "LOT_SIZE")
        step        = float(lot_filter["stepSize"])
        min_qty     = float(lot_filter["minQty"])
        quantity    = math.floor(quantity / step) * step
        if quantity < min_qty:
            return 0.0
        return float(f"{quantity:.8f}")

    def place_market_buy(self, usdt_amount: float, note: str = "") -> dict:
        """Market BUY quoteOrderQty=usdt_amount. Returns fill summary."""
        params = {
            "symbol":        "BTCUSDT",
            "side":          "BUY",
            "type":          "MARKET",
            "quoteOrderQty": f"{usdt_amount:.2f}",
        }
        data = self._post("/api/v3/order", params)
        fill = self._parse_fill(data)
        _log_trade({**fill, "side": "BUY", "note": note})
        print(f"  [BUY]  {fill['btc_qty']:.6f} BTC @ {fill['fill_price']:.2f} USDT"
              f"  (${fill['usdt_value']:.2f})")
        return fill

    def place_market_sell(self, btc_qty: float, note: str = "") -> dict:
        """Market SELL quantity=btc_qty BTC. Returns fill summary."""
        btc_qty = self._normalize_quantity("BTCUSDT", btc_qty)
        if btc_qty == 0:
            print("[OrderExecutor] Quantity below minQty — skipping order")
            return {}
        params = {
            "symbol":   "BTCUSDT",
            "side":     "SELL",
            "type":     "MARKET",
            "quantity": f"{btc_qty:.8f}",
        }
        data = self._post("/api/v3/order", params)
        fill = self._parse_fill(data)
        _log_trade({**fill, "side": "SELL", "note": note})
        print(f"  [SELL] {fill['btc_qty']:.6f} BTC @ {fill['fill_price']:.2f} USDT"
              f"  (${fill['usdt_value']:.2f})")
        return fill

    @staticmethod
    def _parse_fill(order_resp: dict) -> dict:
        fills     = order_resp.get("fills", [])
        total_qty = sum(float(f["qty"]) for f in fills)
        total_val = sum(float(f["qty"]) * float(f["price"]) for f in fills)
        total_fee = sum(float(f.get("commission", 0)) for f in fills)
        avg_price = total_val / total_qty if total_qty > 0 else 0.0
        now_utc   = datetime.now(timezone.utc)
        return {
            "timestamp":  now_utc.isoformat(),
            "date":       now_utc.strftime("%Y-%m-%d"),
            "symbol":     order_resp.get("symbol", "BTCUSDT"),
            "btc_qty":    round(total_qty, 6),
            "fill_price": round(avg_price, 2),
            "usdt_value": round(total_val, 2),
            "fees_usdt":  round(total_fee, 4),
            "order_id":   str(order_resp.get("orderId", "")),
        }

    def get_open_orders(self) -> list[dict]:
        return self._get("/api/v3/openOrders", {"symbol": "BTCUSDT"}, signed=True)

    def read_trade_log(self) -> list[dict]:
        if not TRADE_LOG.exists():
            return []
        with open(TRADE_LOG) as fh:
            return list(csv.DictReader(fh))
