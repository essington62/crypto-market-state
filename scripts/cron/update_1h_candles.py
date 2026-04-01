"""
Atualiza candles 1h do BTC via Binance API (endpoint público — sem key).
Roda a cada hora via cron (minuto :50).
Append incremental — busca apenas candles novos desde o último timestamp.

Output:
    data/01_raw/spot/crypto/1h/BTCUSDT_1h.parquet

Index: timestamp (UTC, tz-aware)
Cols:  open, high, low, close, volume, quote_volume, trades,
       taker_buy_volume, taker_buy_quote_volume
"""

import json
import logging
import sys
import urllib.request
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
PARQUET     = ROOT / "data/01_raw/spot/crypto/1h/BTCUSDT_1h.parquet"
LOG_DIR     = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
COLS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "trades",
    "taker_buy_volume", "taker_buy_quote_volume",
]
FALLBACK_DAYS = 30


def fetch_candles(start_ms: int, limit: int = 500) -> pd.DataFrame:
    """Busca candles 1h da Binance API a partir de start_ms (epoch ms)."""
    url = (
        f"{BINANCE_KLINES}?symbol=BTCUSDT&interval=1h"
        f"&startTime={start_ms}&limit={limit}"
    )
    req  = urllib.request.Request(url, headers={"User-Agent": "CryptoTrader/1.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    raw  = json.loads(resp.read())

    if not raw:
        return pd.DataFrame(columns=COLS)

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_volume", "taker_buy_quote_volume", "_ignore",
    ])

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")

    for col in ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = df[col].astype(float)
    df["trades"] = df["trades"].astype(int)

    return df[COLS]


def update() -> None:
    # ── Ler parquet existente ─────────────────────────────────────────────────
    if PARQUET.exists():
        existing = pd.read_parquet(PARQUET)
        existing.index = pd.to_datetime(existing.index, utc=True)
        existing = existing.sort_index()
        last_ts  = existing.index[-1]
        # +1ms para não duplicar o candle existente
        start_ms = int(last_ts.timestamp() * 1000) + 1
        logger.info(f"Existing: {len(existing)} candles, last: {last_ts}")
    else:
        PARQUET.parent.mkdir(parents=True, exist_ok=True)
        now_ms   = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        start_ms = now_ms - FALLBACK_DAYS * 24 * 3600 * 1000
        existing = pd.DataFrame(columns=COLS)
        logger.info(f"Parquet not found — fetching last {FALLBACK_DAYS} days")

    # ── Buscar novos candles (pode precisar de múltiplos requests) ────────────
    all_new: list[pd.DataFrame] = []
    now_ms  = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    while start_ms < now_ms:
        chunk = fetch_candles(start_ms, limit=500)
        if chunk.empty:
            break
        all_new.append(chunk)
        last_chunk_ms = int(chunk.index[-1].timestamp() * 1000)
        if last_chunk_ms >= now_ms or len(chunk) < 500:
            break
        start_ms = last_chunk_ms + 1

    if not all_new:
        logger.info("No new candles — already up to date")
        return

    new = pd.concat(all_new)
    new = new[~new.index.duplicated(keep="last")]

    # ── Concat e dedup ────────────────────────────────────────────────────────
    if len(existing) > 0:
        combined = pd.concat([existing, new])
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new

    combined = combined.sort_index()

    # Remover candle incompleto (candle atual ainda aberto)
    now_utc     = pd.Timestamp.now(tz="UTC")
    closed_only = combined[combined.index < now_utc.floor("h")]

    closed_only.to_parquet(PARQUET)
    logger.info(
        f"Updated: {len(new)} new candles fetched | "
        f"total {len(closed_only)} | "
        f"last closed: {closed_only.index[-1]}"
    )


if __name__ == "__main__":
    try:
        update()
    except Exception as e:
        logger.error(f"update_1h_candles FAILED: {e}")
        sys.exit(1)
