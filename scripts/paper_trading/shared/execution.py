"""
Módulo compartilhado de execução de trades — ÚNICA fonte de verdade.

Usado por: specialist_4h_paper_trader.py e specialist_4h_monitor_1h.py

Responsabilidades:
  - Paths canônicos (derivados do PROJECT root, nunca hardcoded)
  - Portfolio read/write (file-locked, atomic)
  - Pending signal read/write/cancel (validação explícita)
  - execute_buy() / execute_sell() — position sizing, fees, portfolio update
  - _SIGNALS_FIELDS — lista canônica de colunas do signals.csv

NÃO é responsabilidade deste módulo:
  - Features de modelo (Group A/B)
  - Control layer (gates, stops)
  - Signal logging (cada script mantém seu próprio _append_signal)
  - Mark-to-market
"""
from __future__ import annotations

import csv
import fcntl
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths canônicos (derivados de PROJECT root) ───────────────────────────────
PROJECT        = Path(__file__).resolve().parents[3]   # crypto-market-state/
STATE_DIR      = PROJECT / "scripts" / "paper_trading" / "state" / "specialist_4h"
PORTFOLIO_PATH = STATE_DIR / "portfolio.json"
SIGNALS_PATH   = STATE_DIR / "signals.csv"
PENDING_PATH   = STATE_DIR / "pending_signal.json"
CONFIG_PATH    = STATE_DIR / "config.json"

# ── Canonical signals.csv field list ─────────────────────────────────────────
# Keep in sync — this is the single definition; both scripts import from here.
SIGNALS_FIELDS = [
    "timestamp", "candle_close", "price_close",
    "allocation_raw", "allocation_final",
    "r11_prob_bull", "r11_entropy", "regime_age_log", "stress_score",
    "returns_4h", "volatility_24h", "volume_zscore", "buy_pressure", "price_range_4h",
    "position_btc", "position_usdt", "portfolio_value",
    "action", "delta_allocation", "fee_paid",
    "stop_loss_triggered", "stop_gain_triggered",
    "regime_gate_triggered", "timing_gate_score", "rsi_4h", "bb_pct_b",
    "news_gate_triggered",
    "macro_gate_mult", "deriv_gate_mult",
    "entry_price", "drawdown_pct",
    "source",
    "entry_source", "timing_score_1h", "pending_age_hours",
]

# ── Pending signal validation ─────────────────────────────────────────────────
# Type-checked fields — wrong type is treated same as missing (corrupt data)
_REQUIRED_TYPES: dict[str, type | tuple] = {
    "has_pending":      bool,
    "allocation_final": (int, float),
    "created_at":       str,
    "expires_at":       str,
}


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def parse_utc(ts: str) -> datetime:
    """Parse ISO timestamp, assume UTC if tz-naive."""
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def atomic_write_json(path: Path, data: dict) -> None:
    """
    Write JSON atomically via temp file + os.replace().
    Safe for local filesystem and future NFS/container use.
    fcntl lock on writes is replaced by the atomicity of os.replace().
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        suffix=".tmp",
        prefix=path.stem + "_",
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_json_locked(path: Path) -> dict:
    """Read JSON with shared fcntl lock."""
    with open(path, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_SH)
        try:
            return json.load(fh)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    return read_json_locked(CONFIG_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio
# ══════════════════════════════════════════════════════════════════════════════

def read_portfolio() -> dict:
    return read_json_locked(PORTFOLIO_PATH)


def write_portfolio(data: dict) -> None:
    atomic_write_json(PORTFOLIO_PATH, data)


def write_portfolio_safe(path: Path, new_portfolio: dict) -> None:
    """
    Write portfolio only if new_portfolio is not stale vs what's already on disk.

    Compares last_update timestamps: if the on-disk version is newer than
    new_portfolio, the write is skipped (stale-write protection).
    Falls back to unconditional write if timestamps are absent or unreadable.
    """
    try:
        current = read_json_locked(path)
        current_ts = current.get("last_update")
        new_ts     = new_portfolio.get("last_update")
        if current_ts and new_ts and current_ts > new_ts:
            logger.warning(
                "Skipping stale portfolio write: disk=%s > new=%s",
                current_ts, new_ts,
            )
            return
    except Exception:
        pass  # File missing or unreadable — proceed with write
    atomic_write_json(path, new_portfolio)


# ══════════════════════════════════════════════════════════════════════════════
# Pending signal
# ══════════════════════════════════════════════════════════════════════════════

def read_pending_signal() -> dict:
    """
    Read pending_signal.json with strong validation.
    Returns {"has_pending": False} on any error:
      - file missing
      - JSON corrupt
      - required field absent
      - field has wrong type
      - timestamps not parseable as UTC
    Never raises.
    """
    if not PENDING_PATH.exists():
        return {"has_pending": False}
    try:
        data = read_json_locked(PENDING_PATH)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read pending_signal.json: %s", exc)
        return {"has_pending": False}

    # has_pending must be bool
    if "has_pending" not in data or not isinstance(data["has_pending"], bool):
        logger.error("pending_signal.json: 'has_pending' missing or not bool")
        return {"has_pending": False}

    # Inject executed=False if missing (backward compat)
    if "executed" not in data:
        data["executed"] = False

    # When signal is inactive, no further validation needed
    if not data["has_pending"]:
        return data

    # Active signal — validate remaining required fields with type checks
    for field, expected_type in _REQUIRED_TYPES.items():
        if field == "has_pending":
            continue  # already checked above
        if field not in data:
            logger.error("pending_signal.json missing field '%s'", field)
            return {"has_pending": False}
        if not isinstance(data[field], expected_type):
            logger.error(
                "pending_signal.json field '%s' has wrong type: got %s, expected %s",
                field, type(data[field]).__name__, expected_type,
            )
            return {"has_pending": False}

    # Validate timestamps are parseable
    try:
        parse_utc(data["created_at"])
        parse_utc(data["expires_at"])
    except Exception as exc:
        logger.error("pending_signal.json has unparseable timestamp: %s", exc)
        return {"has_pending": False}

    return data


def write_pending_signal(data: dict) -> None:
    """Write pending_signal.json atomically. Always ensures 'executed' field is present."""
    payload = {**data}
    if "executed" not in payload:
        payload["executed"] = False
    atomic_write_json(PENDING_PATH, payload)
    logger.info(
        "Pending signal saved: has_pending=%s alloc=%.3f executed=%s",
        payload.get("has_pending"), payload.get("allocation_final", 0.0),
        payload.get("executed"),
    )


def cancel_pending_signal(reason: str = "cancelled") -> None:
    """Reset pending_signal.json to empty state."""
    atomic_write_json(PENDING_PATH, {
        "has_pending": False,
        "executed": False,
        "allocation_raw": 0.0,
        "allocation_final": 0.0,
        "specialist_candle": None,
        "specialist_price": None,
        "created_at": None,
        "expires_at": None,
        "r11_prob_bull": None,
        "r11_regime": None,
        "news_sentiment_4h": None,
        "regime_gate_passed": False,
        "news_gate_passed": False,
    })
    logger.info("Pending signal cancelled: reason=%s", reason)


def is_pending_expired(pending: dict) -> bool:
    """Return True if the pending signal's expires_at is in the past."""
    expires_str = pending.get("expires_at")
    if not expires_str:
        return False
    try:
        return datetime.now(timezone.utc) > parse_utc(expires_str)
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Trade execution — ÚNICA fonte de verdade para BUY/SELL
# ══════════════════════════════════════════════════════════════════════════════

def execute_buy(
    price:              float,
    allocation:         float,
    config:             dict,
    portfolio:          Optional[dict] = None,
    entry_source:       str   = "specialist",
    timing_score_1h:    Optional[float] = None,
    pending_age_hours:  Optional[float] = None,
) -> tuple[dict, float]:
    """
    Execute a BUY trade. Returns (updated_portfolio, fee_paid).

    Does NOT write portfolio to disk — caller is responsible for write_portfolio().
    This allows the caller to add extra fields (e.g. r11_regime, entry_price)
    before persisting.

    Position sizing:
        target_exposure = clip((allocation - threshold) / (1 - threshold), 0, 1)
        delta = target_exposure - current_exposure
        If delta < min_delta_allocation → skip (returns portfolio unchanged, fee=0)
    """
    if portfolio is None:
        portfolio = read_portfolio()

    cl           = config.get("control_layer", {})
    fee_rate     = float(config.get("fee_rate", 0.0004))
    threshold    = float(cl.get("entry_threshold", 0.38))
    min_delta    = float(config.get("min_delta_allocation", 0.05))

    if allocation <= threshold:
        logger.info("[execute_buy] allocation=%.3f <= threshold=%.3f — skip", allocation, threshold)
        return portfolio, 0.0

    target_exposure  = min((allocation - threshold) / (1.0 - threshold), 1.0)
    current_exposure = portfolio["position_pct"]
    delta            = target_exposure - current_exposure

    if delta < min_delta:
        logger.info("[execute_buy] delta=%.3f < min_delta=%.3f — skip", delta, min_delta)
        return portfolio, 0.0

    total_usdt   = portfolio["total_usdt"]
    delta_usdt   = delta * total_usdt
    fee_paid     = delta_usdt * fee_rate
    usdt_to_spend = delta_usdt + fee_paid          # full debit from usdt_free
    if usdt_to_spend > portfolio["usdt_free"]:     # cap at available cash
        usdt_to_spend = portfolio["usdt_free"]
        delta_usdt    = usdt_to_spend / (1.0 + fee_rate)
        fee_paid      = usdt_to_spend - delta_usdt
    btc_bought = delta_usdt / price

    p = portfolio.copy()
    p["usdt_free"]   = round(portfolio["usdt_free"] - usdt_to_spend, 8)
    p["btc_held"]    = round(portfolio["btc_held"] + btc_bought, 8)
    p["btc_price"]   = price
    p["total_usdt"]  = round(p["usdt_free"] + p["btc_held"] * price, 4)
    p["position_pct"] = (p["btc_held"] * price) / p["total_usdt"] if p["total_usdt"] > 0 else 0.0
    p["last_update"] = datetime.now(timezone.utc).isoformat()

    logger.info(
        "[execute_buy] price=$%.2f alloc=%.3f delta=%.3f btc=+%.6f fee=$%.4f "
        "source=%s timing=%.3f",
        price, allocation, delta, btc_bought, fee_paid,
        entry_source, timing_score_1h or 0.0,
    )
    return p, round(fee_paid, 6)


def execute_sell(
    price:       float,
    btc_to_sell: float,
    config:      dict,
    reason:      str            = "SELL",
    portfolio:   Optional[dict] = None,
) -> tuple[dict, float]:
    """
    Execute a SELL trade. Returns (updated_portfolio, fee_paid).

    Does NOT write portfolio to disk — caller is responsible for write_portfolio().
    Caller sets cooldown, entry_price=None, etc. after this call.
    """
    if portfolio is None:
        portfolio = read_portfolio()

    fee_rate    = float(config.get("fee_rate", 0.0004))
    btc_to_sell = min(btc_to_sell, float(portfolio["btc_held"]))

    gross    = btc_to_sell * price
    fee_paid = gross * fee_rate
    net      = gross - fee_paid

    p = portfolio.copy()
    p["usdt_free"]   = round(portfolio["usdt_free"] + net, 4)
    p["btc_held"]    = round(portfolio["btc_held"] - btc_to_sell, 8)
    if p["btc_held"] < 1e-8:
        p["btc_held"] = 0.0
    p["btc_price"]   = price
    p["total_usdt"]  = round(p["usdt_free"] + p["btc_held"] * price, 4)
    p["position_pct"] = (p["btc_held"] * price) / p["total_usdt"] if p["total_usdt"] > 0 else 0.0
    p["last_update"] = datetime.now(timezone.utc).isoformat()

    pnl_pct = (price - float(portfolio.get("entry_price") or price)) / float(portfolio.get("entry_price") or price)
    logger.info(
        "[execute_sell] reason=%s price=$%.2f btc=%.6f fee=$%.4f pnl=%+.2f%%",
        reason, price, btc_to_sell, fee_paid, pnl_pct * 100,
    )
    return p, round(fee_paid, 6)
