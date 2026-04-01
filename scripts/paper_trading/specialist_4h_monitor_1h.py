#!/usr/bin/env python3
"""
Monitor 1h — Stop Loss / Stop Gain / Trailing Stop check.

Runs every hour at :02 via cron.
NEVER opens positions — only monitors and executes exits.
Shares state with specialist_4h_paper_trader.py via file lock.

Price logic:
  raw_price      → stop loss  (sensitive — protects capital)
  smoothed_price → stop gain + trailing stop (anti-spike)
  execute_exit uses raw_price (real market price)

Trailing stop:
  Only activates after price rises >= min_profit_to_activate from entry.
  Triggers when smoothed_price falls >= pct from max_price_since_entry.

Usage:
    python scripts/paper_trading/specialist_4h_monitor_1h.py
    python scripts/paper_trading/specialist_4h_monitor_1h.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import logging
import math
import os
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Absolute paths (works in cron without cd) ─────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent.parent
BASE_SCRIPTS = BASE_DIR / "scripts" / "paper_trading"
STATE_DIR    = BASE_SCRIPTS / "state" / "specialist_4h"
PORTFOLIO_PATH = STATE_DIR / "portfolio.json"
CONFIG_PATH    = STATE_DIR / "config.json"
SIGNALS_PATH   = STATE_DIR / "signals.csv"

# ── Shared execution module ───────────────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(BASE_SCRIPTS))
from shared.execution import (
    PENDING_PATH           as PENDING_SIGNAL_PATH,
    PORTFOLIO_PATH         as _SHARED_PORTFOLIO_PATH,
    SIGNALS_FIELDS         as _SHARED_SIGNALS_FIELDS,
    cancel_pending_signal,
    execute_buy            as _shared_execute_buy,
    is_pending_expired,
    read_pending_signal,
    write_pending_signal   as _shared_write_pending,
    write_portfolio_safe   as _write_portfolio_safe,
)

# Binance public API
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

# ── Canonical signals fields — imported from shared.execution ─────────────────
_SIGNALS_FIELDS = _SHARED_SIGNALS_FIELDS

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MONITOR-1H] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# File-lock helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_json_locked(path: Path) -> dict:
    """Read JSON with shared lock."""
    with open(path, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_SH)
        try:
            return json.load(fh)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _save_json_locked(path: Path, data: dict) -> None:
    """Write JSON with exclusive lock."""
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _append_signal_locked(row: dict) -> None:
    """Append one row to signals.csv using the canonical field list.

    Reads existing header first to stay backward-compatible.
    Adds 'source' column if missing (migration).
    """
    # Determine columns: use existing CSV header if available
    if SIGNALS_PATH.exists() and SIGNALS_PATH.stat().st_size > 0:
        with open(SIGNALS_PATH, "r", newline="") as fh:
            fcntl.flock(fh, fcntl.LOCK_SH)
            try:
                reader = csv.reader(fh)
                cols = next(reader, None) or _SIGNALS_FIELDS
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        # Add any missing columns from _SIGNALS_FIELDS (migration for older CSV files)
        missing = [c for c in _SIGNALS_FIELDS if c not in cols]
        if missing:
            cols = list(cols) + missing
            try:
                import pandas as pd
                df = pd.read_csv(SIGNALS_PATH)
                for c in missing:
                    df[c] = ""
                df.to_csv(SIGNALS_PATH, index=False)
                logger.info("[migrate] Added columns to signals.csv: %s", missing)
            except Exception as exc:
                logger.warning("[migrate] Could not add columns %s: %s", missing, exc)
    else:
        cols = _SIGNALS_FIELDS

    with open(SIGNALS_PATH, "a", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(fh, fieldnames=cols)
            if not SIGNALS_PATH.exists() or SIGNALS_PATH.stat().st_size == 0:
                writer.writeheader()
            writer.writerow({c: row.get(c, "") for c in cols})
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ══════════════════════════════════════════════════════════════════════════════
# Price helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_raw_price() -> float:
    """Fetch live BTCUSDT price via Binance ticker (instantaneous)."""
    url      = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    response = urllib.request.urlopen(url, timeout=10)
    return float(json.loads(response.read())["price"])


def _get_prices(portfolio: dict) -> tuple[float, float]:
    """
    Returns (raw_price, smoothed_price).

    raw      → used for stop_loss   (sensitive, protects capital)
    smoothed → used for stop_gain + trailing (anti-spike noise)
    execute_exit always uses raw_price (real market price).
    """
    raw      = _fetch_raw_price()
    last     = float(portfolio.get("last_monitor_price") or raw)
    smoothed = 0.7 * raw + 0.3 * last
    return raw, smoothed


# ══════════════════════════════════════════════════════════════════════════════
# Exit execution
# ══════════════════════════════════════════════════════════════════════════════

def _execute_exit(
    portfolio:   dict,
    config:      dict,
    exit_price:  float,
    reason:      str,
    dry_run:     bool,
) -> None:
    """
    Sell all BTC, update portfolio, log signal row.

    reason: "STOP_LOSS" | "TAKE_PROFIT" | "TRAILING_STOP"
    """
    btc_held    = float(portfolio["btc_held"])
    entry_price = float(portfolio["entry_price"])
    fee_rate    = float(config.get("fee_rate", 0.0004))

    gross       = btc_held * exit_price
    fee         = gross * fee_rate
    net         = gross - fee
    pnl_pct     = (exit_price - entry_price) / entry_price

    p = portfolio.copy()
    p["usdt_free"]            = float(p.get("usdt_free", 0.0)) + net
    p["btc_held"]             = 0.0
    p["total_usdt"]           = p["usdt_free"]
    p["btc_price"]            = exit_price
    p["position_pct"]         = 0.0
    p["last_update"]          = datetime.now(timezone.utc).isoformat()
    p["entry_price"]          = None
    p["entry_time"]           = None
    p["max_price_since_entry"] = None
    p["last_monitor_price"]   = None

    if reason == "STOP_LOSS":
        cl               = config.get("control_layer", {})
        cooldown_candles = int(cl.get("stop_loss", {}).get("cooldown_candles", 1))
        cooldown_end     = datetime.now(timezone.utc) + timedelta(hours=4 * cooldown_candles)
        p["stop_loss_cooldown_until"] = cooldown_end.isoformat()
        logger.info("[monitor] Stop loss cooldown until: %s", cooldown_end.isoformat())
    else:
        # TAKE_PROFIT / TRAILING_STOP: clear cooldown → specialist can re-enter
        p["stop_loss_cooldown_until"] = None

    emoji = {"TAKE_PROFIT": "🎯", "STOP_LOSS": "🛑", "TRAILING_STOP": "📉"}.get(reason, "⚠️")
    logger.info(
        "%s %s — exit=$%.2f  entry=$%.2f  P&L=%+.2f%%  fee=$%.4f  btc=%.6f",
        emoji, reason, exit_price, entry_price, pnl_pct * 100, fee, btc_held,
    )

    if dry_run:
        logger.info("[dry-run] No state written.")
        return

    _write_portfolio_safe(PORTFOLIO_PATH, p)

    now_str = datetime.now(timezone.utc).isoformat()
    _append_signal_locked({
        "timestamp":             now_str,
        "candle_close":          now_str,
        "price_close":           round(exit_price, 2),
        "allocation_raw":        "",
        "allocation_final":      0.0,
        "r11_prob_bull":         "",
        "r11_entropy":           "",
        "regime_age_log":        "",
        "stress_score":          "",
        "returns_4h":            "",
        "volatility_24h":        "",
        "volume_zscore":         "",
        "buy_pressure":          "",
        "price_range_4h":        "",
        "position_btc":          0.0,
        "position_usdt":         round(p["usdt_free"], 2),
        "portfolio_value":       round(p["total_usdt"], 2),
        "action":                reason,
        "delta_allocation":      round(-float(portfolio.get("position_pct", 0.0)), 6),
        "fee_paid":              round(fee, 4),
        "stop_loss_triggered":   reason == "STOP_LOSS",
        "stop_gain_triggered":   reason == "TAKE_PROFIT",
        "regime_gate_triggered": "",
        "timing_gate_score":     "",
        "rsi_4h":                "",
        "bb_pct_b":              "",
        "macro_gate_mult":       "",
        "deriv_gate_mult":       "",
        "entry_price":           round(entry_price, 2),
        "drawdown_pct":          round(pnl_pct, 6),
        "source":                "monitor_1h",
    })


# ══════════════════════════════════════════════════════════════════════════════
# Timing execution helpers (multi-horizon pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_1h_candles(limit: int = 50) -> list[dict]:
    """Fetch last N closed 1h candles from Binance public API."""
    url = (
        f"{BINANCE_KLINES}?symbol=BTCUSDT&interval=1h"
        f"&limit={limit + 1}"
    )
    req  = urllib.request.Request(url, headers={"User-Agent": "CryptoTrader/1.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    raw  = json.loads(resp.read())
    candles = []
    now_h = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for item in raw:
        open_time = datetime.fromtimestamp(item[0] / 1000, tz=timezone.utc)
        if open_time >= now_h:
            continue  # skip current (incomplete) candle
        candles.append({
            "open_time": open_time,
            "open":   float(item[1]),
            "high":   float(item[2]),
            "low":    float(item[3]),
            "close":  float(item[4]),
            "volume": float(item[5]),
        })
    return candles[-limit:]


def _compute_timing_features(candles: list[dict]) -> dict:
    """
    Compute RSI-14, Bollinger %B (20 periods, 2 std), volume z-score (50 periods)
    from list of 1h candle dicts. Returns dict with rsi_14, bb_pct_b, volume_zscore.
    """
    if len(candles) < 20:
        return {"rsi_14": float("nan"), "bb_pct_b": float("nan"), "volume_zscore": 0.0}

    closes  = [c["close"]  for c in candles]
    volumes = [c["volume"] for c in candles]

    # RSI-14
    rsi_14 = float("nan")
    if len(closes) >= 15:
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains  = [max(d, 0.0) for d in deltas]
        losses = [max(-d, 0.0) for d in deltas]
        avg_gain = sum(gains[-14:]) / 14
        avg_loss = sum(losses[-14:]) / 14
        if avg_loss < 1e-12:
            rsi_14 = 100.0
        else:
            rs     = avg_gain / avg_loss
            rsi_14 = 100.0 - 100.0 / (1.0 + rs)

    # Bollinger %B (20 periods, 2 std)
    bb_pct_b = float("nan")
    if len(closes) >= 20:
        window  = closes[-20:]
        mean    = sum(window) / 20
        variance = sum((x - mean) ** 2 for x in window) / 20
        std     = math.sqrt(variance) if variance > 0 else 0.0
        price   = closes[-1]
        upper   = mean + 2 * std
        lower   = mean - 2 * std
        if upper > lower:
            bb_pct_b = (price - lower) / (upper - lower)
        else:
            bb_pct_b = 0.5

    # Volume z-score (rolling 50)
    vol_zscore = 0.0
    n_vol = min(50, len(volumes))
    if n_vol >= 5:
        vol_window = volumes[-n_vol:]
        v_mean = sum(vol_window) / n_vol
        v_var  = sum((x - v_mean) ** 2 for x in vol_window) / n_vol
        v_std  = math.sqrt(v_var) if v_var > 0 else 0.0
        if v_std > 0:
            vol_zscore = (volumes[-1] - v_mean) / v_std

    return {"rsi_14": rsi_14, "bb_pct_b": bb_pct_b, "volume_zscore": vol_zscore}


def _calc_timing_score(features: dict, cfg_timing: dict) -> float:
    """Compute timing score from RSI, BB, volume features."""
    weights = cfg_timing.get("weights", {"rsi": 0.4, "bollinger": 0.3, "volume": 0.3})
    rsi_threshold = float(cfg_timing.get("rsi_threshold", 40))
    bb_threshold  = float(cfg_timing.get("bb_threshold", 0.30))

    rsi = features["rsi_14"]
    bb  = features["bb_pct_b"]
    vol_z = features["volume_zscore"]

    rsi_score = max(0.0, (rsi_threshold - rsi) / rsi_threshold) if (not math.isnan(rsi) and rsi < rsi_threshold) else 0.0
    bb_score  = max(0.0, (bb_threshold - bb) / bb_threshold)    if (not math.isnan(bb)  and bb  < bb_threshold)  else 0.0
    vol_score = 1.0 if vol_z > 0 else 0.0

    return (weights.get("rsi", 0.4) * rsi_score
            + weights.get("bollinger", 0.3) * bb_score
            + weights.get("volume", 0.3) * vol_score)


def _execute_entry_via_timing(
    portfolio:    dict,
    config:       dict,
    price:        float,
    pending:      dict,
    timing_score: float,
    dry_run:      bool,
) -> None:
    """
    Execute BUY from pending signal via timing layer.

    Uses shared execute_buy() for position sizing + portfolio update.
    Handles signal logging and entry tracking here.
    """
    allocation = float(pending["allocation_final"])

    # Compute pending age
    pending_age_hours: float | str = ""
    created_at_str = pending.get("created_at")
    if created_at_str:
        try:
            from shared.execution import parse_utc as _parse_utc
            pending_age_hours = round(
                (datetime.now(timezone.utc) - _parse_utc(created_at_str)).total_seconds() / 3600,
                2,
            )
        except Exception:
            pass

    if dry_run:
        logger.info(
            "[dry-run] Would execute BUY: price=$%.2f alloc=%.3f timing=%.3f age=%.2fh",
            price, allocation, timing_score,
            pending_age_hours if pending_age_hours != "" else 0.0,
        )
        return

    # Execute trade via shared module — single source of truth
    portfolio, fee_paid = _shared_execute_buy(
        price, allocation, config,
        portfolio=portfolio,
        entry_source="timing_layer",
        timing_score_1h=timing_score,
        pending_age_hours=pending_age_hours if pending_age_hours != "" else None,
    )

    # Entry tracking fields (specialist sets these on direct BUY; we mirror here)
    portfolio["entry_price"]           = price
    portfolio["entry_time"]            = datetime.now(timezone.utc).isoformat()
    portfolio["max_price_since_entry"] = price
    portfolio["stop_loss_cooldown_until"] = None

    _write_portfolio_safe(_SHARED_PORTFOLIO_PATH, portfolio)

    logger.info(
        "[MONITOR] BUY executed via timing layer: price=$%.2f alloc=%.3f "
        "btc=%.6f fee=$%.4f timing=%.3f age=%.2fh",
        price, allocation,
        float(portfolio.get("btc_held", 0)),
        fee_paid, timing_score,
        pending_age_hours if pending_age_hours != "" else 0.0,
    )

    now_str = datetime.now(timezone.utc).isoformat()
    _append_signal_locked({
        "timestamp":             now_str,
        "candle_close":          now_str,
        "price_close":           round(price, 2),
        "allocation_raw":        round(float(pending.get("allocation_raw", 0.0)), 6),
        "allocation_final":      round(allocation, 6),
        "r11_prob_bull":         pending.get("r11_prob_bull", ""),
        "r11_entropy":           "",
        "regime_age_log":        "",
        "stress_score":          "",
        "returns_4h":            "",
        "volatility_24h":        "",
        "volume_zscore":         "",
        "buy_pressure":          "",
        "price_range_4h":        "",
        "position_btc":          round(float(portfolio.get("btc_held", 0)), 8),
        "position_usdt":         round(float(portfolio.get("usdt_free", 0)), 2),
        "portfolio_value":       round(float(portfolio.get("total_usdt", 0)), 2),
        "action":                "BUY",
        "delta_allocation":      round(float(portfolio.get("position_pct", 0)), 6),
        "fee_paid":              round(fee_paid, 4),
        "stop_loss_triggered":   False,
        "stop_gain_triggered":   False,
        "regime_gate_triggered": not pending.get("regime_gate_passed", True),
        "timing_gate_score":     round(timing_score, 4),
        "rsi_4h":                "",
        "bb_pct_b":              "",
        "news_gate_triggered":   not pending.get("news_gate_passed", True),
        "macro_gate_mult":       "",
        "deriv_gate_mult":       "",
        "entry_price":           round(price, 2),
        "drawdown_pct":          0.0,
        "source":                "timing_layer",
        "entry_source":          "timing_layer",
        "timing_score_1h":       round(timing_score, 4),
        "pending_age_hours":     pending_age_hours,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Main monitor
# ══════════════════════════════════════════════════════════════════════════════

def run_monitor(dry_run: bool = False) -> None:
    """
    1h stop-check loop.
    Order: stop_gain → stop_loss → trailing_stop.
    """
    config    = _load_json_locked(CONFIG_PATH)
    portfolio = _load_json_locked(PORTFOLIO_PATH)
    snapshot_update = portfolio.get("last_update")

    # ── 1. No position → check for pending signal (timing execution) ─────────
    if float(portfolio.get("btc_held", 0.0)) <= 1e-8:
        # read_pending_signal() validates JSON and required fields; never raises
        pending = read_pending_signal()

        if not pending.get("has_pending"):
            logger.info("[MONITOR] No position, no pending signal — skip")
            return

        # Check expiry via shared helper (timezone-safe)
        if is_pending_expired(pending):
            logger.info(
                "[MONITOR] Pending signal expired (%s) — cancelling",
                pending.get("expires_at"),
            )
            cancel_pending_signal("expired")
            return

        # Idempotency guard — prevent double execution if monitor crashed mid-cycle
        if pending.get("executed"):
            logger.info("[MONITOR] Pending already executed — skipping (cleanup on next cycle)")
            return

        # Compute timing score from 1h candles
        try:
            candles_1h = _fetch_1h_candles(limit=50)
        except Exception as exc:
            logger.warning("[MONITOR] Failed to fetch 1h candles: %s — keeping pending", exc)
            return

        features = _compute_timing_features(candles_1h)
        timing_score = _calc_timing_score(
            features,
            config.get("control_layer", {}).get("timing_execution", {}),
        )
        min_score = float(
            config.get("control_layer", {})
                  .get("timing_execution", {})
                  .get("min_score", 0.30)
        )

        logger.info(
            "[MONITOR] Timing check: score=%.3f (RSI=%.1f BB=%.2f vol_z=%.2f) min=%.2f — %s",
            timing_score,
            features["rsi_14"]     if not math.isnan(features["rsi_14"])     else -1.0,
            features["bb_pct_b"]   if not math.isnan(features["bb_pct_b"])   else -1.0,
            features["volume_zscore"],
            min_score,
            "EXECUTING" if timing_score >= min_score else "WAITING",
        )

        if timing_score >= min_score:
            try:
                price = _fetch_raw_price()
            except Exception as exc:
                logger.warning("[MONITOR] Price fetch failed: %s — keeping pending signal", exc)
                return
            # execute_entry_via_timing uses shared execute_buy — single source of truth
            _execute_entry_via_timing(portfolio, config, price, pending, timing_score, dry_run)
            if not dry_run:
                # Mark executed BEFORE cancelling — crash-safe idempotency window
                _shared_write_pending({**pending, "executed": True})
                cancel_pending_signal("executed")
        else:
            logger.info(
                "[MONITOR] Timing not ready (%.3f < %.2f) — keeping pending signal",
                timing_score, min_score,
            )
        return

    entry_price = portfolio.get("entry_price")
    if not entry_price or float(entry_price) <= 0:
        logger.warning("btc_held > 0 but entry_price missing — state inconsistency. Skip.")
        return
    entry_price = float(entry_price)

    # ── 2. Cooldown check (only matters if somehow position survived cooldown) ──
    cooldown_str = portfolio.get("stop_loss_cooldown_until")
    if cooldown_str:
        try:
            from shared.execution import parse_utc as _parse_utc_local
            if datetime.now(timezone.utc) < _parse_utc_local(cooldown_str):
                logger.info("In cooldown until %s — skip", cooldown_str)
                return
        except (ValueError, TypeError):
            pass

    # ── 3. Fetch price ─────────────────────────────────────────────────────────
    try:
        raw_price, smoothed_price = _get_prices(portfolio)
    except Exception as exc:
        logger.warning("Price fetch failed: %s — skip (retry next cycle)", exc)
        return

    # ── 4. Race condition check (double-read) ──────────────────────────────────
    portfolio_recheck = _load_json_locked(PORTFOLIO_PATH)
    if portfolio_recheck.get("last_update") != snapshot_update:
        logger.warning("Portfolio modified during price fetch — abort (concurrent write).")
        return

    # ── 5. Update tracking fields ─────────────────────────────────────────────
    max_price = float(portfolio.get("max_price_since_entry") or entry_price)
    if raw_price > max_price:
        max_price = raw_price
    portfolio["max_price_since_entry"] = max_price
    portfolio["last_monitor_price"]    = raw_price

    cl = config.get("control_layer", {})

    # ── 6. STOP GAIN — smoothed price (anti-spike) ────────────────────────────
    sg = cl.get("stop_gain", {})
    if sg.get("enabled", False):
        gain_pct = float(sg.get("pct", 0.015))
        profit   = (smoothed_price - entry_price) / entry_price
        if profit >= gain_pct:
            logger.info(
                "TAKE_PROFIT — smoothed=%+.2f%% >= threshold=%+.2f%%",
                profit * 100, gain_pct * 100,
            )
            _execute_exit(portfolio, config, raw_price, "TAKE_PROFIT", dry_run)
            return

    # ── 7. STOP LOSS — raw price (sensitive, protects capital) ────────────────
    sl = cl.get("stop_loss", {})
    if sl.get("enabled", False):
        stop_pct  = float(sl.get("pct", 0.02))
        drawdown  = (raw_price - entry_price) / entry_price
        if drawdown <= -stop_pct:
            logger.info(
                "STOP_LOSS — raw=%+.2f%% <= threshold=%.2f%%",
                drawdown * 100, -stop_pct * 100,
            )
            _execute_exit(portfolio, config, raw_price, "STOP_LOSS", dry_run)
            return

    # ── 8. TRAILING STOP — smoothed price, only after min_profit ──────────────
    ts = cl.get("trailing_stop", {})
    if ts.get("enabled", False):
        trail_pct   = float(ts.get("pct", 0.01))
        min_profit  = float(ts.get("min_profit_to_activate", 0.005))

        profit_from_entry = (max_price - entry_price) / entry_price
        if profit_from_entry >= min_profit:
            trail_dd = (smoothed_price - max_price) / max_price
            if trail_dd <= -trail_pct:
                logger.info(
                    "TRAILING_STOP — profit_from_entry=%+.2f%% >= %.2f%% (active), "
                    "trail_dd=%+.2f%% <= -%.2f%%",
                    profit_from_entry * 100, min_profit * 100,
                    trail_dd * 100, trail_pct * 100,
                )
                _execute_exit(portfolio, config, raw_price, "TRAILING_STOP", dry_run)
                return

    # ── 9. No stop triggered — log status ─────────────────────────────────────
    drawdown_now    = (raw_price - entry_price) / entry_price
    dist_from_max   = (raw_price - max_price) / max_price if max_price else 0.0
    sl_pct  = float(sl.get("pct", 0.02))
    sg_pct  = float(sg.get("pct", 0.015))

    logger.info(
        "OK — raw=$%.2f smooth=$%.2f | entry=$%.2f max=$%.2f | "
        "P&L=%+.2f%% from_max=%+.2f%% | SL@%.1f%% SG@+%.1f%%",
        raw_price, smoothed_price, entry_price, max_price,
        drawdown_now * 100, dist_from_max * 100,
        -sl_pct * 100, sg_pct * 100,
    )

    # ── 10. Persist updated tracking (max_price, last_monitor_price) ──────────
    if not dry_run:
        _write_portfolio_safe(PORTFOLIO_PATH, portfolio)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Specialist 4h Monitor — 1h stop loss/gain/trailing check"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute checks but do not write state files",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN — no state writes.")

    run_monitor(dry_run=args.dry_run)
