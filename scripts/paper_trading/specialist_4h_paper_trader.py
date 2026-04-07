"""
Specialist 4h Paper Trader — LightGBM SET_B inference at 4h cadence.

Model : SET_B_split_4_recent.pkl  (técnico 4h + R11 regime context)
Features (9):
  Group A (spot tech): returns_4h, volatility_24h, volume_zscore,
                       buy_pressure, price_range_4h
  Group B (regime):    r11_prob_bull, r11_entropy, regime_age_log,
                       stress_score

Usage:
    # First-time setup:
    python scripts/paper_trading/specialist_4h_paper_trader.py --init

    # Regular run (cron every 4h):
    python scripts/paper_trading/specialist_4h_paper_trader.py

    # Status only (no state write):
    python scripts/paper_trading/specialist_4h_paper_trader.py --status

Flow:
    1. Fetch latest 4h BTCUSDT candles (Binance public API)
    2. Update 4h OHLCV buffer → state/specialist_4h/ohlcv_4h_buffer.parquet
    3. Compute Group A spot features (same windows as L3 primary.spot_4h)
    4. Compute Group B regime features via R11 signal generator + L4 forward-fill
    5. Load SET_B model from config["model_path"]
    6. predict_proba(X) → allocation_raw  (class 1 probability)
    7. apply_control_layer(allocation_raw, context, config) → (allocation_final, gate_log)
       Phase 1: stop_loss — functional
       Phase 2: macro_gate — stub (enabled: false)
       Phase 3: derivatives_gate — stub (enabled: false)
    8. Compute target_exposure; execute simulated trade if delta > threshold
    9. Update entry tracking fields in portfolio.json
   10. Append row (with gate_log columns) to signals.csv; update portfolio.json

L5 regime-shift contract (mandatory):
    Regime of day D is applied to candles starting D+1 00:00 UTC.
    Implementation: exclude daily candles with open_time ≥ today UTC midnight.

Portfolio simulation:
    Fully local — no Binance Testnet. BTC price fetched from public API for
    mark-to-market. Fees simulated as abs(delta_notional) × fee_rate.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.util
import json
import logging
import pickle
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
_PT_DIR      = Path(__file__).parent
STATE_DIR    = _PT_DIR / "state" / "specialist_4h"
CONFIG_JS    = STATE_DIR / "config.json"
PORTFOLIO_JS = STATE_DIR / "portfolio.json"
SIGNALS_CSV  = STATE_DIR / "signals.csv"
BUFFER_4H    = STATE_DIR / "ohlcv_4h_buffer.parquet"
OHLCV_1H     = ROOT / "data" / "01_raw" / "spot" / "crypto" / "1h" / "BTCUSDT_1h.parquet"

# ── Shared execution module ───────────────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.execution import (
    PENDING_PATH as PENDING_SIGNAL_PATH,
    SIGNALS_FIELDS as _SHARED_SIGNALS_FIELDS,
    cancel_pending_signal as _cancel_pending_signal_shared,
    execute_buy   as _shared_execute_buy,
    execute_sell  as _shared_execute_sell,
    is_pending_expired,
    write_pending_signal as _write_pending_signal_shared,
    write_portfolio       as _shared_write_portfolio,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("specialist_4h")

# ── Binance public API ────────────────────────────────────────────────────────
BINANCE_API    = "https://api.binance.com"
SYMBOL         = "BTCUSDT"
SENTIMENT_PATH = ROOT / "data/01_raw/news/cryptocompare/sentiment_metrics.json"

# Canonical signals fields — imported from shared.execution (single definition)
_SIGNALS_FIELDS = _SHARED_SIGNALS_FIELDS

# Legacy columns present in signals.csv files written before the control layer.
_LEGACY_FIELDS = [
    "timestamp", "candle_close", "price_close",
    "allocation_raw", "allocation_final",
    "r11_prob_bull", "r11_entropy", "regime_age_log", "stress_score",
    "returns_4h", "volatility_24h", "volume_zscore", "buy_pressure", "price_range_4h",
    "position_btc", "position_usdt", "portfolio_value",
    "action", "delta_allocation", "fee_paid",
]


# ══════════════════════════════════════════════════════════════════════════════
# Config helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_config() -> dict:
    if not CONFIG_JS.exists():
        raise FileNotFoundError(
            f"Config not found: {CONFIG_JS}\n"
            "  Run: python scripts/paper_trading/specialist_4h_paper_trader.py --init"
        )
    return json.loads(CONFIG_JS.read_text())


# ── Thin wrappers — delegate to shared.execution ─────────────────────────────
def _save_pending_signal(data: dict) -> None:
    _write_pending_signal_shared(data)

def _cancel_pending_signal() -> None:
    _cancel_pending_signal_shared("specialist_no_entry")


# ══════════════════════════════════════════════════════════════════════════════
# Binance public API helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_4h_candles(limit: int = 205) -> pd.DataFrame:
    """Fetch most recent `limit` 4h BTCUSDT candles from Binance public API."""
    url    = f"{BINANCE_API}/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": "4h", "limit": limit}
    resp   = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    records = []
    for k in resp.json():
        records.append({
            "open_time":              pd.Timestamp(k[0], unit="ms", tz="UTC"),
            "open":                   float(k[1]),
            "high":                   float(k[2]),
            "low":                    float(k[3]),
            "close":                  float(k[4]),
            "volume":                 float(k[5]),
            "quote_volume":           float(k[7]),
            "taker_buy_base_volume":  float(k[9]),
            "taker_buy_quote_volume": float(k[10]),
        })
    df = pd.DataFrame(records).set_index("open_time").sort_index()
    return df


def _fetch_1d_candles(limit: int = 205) -> pd.DataFrame:
    """Fetch most recent `limit` daily BTCUSDT candles for R11 feature computation."""
    url    = f"{BINANCE_API}/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": "1d", "limit": limit}
    resp   = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    records = []
    for k in resp.json():
        records.append({
            "open_time": pd.Timestamp(k[0], unit="ms", tz="UTC"),
            "open":   float(k[1]),
            "high":   float(k[2]),
            "low":    float(k[3]),
            "close":  float(k[4]),
            "volume": float(k[5]),
        })
    return pd.DataFrame(records).set_index("open_time").sort_index()


def _fetch_btc_price() -> float:
    resp = requests.get(f"{BINANCE_API}/api/v3/ticker/price",
                        params={"symbol": SYMBOL}, timeout=10)
    resp.raise_for_status()
    return float(resp.json()["price"])


# ══════════════════════════════════════════════════════════════════════════════
# 4h OHLCV buffer management
# ══════════════════════════════════════════════════════════════════════════════

def _load_4h_buffer() -> pd.DataFrame:
    if BUFFER_4H.exists():
        df = pd.read_parquet(BUFFER_4H)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    return pd.DataFrame()


def _save_4h_buffer(df: pd.DataFrame) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BUFFER_4H)


def update_4h_buffer(cfg: dict) -> pd.DataFrame:
    """Fetch fresh 4h candles, merge with buffer, return updated buffer."""
    limit  = int(cfg.get("buffer_4h_candles", 200)) + 5
    fresh  = _fetch_4h_candles(limit=limit)
    stored = _load_4h_buffer()
    if stored.empty:
        combined = fresh
    else:
        combined = pd.concat([stored, fresh])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index().tail(limit)
    _save_4h_buffer(combined)
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# Group A — spot 4h feature computation (matches L3 primary.spot_4h exactly)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_group_a(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 5 spot 4h features identical to L3 primary.spot_4h.nodes.py.

    returns_4h     = close.pct_change()
    volatility_24h = returns_4h.rolling(6).std()
    volume_zscore  = (volume - volume.rolling(30).mean()) / volume.rolling(30).std()
    buy_pressure   = taker_buy_quote_volume / quote_volume, clipped [0, 1]
    price_range_4h = (high - low) / close
    """
    out = df.copy()

    out["returns_4h"]     = out["close"].pct_change()
    out["volatility_24h"] = out["returns_4h"].rolling(6).std()

    vol_mean = out["volume"].rolling(30).mean()
    vol_std  = out["volume"].rolling(30).std().replace(0.0, np.nan)
    out["volume_zscore"]  = (out["volume"] - vol_mean) / vol_std

    qv = out["quote_volume"].replace(0.0, np.nan)
    out["buy_pressure"]   = (out["taker_buy_quote_volume"] / qv).clip(0.0, 1.0)

    close_safe = out["close"].replace(0.0, np.nan)
    out["price_range_4h"] = (out["high"] - out["low"]) / close_safe

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Timing indicators (for timing gate)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_timing_indicators(ohlcv_buffer: pd.DataFrame) -> dict:
    """
    Calculate RSI, Bollinger %B and volume_zscore for timing gate.
    Uses the 4h OHLCV buffer already maintained by the paper trader.
    Returns values for the last row. NaN → safe fallback (blocks entry).
    """
    close  = ohlcv_buffer["close"]
    volume = ohlcv_buffer["volume"]

    # RSI 14 periods
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = (100 - 100 / (1 + rs)).iloc[-1]

    # Bollinger %B (20 periods, 2 std)
    bb_mid   = close.rolling(20).mean().iloc[-1]
    bb_std   = close.rolling(20).std().iloc[-1]
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pct_b = (
        (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower)
        if bb_upper != bb_lower else 0.5
    )

    # Volume z-score (50 periods)
    vol_mean = volume.rolling(50).mean().iloc[-1]
    vol_std  = volume.rolling(50).std().iloc[-1]
    volume_z = (volume.iloc[-1] - vol_mean) / vol_std if vol_std and vol_std > 0 else 0.0

    return {
        "rsi_4h":               float(rsi)     if not pd.isna(rsi)     else float("nan"),
        "bb_pct_b":             float(bb_pct_b) if not pd.isna(bb_pct_b) else float("nan"),
        "volume_zscore_timing": float(volume_z) if not pd.isna(volume_z) else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# R11 feature helpers (mirrors r11_paper_trader._compute_r11_features)
# ══════════════════════════════════════════════════════════════════════════════

def _slope_normalized(arr: np.ndarray) -> float:
    n = len(arr)
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, arr, 1)[0])
    last  = arr[-1]
    return slope / last if last != 0.0 else 0.0


def _compute_r11_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute R11 features from daily OHLCV. Matches r11_paper_trader exactly."""
    close  = df["close"]
    volume = df["volume"]

    log_ret  = np.log(close / close.shift(1))
    vol_s    = log_ret.rolling(7).std()
    vol_l    = log_ret.rolling(30).std()
    vol_r    = vol_s / vol_l.replace(0, np.nan)

    roll_max = close.rolling(90, min_periods=1).max()
    drawdown = close / roll_max - 1.0

    vol_mean = volume.rolling(30).mean()
    vol_std  = volume.rolling(30).std()
    volume_z = (volume - vol_mean) / vol_std.replace(0, np.nan)

    slope_21d = close.rolling(21).apply(_slope_normalized, raw=True)

    out = df.copy()
    out["log_return"] = log_ret
    out["vol_short"]  = vol_s
    out["vol_ratio"]  = vol_r
    out["drawdown"]   = drawdown
    out["volume_z"]   = volume_z
    out["slope_21d"]  = slope_21d
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Group B — regime context feature computation
# ══════════════════════════════════════════════════════════════════════════════

def _load_r5c_signal_generator(cfg: dict):
    """Import R5CSignalGenerator from r5c_signal_generator.py via local import."""
    r5c_model_path  = ROOT / cfg["r5c_model_path"]
    sig_module_path = _PT_DIR / "r5c_signal_generator.py"
    spec = importlib.util.spec_from_file_location("r5c_signal_generator", sig_module_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["r5c_signal_generator"] = mod
    spec.loader.exec_module(mod)
    return mod.R5CSignalGenerator(model_path=r5c_model_path)


def _load_r11_signal_generator(cfg: dict):
    """Import SignalGenerator from r11_signal_generator.py via local import. (backup)"""
    r11_model_path  = ROOT / cfg["r11_model_path"]
    sig_module_path = _PT_DIR / "r11_signal_generator.py"
    spec = importlib.util.spec_from_file_location("r11_signal_generator", sig_module_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["r11_signal_generator"] = mod
    spec.loader.exec_module(mod)
    return mod.SignalGenerator(model_path=r11_model_path)


def _compute_group_b(
    cfg:          dict,
    portfolio:    dict,
    candle_close: pd.Timestamp,
) -> dict:
    """
    Compute Group B regime features for the current 4h candle.

    Layer 1: R5C HMM (3 states — Bull/Sideways/Bear).

    Opção A (sem retreino): os nomes das features enviadas ao LightGBM mantêm-se
    como r11_prob_bull, r11_entropy, regime_age_log, stress_score — mas os valores
    vêm do R5C. Isso preserva compatibilidade com o modelo SET_B treinado.

      r11_prob_bull  ← r5c_prob_bull  (ambos medem "bullishness")
      r11_entropy    ← r5c_entropy    (fórmula com 3 estados, conceito similar)
      regime_age_log ← mesma fórmula

    Retorna também _r5c_regime, _r5c_prob_bull, _r5c_prob_bear, _r5c_prob_sideways
    para portfolio state e display.

    L5 day-shift contract: regime of day D applied to 4h candles starting D+1.
    """
    limit    = int(cfg.get("r11_buffer_days", 200)) + 5
    daily_df = _fetch_1d_candles(limit=limit)

    gen = _load_r5c_signal_generator(cfg)

    today_utc = candle_close.normalize()
    # Day-shift: exclude today's candle (not closed yet)
    daily_filtered = daily_df[daily_df.index < today_utc]

    if len(daily_filtered) < 30:
        raise ValueError("[specialist_4h] Not enough daily rows for R5C (day-shift filter).")

    signal = gen.predict(daily_filtered, proba_window=gen.proba_window)

    r5c_prob_bull     = float(signal["prob_bull"])
    r5c_prob_bear     = float(signal["prob_bear"])
    r5c_prob_sideways = float(signal["prob_sideways"])
    r5c_entropy       = float(signal["entropy"])
    current_regime    = signal["regime"]   # "Bull" | "Sideways" | "Bear"

    # regime_age: track consecutive days in current regime
    prev_regime     = portfolio.get("r5c_regime", portfolio.get("r11_regime", current_regime))
    regime_age_days = int(portfolio.get("regime_age_days", 0))
    regime_age_days = 1 if current_regime != prev_regime else regime_age_days + 1
    regime_age_log  = float(np.log1p(regime_age_days))

    # stress_score: forward-filled from L4 (unchanged)
    l4_path      = ROOT / cfg["l4_regime_path"]
    stress_score = float(portfolio.get("last_stress_score", 0.0))
    if l4_path.exists():
        try:
            l4 = pd.read_parquet(l4_path)
            if l4.index.tz is None:
                l4.index = l4.index.tz_localize("UTC")
            l4_valid = l4[l4.index < today_utc]
            if not l4_valid.empty and "stress_score" in l4_valid.columns:
                stress_score = float(l4_valid["stress_score"].dropna().iloc[-1])
        except Exception as exc:
            log.warning("[Group B] stress_score L4 read failed: %s — using last known", exc)

    return {
        # Opção A: feature names for LightGBM model (unchanged from training)
        "r11_prob_bull":    r5c_prob_bull,      # r5c_prob_bull mapped → r11_prob_bull
        "r11_entropy":      r5c_entropy,         # r5c_entropy mapped → r11_entropy
        "regime_age_log":   regime_age_log,
        "stress_score":     stress_score,
        # Private: R5C full context (portfolio state + display)
        "_r5c_regime":         current_regime,
        "_r5c_prob_bull":      r5c_prob_bull,
        "_r5c_prob_bear":      r5c_prob_bear,
        "_r5c_prob_sideways":  r5c_prob_sideways,
        "_regime_age_days":    regime_age_days,
        "_stress_score":       stress_score,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Model inference
# ══════════════════════════════════════════════════════════════════════════════

def _load_model(cfg: dict):
    model_path = ROOT / cfg["model_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"[specialist_4h] Model not found: {model_path}")
    with open(model_path, "rb") as fh:
        return pickle.load(fh)


def _predict_allocation(model, features_row: dict, feature_names: list[str]) -> float:
    X = pd.DataFrame([{f: features_row[f] for f in feature_names}])
    return float(model.predict_proba(X)[0][1])


# ══════════════════════════════════════════════════════════════════════════════
# Control layer — Phase 1 active, Phase 2-3 stubs
# ══════════════════════════════════════════════════════════════════════════════

def _apply_stop_loss(
    allocation: float,
    context:    dict,
    cfg:        dict,
) -> tuple[float, bool]:
    """
    Stop loss fixo: if position open and drawdown from entry > pct, force exit.
    Post-stop cooldown: block re-entry for cooldown_candles × 4h.

    Returns (allocation, triggered).
    Gates NEVER increase allocation — only reduce or zero.
    """
    entry_price    = context.get("entry_price")
    current_price  = context.get("current_price")
    cooldown_until = context.get("stop_loss_cooldown_until")   # pd.Timestamp or None
    current_time   = context.get("current_time")               # pd.Timestamp candle open

    stop_pct = float(cfg.get("pct", 0.03))

    # ── Cooldown check: still in cooldown → force cash ────────────────────────
    if cooldown_until is not None and current_time is not None:
        if current_time < cooldown_until:
            remaining = int((cooldown_until - current_time).total_seconds() / 3600)
            log.info("[stop_loss] Cooldown active — %dh remaining. Forcing cash.", remaining)
            return 0.0, False

    # ── Stop loss check: only when in position ────────────────────────────────
    if entry_price and current_price and float(entry_price) > 0:
        drawdown = (float(current_price) - float(entry_price)) / float(entry_price)
        if drawdown < -stop_pct:
            log.info(
                "[stop_loss] TRIGGERED — entry=%.2f  current=%.2f  drawdown=%.2f%%  threshold=%.1f%%",
                entry_price, current_price, drawdown * 100, stop_pct * 100,
            )
            return 0.0, True

    return allocation, False


def _apply_stop_gain(
    allocation: float,
    context:    dict,
    cfg:        dict,
) -> tuple[float, bool]:
    """
    Stop gain: if position open and profit from entry >= pct, force exit (TAKE_PROFIT).
    Unlike stop loss, stop gain does NOT trigger cooldown — can re-enter immediately.

    Returns (allocation, triggered).
    """
    entry_price   = context.get("entry_price")
    current_price = context.get("current_price")
    gain_pct      = float(cfg.get("pct", 0.015))

    if entry_price and current_price and float(entry_price) > 0:
        profit = (float(current_price) - float(entry_price)) / float(entry_price)
        if profit >= gain_pct:
            log.info(
                "[stop_gain] TRIGGERED — entry=%.2f  current=%.2f  profit=%.2f%%  target=%.1f%%",
                float(entry_price), float(current_price), profit * 100, gain_pct * 100,
            )
            return 0.0, True

    return allocation, False


def _apply_regime_gate(
    allocation: float,
    context:    dict,
    cfg:        dict,
) -> tuple[float, bool, str]:
    """
    Regime gate: block NEW entries when R11 indicates Bear/uncertain regime.

    Logic (3-state R5C):
    - Bear regime OR prob_bear > max_prob_bear   → BLOCK (return 0)
    - Sideways regime                            → allow full allocation (sizing reduced downstream)
    - Bull regime                                → allow full
    - entropy > max_entropy                      → BLOCK (no conviction)
    - NaN in any indicator                       → BLOCK (safety)

    Returns (allocation, triggered, regime_label).
    Does NOT affect open positions — let stop loss/gain handle exits.
    """
    prob_bull     = context.get("r11_prob_bull", float("nan"))   # r5c_prob_bull mapped
    r5c_regime    = context.get("_r5c_regime", None)
    r5c_prob_bear = context.get("_r5c_prob_bear", float("nan"))
    entropy       = context.get("r11_entropy",   float("nan"))

    min_prob_bull       = float(cfg.get("min_prob_bull",           0.30))
    max_prob_bear       = float(cfg.get("max_prob_bear",           0.60))
    sideways_factor     = float(cfg.get("sideways_allocation_factor", 0.5))  # noqa: F841 — used in sizing
    max_entropy         = float(cfg.get("max_entropy",             0.85))

    # If already in position, do not interfere
    if float(context.get("position_btc", 0.0)) > 1e-6:
        return allocation, False, ""

    # NaN safety
    if pd.isna(prob_bull) or pd.isna(entropy):
        log.warning("[regime_gate] NaN in regime indicators — blocking entry.")
        return 0.0, True, ""

    # Entropy check (no conviction)
    if entropy > max_entropy:
        log.info("[regime_gate] TRIGGERED — high entropy=%.3f (max=%.2f)", entropy, max_entropy)
        return 0.0, True, ""

    # 3-state logic
    if r5c_regime == "Bear" or (not pd.isna(r5c_prob_bear) and r5c_prob_bear > max_prob_bear):
        log.info(
            "[regime_gate] TRIGGERED Bear — regime=%s prob_bear=%.3f (max=%.2f)",
            r5c_regime, r5c_prob_bear if not pd.isna(r5c_prob_bear) else 0.0, max_prob_bear,
        )
        return 0.0, True, "bear"

    if r5c_regime == "Sideways":
        # Sideways: passa allocation integral ao specialist
        # O fator de redução é aplicado no POSITION SIZING, não na decisão
        # Isso permite que o modelo decida (threshold) e o gate ajuste o risco
        log.info(
            "[regime_gate] Sideways — passing allocation %.3f intact (sizing reduction applied downstream)",
            allocation,
        )
        return allocation, False, "sideways"

    # Bull (or r5c_regime unknown but prob_bull passes threshold)
    if prob_bull < min_prob_bull:
        log.info(
            "[regime_gate] TRIGGERED — prob_bull=%.3f < min=%.2f", prob_bull, min_prob_bull,
        )
        return 0.0, True, ""

    return allocation, False, "bull"


def _apply_timing_gate(
    allocation: float,
    context:    dict,
    cfg:        dict,
) -> tuple[float, float]:
    """
    Timing gate: block entry when technical indicators signal a local top.

    Score (0–1):
    - Low RSI   = good (oversold, room to rise)
    - Low BB %B = good (near support)
    - High vol  = good (movement confirmation)

    score < min_score → block entry.
    Does NOT affect open positions.
    NaN in any timing indicator → block by safety (score = 0.0).
    """
    # If already in position, do not interfere
    if float(context.get("position_btc", 0.0)) > 1e-6:
        return allocation, 1.0

    # If allocation already zeroed (regime gate blocked), skip
    if allocation <= 0:
        return 0.0, 0.0

    rsi   = context.get("rsi_4h",               float("nan"))
    bb    = context.get("bb_pct_b",              float("nan"))
    vol_z = context.get("volume_zscore_timing",  0.0)

    # NaN safety: block
    if pd.isna(rsi) or pd.isna(bb):
        log.warning("[timing_gate] NaN in timing indicators — blocking entry.")
        return 0.0, 0.0

    weights   = cfg.get("weights", {"rsi": 0.4, "bollinger": 0.3, "volume": 0.3})
    rsi_t     = float(cfg.get("rsi_threshold", 40))
    bb_t      = float(cfg.get("bb_threshold",  0.30))
    min_score = float(cfg.get("min_score",     0.30))

    rsi_component = max(0.0, (rsi_t - rsi) / rsi_t)
    bb_component  = max(0.0, (bb_t  - bb)  / bb_t)
    vol_component = max(0.0, min(vol_z / 2.0, 1.0))

    timing_score = (
        weights["rsi"]       * rsi_component +
        weights["bollinger"] * bb_component  +
        weights["volume"]    * vol_component
    )

    if timing_score < min_score:
        log.info(
            "[timing_gate] TRIGGERED — score=%.3f < min=%.2f  "
            "(rsi=%.1f  bb=%.3f  vol_z=%.2f)",
            timing_score, min_score, rsi, bb, vol_z,
        )
        return 0.0, float(timing_score)

    return allocation, float(timing_score)


def _apply_news_gate(
    allocation: float,
    context:    dict,
    cfg:        dict,
) -> tuple[float, bool]:
    """
    News gate: block NEW entries when news sentiment is bearish.

    Rules:
    - position_btc > 0       → skip (do not interfere with open positions)
    - allocation <= 0        → skip (already blocked upstream)
    - File not found / stale > max_stale_hours → fail open (allow entry)
    - high_stress = True     → block
    - sentiment_4h < min_sentiment_4h → block

    Returns (allocation, triggered).
    """
    # Do not interfere with open positions
    if float(context.get("position_btc", 0.0)) > 1e-6:
        return allocation, False

    # Already zeroed by a previous gate — nothing to do
    if allocation <= 0:
        return 0.0, False

    max_stale_h   = float(cfg.get("max_stale_hours",   2))
    min_sentiment = float(cfg.get("min_sentiment_4h", -0.30))

    # Fail open: file not found
    if not SENTIMENT_PATH.exists():
        log.warning("[news_gate] sentiment_metrics.json not found — fail open.")
        return allocation, False

    try:
        data = json.loads(SENTIMENT_PATH.read_text())
    except Exception as exc:
        log.warning("[news_gate] Read error: %s — fail open.", exc)
        return allocation, False

    # Stale check
    updated_at_str = data.get("impact_updated_at", data.get("updated_at", ""))
    if updated_at_str:
        try:
            updated_at = datetime.fromisoformat(updated_at_str)
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            age_h = (datetime.now(timezone.utc) - updated_at).total_seconds() / 3600
            if age_h > max_stale_h:
                log.warning(
                    "[news_gate] Stale data (%.1fh > %.1fh) — fail open.", age_h, max_stale_h
                )
                return allocation, False
        except Exception:
            pass

    # ── Nova estrutura: combined_news.4h.regime (Bull/Sideways/Bear) ──────────
    #combined_news_4h = data.get("combined_news", {}).get("4h", {})
    combined_news_4h = data.get("combined_news", data.get("combined", {})).get("4h", {})
    news_regime      = combined_news_4h.get("regime", None)
    news_score       = float(combined_news_4h.get("score", 0.0))

    if news_regime is not None:
        if news_regime == "BEAR" and news_score < -3:
            log.info("[news_gate] TRIGGERED — news BEAR forte: score=%.2f", news_score)
            return 0.0, True

        if news_regime == "BEAR":
            reduced = allocation * 0.5
            log.info("[news_gate] news BEAR moderado: score=%.2f → alloc × 0.5 = %.3f", news_score, reduced)
            return reduced, False

        if news_regime == "BULL" and news_score > 3:
            boosted = min(allocation * 1.15, 1.0)
            log.info("[news_gate] news BULL boost: score=%.2f → alloc=%.3f", news_score, boosted)
            return boosted, False

        # SIDEWAYS → pass through
        log.info("[news_gate] news regime=%s score=%.2f — pass through", news_regime, news_score)
        return allocation, False

    # ── Fallback: estrutura antiga (combined.4h.combined_score) ──────────────
    if data.get("high_stress", False):
        log.info("[news_gate] TRIGGERED — high_stress=True")
        return 0.0, True

    combined_4h    = data.get("combined", {}).get("4h", {})
    combined_score = float(combined_4h.get("combined_score", 0.0)) if combined_4h else 0.0

    if combined_score < -0.30:
        log.info("[news_gate] TRIGGERED (fallback) — combined_score=%.3f", combined_score)
        return 0.0, True

    macro_4h     = data.get("macro", {}).get("4h", {})
    macro_stress = float(macro_4h.get("macro_score", 0.0)) if macro_4h else 0.0

    if macro_stress < -0.50:
        log.info("[news_gate] TRIGGERED (fallback) — macro_stress=%.3f", macro_stress)
        return 0.0, True

    return allocation, False


def _compute_technical_gate_features() -> dict:
    """Calcular RSI 14 e BB %B 20 a partir dos candles 1h mais recentes."""
    try:
        df = pd.read_parquet(OHLCV_1H)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        close = df["close"].iloc[-50:]

        # RSI 14
        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs    = gain / loss
        rsi_14 = float((100 - (100 / (1 + rs))).iloc[-1])

        # Bollinger Bands %B (20 períodos)
        sma20   = close.rolling(20).mean()
        std20   = close.rolling(20).std()
        upper   = sma20 + 2 * std20
        lower   = sma20 - 2 * std20
        bb_pct_b = float(
            (close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        )

        return {"rsi_14": rsi_14, "bb_pct_b": bb_pct_b, "valid": True}
    except Exception as exc:
        log.warning("[tech_gate] Failed to compute features: %s — fail open", exc)
        return {"rsi_14": 50.0, "bb_pct_b": 0.50, "valid": False}


def _apply_technical_gate(
    allocation: float,
    context:    dict,
    cfg:        dict,
) -> tuple[float, bool, str, float, float]:
    """
    Gate técnico: RSI + BB %B combinados.
    Bloqueia entradas no topo do range, reduz em zona alta,
    boost em oversold. Não interfere em posições abertas.

    Retorna (allocation_ajustada, triggered, reason, rsi_14, bb_pct_b).
    """
    _nan = float("nan")

    # Do not interfere with open positions
    if float(context.get("position_btc", 0.0)) > 1e-6:
        return allocation, False, "in_position", _nan, _nan

    # Already zeroed upstream — skip
    if allocation <= 0:
        return 0.0, False, "upstream_blocked", _nan, _nan

    tg_cfg = cfg.get("control_layer", {}).get("technical_gate", {})
    if not tg_cfg.get("enabled", True):
        return allocation, False, "disabled", _nan, _nan

    features = _compute_technical_gate_features()
    rsi = features["rsi_14"]
    bb  = features["bb_pct_b"]

    rsi_block     = float(tg_cfg.get("rsi_block",          65))
    rsi_reduce    = float(tg_cfg.get("rsi_reduce",         50))
    rsi_boost     = float(tg_cfg.get("rsi_boost",          35))
    rsi_strong    = float(tg_cfg.get("rsi_strong_boost",   30))
    bb_block      = float(tg_cfg.get("bb_block",           0.75))
    bb_reduce     = float(tg_cfg.get("bb_reduce",          0.50))
    bb_boost      = float(tg_cfg.get("bb_boost",           0.25))
    bb_strong     = float(tg_cfg.get("bb_strong_boost",    0.15))
    reduce_factor = float(tg_cfg.get("reduce_factor",      0.7))
    boost_factor  = float(tg_cfg.get("boost_factor",       1.2))
    strong_factor = float(tg_cfg.get("strong_boost_factor",1.3))

    log.info(
        "[tech_gate] RSI=%.1f BB=%.2f (block: RSI>%.0f or BB>%.2f)",
        rsi, bb, rsi_block, bb_block,
    )

    # BLOQUEIA: preço esticado
    if rsi > rsi_block or bb > bb_block:
        log.info(
            "[tech_gate] BLOCKED — preço esticado (RSI=%.1f>%.0f or BB=%.2f>%.2f)",
            rsi, rsi_block, bb, bb_block,
        )
        return 0.0, True, f"overbought_rsi{rsi:.0f}_bb{bb:.2f}", rsi, bb

    # FORTE BOOST: fortemente oversold (ambas as condições)
    if rsi < rsi_strong and bb < bb_strong:
        boosted = min(allocation * strong_factor, 1.0)
        log.info(
            "[tech_gate] STRONG BOOST — oversold (RSI=%.1f BB=%.2f) alloc %.3f → %.3f",
            rsi, bb, allocation, boosted,
        )
        return boosted, False, f"strong_oversold_rsi{rsi:.0f}_bb{bb:.2f}", rsi, bb

    # BOOST: oversold (ambas)
    if rsi < rsi_boost and bb < bb_boost:
        boosted = min(allocation * boost_factor, 1.0)
        log.info(
            "[tech_gate] BOOST — oversold (RSI=%.1f BB=%.2f) alloc %.3f → %.3f",
            rsi, bb, allocation, boosted,
        )
        return boosted, False, f"oversold_rsi{rsi:.0f}_bb{bb:.2f}", rsi, bb

    # REDUZ: zona neutra/alta (ambas)
    if rsi > rsi_reduce and bb > bb_reduce:
        reduced = allocation * reduce_factor
        log.info(
            "[tech_gate] REDUCED — zona alta (RSI=%.1f BB=%.2f) alloc %.3f → %.3f",
            rsi, bb, allocation, reduced,
        )
        return reduced, False, f"high_zone_rsi{rsi:.0f}_bb{bb:.2f}", rsi, bb

    # NEUTRO: passa
    log.info("[tech_gate] NEUTRAL — RSI=%.1f BB=%.2f — pass through", rsi, bb)
    return allocation, False, f"neutral_rsi{rsi:.0f}_bb{bb:.2f}", rsi, bb


def _apply_macro_gate(
    allocation: float,
    context:    dict,
    cfg:        dict,
) -> tuple[float, float]:
    """
    Macro gate: MOVE-based continuous gate (Phase 2 — stub).

    Formula (validated in daily specialist):
      stress_factor    = clip((MOVE - min) / (max - min), 0, 1)
      bearish_strength = clip(1 - allocation_r5c, 0, 1)
      multiplier       = clip(1 - stress_factor * bearish_strength * max_cut, 0, 1)

    TODO: implement live MOVE index read (Yahoo Finance / yfinance).
    """
    multiplier = 1.0  # pass-through
    return allocation * multiplier, multiplier


def _apply_derivatives_gate(
    allocation: float,
    context:    dict,
    cfg:        dict,
) -> tuple[float, float]:
    """
    Derivatives gate: funding/OI extreme filter (Phase 3 — stub).

    Logic (from feature importance analysis):
      funding_z and oi_price_div have signal strength but degrade model Sharpe
      as features → applied as post-model rule-based gate instead.

    Gate rule:
      if funding_z >= funding_z_threshold  AND  oi_price_div >= oi_div_threshold:
          multiplier = 1 - max_cut
      else:
          multiplier = 1.0

    TODO: implement live CoinGlass funding_z and oi_price_div read.
    """
    multiplier = 1.0  # pass-through
    return allocation * multiplier, multiplier


def _compute_entry_score(
    allocation: float,
    gate_log:   dict,
    cfg:        dict,
) -> dict:
    """
    Scoring system para decisão de entrada.
    Cada condição contribui com um peso. Score >= threshold → enter.
    Mais robusto que filtros AND (não precisa que todas as condições sejam perfeitas).

    Se entry_scoring.enabled=False, usa threshold fixo (fallback).
    Retorna: {"enter": bool, "score": float, "components": dict, "reason": str}
    """
    score_cfg = cfg.get("control_layer", {}).get("entry_scoring", {})
    if not score_cfg.get("enabled", True):
        threshold = float(cfg.get("control_layer", {}).get("entry_threshold", 0.38))
        return {
            "enter":      allocation > threshold,
            "score":      allocation,
            "components": {},
            "reason":     "threshold_fixed",
        }

    min_score = float(score_cfg.get("min_score", 2.5))
    components: dict = {}
    total = 0.0

    # ── 1. BB score (sinal dominante) ──
    bb = gate_log.get("tech_gate_bb", 0.50)
    import math as _math
    if _math.isnan(bb):
        bb = 0.50
    if bb > 0.80:
        bb_score = -2.0   # kill switch no topo (win 43%, ret negativo)
    elif bb < 0.20:
        bb_score = 3.0    # sinal forte (win 88%, ret_3d +1.75%)
    elif bb < 0.30:
        bb_score = 2.0    # sinal bom (win 77%, ret_3d +1.65%)
    elif bb < 0.40:
        bb_score = 0.5    # zona favorável mas fraca
    else:
        bb_score = 0.0
    components["bb"] = bb_score
    total += bb_score

    # ── 2. RSI score (complementar) ──
    rsi = gate_log.get("tech_gate_rsi", 50.0)
    if _math.isnan(rsi):
        rsi = 50.0
    if rsi < 35:
        rsi_score = 1.0
    elif rsi < 45:
        rsi_score = 0.5
    elif rsi > 60:
        rsi_score = -1.0
    else:
        rsi_score = 0.0
    components["rsi"] = rsi_score
    total += rsi_score

    # ── 3. Alloc score (zona intermediária do specialist) ──
    alloc_low      = float(score_cfg.get("alloc_low",      0.50))
    alloc_high     = float(score_cfg.get("alloc_high",     0.54))
    alloc_marginal = float(score_cfg.get("alloc_marginal", 0.48))
    if alloc_low <= allocation <= alloc_high:
        alloc_score = 0.5   # fraco — alloc não tem edge forte
    elif alloc_marginal <= allocation < alloc_low:
        alloc_score = 0.25
    else:
        alloc_score = 0.0
    components["alloc"] = alloc_score
    total += alloc_score

    # ── 4. News score ──
    try:
        import json as _json
        with open(SENTIMENT_PATH) as _f:
            _metrics = _json.load(_f)
        _combined     = _metrics.get("combined", {}).get("4h", {})
        news_regime   = _combined.get("regime", "SIDEWAYS")
        news_score_val = float(_combined.get("combined_score", 0))
    except Exception:
        news_regime    = "SIDEWAYS"
        news_score_val = 0.0

    if news_regime == "BULL" and news_score_val > 3:
        news_score = 1.0
    elif news_regime == "BULL":
        news_score = 0.5
    elif news_regime == "BEAR" and news_score_val < -3:
        news_score = -1.5
    elif news_regime == "BEAR":
        news_score = -0.5
    else:
        news_score = 0.0
    components["news"] = news_score
    total += news_score

    # ── Decisão ──
    enter = total >= min_score

    # Regime check (hard block)
    if gate_log.get("regime_gate_regime", "").lower() == "bear":
        enter = False
        total = -99.0
        components["regime_block"] = True

    reason_parts = [f"{k}={v:+.1f}" for k, v in components.items() if isinstance(v, (int, float))]
    reason = f"score={total:.1f} ({' '.join(reason_parts)}) {'ENTER' if enter else 'HOLD'}"

    log.info(
        "[entry_score] total=%.1f (bb=%.1f rsi=%.1f alloc=%.1f news=%.1f) min=%.1f → %s",
        total, bb_score, rsi_score, alloc_score, news_score, min_score,
        "ENTER" if enter else "HOLD",
    )

    return {
        "enter":      enter,
        "score":      total,
        "components": components,
        "reason":     reason,
    }


def apply_control_layer(
    allocation_raw: float,
    context:        dict,
    config:         dict,
) -> tuple[float, dict]:
    """
    Control layer: sequential gate pipeline.
    Gates reduce allocation — never increase it.

    Args:
        allocation_raw: model output (class 1 probability, 0–1)
        context:        dict with current_price, entry_price, current_time,
                        stop_loss_cooldown_until, and feature values
        config:         full config dict (reads config["control_layer"])

    Returns:
        allocation_final: float (0–1)
        gate_log:         dict with trigger info for signals.csv
    """
    allocation = allocation_raw
    gate_log   = {
        "stop_loss_triggered":    False,
        "stop_gain_triggered":    False,
        "regime_gate_triggered":  False,
        "regime_gate_regime":     "",
        "timing_context_score":   None,
        "timing_gate_active":     False,
        "news_gate_triggered":        False,
        "technical_gate_triggered":   False,
        "technical_gate_reason":      "",
        "tech_gate_rsi":              float("nan"),
        "tech_gate_bb":               float("nan"),
        "entry_score_total":          float("nan"),
        "entry_score_bb":             float("nan"),
        "entry_score_rsi":            float("nan"),
        "entry_score_alloc":          float("nan"),
        "entry_score_news":           float("nan"),
        "entry_score_reason":         "",
        "macro_gate_multiplier":      1.0,
        "deriv_gate_multiplier":      1.0,
    }
    cl = config.get("control_layer", {})

    # ── Stop Gain — check BEFORE stop loss (take profit beats protection) ─────
    sg_cfg = cl.get("stop_gain", {})
    if sg_cfg.get("enabled", False):
        allocation, triggered = _apply_stop_gain(allocation, context, sg_cfg)
        gate_log["stop_gain_triggered"] = triggered
        if triggered:
            return allocation, gate_log  # exit with profit — skip remaining gates

    # ── Phase 1: Stop Loss ────────────────────────────────────────────────────
    sl_cfg = cl.get("stop_loss", {})
    if sl_cfg.get("enabled", False):
        allocation, triggered = _apply_stop_loss(allocation, context, sl_cfg)
        gate_log["stop_loss_triggered"] = triggered

    # ── Phase 2: Regime Gate (hard filter — Bear / uncertain regime) ──────────
    rg_cfg = cl.get("regime_gate", {})
    if rg_cfg.get("enabled", False):
        allocation, triggered, rg_regime = _apply_regime_gate(allocation, context, rg_cfg)
        gate_log["regime_gate_triggered"] = triggered
        gate_log["regime_gate_regime"]    = rg_regime

    # ── Phase 3: Timing Gate — desativado no specialist (executado apenas no monitor 1h)
    timing_score = None  # noqa: F841  (preservado para referência; avaliado no monitor 1h)
    gate_log["timing_context_score"] = None
    gate_log["timing_gate_active"]   = False

    # ── Phase 4: News Gate ────────────────────────────────────────────────────
    ng_cfg = cl.get("news_gate", {})
    if ng_cfg.get("enabled", False):
        allocation, triggered = _apply_news_gate(allocation, context, ng_cfg)
        gate_log["news_gate_triggered"] = triggered

    # ── Phase 5: Technical Gate (RSI + BB) ───────────────────────────────────
    allocation, tg_triggered, tg_reason, tg_rsi, tg_bb = _apply_technical_gate(
        allocation, context, cl
    )
    gate_log["technical_gate_triggered"] = tg_triggered
    gate_log["technical_gate_reason"]    = tg_reason
    gate_log["tech_gate_rsi"]            = tg_rsi
    gate_log["tech_gate_bb"]             = tg_bb

    # ── Phase 6: Macro Gate (stub) ────────────────────────────────────────────
    mg_cfg = cl.get("macro_gate", {})
    if mg_cfg.get("enabled", False):
        allocation, mult = _apply_macro_gate(allocation, context, mg_cfg)
        gate_log["macro_gate_multiplier"] = mult

    # ── Phase 7: Derivatives Gate (stub) ─────────────────────────────────────
    dg_cfg = cl.get("derivatives_gate", {})
    if dg_cfg.get("enabled", False):
        allocation, mult = _apply_derivatives_gate(allocation, context, dg_cfg)
        gate_log["deriv_gate_multiplier"] = mult

    return allocation, gate_log


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio state management
# ══════════════════════════════════════════════════════════════════════════════

def _load_portfolio(cfg: dict) -> dict:
    if PORTFOLIO_JS.exists():
        return json.loads(PORTFOLIO_JS.read_text())
    initial = float(cfg["initial_capital"])
    return {
        "usdt_free":               initial,
        "btc_held":                0.0,
        "total_usdt":              initial,
        "btc_price":               0.0,
        "position_pct":            0.0,
        "r5c_regime":              "Bear",
        "r5c_prob_bull":           0.0,
        "r5c_prob_bear":           1.0,
        "r5c_prob_sideways":       0.0,
        "regime_age_days":         0,
        "last_stress_score":       0.0,
        "last_update":             "",
        "entry_price":             None,
        "entry_time":              None,
        "max_price_since_entry":   None,
        "stop_loss_cooldown_until": None,
    }


def _save_portfolio(portfolio: dict) -> None:
    """Write portfolio.json — delegates to shared atomic writer."""
    _shared_write_portfolio(portfolio)


def _mark_to_market(portfolio: dict, btc_price: float) -> dict:
    p = portfolio.copy()
    p["btc_price"]    = btc_price
    p["total_usdt"]   = p["usdt_free"] + p["btc_held"] * btc_price
    p["position_pct"] = (p["btc_held"] * btc_price) / p["total_usdt"] \
                        if p["total_usdt"] > 0 else 0.0
    return p


# ══════════════════════════════════════════════════════════════════════════════
# Trade execution (simulated)
# ══════════════════════════════════════════════════════════════════════════════

def _execute_simulated_trade(
    portfolio:       dict,
    target_exposure: float,
    btc_price:       float,
    fee_rate:        float,
) -> tuple[dict, str, float, float]:
    """Simulate BTC trade to target_exposure. Returns (portfolio, action, delta, fee)."""
    total_usdt      = portfolio["total_usdt"]
    current_btc_val = portfolio["btc_held"] * btc_price
    target_btc_val  = target_exposure * total_usdt
    delta_usdt      = target_btc_val - current_btc_val
    fee_paid        = abs(delta_usdt) * fee_rate

    p = portfolio.copy()

    if delta_usdt > 0:
        usdt_spent = delta_usdt + fee_paid
        if usdt_spent > p["usdt_free"]:
            usdt_spent = p["usdt_free"]
            delta_usdt = usdt_spent / (1 + fee_rate)
            fee_paid   = usdt_spent - delta_usdt
        p["btc_held"]  += delta_usdt / btc_price
        p["usdt_free"] -= usdt_spent
        action = "BUY"
    elif delta_usdt < 0:
        btc_to_sell    = min(abs(delta_usdt) / btc_price, p["btc_held"])
        p["btc_held"]  -= btc_to_sell
        p["usdt_free"] += btc_to_sell * btc_price * (1 - fee_rate)
        fee_paid        = btc_to_sell * btc_price * fee_rate
        action = "SELL"
    else:
        action   = "HOLD"
        fee_paid = 0.0

    p["total_usdt"]   = p["usdt_free"] + p["btc_held"] * btc_price
    p["position_pct"] = (p["btc_held"] * btc_price) / p["total_usdt"] \
                        if p["total_usdt"] > 0 else 0.0

    delta_alloc = target_exposure - portfolio["position_pct"]
    return p, action, delta_alloc, fee_paid


# ══════════════════════════════════════════════════════════════════════════════
# Signals CSV — backward compatible migration
# ══════════════════════════════════════════════════════════════════════════════

def _migrate_signals_csv() -> None:
    """
    If signals.csv exists with legacy columns (no control layer columns),
    add the new columns with empty values and rewrite the file.
    Idempotent — safe to call on every run.
    """
    if not SIGNALS_CSV.exists() or SIGNALS_CSV.stat().st_size == 0:
        return
    try:
        df = pd.read_csv(SIGNALS_CSV)
    except Exception:
        return

    new_cols = [c for c in _SIGNALS_FIELDS if c not in df.columns]
    if not new_cols:
        return  # already up to date

    for c in new_cols:
        df[c] = ""

    df = df[[c for c in _SIGNALS_FIELDS if c in df.columns]]
    df.to_csv(SIGNALS_CSV, index=False)
    log.info("[migrate] signals.csv updated — added columns: %s", new_cols)


def _ensure_signals_csv() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _migrate_signals_csv()
    if not SIGNALS_CSV.exists() or SIGNALS_CSV.stat().st_size == 0:
        with open(SIGNALS_CSV, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=_SIGNALS_FIELDS).writeheader()


def _append_signal(row: dict) -> None:
    _ensure_signals_csv()
    with open(SIGNALS_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SIGNALS_FIELDS)
        writer.writerow({f: row.get(f, "") for f in _SIGNALS_FIELDS})


def _candle_already_logged(candle_ts: pd.Timestamp) -> bool:
    if not SIGNALS_CSV.exists() or SIGNALS_CSV.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(SIGNALS_CSV)
        if "candle_close" not in df.columns or df.empty:
            return False
        return any(pd.to_datetime(df["candle_close"], utc=True) == candle_ts)
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_4h(status_only: bool = False) -> None:
    cfg       = _load_config()
    fee_rate  = float(cfg["fee_rate"])
    min_delta = float(cfg["min_delta_allocation"])

    # ── 1. Update 4h buffer ──────────────────────────────────────────────────
    log.info("Fetching latest 4h BTCUSDT candles …")
    buf_4h = update_4h_buffer(cfg)
    log.info("4h buffer: %d rows  (%s → %s)",
             len(buf_4h), buf_4h.index[0].date(), buf_4h.index[-1].date())

    # ── 2. Compute Group A features ──────────────────────────────────────────
    buf_feat     = _compute_group_a(buf_4h)
    group_a_cols = ["returns_4h", "volatility_24h", "volume_zscore",
                    "buy_pressure", "price_range_4h"]
    last         = buf_feat.dropna(subset=group_a_cols).iloc[-1]
    candle_ts    = last.name
    candle_close = float(last["close"])

    # ── 3. Idempotency guard ─────────────────────────────────────────────────
    if not status_only and _candle_already_logged(candle_ts):
        log.info("Candle %s already processed — skipping.", candle_ts)
        return

    # ── 4. Load portfolio + BTC price ────────────────────────────────────────
    btc_price = _fetch_btc_price()
    portfolio = _load_portfolio(cfg)
    portfolio = _mark_to_market(portfolio, btc_price)

    # ── 5. Compute Group B regime features ───────────────────────────────────
    log.info("Computing Group B regime features …")
    group_b = _compute_group_b(cfg, portfolio, candle_ts)

    # ── 6. Assemble feature row ───────────────────────────────────────────────
    feature_names = cfg["set_b_features"]
    features_row  = {
        "returns_4h":     float(last["returns_4h"]),
        "volatility_24h": float(last["volatility_24h"]),
        "volume_zscore":  float(last["volume_zscore"]),
        "buy_pressure":   float(last["buy_pressure"]),
        "price_range_4h": float(last["price_range_4h"]),
        "r11_prob_bull":  group_b["r11_prob_bull"],
        "r11_entropy":    group_b["r11_entropy"],
        "regime_age_log": group_b["regime_age_log"],
        "stress_score":   group_b["stress_score"],
    }

    # ── 7. Model inference ───────────────────────────────────────────────────
    model          = _load_model(cfg)
    allocation_raw = _predict_allocation(model, features_row, feature_names)

    # ── 8. Control layer ─────────────────────────────────────────────────────
    entry_price_raw    = portfolio.get("entry_price")
    cooldown_until_str = portfolio.get("stop_loss_cooldown_until")
    cooldown_ts        = (pd.Timestamp(cooldown_until_str, tz="UTC")
                         if cooldown_until_str else None)

    timing_indicators = calculate_timing_indicators(buf_feat)

    context = {
        "candle_ts":                candle_ts,
        "current_price":            btc_price,
        "entry_price":              entry_price_raw,
        "current_time":             candle_ts,
        "stop_loss_cooldown_until": cooldown_ts,
        "position_btc":             float(portfolio.get("btc_held", 0.0)),
        # R5C regime context (used by 3-state regime gate)
        "_r5c_regime":      group_b["_r5c_regime"],
        "_r5c_prob_bear":   group_b["_r5c_prob_bear"],
        "_r5c_prob_bull":   group_b["_r5c_prob_bull"],
        "_r5c_prob_sideways": group_b["_r5c_prob_sideways"],
        **features_row,
        **timing_indicators,
    }

    allocation_final, gate_log = apply_control_layer(allocation_raw, context, cfg)

    # ── 9. Position sizing ───────────────────────────────────────────────────
    entry_threshold  = float(cfg.get("control_layer", {}).get("entry_threshold", 0.50))
    if allocation_final > entry_threshold:
        target_exposure = (allocation_final - entry_threshold) / (1.0 - entry_threshold)
        target_exposure = min(target_exposure, 1.0)
    else:
        target_exposure = 0.0

    # ── Sideways sizing reduction ──
    # Se regime Sideways, reduz exposição mas NÃO bloqueia entrada
    if gate_log.get("regime_gate_regime") == "sideways" and target_exposure > 0:
        _sw_factor  = float(cfg.get("control_layer", {})
                               .get("regime_gate", {})
                               .get("sideways_allocation_factor", 0.5))
        _original   = target_exposure
        target_exposure = target_exposure * _sw_factor
        log.info(
            "[sizing] Sideways factor %.1f applied: exposure %.1f%% → %.1f%%",
            _sw_factor, _original * 100, target_exposure * 100,
        )

    current_exposure = portfolio["position_pct"]
    delta            = abs(target_exposure - current_exposure)

    # ── 10. Execute simulated trade ──────────────────────────────────────────
    action      = "HOLD"
    delta_alloc = 0.0
    fee_paid    = 0.0
    was_long    = float(portfolio.get("btc_held", 0.0)) > 1e-6

    if not status_only:
        # ── Fail-safe: detect expired pending signal (timing layer may have failed) ──
        if not was_long and PENDING_SIGNAL_PATH.exists():
            from shared.execution import read_pending_signal as _read_ps
            _ps = _read_ps()
            if _ps.get("has_pending") and is_pending_expired(_ps):
                log.warning(
                    "[SPECIALIST] Pending signal EXPIRED — timing layer may have failed. "
                    "Cancelling. fallback_direct_execution=%s",
                    cfg.get("fallback_direct_execution", False),
                )
                _cancel_pending_signal()
                # Fallback: execute direct BUY if config enables it
                if cfg.get("fallback_direct_execution", False) and allocation_final > float(
                    cfg.get("control_layer", {}).get("entry_threshold", 0.38)
                ):
                    log.warning("[SPECIALIST] Executing direct BUY as fallback.")
                    _pre_fb = portfolio.copy()
                    portfolio, fee_paid = _shared_execute_buy(
                        btc_price, allocation_final, cfg,
                        portfolio=portfolio, entry_source="specialist_fallback",
                    )
                    delta_alloc = portfolio["position_pct"] - _pre_fb["position_pct"]
                    action = "BUY"

        if gate_log["stop_gain_triggered"]:
            action = "TAKE_PROFIT"
            if portfolio["btc_held"] > 1e-6:
                _pre = portfolio.copy()
                portfolio, fee_paid = _shared_execute_sell(
                    btc_price, portfolio["btc_held"], cfg,
                    reason="TAKE_PROFIT", portfolio=portfolio,
                )
                delta_alloc = portfolio["position_pct"] - _pre["position_pct"]
        elif gate_log["stop_loss_triggered"]:
            action = "STOP_LOSS"
            if portfolio["btc_held"] > 1e-6:
                _pre = portfolio.copy()
                portfolio, fee_paid = _shared_execute_sell(
                    btc_price, portfolio["btc_held"], cfg,
                    reason="STOP_LOSS", portfolio=portfolio,
                )
                delta_alloc = portfolio["position_pct"] - _pre["position_pct"]
        elif delta >= min_delta:
            if not was_long and target_exposure > 0:
                # ── Entry scoring: BB + RSI + alloc + news ───────────────────
                entry = _compute_entry_score(allocation_final, gate_log, cfg)
                gate_log["entry_score_total"] = entry["score"]
                gate_log["entry_score_bb"]    = entry["components"].get("bb",    float("nan"))
                gate_log["entry_score_rsi"]   = entry["components"].get("rsi",   float("nan"))
                gate_log["entry_score_alloc"] = entry["components"].get("alloc", float("nan"))
                gate_log["entry_score_news"]  = entry["components"].get("news",  float("nan"))
                gate_log["entry_score_reason"] = entry["reason"]
                log.info("[entry] %s", entry["reason"])

                if entry["enter"]:
                    # ── Score-based position sizing ──────────────────────────
                    _score = entry["score"]
                    if _score >= 3.5:
                        _size_mult = 1.2
                    elif _score >= 2.5:
                        _size_mult = 1.0
                    else:
                        _size_mult = 0.8
                    target_exposure = min(target_exposure * _size_mult, 1.0)
                    log.info(
                        "[sizing] score=%.1f → size_mult=%.1f  exposure=%.1f%%",
                        _score, _size_mult, target_exposure * 100,
                    )

                    # New entry → delegate to timing layer via pending_signal.json
                    try:
                        _nd = json.loads(SENTIMENT_PATH.read_text()) if SENTIMENT_PATH.exists() else {}
                        _news_sent = float(_nd.get("4h", {}).get("sentiment_score", 0.0))
                    except Exception:
                        _news_sent = None
                    _save_pending_signal({
                        "has_pending": True,
                        "allocation_raw": round(allocation_raw, 6),
                        "allocation_final": round(allocation_final, 6),
                        "specialist_candle": candle_ts.isoformat(),
                        "specialist_price": round(btc_price, 2),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "expires_at": (candle_ts + timedelta(hours=4)).isoformat(),
                        "r5c_prob_bull": round(group_b["_r5c_prob_bull"], 6),
                        "r5c_prob_bear": round(group_b["_r5c_prob_bear"], 6),
                        "r5c_prob_sideways": round(group_b["_r5c_prob_sideways"], 6),
                        "r5c_regime": group_b["_r5c_regime"],
                        # legacy aliases for backward compat with monitor/dashboard
                        "r11_prob_bull": round(group_b["_r5c_prob_bull"], 6),
                        "r11_regime": group_b["_r5c_regime"],
                        "news_sentiment_4h": _news_sent,
                        "regime_gate_passed": not gate_log["regime_gate_triggered"],
                        "news_gate_passed": not gate_log["news_gate_triggered"],
                    })
                    action = "PENDING"
                    log.info(
                        "[SPECIALIST] PENDING ENTRY: alloc=%.3f price=$%.2f expires=%s",
                        allocation_final, btc_price,
                        (candle_ts + timedelta(hours=4)).isoformat(),
                    )
                else:
                    log.info("[SPECIALIST] Entry filtered: %s", entry["reason"])
            else:
                # Exit (was_long, target=0) or position scaling (was_long, target>0)
                _pre = portfolio.copy()
                if target_exposure < current_exposure:
                    portfolio, fee_paid = _shared_execute_sell(
                        btc_price, portfolio["btc_held"], cfg,
                        reason="SELL", portfolio=portfolio,
                    )
                    action = "SELL"
                else:
                    portfolio, fee_paid = _shared_execute_buy(
                        btc_price, allocation_final, cfg,
                        portfolio=portfolio, entry_source="specialist",
                    )
                    action = "BUY"
                delta_alloc = portfolio["position_pct"] - _pre["position_pct"]
                log.info(
                    "[SPECIALIST] Trade: %s  target=%.1f%%  delta=%.1f%%  fee=$%.4f",
                    action, target_exposure * 100, delta * 100, fee_paid,
                )
        else:
            log.info(
                "[SPECIALIST] No trade — delta=%.1f%% < threshold=%.0f%%",
                delta * 100, min_delta * 100,
            )
            # Cancel stale pending signal when specialist no longer wants entry
            if not was_long and PENDING_SIGNAL_PATH.exists():
                from shared.execution import read_pending_signal as _read_ps2
                _ps2 = _read_ps2()
                if _ps2.get("has_pending"):
                    _cancel_pending_signal()
                    log.info("[SPECIALIST] CANCELLED pending signal — allocation below threshold")

    # ── 11. Update entry tracking and portfolio state ─────────────────────────
    is_now_long = float(portfolio.get("btc_held", 0.0)) > 1e-6

    if not status_only:
        if not was_long and is_now_long:
            # New entry
            portfolio["entry_price"]           = candle_close
            portfolio["entry_time"]            = candle_ts.isoformat()
            portfolio["max_price_since_entry"] = candle_close
            log.info("Entry recorded at $%.2f", candle_close)

        elif was_long and is_now_long:
            # Still in position — update max price
            prev_max = portfolio.get("max_price_since_entry") or candle_close
            portfolio["max_price_since_entry"] = max(float(prev_max), candle_close)

        elif was_long and not is_now_long:
            # Exit (model signal or stop loss)
            portfolio["entry_price"]           = None
            portfolio["entry_time"]            = None
            portfolio["max_price_since_entry"] = None

        # Stop loss cooldown — stop gain does NOT trigger cooldown
        if gate_log["stop_loss_triggered"]:
            cooldown_candles = (cfg.get("control_layer", {})
                                   .get("stop_loss", {})
                                   .get("cooldown_candles", 1))
            cooldown_end = candle_ts + timedelta(hours=4 * int(cooldown_candles))
            portfolio["stop_loss_cooldown_until"] = cooldown_end.isoformat()
            log.info("Stop loss cooldown until: %s", cooldown_end.isoformat())
        elif gate_log["stop_gain_triggered"]:
            # Take profit: clear any existing cooldown so re-entry is immediate
            portfolio["stop_loss_cooldown_until"] = None
            log.info("Take profit: no cooldown applied — can re-enter next candle")

        portfolio["last_update"]       = candle_ts.isoformat()
        portfolio["r5c_regime"]        = group_b["_r5c_regime"]
        portfolio["r5c_prob_bull"]     = group_b["_r5c_prob_bull"]
        portfolio["r5c_prob_bear"]     = group_b["_r5c_prob_bear"]
        portfolio["r5c_prob_sideways"] = group_b["_r5c_prob_sideways"]
        portfolio["regime_age_days"]   = group_b["_regime_age_days"]
        portfolio["last_stress_score"] = group_b["_stress_score"]
        _save_portfolio(portfolio)

    # ── 12. Compute drawdown_pct for log — use pre-trade entry_price ─────────
    ep = entry_price_raw  # captured before any trade/state update
    drawdown_pct = round((btc_price - float(ep)) / float(ep), 6) if ep else ""

    # ── 13. Log signal ────────────────────────────────────────────────────────
    if not status_only:
        _append_signal({
            "timestamp":           datetime.now(timezone.utc).isoformat(),
            "candle_close":        candle_ts.isoformat(),
            "price_close":         round(candle_close, 2),
            "allocation_raw":      round(allocation_raw, 6),
            "allocation_final":    round(allocation_final, 6),
            "r11_prob_bull":       round(group_b["r11_prob_bull"], 6),
            "r11_entropy":         round(group_b["r11_entropy"], 6),
            "regime_age_log":      round(group_b["regime_age_log"], 6),
            "stress_score":        group_b["stress_score"],
            "returns_4h":          round(features_row["returns_4h"], 6),
            "volatility_24h":      round(features_row["volatility_24h"], 6),
            "volume_zscore":       round(features_row["volume_zscore"], 6),
            "buy_pressure":        round(features_row["buy_pressure"], 6),
            "price_range_4h":      round(features_row["price_range_4h"], 6),
            "position_btc":        round(portfolio["btc_held"], 6),
            "position_usdt":       round(portfolio["usdt_free"], 2),
            "portfolio_value":     round(portfolio["total_usdt"], 2),
            "action":              action,
            "delta_allocation":    round(delta_alloc, 6),
            "fee_paid":            round(fee_paid, 4),
            "stop_loss_triggered":   gate_log["stop_loss_triggered"],
            "stop_gain_triggered":   gate_log["stop_gain_triggered"],
            "regime_gate_triggered": gate_log["regime_gate_triggered"],
            "timing_context_score":  "",
            "timing_gate_active":    False,
            "rsi_4h":                round(timing_indicators["rsi_4h"], 4)
                                     if not pd.isna(timing_indicators["rsi_4h"]) else "",
            "bb_pct_b":              round(timing_indicators["bb_pct_b"], 4)
                                     if not pd.isna(timing_indicators["bb_pct_b"]) else "",
            "news_gate_triggered":   gate_log["news_gate_triggered"],
            "tech_gate_triggered":   gate_log["technical_gate_triggered"],
            "tech_gate_rsi":         round(gate_log["tech_gate_rsi"], 2)
                                     if not pd.isna(gate_log["tech_gate_rsi"]) else "",
            "tech_gate_bb":          round(gate_log["tech_gate_bb"], 4)
                                     if not pd.isna(gate_log["tech_gate_bb"]) else "",
            "tech_gate_reason":      gate_log["technical_gate_reason"],
            "entry_score_total":     round(gate_log.get("entry_score_total", float("nan")), 2)
                                     if not pd.isna(gate_log.get("entry_score_total", float("nan"))) else "",
            "entry_score_bb":        round(gate_log.get("entry_score_bb",    float("nan")), 1)
                                     if not pd.isna(gate_log.get("entry_score_bb",    float("nan"))) else "",
            "entry_score_rsi":       round(gate_log.get("entry_score_rsi",   float("nan")), 1)
                                     if not pd.isna(gate_log.get("entry_score_rsi",   float("nan"))) else "",
            "entry_score_alloc":     round(gate_log.get("entry_score_alloc", float("nan")), 1)
                                     if not pd.isna(gate_log.get("entry_score_alloc", float("nan"))) else "",
            "entry_score_news":      round(gate_log.get("entry_score_news",  float("nan")), 1)
                                     if not pd.isna(gate_log.get("entry_score_news",  float("nan"))) else "",
            "entry_score_reason":    gate_log.get("entry_score_reason", ""),
            "entry_rsi":             round(gate_log["tech_gate_rsi"], 2)
                                     if not pd.isna(gate_log.get("tech_gate_rsi", float("nan"))) else "",
            "entry_prev_ret":        round(features_row.get("returns_4h", float("nan")), 6)
                                     if not pd.isna(features_row.get("returns_4h", float("nan"))) else "",
            "macro_gate_mult":       round(gate_log["macro_gate_multiplier"], 4),
            "deriv_gate_mult":       round(gate_log["deriv_gate_multiplier"], 4),
            "entry_price":           round(float(ep), 2) if ep else "",
            "drawdown_pct":          drawdown_pct,
            "source":                "specialist_4h",
            "entry_source":          "specialist_4h" if action not in ("PENDING",) else "",
            "timing_score_1h":       "",
            "pending_age_hours":     "",
        })

    # ── 14. Report ────────────────────────────────────────────────────────────
    _print_report(
        candle_ts        = candle_ts,
        candle_close     = candle_close,
        features_row     = features_row,
        group_b          = group_b,
        allocation_raw   = allocation_raw,
        allocation_final = allocation_final,
        target_exposure  = target_exposure,
        current_exposure = current_exposure,
        action           = action,
        portfolio        = portfolio,
        gate_log         = gate_log,
        drawdown_pct     = drawdown_pct,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def _print_report(
    candle_ts:        pd.Timestamp,
    candle_close:     float,
    features_row:     dict,
    group_b:          dict,
    allocation_raw:   float,
    allocation_final: float,
    target_exposure:  float,
    current_exposure: float,
    action:           str,
    portfolio:        dict,
    gate_log:         dict,
    drawdown_pct,
) -> None:
    bar = "=" * 60
    log.info(bar)
    log.info("  Specialist 4h — %s", candle_ts.isoformat())
    log.info(bar)
    log.info("  BTC close       : $%s", f"{candle_close:,.2f}")
    log.info("  allocation_raw  : %.4f", allocation_raw)
    log.info("  allocation_final: %.4f", allocation_final)
    log.info("  target_exposure : %.1f%%  current: %.1f%%  → %s",
             target_exposure * 100, current_exposure * 100, action)
    log.info("")
    log.info("  Control layer:")
    log.info("    stop_gain      : triggered=%s",  gate_log["stop_gain_triggered"])
    log.info("    stop_loss      : triggered=%s",  gate_log["stop_loss_triggered"])
    log.info("    regime_gate    : triggered=%s",  gate_log["regime_gate_triggered"])
    log.info("    timing_gate    : disabled in specialist (monitor 1h only)")
    log.info("    news_gate      : triggered=%s",  gate_log["news_gate_triggered"])
    log.info("    technical_gate : triggered=%s  reason=%s",
             gate_log["technical_gate_triggered"], gate_log.get("technical_gate_reason", ""))
    log.info("    entry_score    : total=%.1f (bb=%.1f rsi=%.1f alloc=%.1f news=%.1f) → %s",
             gate_log.get("entry_score_total", float("nan")),
             gate_log.get("entry_score_bb",    float("nan")),
             gate_log.get("entry_score_rsi",   float("nan")),
             gate_log.get("entry_score_alloc", float("nan")),
             gate_log.get("entry_score_news",  float("nan")),
             "ENTER" if gate_log.get("entry_score_total", -99) >= 2.0 else "HOLD")
    log.info("    macro_gate     : mult=%.4f",     gate_log["macro_gate_multiplier"])
    log.info("    deriv_gate     : mult=%.4f",     gate_log["deriv_gate_multiplier"])
    ep = portfolio.get("entry_price")
    if ep:
        log.info("    entry_price   : $%.2f  drawdown=%.2f%%",
                 float(ep),
                 float(drawdown_pct) * 100 if drawdown_pct != "" else 0.0)
    log.info("")
    log.info("  Group A (spot tech):")
    log.info("    returns_4h    : %.6f", features_row["returns_4h"])
    log.info("    volatility_24h: %.6f", features_row["volatility_24h"])
    log.info("    volume_zscore : %.4f",  features_row["volume_zscore"])
    log.info("    buy_pressure  : %.4f",  features_row["buy_pressure"])
    log.info("    price_range_4h: %.6f", features_row["price_range_4h"])
    log.info("")
    log.info("  Group B (R5C regime):")
    log.info("    r5c_regime    : %s",    group_b["_r5c_regime"])
    log.info("    prob_bull     : %.4f",  group_b["_r5c_prob_bull"])
    log.info("    prob_bear     : %.4f",  group_b["_r5c_prob_bear"])
    log.info("    prob_sideways : %.4f",  group_b["_r5c_prob_sideways"])
    log.info("    r11_entropy   : %.4f (mapped from r5c_entropy)", group_b["r11_entropy"])
    log.info("    regime_age_log: %.4f (days=%d)",
             group_b["regime_age_log"], group_b["_regime_age_days"])
    log.info("    stress_score  : %.1f",  group_b["stress_score"])
    log.info("")
    log.info("  Portfolio:")
    log.info("    Total USDT    : $%s",   f"{portfolio['total_usdt']:,.2f}")
    log.info("    USDT free     : $%s",   f"{portfolio['usdt_free']:,.2f}")
    log.info("    BTC held      : %.6f BTC", portfolio["btc_held"])
    log.info("    Position      : %.1f%%",   portfolio["position_pct"] * 100)
    log.info(bar)


# ══════════════════════════════════════════════════════════════════════════════
# Init
# ══════════════════════════════════════════════════════════════════════════════

def run_init() -> None:
    cfg = _load_config()
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ROOT / cfg["model_path"]
    if not model_path.exists():
        log.error("[init] SET_B model not found: %s", model_path)
        sys.exit(1)
    log.info("[init] SET_B model found ✓  %s", model_path.name)

    r11_path = ROOT / cfg["r11_model_path"]
    if not r11_path.exists():
        log.error("[init] R11 model not found: %s — run freeze_r11_model.py first", r11_path)
        sys.exit(1)
    log.info("[init] R11 model found ✓")

    log.info("[init] Seeding 4h OHLCV buffer …")
    buf = update_4h_buffer(cfg)
    log.info("[init] Buffer: %d candles  (%s → %s)",
             len(buf), buf.index[0].date(), buf.index[-1].date())

    if PORTFOLIO_JS.exists():
        # Ensure new fields are present (migration)
        p = json.loads(PORTFOLIO_JS.read_text())
        for field, default in [
            ("entry_price",             None),
            ("entry_time",              None),
            ("max_price_since_entry",   None),
            ("stop_loss_cooldown_until", None),
        ]:
            if field not in p:
                p[field] = default
        _save_portfolio(p)
        log.info("[init] portfolio.json updated with entry tracking fields.")
    else:
        initial = float(cfg["initial_capital"])
        _save_portfolio({
            "usdt_free":               initial,
            "btc_held":                0.0,
            "total_usdt":              initial,
            "btc_price":               0.0,
            "position_pct":            0.0,
            "r11_regime":              "Bear",
            "regime_age_days":         0,
            "last_stress_score":       0.0,
            "last_update":             "",
            "entry_price":             None,
            "entry_time":              None,
            "max_price_since_entry":   None,
            "stop_loss_cooldown_until": None,
        })
        log.info("[init] portfolio.json created  (capital=$%.0f)", initial)

    _ensure_signals_csv()
    log.info("[init] signals.csv ready")
    log.info("[init] Done. Run without --init to start 4h paper trading.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Specialist 4h Paper Trader — SET_B LightGBM + Control Layer"
    )
    parser.add_argument("--init",   action="store_true", help="Initialise state files")
    parser.add_argument("--status", action="store_true", help="Status only — no state write")
    args = parser.parse_args()

    if args.init:
        run_init()
    else:
        run_4h(status_only=args.status)
