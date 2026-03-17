"""
R11 Paper Trader — daily orchestrator for Binance Testnet.

Usage:
    # First-time setup (freeze model + initialise):
    python scripts/paper_trading/freeze_r11_model.py
    python scripts/paper_trading/r11_paper_trader.py --init

    # Daily run (cron / manual):
    python scripts/paper_trading/r11_paper_trader.py

    # Status only (no trade, no state write):
    python scripts/paper_trading/r11_paper_trader.py --status

Flow:
    1. Fetch yesterday's BTCUSDT daily candle (Binance public API)
    2. Update OHLCV buffer  →  state/ohlcv_buffer.parquet
    3. Compute R11 features (rolling, matches L3 pipeline exactly)
    4. Generate signal  (Bull / Bear + p_bull)
    5. Compute target position  (f=0.75 if Bull, 0 if Bear)
    6. Compare with current position; skip if delta < 5% of capital
    7. Execute order on Binance Testnet if needed
    8. Update equity curve + live metrics
    9. Print daily report

Constants:
    BUFFER_DAYS  = 200   (enough for 90d drawdown + 21d slope + vol windows)
    FRACTION     = 0.75  (Fixed Fractional f)
    MIN_DELTA    = 0.05  (position delta threshold before placing an order)
    TC_BPS       = 12    (transaction cost estimate, informational only)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT       = Path(__file__).resolve().parents[2]
STATE_DIR  = Path(__file__).parent / "state"
BUFFER_PKL = STATE_DIR / "ohlcv_buffer.parquet"
TRADE_LOG  = STATE_DIR / "trade_log.csv"
EQUITY_CSV = STATE_DIR / "equity_curve.csv"

_TRADE_LOG_FIELDS = [
    "timestamp", "date", "side", "symbol",
    "btc_qty", "fill_price", "usdt_value", "fees_usdt", "order_id", "note",
]

BINANCE_API   = "https://api.binance.com"
SYMBOL        = "BTCUSDT"
INTERVAL      = "1d"
BUFFER_DAYS   = 200
FRACTION      = 0.75
MIN_DELTA     = 0.05    # minimum position change to place an order
TC_BPS        = 12


# ---------------------------------------------------------------------------
# Feature computation (mirrors L3 pipeline exactly)
# ---------------------------------------------------------------------------

def _slope_normalized(arr: np.ndarray) -> float:
    """OLS slope / last_close — same as L3 _slope_normalized."""
    n = len(arr)
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, arr, 1)[0])
    last  = arr[-1]
    return slope / last if last != 0.0 else 0.0


def _compute_r11_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 6 R11 features from OHLCV data.

    Input columns required: open, high, low, close, volume
    Returns df with added feature columns (NaN in burn-in rows).
    Windows: vol_short=7, vol_long=30, drawdown=90, volume_z=30, slope=21
    """
    close  = df["close"]
    volume = df["volume"]

    log_ret   = np.log(close / close.shift(1))
    vol_short = log_ret.rolling(7).std()
    vol_long  = log_ret.rolling(30).std()
    vol_ratio = vol_short / vol_long.replace(0, np.nan)

    roll_max  = close.rolling(90, min_periods=1).max()
    drawdown  = close / roll_max - 1.0

    vol_mean  = volume.rolling(30).mean()
    vol_std   = volume.rolling(30).std()
    volume_z  = (volume - vol_mean) / vol_std.replace(0, np.nan)

    slope_21d = close.rolling(21).apply(_slope_normalized, raw=True)

    df = df.copy()
    df["log_return"] = log_ret
    df["vol_short"]  = vol_short
    df["vol_ratio"]  = vol_ratio
    df["drawdown"]   = drawdown
    df["volume_z"]   = volume_z
    df["slope_21d"]  = slope_21d
    return df


# ---------------------------------------------------------------------------
# OHLCV buffer management
# ---------------------------------------------------------------------------

def _fetch_candles(limit: int = BUFFER_DAYS + 5) -> pd.DataFrame:
    """
    Fetch the most recent `limit` daily BTCUSDT candles from Binance public API.
    Returns DataFrame with columns: open_time, open, high, low, close, volume.
    open_time is UTC DatetimeIndex.
    """
    url    = f"{BINANCE_API}/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": limit}
    resp   = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    records = []
    for k in raw:
        records.append({
            "open_time": pd.Timestamp(k[0], unit="ms", tz="UTC"),
            "open":      float(k[1]),
            "high":      float(k[2]),
            "low":       float(k[3]),
            "close":     float(k[4]),
            "volume":    float(k[5]),
        })
    df = pd.DataFrame(records).set_index("open_time").sort_index()
    return df


def _load_buffer() -> pd.DataFrame:
    if BUFFER_PKL.exists():
        df = pd.read_parquet(BUFFER_PKL)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    return pd.DataFrame()


def _save_buffer(df: pd.DataFrame) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BUFFER_PKL)


def update_buffer() -> pd.DataFrame:
    """Fetch latest candles, merge with existing buffer, return updated buffer."""
    fresh  = _fetch_candles(limit=BUFFER_DAYS + 5)
    stored = _load_buffer()

    if stored.empty:
        combined = fresh
    else:
        combined = pd.concat([stored, fresh])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index().tail(BUFFER_DAYS + 5)

    _save_buffer(combined)
    return combined


def get_yesterday_date() -> str:
    """Return yesterday's date (UTC) as YYYY-MM-DD string."""
    return (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def _target_position(regime: str) -> float:
    return FRACTION if regime == "Bull" else 0.0


def _get_current_position(portfolio: dict) -> float:
    """Return current BTC position as fraction of total portfolio."""
    return portfolio.get("position_pct", 0.0)


def _maybe_log_initial_position(portfolio: dict, signal: dict) -> None:
    """
    If the earliest trade in trade_log is a SELL with no prior BUY, prepend a
    reconstructed initial BUY row.  Idempotent — safe to call on every run.

    Reconstruction:
        btc_qty    = first_sell.btc_qty + portfolio.btc_held
        fill_price = (equity_curve[0].portfolio_value * FRACTION) / btc_qty
        usdt_value = equity_curve[0].portfolio_value * FRACTION
        timestamp  = equity_curve[0].date @ 00:00:00 UTC
    """
    # trade_log must exist and have content
    if not TRADE_LOG.exists() or TRADE_LOG.stat().st_size == 0:
        return

    existing = pd.read_csv(TRADE_LOG)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
    existing = existing.sort_values("timestamp").reset_index(drop=True)

    # Idempotency: skip if any BUY exists before (or at same timestamp as) the first SELL
    sell_mask = existing["side"] == "SELL"
    if not sell_mask.any():
        return  # no SELL at all — nothing to prepend before

    first_sell_pos = sell_mask.idxmax()  # index of first SELL in sorted order
    if (existing.loc[:first_sell_pos - 1, "side"] == "BUY").any() if first_sell_pos > 0 else False:
        return  # BUY already precedes the first SELL

    # Need equity_curve for portfolio_value[0]
    if not EQUITY_CSV.exists() or EQUITY_CSV.stat().st_size == 0:
        return
    try:
        eq = pd.read_csv(EQUITY_CSV, parse_dates=["date"])
    except Exception:
        return
    if eq.empty:
        return

    pv0       = float(eq["portfolio_value"].iloc[0])
    init_ts   = pd.Timestamp(eq["date"].iloc[0], tz="UTC")

    # Reconstruct initial BTC quantity and price
    first_sell  = existing[existing["side"] == "SELL"].iloc[0]
    sell_qty    = float(first_sell["btc_qty"])
    btc_held    = portfolio.get("btc_held", 0.0)
    btc_qty     = round(sell_qty + btc_held, 6)

    if btc_qty <= 0:
        return

    usdt_value  = round(pv0 * FRACTION, 2)
    fill_price  = round(usdt_value / btc_qty, 2)

    buy_row = {
        "timestamp":  init_ts.isoformat(),
        "date":       init_ts.strftime("%Y-%m-%d"),
        "side":       "BUY",
        "symbol":     "BTCUSDT",
        "btc_qty":    btc_qty,
        "fill_price": fill_price,
        "usdt_value": usdt_value,
        "fees_usdt":  0.0,
        "order_id":   "",
        "note":       "INITIAL_POSITION (reconstructed)",
    }
    new_row = pd.DataFrame([buy_row])
    new_row["timestamp"] = pd.to_datetime(new_row["timestamp"], utc=True)

    combined = pd.concat([existing, new_row], ignore_index=True)
    combined  = combined.sort_values("timestamp").reset_index(drop=True)

    # Serialize timestamps back to ISO-8601 with UTC offset
    combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    combined[_TRADE_LOG_FIELDS].to_csv(TRADE_LOG, index=False)

    print(f"[PaperTrader] Injected INITIAL_POSITION BUY: "
          f"{btc_qty:.6f} BTC @ ${fill_price:,.2f}  (usdt_value=${usdt_value:,.2f})")


# ---------------------------------------------------------------------------
# Daily report
# ---------------------------------------------------------------------------

def _print_daily_report(
    date_str:    str,
    signal:      dict,
    portfolio:   dict,
    target_pos:  float,
    current_pos: float,
    traded:      bool,
    metrics:     dict,
) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  R11 Paper Trader — {date_str}")
    print(bar)
    print(f"  Signal  : {signal['regime']:4s}  |  p_bull={signal['p_bull']:.3f}  p_bear={signal['p_bear']:.3f}")
    print(f"  Position: {current_pos*100:.1f}% → target {target_pos*100:.1f}%"
          f"  ({'TRADED' if traded else 'no trade'})")
    print()
    print(f"  Portfolio:")
    print(f"    Total USDT : ${portfolio['total_usdt']:>12,.2f}")
    print(f"    USDT free  : ${portfolio['usdt_free']:>12,.2f}")
    print(f"    BTC held   : {portfolio['btc_held']:>12.6f}  BTC")
    print(f"    BTC value  : ${portfolio['btc_usdt']:>12,.2f}")
    print(f"    BTC price  : ${portfolio['btc_price']:>12,.2f}")
    print()

    n = metrics.get("n_days", 0)
    if n >= 20:
        print(f"  Live metrics ({n} days):")

        def _line(label, live, ref, pct=False):
            if live is None:
                return
            fmt = "{:+.2f}%" if pct else "{:+.4f}"
            lv  = live * 100 if pct else live
            rv  = ref  * 100 if pct else ref
            dv  = (live - ref) * 100 if pct else (live - ref)
            d_s = "+" if dv >= 0 else ""
            print(f"    {label:<10}: {fmt.format(lv):<10}  (bt={fmt.format(rv)}  Δ={d_s}{fmt.format(dv)})")

        _line("Sharpe",  metrics["sharpe"],  0.775)
        _line("CAGR",    metrics["cagr"],    0.0977, pct=True)
        _line("MaxDD",   metrics["maxdd"],   -0.0987, pct=True)
        _line("HitRate", metrics["hitrate"], 0.4851,  pct=True)
    else:
        print(f"  Live metrics: {n}/20 days accumulated (need ≥20)")

    if metrics.get("maxdd") is not None and metrics["maxdd"] <= -0.12:
        print()
        print(f"  *** DRAWDOWN ALERT: MaxDD = {metrics['maxdd']*100:.2f}% ***")
        print(f"  *** CONSIDER HALTING or REDUCING POSITION SIZE ***")

    print(bar)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_daily(status_only: bool = False, init_mode: bool = False) -> None:
    import importlib.util, sys as _sys

    def _local_import(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod  = importlib.util.module_from_spec(spec)
        _sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _pt_dir = Path(__file__).parent
    _sig    = _local_import("r11_signal_generator",   _pt_dir / "r11_signal_generator.py")
    _oe     = _local_import("r11_order_executor",     _pt_dir / "r11_order_executor.py")
    _perf   = _local_import("r11_performance_tracker",_pt_dir / "r11_performance_tracker.py")

    SignalGenerator    = _sig.SignalGenerator
    OrderExecutor      = _oe.OrderExecutor
    PerformanceTracker = _perf.PerformanceTracker
    load_equity_curve  = _perf.load_equity_curve

    tracker  = PerformanceTracker()
    executor = OrderExecutor()
    gen      = SignalGenerator()

    # 1. Update OHLCV buffer
    print("[PaperTrader] Fetching latest BTCUSDT candles …")
    buffer = update_buffer()
    print(f"[PaperTrader] Buffer: {len(buffer)} rows  "
          f"({buffer.index[0].date()} → {buffer.index[-1].date()})")

    # 2. Compute features
    buf_feat = _compute_r11_features(buffer)
    last_row = buf_feat.dropna(subset=["log_return", "vol_short", "vol_ratio",
                                        "drawdown", "volume_z", "slope_21d"]).iloc[-1]
    date_str = str(last_row.name.date())

    # 3. Idempotency guard
    if not init_mode and tracker.date_exists(date_str):
        print(f"[PaperTrader] Already processed {date_str} — skipping.")
        tracker.print_metrics()
        return

    # 4. Generate signal
    signal = gen.predict(last_row, signal_date=date_str, context_df=buf_feat)
    print(f"[PaperTrader] Signal: {signal['regime']}  p_bull={signal['p_bull']:.3f}")

    # 5. Get current portfolio
    portfolio    = executor.get_portfolio_value()
    current_pos  = _get_current_position(portfolio)
    target_pos   = _target_position(signal["regime"])
    delta        = abs(target_pos - current_pos)

    # Log initial BUY if never recorded
    if not status_only:
        _maybe_log_initial_position(portfolio, signal)

    traded = False

    if not status_only:
        # 6. Execute order if delta is meaningful
        if delta >= MIN_DELTA:
            total = portfolio["total_usdt"]
            if target_pos > current_pos:
                # BUY: invest the additional fraction
                usdt_to_buy = (target_pos - current_pos) * total
                note = f"R11 signal=Bull p={signal['p_bull']:.3f}"
                executor.place_market_buy(usdt_to_buy, note=note)
                traded = True
            else:
                # SELL: reduce BTC to match target fraction
                btc_to_sell = portfolio["btc_held"] * (current_pos - target_pos) / current_pos \
                              if current_pos > 0 else portfolio["btc_held"]
                if btc_to_sell > 0.000001:
                    note = f"R11 signal=Bear p={signal['p_bear']:.3f}"
                    executor.place_market_sell(btc_to_sell, note=note)
                    traded = True
        else:
            print(f"[PaperTrader] Position delta={delta*100:.1f}% < {MIN_DELTA*100:.0f}% — no trade.")

        # 7. Refresh portfolio after trade
        portfolio = executor.get_portfolio_value()

        # 8. Update equity curve
        prev_df = load_equity_curve()

        if len(prev_df) > 0:
            prev_val  = float(prev_df["portfolio_value"].iloc[-1])
            daily_ret = (portfolio["total_usdt"] / prev_val) - 1.0
        else:
            daily_ret = 0.0

        tracker.append_row(
            date_str        = date_str,
            portfolio_value = portfolio["total_usdt"],
            daily_ret       = daily_ret,
            position_pct    = portfolio["position_pct"],
            regime          = signal["regime"],
            p_bull          = signal["p_bull"],
        )

    # 9. Get live metrics
    metrics = tracker.get_metrics()

    # 10. Daily report
    _print_daily_report(
        date_str    = date_str,
        signal      = signal,
        portfolio   = portfolio,
        target_pos  = target_pos,
        current_pos = portfolio["position_pct"],
        traded      = traded,
        metrics     = metrics,
    )

    if not status_only:
        # MaxDD alert check
        alert, dd_live = tracker.check_maxdd_alert()
        if alert:
            print(f"\n[PaperTrader] *** MAXDD ALERT: {dd_live*100:.2f}% ***")
            print("[PaperTrader] Review position sizing immediately.")


def run_init() -> None:
    """
    Initialise paper trading:
      - Verify frozen model exists
      - Seed OHLCV buffer
      - Print portfolio status (no trade placed)
    """
    import importlib.util, sys as _sys

    def _local_import(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod  = importlib.util.module_from_spec(spec)
        _sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _pt_dir = Path(__file__).parent
    _sig    = _local_import("r11_signal_generator", _pt_dir / "r11_signal_generator.py")
    _oe     = _local_import("r11_order_executor",   _pt_dir / "r11_order_executor.py")

    SignalGenerator = _sig.SignalGenerator
    OrderExecutor   = _oe.OrderExecutor

    model_pkl = Path(__file__).parent / "state" / "r11_hmm_model.pkl"
    if not model_pkl.exists():
        print("[init] ERROR: Frozen model not found.")
        print("  Run: python scripts/paper_trading/freeze_r11_model.py")
        sys.exit(1)

    print("[init] Frozen model found ✓")

    print("[init] Seeding OHLCV buffer …")
    buffer = update_buffer()
    print(f"[init] Buffer seeded: {len(buffer)} rows  "
          f"({buffer.index[0].date()} → {buffer.index[-1].date()})")

    gen    = SignalGenerator()
    buf_f  = _compute_r11_features(buffer)
    last   = buf_f.dropna(subset=["log_return", "vol_short", "vol_ratio",
                                   "drawdown", "volume_z", "slope_21d"]).iloc[-1]
    signal = gen.predict(last, signal_date=str(last.name.date()), context_df=buf_f)
    print(f"[init] Current signal: {signal['regime']}  p_bull={signal['p_bull']:.3f}")

    executor  = OrderExecutor()
    portfolio = executor.get_portfolio_value()
    print(f"[init] Portfolio: ${portfolio['total_usdt']:,.2f} USDT total")
    print(f"         BTC: {portfolio['btc_held']:.6f}  USDT free: ${portfolio['usdt_free']:,.2f}")
    print("[init] Ready. Run without --init to start daily trading.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R11 Paper Trader — Binance Testnet")
    parser.add_argument("--init",   action="store_true", help="Initialise (seed buffer, verify model)")
    parser.add_argument("--status", action="store_true", help="Status only — no trade, no state write")
    args = parser.parse_args()

    if args.init:
        run_init()
    else:
        run_daily(status_only=args.status)
