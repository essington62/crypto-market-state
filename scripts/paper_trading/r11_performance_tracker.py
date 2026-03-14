"""
R11 Performance Tracker — live equity curve and metrics.

Maintains state/equity_curve.csv and computes rolling metrics vs backtest targets.

Backtest targets (Fixed Fractional f=0.75):
  Sharpe = +0.775
  CAGR   = +9.77%
  MaxDD  = -9.87%
  HitRate = 48.51%

Alert threshold: MaxDD ≤ -12.0% (150% of backtest MaxDD)
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

STATE_DIR   = Path(__file__).parent / "state"
EQUITY_CSV  = STATE_DIR / "equity_curve.csv"
PERIODS_YR  = 252
MIN_DAYS    = 20          # minimum rows for metric computation
DD_ALERT    = -0.12       # MaxDD alert threshold

# Backtest reference targets (f=0.75)
BT_SHARPE  = 0.775
BT_CAGR    = 0.0977
BT_MAXDD   = -0.0987
BT_HITRATE = 0.4851

_EQUITY_FIELDS = [
    "date", "portfolio_value", "daily_ret", "position_pct", "regime", "p_bull",
]


def _ensure_equity_csv() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not EQUITY_CSV.exists():
        with open(EQUITY_CSV, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=_EQUITY_FIELDS).writeheader()


def load_equity_curve() -> pd.DataFrame:
    """Load equity_curve.csv as a DataFrame indexed by date."""
    _ensure_equity_csv()
    if EQUITY_CSV.stat().st_size == 0:
        return pd.DataFrame(columns=_EQUITY_FIELDS)
    df = pd.read_csv(EQUITY_CSV, parse_dates=["date"])
    if df.empty:
        return df
    df = df.set_index("date").sort_index()
    for col in ["portfolio_value", "daily_ret", "position_pct", "p_bull"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _compute_metrics(df: pd.DataFrame) -> dict:
    """Compute Sharpe, CAGR, MaxDD, HitRate from equity curve DataFrame."""
    if len(df) < MIN_DAYS:
        return {
            "sharpe":   None,
            "cagr":     None,
            "maxdd":    None,
            "hitrate":  None,
            "n_days":   len(df),
            "n_trades": None,
        }

    rets = df["daily_ret"].dropna().values

    # Sharpe (annualised)
    mu    = rets.mean()
    sigma = rets.std(ddof=1)
    sharpe = float((mu / (sigma + 1e-10)) * np.sqrt(PERIODS_YR)) if sigma > 0 else 0.0

    # CAGR
    total_ret = float(df["portfolio_value"].iloc[-1] / df["portfolio_value"].iloc[0]) - 1.0
    n_years   = len(df) / PERIODS_YR
    cagr      = float((1.0 + total_ret) ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0

    # MaxDD (running)
    pv     = df["portfolio_value"].values
    peak   = np.maximum.accumulate(pv)
    dd_arr = pv / peak - 1.0
    maxdd  = float(dd_arr.min())

    # Hit rate — count days with positive position where daily return > 0
    pos_days = df[df["position_pct"] > 0]
    if len(pos_days) > 0:
        hitrate = float((pos_days["daily_ret"] > 0).mean())
    else:
        hitrate = float("nan")

    # Trade count: transitions where position_pct changes significantly
    pos_arr  = df["position_pct"].values
    n_trades = int(np.sum(np.abs(np.diff(pos_arr)) > 0.05))

    return {
        "sharpe":   round(sharpe, 4),
        "cagr":     round(cagr,   4),
        "maxdd":    round(maxdd,  4),
        "hitrate":  round(hitrate, 4) if not np.isnan(hitrate) else None,
        "n_days":   len(df),
        "n_trades": n_trades,
    }


def _vs_backtest(metrics: dict) -> dict:
    """Compute delta vs backtest targets."""
    def _d(v, ref):
        return round(v - ref, 4) if v is not None else None

    return {
        "sharpe_vs_bt":  _d(metrics["sharpe"],  BT_SHARPE),
        "cagr_vs_bt":    _d(metrics["cagr"],    BT_CAGR),
        "maxdd_vs_bt":   _d(metrics["maxdd"],   BT_MAXDD),
        "hitrate_vs_bt": _d(metrics["hitrate"], BT_HITRATE),
    }


class PerformanceTracker:
    """Tracks live equity curve and computes rolling metrics."""

    def __init__(self) -> None:
        _ensure_equity_csv()

    def date_exists(self, date_str: str) -> bool:
        """Return True if a row for date_str already exists (idempotency guard)."""
        df = load_equity_curve()
        if df.empty:
            return False
        return date_str in [str(d.date()) if hasattr(d, "date") else str(d)
                            for d in df.index]

    def append_row(
        self,
        date_str:        str,
        portfolio_value: float,
        daily_ret:       float,
        position_pct:    float,
        regime:          str,
        p_bull:          float,
    ) -> None:
        """Append one row to equity_curve.csv."""
        _ensure_equity_csv()
        row = {
            "date":            date_str,
            "portfolio_value": round(portfolio_value, 4),
            "daily_ret":       round(daily_ret, 6),
            "position_pct":    round(position_pct, 4),
            "regime":          regime,
            "p_bull":          round(p_bull, 4),
        }
        with open(EQUITY_CSV, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=_EQUITY_FIELDS).writerow(row)

    def get_metrics(self) -> dict:
        """Return current live metrics dict."""
        df      = load_equity_curve()
        metrics = _compute_metrics(df)
        deltas  = _vs_backtest(metrics)
        return {**metrics, **deltas}

    def check_maxdd_alert(self) -> tuple[bool, float | None]:
        """Return (alert_triggered, current_maxdd). Alert if maxdd ≤ DD_ALERT."""
        df = load_equity_curve()
        if len(df) < 2:
            return False, None
        pv    = df["portfolio_value"].values
        peak  = np.maximum.accumulate(pv)
        dd    = float((pv / peak - 1.0).min())
        alert = dd <= DD_ALERT
        return alert, round(dd, 4)

    def print_metrics(self) -> None:
        """Print a formatted metrics table to stdout."""
        m = self.get_metrics()
        alert, dd_live = self.check_maxdd_alert()

        print("\n" + "=" * 56)
        print("  R11 Live Performance")
        print("=" * 56)
        print(f"  Days tracked : {m['n_days']}")
        print(f"  Trades       : {m['n_trades']}")
        print()

        def _fmt(val, ref, pct=False):
            if val is None:
                return "  n/a         (need ≥20 days)"
            sign  = "+" if val >= ref else " "
            delta = val - ref
            d_sign = "+" if delta >= 0 else ""
            if pct:
                return f"  {val*100:+.2f}%   (bt={ref*100:.2f}%  Δ={d_sign}{delta*100:.2f}%)"
            return f"  {val:+.4f}   (bt={ref:+.4f}  Δ={d_sign}{delta:.4f})"

        print(f"  Sharpe   {_fmt(m['sharpe'],  BT_SHARPE)}")
        print(f"  CAGR     {_fmt(m['cagr'],    BT_CAGR,    pct=True)}")
        print(f"  MaxDD    {_fmt(m['maxdd'],   BT_MAXDD,   pct=True)}")
        print(f"  HitRate  {_fmt(m['hitrate'], BT_HITRATE, pct=True)}")

        if alert:
            print()
            print(f"  *** ALERT: MaxDD = {dd_live*100:.2f}% ≤ threshold {DD_ALERT*100:.0f}% ***")
            print(f"  Consider reducing position size or halting trading.")

        print("=" * 56)
