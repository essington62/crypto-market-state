"""
Specialist 4h Dashboard — LightGBM SET_B paper trading monitor.

Run:
    streamlit run apps/specialist_4h_dashboard.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
STATE_DIR  = ROOT / "scripts" / "paper_trading" / "state" / "specialist_4h"
R11_STATE  = ROOT / "scripts" / "paper_trading" / "state"

SIGNALS_CSV  = STATE_DIR / "signals.csv"
PORTFOLIO_JS = STATE_DIR / "portfolio.json"
CONFIG_JS    = STATE_DIR / "config.json"

# For buy&hold comparison: L3 daily BTC price
L3_DAILY = ROOT / "data" / "03_primary" / "spot" / "daily" / "BTCUSDT.parquet"

PERIODS_YR = 365 * 6   # 4h periods per year

BULL_CLR = "#2E7D32"
BEAR_CLR = "#C62828"
LINE_CLR = "#1565C0"
ORANGE   = "#FB8C00"


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Specialist 4h Dashboard",
    page_icon="⚡",
    layout="wide",
)
st.title("⚡ Specialist 4h — SET_B LightGBM Paper Trading")

with st.sidebar:
    st.header("Controls")
    st.button("🔄 Refresh", use_container_width=True)
    st.markdown("---")
    st.markdown("**Model**")
    try:
        cfg = json.loads(CONFIG_JS.read_text()) if CONFIG_JS.exists() else {}
        st.code(
            f"SET_B split_4_recent\n"
            f"Features: {len(cfg.get('set_b_features', []))}\n"
            f"Capital: ${cfg.get('initial_capital', 0):,.0f}\n"
            f"Fee: {cfg.get('fee_rate', 0)*100:.2f}%",
            language=None,
        )
    except Exception:
        st.code("config not loaded", language=None)
    st.markdown("---")
    st.caption(f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")


# ══════════════════════════════════════════════════════════════════════════════
# Loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_signals() -> pd.DataFrame:
    if not SIGNALS_CSV.exists() or SIGNALS_CSV.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(SIGNALS_CSV)
    if "candle_close" in df.columns:
        df["candle_close"] = pd.to_datetime(df["candle_close"], utc=True)
        df = df.sort_values("candle_close").reset_index(drop=True)
    num_cols = [
        "price_close", "allocation_raw", "allocation_final",
        "r11_prob_bull", "r11_entropy", "regime_age_log", "stress_score",
        "returns_4h", "volatility_24h", "volume_zscore", "buy_pressure", "price_range_4h",
        "position_btc", "position_usdt", "portfolio_value",
        "delta_allocation", "fee_paid",
        "timing_gate_score", "rsi_4h", "bb_pct_b",
        "macro_gate_mult", "deriv_gate_mult", "entry_price", "drawdown_pct",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=300)
def load_portfolio() -> dict:
    if not PORTFOLIO_JS.exists():
        return {}
    try:
        return json.loads(PORTFOLIO_JS.read_text())
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_btc_price_history() -> pd.DataFrame:
    """Load daily close prices for buy&hold reference."""
    if not L3_DAILY.exists():
        return pd.DataFrame()
    df = pd.read_parquet(L3_DAILY)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[["close"]].sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _stale(date_val, max_hours: int = 8) -> bool:
    """Return True if date_val is older than max_hours."""
    try:
        dt = pd.Timestamp(date_val)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return (pd.Timestamp.now(tz="UTC") - dt).total_seconds() / 3600 > max_hours
    except Exception:
        return True


_TABLE_CELL = {"background-color": "#0e0e0e", "color": "#00ff88", "border-color": "#1a1a1a"}
_TABLE_STYLES = [
    {"selector": "th", "props": [
        ("background-color", "#111111"), ("color", "#00ff88"),
        ("border-color", "#1a1a1a"), ("font-weight", "600"),
    ]},
    {"selector": "td", "props": [
        ("background-color", "#0e0e0e"), ("color", "#00ff88"), ("border-color", "#1a1a1a"),
    ]},
]


def _dark_table(styler):
    return styler.set_properties(**_TABLE_CELL).set_table_styles(_TABLE_STYLES)


def _action_color(row):
    c = row.get("action", "")
    if c == "BUY":
        return ["background-color:#0e0e0e; color:#00ff88"] * len(row)
    if c == "SELL":
        return ["background-color:#0e0e0e; color:#66ccaa"] * len(row)
    return ["background-color:#0e0e0e; color:#888888"] * len(row)


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — KPI Row
# ══════════════════════════════════════════════════════════════════════════════

def section_kpis(signals: pd.DataFrame, portfolio: dict) -> None:
    st.subheader("① Model Status")

    # Stop loss alert banner
    if not signals.empty and "stop_loss_triggered" in signals.columns:
        recent_sl = signals[signals["stop_loss_triggered"].astype(str).isin(["True", "true", "1"])]
        if not recent_sl.empty:
            last_sl_ts = str(recent_sl["candle_close"].max())[:16]
            st.error(f"⛔ Stop Loss triggered at {last_sl_ts} UTC")

    cooldown_until = portfolio.get("stop_loss_cooldown_until")
    if cooldown_until:
        try:
            cd_ts = pd.Timestamp(cooldown_until, tz="UTC")
            now   = pd.Timestamp.now(tz="UTC")
            if cd_ts > now:
                remaining_h = round((cd_ts - now).total_seconds() / 3600, 1)
                st.warning(f"⏸ Stop loss cooldown active — re-entry blocked for {remaining_h}h "
                           f"(until {str(cd_ts)[:16]} UTC)")
        except Exception:
            pass

    if not signals.empty and "portfolio_value" in signals.columns:
        last = signals.iloc[-1]
        pv_last   = float(last.get("portfolio_value", 0))
        alloc_raw = float(last.get("allocation_raw", float("nan")))
        alloc_fin = float(last.get("allocation_final", float("nan")))
        pos_pct   = float(portfolio.get("position_pct", float("nan")))
        regime    = str(portfolio.get("r11_regime", "n/a"))
        p_bull    = float(last.get("r11_prob_bull", float("nan")))
        action    = str(last.get("action", "n/a"))

        pv_series = signals["portfolio_value"].dropna()
        if len(pv_series) >= 2:
            tot_ret = float(pv_series.iloc[-1] / pv_series.iloc[0] - 1)
        else:
            tot_ret = None

        # Sharpe from 4h returns
        if "portfolio_value" in signals.columns and len(signals) >= 24:
            rets = signals["portfolio_value"].pct_change().dropna()
            mu, std = rets.mean(), rets.std(ddof=1)
            sharpe = float((mu / max(std, 1e-10)) * np.sqrt(PERIODS_YR)) if std > 0 else float("nan")
        else:
            sharpe = float("nan")
    else:
        pv_last = alloc_raw = alloc_fin = pos_pct = float("nan")
        regime = action = "n/a"
        p_bull = tot_ret = sharpe = float("nan")

    cols = st.columns(7)
    with cols[0]:
        st.metric("R11 Regime", regime)
        if regime == "Bull":
            st.markdown(f"<span style='color:{BULL_CLR};font-size:1.3em'>●</span>", unsafe_allow_html=True)
        elif regime == "Bear":
            st.markdown(f"<span style='color:{BEAR_CLR};font-size:1.3em'>●</span>", unsafe_allow_html=True)
    with cols[1]:
        st.metric("P(Bull)", f"{p_bull:.3f}" if not np.isnan(p_bull) else "n/a")
    with cols[2]:
        st.metric("Allocation", f"{alloc_fin:.3f}" if not np.isnan(alloc_fin) else "n/a")
    with cols[3]:
        st.metric("Position", f"{pos_pct*100:.0f}%" if not np.isnan(pos_pct) else "n/a")
    with cols[4]:
        st.metric("Portfolio", f"${pv_last:,.0f}" if not np.isnan(pv_last) else "n/a")
    with cols[5]:
        tot_str = f"{tot_ret*100:+.2f}%" if tot_ret is not None else "--"
        st.metric("Total Return", tot_str, delta=tot_str if tot_ret is not None else None,
                  delta_color="normal")
    with cols[6]:
        if np.isnan(sharpe):
            st.metric("Live Sharpe", "n/a", delta="need 24 candles")
        else:
            st.metric("Live Sharpe", f"{sharpe:+.3f}",
                      delta=f"{sharpe - 3.872:+.3f} vs bt",
                      delta_color="normal")


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Equity Curve + Buy&Hold
# ══════════════════════════════════════════════════════════════════════════════

def section_equity(signals: pd.DataFrame) -> None:
    st.subheader("② Equity Curve vs Buy & Hold")
    if signals.empty or "portfolio_value" not in signals.columns:
        st.info("No portfolio data yet.")
        return

    try:
        pv = signals.set_index("candle_close")["portfolio_value"].dropna()
        if pv.empty:
            st.info("No portfolio values recorded yet.")
            return

        start_val = float(pv.iloc[0])

        fig = go.Figure()

        # Specialist equity
        fig.add_trace(go.Scatter(
            x=pv.index, y=pv.values,
            name="Specialist 4h",
            line=dict(color=LINE_CLR, width=2),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.2f}<extra>Specialist</extra>",
        ))

        # Buy & hold reference (daily → resampled to same 4h grid)
        btc_hist = load_btc_price_history()
        if not btc_hist.empty:
            btc_cut = btc_hist.loc[btc_hist.index >= pv.index[0]]
            if not btc_cut.empty:
                btc_start = float(btc_cut["close"].iloc[0])
                bh_norm   = btc_cut["close"] / btc_start * start_val
                fig.add_trace(go.Scatter(
                    x=bh_norm.index, y=bh_norm.values,
                    name="BTC Buy&Hold",
                    line=dict(color=ORANGE, width=1.5, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra>B&H</extra>",
                ))

        fig.add_hline(y=start_val, line_dash="dash", line_color="gray",
                      annotation_text=f"Start ${start_val:,.0f}")
        fig.update_layout(
            yaxis=dict(title="Portfolio (USDT)", tickprefix="$", tickformat=",.0f"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=30, b=0),
            height=380,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Equity chart unavailable: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Recent Signals (last 24h = 6 candles)
# ══════════════════════════════════════════════════════════════════════════════

def section_recent_signals(signals: pd.DataFrame) -> None:
    st.subheader("③ Recent Signals (last 6 candles = 24h)")
    if signals.empty:
        st.info("No signals logged yet.")
        return

    try:
        recent = signals.tail(6).copy()
        show_cols = [c for c in [
            "candle_close", "price_close", "allocation_raw", "allocation_final",
            "r11_prob_bull", "r11_entropy", "regime_age_log", "stress_score",
            "returns_4h", "volatility_24h", "volume_zscore", "buy_pressure",
            "portfolio_value", "action",
        ] if c in recent.columns]

        fmt = {}
        for c in ["allocation_raw", "allocation_final", "r11_prob_bull", "r11_entropy",
                  "regime_age_log", "returns_4h", "volatility_24h", "volume_zscore",
                  "buy_pressure"]:
            if c in show_cols:
                fmt[c] = "{:.4f}"
        if "price_close" in show_cols:
            fmt["price_close"] = "{:,.2f}"
        if "portfolio_value" in show_cols:
            fmt["portfolio_value"] = "{:,.2f}"
        if "stress_score" in show_cols:
            fmt["stress_score"] = "{:.1f}"

        st.dataframe(
            _dark_table(
                recent[show_cols].style
                .format(fmt)
                .apply(_action_color, axis=1)
            ),
            use_container_width=True,
            height=250,
        )
    except Exception as exc:
        st.warning(f"Recent signals unavailable: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Allocation Signal
# ══════════════════════════════════════════════════════════════════════════════

def section_allocation(signals: pd.DataFrame) -> None:
    st.subheader("④ Allocation Signal History")
    if signals.empty or "allocation_final" not in signals.columns:
        st.info("No allocation data yet.")
        return

    try:
        df = signals.set_index("candle_close")[
            ["allocation_raw", "allocation_final"]
        ].dropna()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df["allocation_raw"],
            name="allocation_raw",
            line=dict(color="#9575CD", width=1.2, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>raw=%{y:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["allocation_final"],
            name="allocation_final",
            line=dict(color=LINE_CLR, width=2),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>final=%{y:.4f}<extra></extra>",
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                      annotation_text="0.50 threshold")
        fig.update_layout(
            yaxis=dict(title="Allocation", range=[0, 1], tickformat=".2f"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=30, b=0),
            height=300,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Allocation chart unavailable: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — Regime Context
# ══════════════════════════════════════════════════════════════════════════════

def section_regime(signals: pd.DataFrame, portfolio: dict) -> None:
    st.subheader("⑤ Regime Context (R11)")
    if signals.empty:
        st.info("No signals yet.")
        return

    try:
        col1, col2, col3 = st.columns(3)
        col1.metric("R11 Regime",     portfolio.get("r11_regime", "n/a"))
        col2.metric("Regime Age",     f"{portfolio.get('regime_age_days', 0)}d")
        col3.metric("Stress Score",   f"{portfolio.get('last_stress_score', 0.0):.1f}")

        df_regime = signals.set_index("candle_close")[
            [c for c in ["r11_prob_bull", "r11_entropy", "regime_age_log"]
             if c in signals.columns]
        ].dropna()

        if not df_regime.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_regime.index, y=df_regime["r11_prob_bull"],
                name="r11_prob_bull",
                line=dict(color=BULL_CLR, width=1.8),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>p_bull=%{y:.4f}<extra></extra>",
            ))
            if "r11_entropy" in df_regime.columns:
                fig.add_trace(go.Scatter(
                    x=df_regime.index, y=df_regime["r11_entropy"],
                    name="r11_entropy",
                    line=dict(color=ORANGE, width=1.2, dash="dot"),
                    yaxis="y2",
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>H=%{y:.4f}<extra>entropy</extra>",
                ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
            fig.update_layout(
                yaxis=dict(title="P(Bull)", range=[0, 1]),
                yaxis2=dict(title="Entropy", overlaying="y", side="right",
                            range=[0, 1], showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=60, t=30, b=0),
                height=280,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Regime chart unavailable: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 6 — Control Layer Status
# ══════════════════════════════════════════════════════════════════════════════

def section_control_layer(signals: pd.DataFrame, portfolio: dict) -> None:
    st.subheader("⑥ Control Layer")
    try:
        cfg = json.loads(CONFIG_JS.read_text()) if CONFIG_JS.exists() else {}
        cl  = cfg.get("control_layer", {})

        # ── Gate status table ─────────────────────────────────────────────────
        rows = []
        for gate_key, label in [
            ("stop_loss",        "Stop Loss"),
            ("stop_gain",        "Stop Gain"),
            ("trailing_stop",    "Trailing Stop"),
            ("regime_gate",      "Regime Gate (R11 Bear filter)"),
            ("timing_gate",      "Timing Gate (RSI/BB/Vol)"),
            ("macro_gate",       "Macro Gate (MOVE)"),
            ("derivatives_gate", "Derivatives Gate (funding/OI)"),
        ]:
            gcfg    = cl.get(gate_key, {})
            enabled = gcfg.get("enabled", False)
            rows.append({
                "Gate":    label,
                "Status":  "ENABLED" if enabled else "disabled",
                "Config":  json.dumps({k: v for k, v in gcfg.items() if k != "enabled"}),
            })
        df_gates = pd.DataFrame(rows)
        st.dataframe(
            _dark_table(df_gates.style),
            use_container_width=True,
            height=200,
        )

        # ── Last regime gate + timing gate values ─────────────────────────────
        if not signals.empty:
            last_sig = signals.iloc[-1]
            rg_col1, rg_col2, rg_col3, rg_col4 = st.columns(4)
            rg_triggered = str(last_sig.get("regime_gate_triggered", "")).lower() in ("true", "1")
            rg_col1.metric("Regime Gate", "BLOCKED" if rg_triggered else "PASS",
                           delta="Bear/uncertain" if rg_triggered else "ok",
                           delta_color="inverse" if rg_triggered else "normal")
            timing_score = last_sig.get("timing_gate_score", None)
            rg_col2.metric("Timing Score",
                           f"{float(timing_score):.3f}" if timing_score not in (None, "") else "n/a")
            rsi_val = last_sig.get("rsi_4h", None)
            rg_col3.metric("RSI 4h",
                           f"{float(rsi_val):.1f}" if rsi_val not in (None, "") else "n/a")
            bb_val = last_sig.get("bb_pct_b", None)
            rg_col4.metric("BB %B",
                           f"{float(bb_val):.3f}" if bb_val not in (None, "") else "n/a")

        # ── Regime gate history ────────────────────────────────────────────────
        if not signals.empty and "regime_gate_triggered" in signals.columns:
            rg_events = signals[
                signals["regime_gate_triggered"].astype(str).isin(["True", "true", "1"])
            ][["candle_close", "price_close", "r11_prob_bull", "r11_entropy",
               "rsi_4h", "bb_pct_b", "timing_gate_score"]]
            st.caption(f"Regime gate blocks: {len(rg_events)}")

        # ── Position / entry tracking ─────────────────────────────────────────
        entry_price = portfolio.get("entry_price")
        entry_time  = portfolio.get("entry_time", "")
        btc_held    = float(portfolio.get("btc_held", 0.0))
        btc_price   = float(portfolio.get("btc_price", 0.0))

        c1, c2, c3, c4 = st.columns(4)
        if entry_price and btc_held > 1e-6:
            ep         = float(entry_price)
            drawdown   = (btc_price - ep) / ep if ep > 0 else 0.0
            sl_pct     = float(cl.get("stop_loss", {}).get("pct", 0.03))
            sl_price   = ep * (1 - sl_pct)
            c1.metric("Entry Price",    f"${ep:,.2f}")
            c2.metric("Stop Level",     f"${sl_price:,.2f}  (-{sl_pct*100:.1f}%)")
            dd_str = f"{drawdown*100:+.2f}%"
            c3.metric("Current Drawdown", dd_str,
                      delta=dd_str,
                      delta_color="inverse")
            c4.metric("Entry Time",     str(entry_time)[:16])
            if drawdown < -sl_pct * 0.8:
                st.warning(f"⚠️ Approaching stop level — drawdown={drawdown*100:.2f}%  "
                           f"stop at {sl_pct*100:.1f}%")

            # ── Trailing stop live tracking ────────────────────────────────────
            ts_cfg        = cl.get("trailing_stop", {})
            max_price_raw = portfolio.get("max_price_since_entry")
            last_mon_raw  = portfolio.get("last_monitor_price")
            if ts_cfg.get("enabled") and max_price_raw:
                max_p   = float(max_price_raw)
                trail_pct     = float(ts_cfg.get("pct", 0.01))
                min_profit    = float(ts_cfg.get("min_profit_to_activate", 0.005))
                profit_from_entry = (max_p - ep) / ep
                trail_trigger     = max_p * (1 - trail_pct)
                armed = profit_from_entry >= min_profit

                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("Max Price (peak)",  f"${max_p:,.2f}",
                           delta=f"+{profit_from_entry*100:.2f}% from entry")
                tc2.metric("Trail Trigger",     f"${trail_trigger:,.2f}  (-{trail_pct*100:.1f}%)")
                tc3.metric("Trail Armed",       "YES" if armed else f"need +{min_profit*100:.1f}%",
                           delta="active" if armed else "not yet",
                           delta_color="normal" if armed else "off")
                if last_mon_raw:
                    last_mon = float(last_mon_raw)
                    dist_pct = (last_mon - max_p) / max_p * 100
                    tc4.metric("Last Monitor Price", f"${last_mon:,.2f}",
                               delta=f"{dist_pct:+.2f}% from peak",
                               delta_color="inverse" if dist_pct < 0 else "normal")
                else:
                    tc4.metric("Last Monitor Price", "—  (no 1h check yet)")
        else:
            c1.metric("Entry Price",      "—  (no position)")
            c2.metric("Stop Level",       "—")
            c3.metric("Current Drawdown", "—")
            c4.metric("Entry Time",       "—")

        # ── Monitor 1h status ─────────────────────────────────────────────────
        monitor_log = ROOT / "logs" / "specialist_4h_monitor.log"
        if monitor_log.exists():
            try:
                last_lines = monitor_log.read_text().strip().split("\n")
                last_monitor = last_lines[-1][:35] if last_lines else "—"
                st.caption(f"Monitor 1h last run: `{last_monitor}`")
            except Exception:
                st.caption("Monitor 1h log unreadable")
        else:
            st.caption("Monitor 1h: log not found (cron not yet running?)")

        # ── Source breakdown (specialist_4h vs monitor_1h) ────────────────────
        if not signals.empty and "source" in signals.columns:
            src_counts = signals["source"].value_counts()
            if not src_counts.empty:
                st.caption("Signal sources: " + "  |  ".join(
                    f"{k}: {v}" for k, v in src_counts.items()
                ))

        # ── Stop loss history ─────────────────────────────────────────────────
        if not signals.empty and "stop_loss_triggered" in signals.columns:
            sl_events = signals[
                signals["stop_loss_triggered"].astype(str).isin(["True", "true", "1"])
            ][["candle_close", "price_close", "entry_price", "drawdown_pct"]]
            if not sl_events.empty:
                st.caption(f"Stop loss events: {len(sl_events)}")
                st.dataframe(
                    _dark_table(sl_events.style.format({
                        "price_close": "{:,.2f}",
                        "entry_price": "{:,.2f}",
                        "drawdown_pct": "{:.4f}",
                    })),
                    use_container_width=True,
                    height=120,
                )
            else:
                st.caption("No stop loss events yet.")

    except Exception as exc:
        st.warning(f"Control layer section unavailable: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 7 — System Health
# ══════════════════════════════════════════════════════════════════════════════

def section_health(signals: pd.DataFrame, portfolio: dict) -> None:
    st.subheader("⑦ System Health")
    try:
        def _check(label: str, value: str, is_ok: bool) -> None:
            icon  = "🟢" if is_ok else "🔴"
            color = BULL_CLR if is_ok else BEAR_CLR
            st.markdown(
                f"<span style='color:{color}'>{icon}</span>&nbsp;"
                f"**{label}**: `{value}`",
                unsafe_allow_html=True,
            )

        # Signals freshness
        if not signals.empty and "candle_close" in signals.columns:
            last_candle = str(signals["candle_close"].max())[:19]
            _check("Last signal", last_candle, not _stale(last_candle))
        else:
            _check("Last signal", "no data", False)

        # Portfolio freshness
        last_upd = str(portfolio.get("last_update", ""))[:19]
        _check("Last portfolio update", last_upd or "never", not _stale(last_upd))

        # File checks
        for label, path in [
            ("SET_B model",    ROOT / json.loads(CONFIG_JS.read_text()).get("model_path", "") if CONFIG_JS.exists() else Path("/dev/null")),
            ("R11 model",      ROOT / json.loads(CONFIG_JS.read_text()).get("r11_model_path", "") if CONFIG_JS.exists() else Path("/dev/null")),
            ("4h OHLCV buffer", STATE_DIR / "ohlcv_4h_buffer.parquet"),
            ("signals.csv",    SIGNALS_CSV),
        ]:
            ok = path.exists()
            _check(label, "✓" if ok else "✗", ok)

        # Signal count
        st.markdown(f"🔵&nbsp;**Signals logged**: `{len(signals)}`")

        # Portfolio JSON
        if portfolio:
            st.markdown("**Portfolio state:**")
            st.json(portfolio)

    except Exception as exc:
        st.warning(f"Health check failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    signals   = load_signals()
    portfolio = load_portfolio()

    section_kpis(signals, portfolio)
    st.divider()

    section_equity(signals)
    st.divider()

    section_recent_signals(signals)
    st.divider()

    section_allocation(signals)
    st.divider()

    section_regime(signals, portfolio)
    st.divider()

    section_control_layer(signals, portfolio)
    st.divider()

    section_health(signals, portfolio)


if __name__ == "__main__":
    main()
