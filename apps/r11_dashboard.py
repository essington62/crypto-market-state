"""
R11 BTC Regime Dashboard — Streamlit daily monitoring app.

Run:
    streamlit run apps/r11_dashboard.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "scripts" / "paper_trading" / "state"

EQUITY_CSV  = STATE_DIR / "equity_curve.csv"
TRADE_LOG   = STATE_DIR / "trade_log.csv"
POSITION_JS = STATE_DIR / "position.json"
MODEL_PKL   = STATE_DIR / "r11_hmm_model.pkl"
L3_PATH     = ROOT / "data" / "03_primary" / "spot" / "daily" / "BTCUSDT.parquet"

R11_FEATURES = ["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d"]
PERIODS_YR   = 365
BT_MAXDD     = -0.0987
DD_ALERT     = -0.12

BULL_BG  = "rgba(46,125,50,0.08)"
BEAR_BG  = "rgba(198,40,40,0.08)"
BULL_CLR = "#2E7D32"
BEAR_CLR = "#C62828"
LINE_CLR = "#1565C0"


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="R11 BTC Regime Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("R11 BTC Regime Dashboard")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    st.button("🔄 Refresh", use_container_width=True)
    st.markdown("---")
    st.markdown("**Config**")
    st.code(
        "Features: 6\nPosition: Bull→75% Bear→0%\nData: L3 BTCUSDT",
        language=None,
    )
    st.markdown("---")
    st.caption(f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")


# ══════════════════════════════════════════════════════════════════════════
# Loaders
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_equity() -> pd.DataFrame:
    if not EQUITY_CSV.exists() or EQUITY_CSV.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(EQUITY_CSV, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    for c in ["portfolio_value", "daily_ret", "position_pct", "p_bull"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=300)
def load_trades() -> pd.DataFrame:
    if not TRADE_LOG.exists() or TRADE_LOG.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(TRADE_LOG, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    for c in ["btc_qty", "fill_price", "usdt_value", "fees_usdt"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=300)
def load_l3() -> pd.DataFrame:
    df = pd.read_parquet(L3_PATH)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


@st.cache_data(ttl=300)
def load_position() -> dict:
    if not POSITION_JS.exists():
        return {}
    try:
        return json.loads(POSITION_JS.read_text())
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _compute_metrics(df: pd.DataFrame) -> dict:
    """Sharpe, CAGR, MaxDD from equity DataFrame. Returns NaN if insufficient data."""
    if len(df) < 2:
        return dict(sharpe=float("nan"), cagr=float("nan"), maxdd=float("nan"), n=len(df))
    rets    = df["daily_ret"].dropna().values
    pv      = df["portfolio_value"]
    mu, std = rets.mean(), rets.std(ddof=1)
    sharpe  = float((mu / max(std, 1e-10)) * np.sqrt(PERIODS_YR)) if std > 0 else float("nan")
    n_yr    = len(df) / PERIODS_YR
    tot     = pv.iloc[-1] / pv.iloc[0] - 1.0
    cagr    = float((1 + tot) ** (1 / max(n_yr, 1e-6)) - 1)
    dd_s    = pv / pv.cummax() - 1.0
    maxdd   = float(dd_s.min())
    return dict(sharpe=sharpe, cagr=cagr, maxdd=maxdd, n=len(df),
                cur_dd=float(dd_s.iloc[-1]))


def _regime_shapes(equity: pd.DataFrame, price_min: float, price_max: float) -> list[dict]:
    """Build Plotly vrect shapes from equity regime column."""
    shapes = []
    if equity.empty or "regime" not in equity.columns:
        return shapes
    dates   = equity.index.tolist()
    regimes = equity["regime"].tolist()
    i = 0
    while i < len(regimes):
        r     = regimes[i]
        color = BULL_BG if r == "Bull" else BEAR_BG
        j     = i
        while j < len(regimes) and regimes[j] == r:
            j += 1
        x0 = str(dates[i])[:10]
        x1 = str(dates[j - 1])[:10]
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=x0, x1=x1, y0=0, y1=1,
            fillcolor=color, line_width=0, layer="below",
        ))
        i = j
    return shapes


_TABLE_CELL = {
    "background-color": "#0e0e0e",
    "color":            "#00ff88",
    "border-color":     "#1a1a1a",
}
_TABLE_STYLES = [
    {"selector": "th", "props": [
        ("background-color", "#111111"),
        ("color",            "#00ff88"),
        ("border-color",     "#1a1a1a"),
        ("font-weight",      "600"),
    ]},
    {"selector": "td", "props": [
        ("background-color", "#0e0e0e"),
        ("color",            "#00ff88"),
        ("border-color",     "#1a1a1a"),
    ]},
]


def _style_dark_table(styler: "pd.io.formats.style.Styler") -> "pd.io.formats.style.Styler":
    """Apply green-on-black terminal style to any pandas Styler."""
    return styler.set_properties(**_TABLE_CELL).set_table_styles(_TABLE_STYLES)


def _trade_side_color(row: "pd.Series") -> list[str]:
    """
    Color SELL rows with muted green text on very dark background.
    BUY rows keep the standard bright green.
    No pink/red backgrounds — both sides are fully readable.
    """
    if row.get("side") == "BUY":
        return [f"background-color: #0e0e0e; color: #00ff88"] * len(row)
    if row.get("side") == "SELL":
        return [f"background-color: #0e0e0e; color: #66ccaa"] * len(row)
    return [f"background-color: #0e0e0e; color: #00ff88"] * len(row)


def _stale(date_val, max_days: int = 2) -> bool:
    """Return True if date_val is older than max_days."""
    try:
        dt = pd.Timestamp(date_val)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return (pd.Timestamp.now(tz="UTC") - dt).days > max_days
    except Exception:
        return True


# ══════════════════════════════════════════════════════════════════════════
# Section 1 — KPI Row
# ══════════════════════════════════════════════════════════════════════════

def section_kpis(equity: pd.DataFrame, position: dict) -> None:
    st.subheader("① Model Status")

    if not equity.empty:
        last       = equity.iloc[-1]
        regime     = str(last.get("regime", "n/a"))
        p_bull     = float(last.get("p_bull", float("nan")))
        pos_pct    = float(last.get("position_pct", float("nan")))
        pv         = float(last.get("portfolio_value", float("nan")))
        dret       = float(last.get("daily_ret", float("nan")))
        metrics    = _compute_metrics(equity)
        sharpe     = metrics["sharpe"] if metrics["n"] >= 20 else float("nan")
        maxdd      = metrics["maxdd"]
        cur_dd     = metrics.get("cur_dd", float("nan"))
    else:
        regime = p_bull = pos_pct = pv = dret = float("nan")
        sharpe = maxdd = cur_dd = float("nan")
        regime = "n/a"

    regime_delta = "🟢 Bull" if regime == "Bull" else "🔴 Bear" if regime == "Bear" else "—"

    cols = st.columns(7)
    with cols[0]:
        st.metric("Regime", regime,
                  delta=None,
                  delta_color="normal")
        if regime == "Bull":
            st.markdown("<span style='color:#2E7D32;font-size:1.3em'>●</span>", unsafe_allow_html=True)
        elif regime == "Bear":
            st.markdown("<span style='color:#C62828;font-size:1.3em'>●</span>", unsafe_allow_html=True)

    with cols[1]:
        st.metric("P(Bull)", f"{p_bull:.3f}" if not np.isnan(p_bull) else "n/a")

    with cols[2]:
        st.metric("Position", f"{pos_pct*100:.0f}%" if not np.isnan(pos_pct) else "n/a")

    with cols[3]:
        st.metric("Portfolio", f"${pv:,.0f}" if not np.isnan(pv) else "n/a")

    with cols[4]:
        dret_str   = f"{dret*100:+.2f}%" if not np.isnan(dret) else "n/a"
        dret_delta = f"{dret*100:+.2f}%" if not np.isnan(dret) else None
        st.metric("Daily Return", dret_str,
                  delta=dret_delta,
                  delta_color="normal")

    with cols[5]:
        if np.isnan(sharpe):
            st.metric("Live Sharpe", "n/a", delta="need 20d")
        else:
            st.metric("Live Sharpe", f"{sharpe:+.3f}",
                      delta=f"{sharpe - 0.775:+.3f} vs bt",
                      delta_color="normal")

    with cols[6]:
        if np.isnan(maxdd):
            st.metric("Live MaxDD", "n/a")
        else:
            st.metric("Live MaxDD", f"{maxdd*100:.2f}%",
                      delta=f"{(maxdd - BT_MAXDD)*100:+.2f}% vs bt",
                      delta_color="inverse")

    if not np.isnan(maxdd) and maxdd <= DD_ALERT:
        st.error(f"⚠️ MaxDD ALERT: {maxdd*100:.2f}%  (threshold {DD_ALERT*100:.0f}%) — review position sizing")


# ══════════════════════════════════════════════════════════════════════════
# Section 2 — BTC Price + Regime + Trades
# ══════════════════════════════════════════════════════════════════════════

def section_price_chart(l3: pd.DataFrame, equity: pd.DataFrame, trades: pd.DataFrame) -> None:
    st.subheader("② BTC Price · Regime Background · Trades")
    try:
        window = l3.tail(120).copy()
        shapes = _regime_shapes(
            equity,
            price_min=float(window["close"].min()),
            price_max=float(window["close"].max()),
        )

        fig = go.Figure()

        # BTC close line
        fig.add_trace(go.Scatter(
            x=window.index, y=window["close"],
            mode="lines", name="BTC Close",
            line=dict(color=LINE_CLR, width=1.6),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
        ))

        # Trades overlay
        if not trades.empty and "fill_price" in trades.columns:
            cutoff = window.index[0]
            t_win  = trades[pd.to_datetime(trades["timestamp"], utc=True) >= cutoff].copy()
            for side, color, symbol in [("BUY", BULL_CLR, "triangle-up"),
                                         ("SELL", BEAR_CLR, "triangle-down")]:
                t_side = t_win[t_win["side"] == side]
                if t_side.empty:
                    continue
                hover = (
                    t_side.apply(
                        lambda r: (
                            f"{str(r['timestamp'])[:10]}<br>"
                            f"Price: ${float(r.get('fill_price',0)):,.2f}<br>"
                            f"Qty: {float(r.get('btc_qty',0)):.6f} BTC<br>"
                            f"Value: ${float(r.get('usdt_value',0)):,.2f}"
                        ),
                        axis=1,
                    ).tolist()
                )
                fig.add_trace(go.Scatter(
                    x=t_side["timestamp"],
                    y=t_side["fill_price"],
                    mode="markers",
                    name=side,
                    marker=dict(symbol=symbol, size=14, color=color,
                                line=dict(width=1, color="white")),
                    hovertext=hover,
                    hoverinfo="text",
                ))

        fig.update_layout(
            shapes=shapes,
            xaxis_title="Date",
            yaxis_title="Price (USDT)",
            yaxis_tickprefix="$",
            yaxis_tickformat=",.0f",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=0, r=0, t=30, b=0),
            height=420,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Price chart unavailable: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# Section 3 — Feature Snapshot
# ══════════════════════════════════════════════════════════════════════════

def section_features(l3: pd.DataFrame) -> None:
    st.subheader("③ Feature Snapshot")
    try:
        feat_avail = [f for f in R11_FEATURES if f in l3.columns]
        feat_miss  = [f for f in R11_FEATURES if f not in l3.columns]
        if feat_miss:
            st.warning(f"Features missing from L3: {feat_miss}")
        if not feat_avail:
            st.info("No R11 features found in L3 dataset.")
            return

        snap     = l3[feat_avail].dropna(how="all").tail(5)
        last_row = snap.iloc[-1]

        col_bar, col_tbl = st.columns([1, 2])

        with col_bar:
            colors = [BULL_CLR if v >= 0 else BEAR_CLR for v in last_row.values]
            fig = go.Figure(go.Bar(
                x=last_row.values,
                y=last_row.index.tolist(),
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}: %{x:.6f}<extra></extra>",
            ))
            fig.update_layout(
                title="Latest Feature Values",
                xaxis_title="Value",
                margin=dict(l=0, r=0, t=40, b=0),
                height=280,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_tbl:
            st.caption("Last 5 trading days — R11 model inputs")
            st.dataframe(
                _style_dark_table(snap.style.format("{:.6f}")),
                use_container_width=True,
            )

        st.caption("These are the inputs that drove today's regime decision.")
    except Exception as exc:
        st.warning(f"Feature snapshot unavailable: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# Section 4 — Trade History
# ══════════════════════════════════════════════════════════════════════════

def section_trades(trades: pd.DataFrame) -> None:
    st.subheader("④ Trade History")
    try:
        if trades.empty:
            st.info("No trades executed yet.")
            return

        n_buy   = int((trades["side"] == "BUY").sum())
        n_sell  = int((trades["side"] == "SELL").sum())
        last_ts = str(trades["timestamp"].max())[:10]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", len(trades))
        col2.metric("BUY",  n_buy)
        col3.metric("SELL", n_sell)
        col4.metric("Last Trade", last_ts)

        show_cols = [c for c in ["timestamp", "side", "btc_qty", "fill_price",
                                  "usdt_value", "note"] if c in trades.columns]
        fmt = {}
        if "btc_qty"    in show_cols: fmt["btc_qty"]    = "{:.6f}"
        if "fill_price" in show_cols: fmt["fill_price"] = "{:,.2f}"
        if "usdt_value" in show_cols: fmt["usdt_value"] = "{:,.2f}"

        st.dataframe(
            _style_dark_table(
                trades[show_cols].tail(30).style
                .format(fmt)
                .apply(_trade_side_color, axis=1)
            ),
            use_container_width=True,
            height=300,
        )
    except Exception as exc:
        st.warning(f"Trade history unavailable: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# Section 5 — Portfolio Performance
# ══════════════════════════════════════════════════════════════════════════

def section_equity(equity: pd.DataFrame) -> None:
    st.subheader("⑤ Portfolio Performance")
    try:
        if equity.empty:
            st.info("No equity data yet. Run the paper trader first.")
            return

        pv      = equity["portfolio_value"]
        start   = float(pv.iloc[0])
        end     = float(pv.iloc[-1])
        tot_ret = (end / start - 1.0) * 100

        roll_ret = equity["daily_ret"].rolling(7, min_periods=1).mean() * 100

        fig = go.Figure()

        # Portfolio value
        fig.add_trace(go.Scatter(
            x=pv.index, y=pv.values,
            name="Portfolio",
            line=dict(color=LINE_CLR, width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
        ))
        fig.add_hline(y=start, line_dash="dash", line_color="gray",
                      annotation_text=f"Start ${start:,.0f}", annotation_position="right")

        # Rolling return on secondary axis
        fig.add_trace(go.Scatter(
            x=roll_ret.index, y=roll_ret.values,
            name="7d Rolling Ret %",
            line=dict(color="#FB8C00", width=1.2, dash="dot"),
            yaxis="y2",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>7d Ret</extra>",
        ))

        fig.add_annotation(
            x=pv.index[-1], y=end,
            text=f" Total: {tot_ret:+.1f}%",
            showarrow=False, font=dict(size=12, color=BULL_CLR if tot_ret >= 0 else BEAR_CLR),
            xanchor="left",
        )

        fig.update_layout(
            yaxis=dict(title="Portfolio (USDT)", tickprefix="$", tickformat=",.0f"),
            yaxis2=dict(title="7d Ret %", overlaying="y", side="right",
                        ticksuffix="%", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=60, t=30, b=0),
            height=380,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Equity chart unavailable: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# Section 6 — Drawdown
# ══════════════════════════════════════════════════════════════════════════

def section_drawdown(equity: pd.DataFrame) -> None:
    st.subheader("⑥ Drawdown")
    try:
        if equity.empty:
            st.info("No equity data yet.")
            return

        pv    = equity["portfolio_value"]
        dd    = (pv / pv.cummax() - 1.0) * 100
        maxdd = float(dd.min())
        curdd = float(dd.iloc[-1])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode="lines", name="Drawdown",
            line=dict(color=BEAR_CLR, width=1.5),
            fill="tozeroy", fillcolor="rgba(198,40,40,0.18)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
        ))
        fig.add_hline(y=BT_MAXDD * 100, line_dash="dash", line_color="#FB8C00",
                      annotation_text=f"Backtest MaxDD {BT_MAXDD*100:.2f}%",
                      annotation_position="right")
        fig.add_hline(y=DD_ALERT * 100, line_dash="dash", line_color=BEAR_CLR,
                      annotation_text="Alert −12%",
                      annotation_position="right")
        fig.update_layout(
            yaxis=dict(title="Drawdown (%)", ticksuffix="%"),
            margin=dict(l=0, r=80, t=10, b=0),
            height=300,
            showlegend=False,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max Drawdown",   f"{maxdd:.2f}%")
        c2.metric("Current DD",     f"{curdd:.2f}%")
        c3.metric("Backtest target",f"{BT_MAXDD*100:.2f}%")
        c4.metric("Alert threshold",f"{DD_ALERT*100:.0f}%")

        if curdd <= DD_ALERT * 100:
            st.error(f"⚠️ Drawdown {curdd:.2f}% breaches alert threshold {DD_ALERT*100:.0f}%")
    except Exception as exc:
        st.warning(f"Drawdown chart unavailable: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# Section 7 — Regime Probability
# ══════════════════════════════════════════════════════════════════════════

def section_regime_prob(equity: pd.DataFrame) -> None:
    st.subheader("⑦ Regime Probability")
    try:
        if equity.empty or "p_bull" not in equity.columns:
            st.info("p_bull column not available in equity_curve.csv yet.")
            return

        pb = equity["p_bull"].dropna()
        if pb.empty:
            st.info("p_bull column present but contains no data yet.")
            return

        fig = go.Figure()

        # Fill above 0.5 (bull zone)
        fig.add_trace(go.Scatter(
            x=pb.index, y=pb.values,
            mode="lines", name="P(Bull)",
            line=dict(color=BULL_CLR, width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>P(Bull)=%{y:.4f}<extra></extra>",
        ))
        # Shaded regions
        bull_y = pb.where(pb >= 0.5, 0.5)
        bear_y = pb.where(pb < 0.5,  0.5)

        fig.add_trace(go.Scatter(
            x=pb.index, y=bull_y.values,
            fill="tonexty", fillcolor="rgba(46,125,50,0.15)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
            name="_bull_fill",
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                      annotation_text="0.50", annotation_position="right")
        fig.update_layout(
            yaxis=dict(title="P(Bull)", range=[0, 1], tickformat=".2f"),
            margin=dict(l=0, r=60, t=10, b=0),
            height=300,
            showlegend=False,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Regime probability chart unavailable: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# Section 8 — System Health
# ══════════════════════════════════════════════════════════════════════════

def section_health(l3: pd.DataFrame, equity: pd.DataFrame, trades: pd.DataFrame) -> None:
    st.subheader("⑧ System Health")
    try:
        now = pd.Timestamp.now(tz="UTC")

        def _check(label: str, value: str, is_ok: bool) -> None:
            icon  = "🟢" if is_ok else "🔴"
            color = BULL_CLR if is_ok else BEAR_CLR
            st.markdown(
                f"<span style='color:{color}'>{icon}</span>&nbsp;"
                f"**{label}**: `{value}`",
                unsafe_allow_html=True,
            )

        # L3 last date
        l3_last = str(l3.index[-1])[:10] if not l3.empty else "unknown"
        _check("Last L3 data date",  l3_last,  not _stale(l3_last))

        # Equity last update
        if not equity.empty:
            eq_last = str(equity.index[-1])[:10]
            _check("Last equity update", eq_last, not _stale(eq_last))
        else:
            _check("Last equity update", "no file", False)

        # Last trade
        if not trades.empty and "timestamp" in trades.columns:
            tr_last = str(trades["timestamp"].max())[:10]
            _check("Last trade", tr_last, not _stale(tr_last, max_days=7))
        else:
            _check("Last trade", "no trades", True)   # not an error — just no trades yet

        # Trade count
        n_trades = len(trades)
        st.markdown(f"🔵&nbsp;**Trades executed**: `{n_trades}`")

        # File checks
        for label, path in [
            ("Position file (position.json)", POSITION_JS),
            ("Model pickle (r11_hmm_model.pkl)", MODEL_PKL),
        ]:
            exists = path.exists()
            icon   = "🟢" if exists else "🔴"
            mark   = "✓" if exists else "✗"
            color  = BULL_CLR if exists else BEAR_CLR
            st.markdown(
                f"<span style='color:{color}'>{icon}</span>&nbsp;"
                f"**{label}**: `{mark}`",
                unsafe_allow_html=True,
            )

        # Position JSON content
        pos = load_position()
        if pos:
            st.markdown("**Position state:**")
            st.json(pos)

    except Exception as exc:
        st.warning(f"Health check failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    equity   = load_equity()
    trades   = load_trades()
    position = load_position()

    try:
        l3 = load_l3()
    except Exception as exc:
        st.error(f"Failed to load L3 data from {L3_PATH}: {exc}")
        st.stop()

    section_kpis(equity, position)
    st.divider()

    section_price_chart(l3, equity, trades)
    st.divider()

    section_features(l3)
    st.divider()

    section_trades(trades)
    st.divider()

    section_equity(equity)
    st.divider()

    section_drawdown(equity)
    st.divider()

    section_regime_prob(equity)
    st.divider()

    section_health(l3, equity, trades)


if __name__ == "__main__":
    main()
