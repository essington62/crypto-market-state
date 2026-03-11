"""
R11 Backtest — equity curve, drawdown e métricas.

Gera sinais R11 via walk-forward (4 splits), aplica custo de 10bps por trade.
Regra de execução: signal_t → entra em t+1, holding 5 dias → return_5d[t].

NAO modifica pipeline nem sobrescreve modelos.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

TC         = 0.0010        # 10 bps por trade
PERIODS_YR = 252 / 5      # períodos de 5d por ano
FIG_DIR    = Path("notebooks/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Dados ─────────────────────────────────────────────────────
btc = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
hmm = pd.read_parquet("data/05_models/regime_hmm/btc_states.parquet")
hmm = hmm.rename(columns={"state": "hmm_state"})

df = btc.copy()
df = df.join(hmm[["hmm_state", "bull_prob"]], how="left")
df["regime_strength"] = (df["bull_prob"] - 0.5).abs()
df["return_5d"]   = df["log_return"].rolling(5).sum().shift(-5)
df["label_5d"]    = (df["return_5d"] > 0).astype(int)
df["target_norm"] = df["return_5d"] / df["vol_short"].replace(0, np.nan)


# ── Features ──────────────────────────────────────────────────
TOP10 = [c for c in [
    "atr_14_norm", "bb_width_20d", "vol_ratio", "roc_21",
    "slope_21d", "bb_position", "price_vs_high_30d",
    "range_position_30d", "drawdown", "macd_hist_norm",
] if c in df.columns]

SPLITS = [
    {"train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start":  "2024-07-01", "test_end":  "2024-12-31"},
    {"train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  "2025-03-31"},
    {"train_start": "2023-01-01", "train_end": "2025-03-31",
     "test_start":  "2025-04-01", "test_end":  None},
]


# ── Model factories ────────────────────────────────────────────
def fit_r5(X, y):
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        gamma=0.5, eval_metric="logloss", random_state=42, verbosity=0,
    )
    clf.fit(X, y, verbose=False)
    return clf


def fit_r7b(X, y):
    reg = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=1,
        gamma=0, random_state=42, verbosity=0,
    )
    reg.fit(X, y, verbose=False)
    return reg


# ── Geração de sinais walk-forward ────────────────────────────
print("Gerando sinais R11 (walk-forward 4 splits)...")
records = []
for sp in SPLITS:
    test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
    aux      = ["label_5d", "target_norm", "return_5d"]
    cols     = list(dict.fromkeys(TOP10 + aux))

    train = df[sp["train_start"]:sp["train_end"]][cols].dropna(
        subset=TOP10 + ["label_5d", "target_norm", "return_5d"])
    test  = df[sp["test_start"]:test_end][cols].dropna(
        subset=TOP10 + ["return_5d"])

    if len(train) < 50 or len(test) < 10:
        continue

    clf     = fit_r5(train[TOP10], train["label_5d"])
    prob    = clf.predict_proba(test[TOP10])[:, 1]
    sig_r5  = np.where(prob > 0.60, 1, np.where(prob < 0.40, -1, 0))

    reg        = fit_r7b(train[TOP10], train["target_norm"])
    pred       = reg.predict(test[TOP10])
    train_pred = reg.predict(train[TOP10])
    p_long     = np.percentile(train_pred, 70)
    p_sht      = np.percentile(train_pred, 30)
    sig_r7b    = np.where(pred > p_long, 1, np.where(pred < p_sht, -1, 0))

    sigs                                    = np.zeros(len(sig_r5), dtype=int)
    sigs[(sig_r5 == 1)  & (sig_r7b == 1)]  = 1
    sigs[(sig_r5 == -1) & (sig_r7b == -1)] = -1

    for date, ret, sig in zip(test.index, test["return_5d"].values, sigs):
        records.append({"date": date, "return_5d": float(ret), "signal_r11": int(sig)})

    n_long  = int((sigs == 1).sum())
    n_short = int((sigs == -1).sum())
    print(f"  {sp['test_start']}→{test_end}: n={len(test)} "
          f"long={n_long} short={n_short} neut={len(test)-n_long-n_short}")

bt = (pd.DataFrame(records)
        .set_index("date")
        .sort_index()
        [~pd.DataFrame(records).set_index("date").sort_index().index.duplicated(keep="last")])

print(f"\nBacktest: {len(bt)} observações | "
      f"{bt.index.min().date()} → {bt.index.max().date()}")


# ── Backtest ──────────────────────────────────────────────────
sigs = bt["signal_r11"].values
rets = bt["return_5d"].values

# Retorno bruto da estratégia
strat_ret = np.where(sigs == 1, rets, np.where(sigs == -1, -rets, 0.0))

# Custo de transação: 10bps quando posição muda
prev_sigs  = np.concatenate([[0], sigs[:-1]])
trade_flag = (sigs != prev_sigs).astype(float)
tc_cost    = TC * trade_flag

net_ret = strat_ret - tc_cost

# BTC buy-and-hold (sem custo) para referência
bnh_ret = rets  # sempre long

# Equity curves (cumulativo, base 1)
equity_r11 = pd.Series((1 + net_ret).cumprod(), index=bt.index, name="R11")
equity_bnh = pd.Series((1 + bnh_ret).cumprod(), index=bt.index, name="BTC B&H")

# Drawdown
def drawdown_series(equity):
    peak = equity.cummax()
    return (equity - peak) / peak

dd_r11 = drawdown_series(equity_r11)
dd_bnh = drawdown_series(equity_bnh)


# ── Métricas ──────────────────────────────────────────────────
def compute_metrics(net_ret_arr, equity_s, label=""):
    n          = len(net_ret_arr)
    n_years    = n / PERIODS_YR
    final_eq   = float(equity_s.iloc[-1])
    cagr       = final_eq ** (1 / n_years) - 1
    sharpe     = (net_ret_arr.mean() / (net_ret_arr.std() + 1e-10)) * np.sqrt(PERIODS_YR)
    max_dd     = float(drawdown_series(equity_s).min())
    active     = net_ret_arr[net_ret_arr != 0]
    hit_rate   = float((active > 0).mean()) if len(active) > 0 else float("nan")
    gains      = active[active > 0].sum()
    losses     = abs(active[active < 0].sum())
    prof_factor = gains / losses if losses > 0 else float("nan")
    n_trades   = int((np.diff(np.concatenate([[0], (net_ret_arr != 0).astype(int)])) != 0).sum())
    return {
        "label":         label,
        "CAGR":          cagr,
        "Sharpe":        sharpe,
        "Max Drawdown":  max_dd,
        "Hit Rate":      hit_rate,
        "Profit Factor": prof_factor,
        "n_trades":      n_trades,
        "n_obs":         n,
    }

m_r11 = compute_metrics(net_ret, equity_r11, "R11 Ensemble+TOP10")
m_bnh = compute_metrics(bnh_ret, equity_bnh, "BTC Buy & Hold")


# ── Tabela de métricas ────────────────────────────────────────
print("\n" + "=" * 55)
print("MÉTRICAS")
print("=" * 55)
print(f"{'Métrica':<20} {'R11':>15} {'BTC B&H':>15}")
print("-" * 55)
for key in ["CAGR", "Sharpe", "Max Drawdown", "Hit Rate", "Profit Factor"]:
    v_r11 = m_r11[key]
    v_bnh = m_bnh[key]
    if key in ("CAGR", "Max Drawdown", "Hit Rate"):
        fmt = lambda v: f"{v:>+15.2%}" if not np.isnan(v) else f"{'NaN':>15}"
    elif key == "Profit Factor":
        fmt = lambda v: f"{v:>15.3f}" if not np.isnan(v) else f"{'NaN':>15}"
    else:
        fmt = lambda v: f"{v:>+15.3f}"
    print(f"{key:<20} {fmt(v_r11)} {fmt(v_bnh)}")
print("-" * 55)
print(f"{'n_obs':<20} {m_r11['n_obs']:>15} {m_bnh['n_obs']:>15}")
print(f"{'n_trades':<20} {m_r11['n_trades']:>15} {'—':>15}")
print(f"{'TC total (bps)':<20} {tc_cost.sum() * 10000:>14.1f} {'—':>15}")


# ── Plots ─────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                gridspec_kw={"height_ratios": [3, 1.5]},
                                sharex=True)
fig.suptitle("R11 Backtest — Ensemble+TOP10 (4 splits, 10bps/trade)", fontsize=12)

# Equity curve
ax1.plot(equity_r11.index, equity_r11.values, color="steelblue",
         linewidth=1.2, label=f"R11  CAGR={m_r11['CAGR']:+.1%}  Sharpe={m_r11['Sharpe']:+.2f}")
ax1.plot(equity_bnh.index, equity_bnh.values, color="gray",
         linewidth=0.8, linestyle="--", alpha=0.7,
         label=f"BTC B&H  CAGR={m_bnh['CAGR']:+.1%}")
ax1.axhline(1, color="black", linewidth=0.5, linestyle=":")
ax1.set_ylabel("Equity (base 1)")
ax1.legend(fontsize=9)
ax1.set_title("Equity Curve")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}x"))

# Split boundaries
split_dates = [pd.Timestamp(sp["test_start"]) for sp in SPLITS]
for d in split_dates:
    ax1.axvline(d, color="orange", linewidth=0.7, linestyle="--", alpha=0.6)
    ax2.axvline(d, color="orange", linewidth=0.7, linestyle="--", alpha=0.6)

# Drawdown
ax2.fill_between(dd_r11.index, dd_r11.values, 0,
                 color="steelblue", alpha=0.4, label="R11")
ax2.fill_between(dd_bnh.index, dd_bnh.values, 0,
                 color="gray", alpha=0.2, label="BTC B&H")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_ylabel("Drawdown")
ax2.set_title(f"Drawdown  (R11 max={m_r11['Max Drawdown']:+.2%}  |  BTC max={m_bnh['Max Drawdown']:+.2%})")
ax2.legend(fontsize=9)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=30)

plt.tight_layout()
out_fig = FIG_DIR / "r11_backtest.png"
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigura salva: {out_fig}")
