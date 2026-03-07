import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

BASE = Path("data/01_raw/derivatives/coinglass")
FIG  = Path("notebooks/oi_analysis/figures")
FIG.mkdir(parents=True, exist_ok=True)

# ── Carregar dados ────────────────────────────────────────
oi_agg = pd.read_parquet(BASE / "open_interest/BTCUSDT_aggregated.parquet")
fr_oi  = pd.read_parquet(BASE / "funding/BTCUSDT_oi_weighted.parquet")
btc    = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
hmm    = pd.read_parquet("data/05_models/regime_hmm/btc_states.parquet")

# coluna real é 'state', nao 'hmm_state'
hmm = hmm.rename(columns={"state": "hmm_state"})

# ── Montar DataFrame único ────────────────────────────────
df = btc[["log_return"]].copy()
df = df.join(oi_agg[["open_interest_usd"]], how="left")
df = df.join(fr_oi[["funding_rate_oi_weighted"]], how="left")
df = df.join(hmm[["hmm_state"]], how="left")

# Features de análise
df["oi_change_1d"]   = df["open_interest_usd"].pct_change(fill_method=None)
df["oi_change_5d"]   = df["open_interest_usd"].pct_change(5, fill_method=None)
df["oi_zscore_30d"]  = (
    (df["open_interest_usd"] - df["open_interest_usd"].rolling(30).mean()) /
     df["open_interest_usd"].rolling(30).std()
)
df["price_change_5d"] = df["log_return"].rolling(5).sum()
df["return_5d_fwd"]   = df["log_return"].rolling(5).sum().shift(-5)
df["vol_5d_fwd"]      = df["log_return"].rolling(5).std().shift(-5) * np.sqrt(252)
df["funding"]         = df["funding_rate_oi_weighted"]
df["funding_z30"]     = (
    (df["funding"] - df["funding"].rolling(30).mean()) /
     df["funding"].rolling(30).std()
)
df["oi_funding_div"]  = df["oi_zscore_30d"] - df["funding_z30"]

df_all    = df.dropna()
df_post23 = df[df.index >= "2023-01-01"].dropna()
print(f"Dataset completo:  {df_all.shape}")
print(f"Dataset pos-2023:  {df_post23.shape}")


def corr_report(label, x, y, clip_x=None):
    if clip_x:
        x = x.clip(-clip_x, clip_x)
    mask = x.notna() & y.notna()
    r, p = stats.pearsonr(x[mask], y[mask])
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"
    print(f"  {label}: r={r:+.3f}  p={p:.4f}  {sig}")
    return r, p


# ════════════════════════════════════════════════════════════
# TESTE 2 — OI vs Volatilidade Futura
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TESTE 2 — OI vs Volatilidade Futura")
print("=" * 60)

for label, dset in [("2020+", df_all), ("Pos-2023", df_post23)]:
    print(f"\n{label}:")
    corr_report("OI change 1d  -> vol_5d_fwd",
                dset["oi_change_1d"], dset["vol_5d_fwd"], 0.5)
    corr_report("OI change 1d (shift+1) -> vol_5d_fwd",
                dset["oi_change_1d"].shift(1), dset["vol_5d_fwd"], 0.5)
    corr_report("OI zscore 30d -> vol_5d_fwd",
                dset["oi_zscore_30d"], dset["vol_5d_fwd"], 3)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for col, (dset, dlabel) in enumerate([(df_all, "2020+"), (df_post23, "Pos-2023")]):
    bull = dset[dset["hmm_state"] == 1]
    bear = dset[dset["hmm_state"] == 0]
    for row, (xcol, xlabel) in enumerate([
        ("oi_change_1d",  "OI Change 1d"),
        ("oi_zscore_30d", "OI zscore 30d"),
    ]):
        ax = axes[row][col]
        ax.scatter(bull[xcol].clip(-0.2, 0.2), bull["vol_5d_fwd"],
                   alpha=0.3, s=10, color="green", label="Bull")
        ax.scatter(bear[xcol].clip(-0.2, 0.2), bear["vol_5d_fwd"],
                   alpha=0.3, s=10, color="red",   label="Bear")
        x = dset[xcol].clip(-0.3, 0.3)
        y = dset["vol_5d_fwd"]
        mask = x.notna() & y.notna()
        m, b = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xs, m * xs + b, "k--", linewidth=1.5)
        ax.axvline(0, color="grey", linewidth=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Vol 5d fwd")
        ax.set_title(f"{xlabel} vs Vol Futura ({dlabel})")
        ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIG / "t2_oi_vs_vol.png", dpi=100)
plt.close()
print("-> Salvo: t2_oi_vs_vol.png")

# ════════════════════════════════════════════════════════════
# TESTE 3 — OI em Bull vs Bear
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TESTE 3 — Distribuicao OI por Regime HMM")
print("=" * 60)

for label, dset in [("2020+", df_all), ("Pos-2023", df_post23)]:
    bull = dset[dset["hmm_state"] == 1]["open_interest_usd"]
    bear = dset[dset["hmm_state"] == 0]["open_interest_usd"]
    t, p = stats.ttest_ind(bull, bear)
    print(f"\n{label}:")
    print(f"  Bull OI medio: ${bull.mean()/1e9:.2f}B  |  mediana: ${bull.median()/1e9:.2f}B")
    print(f"  Bear OI medio: ${bear.mean()/1e9:.2f}B  |  mediana: ${bear.median()/1e9:.2f}B")
    print(f"  Razao bull/bear: {bull.mean()/bear.mean():.2f}x")
    print(f"  t-test p={p:.4f}  {'significativo' if p < 0.05 else 'nao significativo'}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (dset, label) in zip(axes, [(df_all, "2020+"), (df_post23, "Pos-2023")]):
    data = [dset[dset["hmm_state"] == 0]["open_interest_usd"] / 1e9,
            dset[dset["hmm_state"] == 1]["open_interest_usd"] / 1e9]
    bp = ax.boxplot(data, labels=["Bear", "Bull"], patch_artist=True, notch=True)
    bp["boxes"][0].set_facecolor("salmon")
    bp["boxes"][1].set_facecolor("lightgreen")
    ax.set_ylabel("OI (USD Bilhoes)")
    ax.set_title(f"OI por Regime HMM ({label})")
    for i, (d, c) in enumerate(zip(data, ["red", "green"])):
        ax.scatter(np.random.normal(i + 1, 0.04, len(d)), d,
                   alpha=0.2, s=4, color=c)
plt.tight_layout()
plt.savefig(FIG / "t3_oi_by_regime.png", dpi=100)
plt.close()
print("-> Salvo: t3_oi_by_regime.png")

# ════════════════════════════════════════════════════════════
# TESTE 4 — Divergência Preço vs OI
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TESTE 4 — Divergencia Preco vs OI (quadrantes)")
print("=" * 60)

dset = df_post23.copy()
oi_thresh_low  = -0.02
oi_thresh_high =  0.05
pr_thresh      =  0.0

quadrants = {
    "Case 1 — Tendencia  (price+ OI+)":
        (dset["price_change_5d"] > pr_thresh) & (dset["oi_change_5d"].between(oi_thresh_low, oi_thresh_high)),
    "Case 2 — Short Squeeze (price+ OI-)":
        (dset["price_change_5d"] > pr_thresh) & (dset["oi_change_5d"] < oi_thresh_low),
    "Case 3 — Topo potencial (price+ OI++)":
        (dset["price_change_5d"] > pr_thresh) & (dset["oi_change_5d"] > oi_thresh_high),
    "Case 4 — Nova baixa   (price- OI+)":
        (dset["price_change_5d"] < pr_thresh) & (dset["oi_change_5d"] > oi_thresh_low),
    "Case 5 — Capitulacao  (price- OI-)":
        (dset["price_change_5d"] < pr_thresh) & (dset["oi_change_5d"] < oi_thresh_low),
}

for case, mask in quadrants.items():
    sub = dset[mask]
    if len(sub) >= 5:
        fwd  = sub["return_5d_fwd"].mean()
        vol  = sub["vol_5d_fwd"].mean()
        bull = (sub["hmm_state"] == 1).mean()
        print(f"\n  {case}")
        print(f"    n={len(sub):3d} | return_5d={fwd:+.4f} | vol_5d={vol:.4f} | bull%={bull:.1%}")

fig, ax = plt.subplots(figsize=(12, 8))
sc = ax.scatter(
    dset["oi_change_5d"].clip(-0.3, 0.3),
    dset["price_change_5d"],
    c=dset["return_5d_fwd"], cmap="RdYlGn",
    alpha=0.6, s=20, vmin=-0.15, vmax=0.15
)
plt.colorbar(sc, ax=ax, label="Return 5d forward")
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.axvline(oi_thresh_high, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xlabel("OI Change 5d")
ax.set_ylabel("Price Change 5d")
ax.set_title("Divergencia Preco vs OI — colorido por retorno futuro 5d (pos-2023)")
ax.text( 0.06,  0.08, "Topo potencial\n(price+ OI++)", fontsize=8, color="darkred")
ax.text(-0.28,  0.08, "Short Squeeze\n(price+ OI-)",   fontsize=8, color="blue")
ax.text( 0.02, -0.08, "Nova Baixa\n(price- OI+)",      fontsize=8, color="red")
ax.text(-0.28, -0.08, "Capitulacao\n(price- OI-)",     fontsize=8, color="orange")
plt.tight_layout()
plt.savefig(FIG / "t4_price_oi_divergence.png", dpi=100)
plt.close()
print("\n-> Salvo: t4_price_oi_divergence.png")

# ════════════════════════════════════════════════════════════
# TESTE 5 — OI zscore por Quintil → retorno futuro
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TESTE 5 — OI zscore por Quintil -> retorno futuro")
print("=" * 60)

dset = df_post23.copy()
dset["oi_q"] = pd.qcut(dset["oi_zscore_30d"], q=5,
                        labels=["Q1\nmuito baixo", "Q2", "Q3", "Q4", "Q5\nmuito alto"])

stats_q = dset.groupby("oi_q", observed=True)["return_5d_fwd"].agg(
    n="count", mean="mean", std="std", median="median"
).round(4)
print(stats_q.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
means  = stats_q["mean"]
stds   = stats_q["std"]
colors = ["green" if m > 0 else "red" for m in means]

ax = axes[0]
ax.bar(range(len(means)), means, yerr=stds / 2,
       color=colors, alpha=0.7, capsize=5, width=0.6)
ax.set_xticks(range(len(means)))
ax.set_xticklabels(stats_q.index, fontsize=8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Return 5d medio")
ax.set_title("OI zscore Quintil -> Return 5d (pos-2023)")

ax = axes[1]
data_q = [dset[dset["oi_q"] == q]["return_5d_fwd"].dropna()
          for q in stats_q.index]
bp = ax.boxplot(data_q, patch_artist=True, notch=False)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.5)
ax.set_xticklabels(stats_q.index, fontsize=8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Distribuicao Return 5d por Quintil OI")
plt.tight_layout()
plt.savefig(FIG / "t5_oi_quintil_return.png", dpi=100)
plt.close()
print("-> Salvo: t5_oi_quintil_return.png")

# ════════════════════════════════════════════════════════════
# TESTE 6 — OI/Funding divergência
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TESTE 6 — OI/Funding divergencia -> retorno e volatilidade")
print("=" * 60)

dset = df_post23.dropna(subset=["oi_funding_div"]).copy()
dset["div_q"] = pd.qcut(dset["oi_funding_div"], q=5,
    labels=["Q1\nfunding dom.", "Q2", "Q3", "Q4", "Q5\nOI dom."])

for col, label in [("return_5d_fwd", "Return 5d"), ("vol_5d_fwd", "Vol 5d")]:
    print(f"\n{label} por quintil OI/Funding divergencia:")
    g = dset.groupby("div_q", observed=True)[col].agg(
        n="count", mean="mean", std="std").round(4)
    print(g.to_string())

for period, dset2 in [("2020+", df_all), ("Pos-2023", df_post23)]:
    print(f"\n{period}:")
    corr_report("oi_funding_div -> return_5d_fwd",
                dset2["oi_funding_div"], dset2["return_5d_fwd"], 3)
    corr_report("oi_funding_div -> vol_5d_fwd",
                dset2["oi_funding_div"], dset2["vol_5d_fwd"], 3)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
g_ret = dset.groupby("div_q", observed=True)["return_5d_fwd"].mean()
g_vol = dset.groupby("div_q", observed=True)["vol_5d_fwd"].mean()

for ax, vals, title, base_color in zip(
    axes,
    [g_ret, g_vol],
    ["OI/Funding Div -> Return 5d (pos-2023)",
     "OI/Funding Div -> Vol 5d (pos-2023)"],
    [None, "darkred"]
):
    cs = ["green" if v > 0 else "red" for v in vals] if base_color is None \
         else [base_color] * 5
    ax.bar(range(len(vals)), vals, color=cs, alpha=0.7, width=0.6)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(vals.index, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
plt.tight_layout()
plt.savefig(FIG / "t6_oi_funding_div.png", dpi=100)
plt.close()
print("-> Salvo: t6_oi_funding_div.png")

# ── Resumo final ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMO — figuras salvas em:", FIG)
print("=" * 60)
for f in sorted(FIG.glob("t[2-6]*.png")):
    print(f"  {f.name}")
