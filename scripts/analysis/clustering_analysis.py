"""
Clustering Analysis — BTC Natural Regimes
============================================
Determines how many natural regimes exist in BTC data
to validate n_states for HMM v2.

Run from project root:
  conda activate crypto_market_state
  python scripts/analysis/clustering_analysis.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import yaml
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

plt.style.use("dark_background")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BTC_PATH     = PROJECT_ROOT / "data/03_primary/spot/daily/BTCUSDT.parquet"
OUT_DIR      = PROJECT_ROOT / "scripts/analysis/output"

TRAIN_START = pd.Timestamp("2023-01-01", tz="UTC")
TODAY       = pd.Timestamp.now(tz="UTC").normalize()

FEATURES = ["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d"]

EVENTS_TS = {
    pd.Timestamp("2024-01-11", tz="UTC"): "ETF approved",
    pd.Timestamp("2024-04-20", tz="UTC"): "Halving",
    pd.Timestamp("2025-01-20", tz="UTC"): "Trump inauguration",
}

CLUSTER_COLORS = ["#4dff88", "#ff4d4d", "#4da6ff", "#ff8c00", "#ff88ff", "#ffff44"]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tbl(rows, headers, title=None):
    if title:
        print(f"\n{'─' * 72}")
        print(f"  {title}")
        print(f"{'─' * 72}")
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="simple"))
    else:
        print("  " + "  ".join(str(h) for h in headers))
        for row in rows:
            print("  " + "  ".join(str(v) for v in row))
    print()


def _fmt(v, pct=False, dec=4):
    if pd.isna(v):
        return "N/A"
    if pct:
        return f"{v*100:+.3f}%"
    return f"{v:+.{dec}f}"


def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def _label_clusters(cluster_profiles):
    """Assign Bull/Bear/Sideways labels based on fwd_5d and vol_short."""
    profiles = cluster_profiles.copy()
    if "fwd_5d" not in profiles.columns or "vol_short" not in profiles.columns:
        return {i: f"C{i}" for i in profiles.index}

    labels = {}
    sorted_by_fwd = profiles["fwd_5d"].sort_values(ascending=False)

    if len(profiles) == 2:
        labels[sorted_by_fwd.index[0]]  = "Bull"
        labels[sorted_by_fwd.index[-1]] = "Bear"
    elif len(profiles) == 3:
        labels[sorted_by_fwd.index[0]]  = "Bull"
        labels[sorted_by_fwd.index[1]]  = "Sideways"
        labels[sorted_by_fwd.index[-1]] = "Bear"
    else:
        for rank, idx in enumerate(sorted_by_fwd.index):
            if rank == 0:
                labels[idx] = "Bull"
            elif rank == len(profiles) - 1:
                labels[idx] = "Bear"
            else:
                labels[idx] = f"Trans{rank}"

    return labels


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Setup and data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    print("Section 0: Loading data...")

    if not BTC_PATH.exists():
        sys.exit(f"[ERROR] BTC parquet not found: {BTC_PATH}")

    btc_raw = pd.read_parquet(BTC_PATH)
    btc_raw.index = pd.to_datetime(btc_raw.index, utc=True)

    # Filter to training period
    mask = (btc_raw.index >= TRAIN_START) & (btc_raw.index <= TODAY)
    df   = btc_raw.loc[mask].copy()

    # Keep available feature columns
    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  [WARN] Missing features: {missing} — will be skipped")

    # Also keep close for forward returns and plotting
    keep = list(dict.fromkeys(["close"] + available))
    df   = df[keep].copy()

    # Forward returns for cluster characterization
    df["fwd_5d"]  = df["close"].pct_change(5).shift(-5)
    df["fwd_21d"] = df["close"].pct_change(21).shift(-21)

    # Drop rows where core features are NaN
    df_clean = df.dropna(subset=available)

    print(f"  Rows  : {len(df_clean)}  ({df_clean.index[0].date()} → {df_clean.index[-1].date()})")
    print(f"  Features used: {available}\n")

    # Scale features
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[available])
    X_df     = pd.DataFrame(X_scaled, index=df_clean.index, columns=available)

    return df_clean, X_df, available, scaler


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Optimal number of clusters
# ──────────────────────────────────────────────────────────────────────────────

def section1_optimal_clusters(X_df):
    print("Section 1: Optimal number of clusters...")

    X = X_df.values
    k_range = range(2, 7)

    km_inertia   = []
    km_silhouette = []
    km_db        = []
    gmm_bic      = []
    gmm_aic      = []
    gmm_ll       = []

    for k in k_range:
        # K-Means
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        km_inertia.append(km.inertia_)
        km_silhouette.append(silhouette_score(X, labels))
        km_db.append(davies_bouldin_score(X, labels))

        # GMM
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=42, n_init=5)
        gmm.fit(X)
        gmm_bic.append(gmm.bic(X))
        gmm_aic.append(gmm.aic(X))
        gmm_ll.append(gmm.score(X) * len(X))

    # ── Table ─────────────────────────────────────────────────────────────────
    rows = []
    for i, k in enumerate(k_range):
        rows.append([k,
                     f"{km_silhouette[i]:.4f}",
                     f"{km_db[i]:.4f}",
                     f"{km_inertia[i]:.1f}",
                     f"{gmm_bic[i]:.1f}",
                     f"{gmm_aic[i]:.1f}"])

    _tbl(rows,
         ["k", "KMeans_silhouette", "KMeans_DB", "KMeans_inertia",
          "GMM_BIC", "GMM_AIC"],
         title="Cluster Evaluation Metrics")

    best_k_sil = list(k_range)[np.argmax(km_silhouette)]
    best_k_bic = list(k_range)[np.argmin(gmm_bic)]
    consensus  = best_k_sil if best_k_sil == best_k_bic else best_k_bic  # BIC preferred

    print(f"  Optimal k (silhouette): {best_k_sil}")
    print(f"  Optimal k (BIC)       : {best_k_bic}")
    print(f"  Consensus             : {consensus}")
    print()

    # ── Plot ──────────────────────────────────────────────────────────────────
    k_list = list(k_range)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax2r = ax.twinx()
    ax.plot(k_list, km_silhouette, color="#4dff88", marker="o", linewidth=1.5,
            label="Silhouette ↑")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score", color="#4dff88")
    ax2r.plot(k_list, km_inertia, color="#ff8c00", marker="s", linewidth=1.5,
              linestyle="--", label="Inertia ↓")
    ax2r.set_ylabel("Inertia", color="#ff8c00")
    ax.axvline(best_k_sil, color="#4dff88", linewidth=0.8, linestyle=":",
               alpha=0.8)
    ax.set_title("K-Means: Elbow + Silhouette")
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8)

    ax = axes[1]
    ax.plot(k_list, gmm_bic, color="#4da6ff", marker="o", linewidth=1.5,
            label="BIC ↓")
    ax.plot(k_list, gmm_aic, color="#ff88ff", marker="s", linewidth=1.5,
            linestyle="--", label="AIC ↓")
    ax.axvline(best_k_bic, color="#4da6ff", linewidth=0.8, linestyle=":",
               alpha=0.8)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Score (lower = better)")
    ax.set_title("GMM: BIC + AIC")
    ax.legend(fontsize=8)

    fig.suptitle("Optimal Number of Clusters / HMM States", y=1.01)
    fig.tight_layout()
    _save(fig, "optimal_clusters.png")

    return consensus, best_k_sil, best_k_bic


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Characterize clusters
# ──────────────────────────────────────────────────────────────────────────────

def section2_cluster_profiles(df, X_df, available, consensus_k):
    print(f"Section 2: Characterizing {consensus_k} clusters...")

    X = X_df.values
    km = KMeans(n_clusters=consensus_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    df = df.copy()
    df["cluster"] = labels

    rows = []
    profile_data = {}

    for c in range(consensus_k):
        sub = df[df["cluster"] == c]
        pct = len(sub) / len(df)
        m   = {feat: sub[feat].mean() for feat in available if feat in sub.columns}
        m["fwd_5d"]  = sub["fwd_5d"].mean()
        m["fwd_21d"] = sub["fwd_21d"].mean()
        profile_data[c] = m

        row = [f"C{c}"]
        row += [_fmt(m.get(f, np.nan), pct=True) if f in ["log_return","slope_21d","drawdown"]
                else _fmt(m.get(f, np.nan)) for f in available]
        row += [_fmt(m["fwd_5d"], pct=True),
                _fmt(m["fwd_21d"], pct=True),
                len(sub),
                f"{pct:.0%}"]
        rows.append(row)

    profiles_df = pd.DataFrame(profile_data).T
    cluster_labels = _label_clusters(profiles_df)

    headers = ["Cluster"] + available + ["fwd_5d", "fwd_21d", "days", "%"]
    _tbl(rows, headers, title=f"Cluster Profiles (k={consensus_k})")

    # Print labels
    for c, lbl in cluster_labels.items():
        p = profile_data[c]
        print(f"  C{c} → {lbl:10s}  fwd_5d={p['fwd_5d']*100:+.2f}%  "
              f"vol={p.get('vol_short', np.nan):+.5f}  "
              f"ret={p.get('log_return', np.nan)*100:+.3f}%")
    print()

    return df, cluster_labels


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Temporal distribution & transition matrix
# ──────────────────────────────────────────────────────────────────────────────

def section3_temporal(df, cluster_labels, consensus_k):
    print("Section 3: Temporal distribution and transition matrix...")

    # ── Plot color-coded BTC price ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    for c in range(consensus_k):
        sub = df[df["cluster"] == c]
        lbl = cluster_labels.get(c, f"C{c}")
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        ax.scatter(sub.index, sub["close"], s=4, color=color,
                   label=f"C{c}: {lbl}", alpha=0.7, zorder=2)

    # Connect dots with thin line
    ax.plot(df.index, df["close"], color="#333333", linewidth=0.5, zorder=1)

    colors_ev = ["cyan", "#ff8c00", "#ff4d4d"]
    for i, (ts, lbl) in enumerate(EVENTS_TS.items()):
        ax.axvline(ts, color=colors_ev[i % len(colors_ev)], linewidth=1.0,
                   linestyle="--", alpha=0.7, label=lbl)

    ax.set_ylabel("BTC Close (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title(f"BTC Cluster Assignments Over Time (k={consensus_k})")
    ax.legend(fontsize=8, markerscale=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    fig.tight_layout()
    _save(fig, "cluster_timeline.png")

    # ── Transition matrix ─────────────────────────────────────────────────────
    labels_seq = df["cluster"].values
    trans = np.zeros((consensus_k, consensus_k), dtype=int)
    for i in range(len(labels_seq) - 1):
        trans[labels_seq[i], labels_seq[i + 1]] += 1

    # Normalize rows to probabilities
    trans_prob = trans / (trans.sum(axis=1, keepdims=True) + 1e-9)

    print("  Cluster transition matrix (row=from, col=to):")
    header = ["From↓ To→"] + [f"C{c}" for c in range(consensus_k)]
    rows = []
    for i in range(consensus_k):
        row = [f"C{i} ({cluster_labels.get(i,'?')})"]
        row += [f"{trans_prob[i, j]:.3f}" for j in range(consensus_k)]
        rows.append(row)
    _tbl(rows, header, title="Markov Transition Probabilities")

    # Check Markov assumption: high diagonal = persistence = valid for HMM
    diag_mean = np.mean([trans_prob[i, i] for i in range(consensus_k)])
    print(f"  Mean self-transition probability: {diag_mean:.3f}")
    if diag_mean > 0.7:
        print("  → Strong state persistence — Markov assumption VALID for HMM ✓")
    else:
        print("  → Low state persistence — HMM may struggle to learn stable regimes ✗")
    print()

    return trans_prob


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PCA visualization
# ──────────────────────────────────────────────────────────────────────────────

def section4_pca(df, X_df, available, cluster_labels):
    print("Section 4: PCA visualization...")

    X = X_df.values
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    ev = pca.explained_variance_ratio_
    comp = pca.components_

    print(f"  PC1: {ev[0]:.1%} variance  |  PC2: {ev[1]:.1%} variance")
    print(f"  Total: {ev.sum():.1%}")

    for i, (pc, label) in enumerate(zip(comp, ["PC1", "PC2"])):
        top = np.argsort(np.abs(pc))[::-1][:3]
        drivers = ", ".join([f"{available[j]}({pc[j]:+.3f})" for j in top])
        print(f"  {label} main drivers: {drivers}")
    print()

    # ── Scatter plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))

    clusters = df["cluster"].values
    for c in np.unique(clusters):
        mask  = clusters == c
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        lbl   = cluster_labels.get(c, f"C{c}")
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=12,
                   color=color, label=f"C{c}: {lbl}", alpha=0.6)

    # Feature loading arrows
    scale = 3.0
    for j, feat in enumerate(available):
        ax.annotate("", xy=(comp[0, j] * scale, comp[1, j] * scale),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="yellow",
                                   lw=1.2, alpha=0.7))
        ax.text(comp[0, j] * scale * 1.1, comp[1, j] * scale * 1.1,
                feat, color="yellow", fontsize=7, ha="center")

    ax.axhline(0, color="white", linewidth=0.3)
    ax.axvline(0, color="white", linewidth=0.3)
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} variance)")
    ax.set_title("PCA: BTC Regime Clusters (feature arrows = loadings)")
    ax.legend(fontsize=9, markerscale=2)
    fig.tight_layout()
    _save(fig, "cluster_pca.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Stability across periods
# ──────────────────────────────────────────────────────────────────────────────

def section5_stability(df, available, consensus_k, scaler):
    print("Section 5: Cluster stability across periods...")

    periods = {
        "2023"   : ("2023-01-01", "2023-12-31"),
        "2024"   : ("2024-01-01", "2024-12-31"),
        "2025+"  : ("2025-01-01", str(TODAY.date())),
    }

    period_profiles = {}
    for pname, (s, e) in periods.items():
        mask = (df.index >= pd.Timestamp(s, tz="UTC")) & \
               (df.index <= pd.Timestamp(e, tz="UTC"))
        sub  = df.loc[mask].dropna(subset=available)
        if len(sub) < consensus_k * 10:
            print(f"  {pname}: insufficient data ({len(sub)} rows)")
            continue

        X_p      = scaler.transform(sub[available])
        km_p     = KMeans(n_clusters=consensus_k, random_state=42, n_init=10)
        lbl_p    = km_p.fit_predict(X_p)
        centers  = km_p.cluster_centers_   # in scaled space

        # Re-invert to original scale for readability
        centers_orig = scaler.inverse_transform(centers)
        rows = []
        for c in range(consensus_k):
            row = [f"C{c}"]
            row += [f"{centers_orig[c, j]:+.5f}" for j in range(len(available))]
            rows.append(row)

        _tbl(rows, ["Cluster"] + available,
             title=f"Period {pname} — cluster means (original scale)")
        period_profiles[pname] = centers_orig

    # Stability check: do vol_short and log_return clusters preserve sign?
    stable = True
    if len(period_profiles) >= 2:
        pnames = list(period_profiles.keys())
        for feat_idx, feat in enumerate(available):
            if feat not in ["log_return", "vol_short"]:
                continue
            vals_by_period = [
                sorted(period_profiles[p][:, feat_idx].tolist())
                for p in pnames if p in period_profiles
            ]
            if len(vals_by_period) < 2:
                continue
            # Check if the ordering of clusters is consistent
            signs_p0 = [np.sign(v) for v in vals_by_period[0]]
            for vp in vals_by_period[1:]:
                signs_pn = [np.sign(v) for v in vp]
                if signs_p0 != signs_pn:
                    stable = False

    stability_label = "STABLE ✓" if stable else "UNSTABLE ✗"
    print(f"  Cluster structure across periods: {stability_label}")
    if not stable:
        print("  → Consider shorter retraining windows for HMM")
    print()

    return stable


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — HMM recommendation
# ──────────────────────────────────────────────────────────────────────────────

def section6_recommendation(df, cluster_labels, consensus_k,
                             best_k_sil, best_k_bic, stable):
    print("Section 6: HMM recommendation...")

    # Collect cluster profiles
    cluster_summary = []
    for c, lbl in cluster_labels.items():
        sub = df[df["cluster"] == c]
        fwd5  = sub["fwd_5d"].mean()
        vol   = sub["vol_short"].mean() if "vol_short" in sub.columns else np.nan
        ret   = sub["log_return"].mean() if "log_return" in sub.columns else np.nan
        cluster_summary.append((c, lbl, fwd5, vol, ret))

    print()
    print("═" * 54)
    print("  CLUSTERING RECOMMENDATION FOR HMM")
    print("═" * 54)
    print(f"  Recommended n_states : {consensus_k}")
    print(f"  Evidence:")
    print(f"    Silhouette peaks at k={best_k_sil}")
    print(f"    GMM BIC minimum at  k={best_k_bic}")
    print(f"    Interpretability    : {'/'.join(cluster_labels.values())}")
    print()
    print(f"  Cluster profiles:")
    for c, lbl, fwd5, vol, ret in cluster_summary:
        vol_str = f"vol={vol:+.5f}" if not pd.isna(vol) else "vol=N/A"
        fwd_str = f"fwd_5d={fwd5*100:+.2f}%" if not pd.isna(fwd5) else "fwd_5d=N/A"
        print(f"    State {c} ({lbl:10s}): {fwd_str}  {vol_str}")
    print()
    if not stable:
        print("  ⚠  WARNING: Cluster structure changes across periods.")
        print("     HMM may need shorter retraining window (≤12 months).")
    else:
        print("  ✓  Cluster structure is stable across 2023/2024/2025.")
    print("═" * 54)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Clustering Analysis — BTC Natural Regimes")
    print(f"  Period : {TRAIN_START.date()} → {TODAY.date()}")
    print("=" * 72)
    print()

    df, X_df, available, scaler = load_data()

    consensus_k, best_k_sil, best_k_bic = section1_optimal_clusters(X_df)

    df, cluster_labels = section2_cluster_profiles(df, X_df, available, consensus_k)

    trans_prob = section3_temporal(df, cluster_labels, consensus_k)

    section4_pca(df, X_df, available, cluster_labels)

    stable = section5_stability(df, available, consensus_k, scaler)

    section6_recommendation(df, cluster_labels, consensus_k,
                            best_k_sil, best_k_bic, stable)

    # Save scores CSV
    out_csv = OUT_DIR / "clustering_results.csv"
    summary_rows = []
    for c, lbl in cluster_labels.items():
        sub = df[df["cluster"] == c]
        row = {"cluster": c, "label": lbl, "n_days": len(sub),
               "pct": len(sub) / len(df)}
        for feat in available:
            if feat in sub.columns:
                row[f"mean_{feat}"] = sub[feat].mean()
        row["mean_fwd_5d"]  = sub["fwd_5d"].mean()
        row["mean_fwd_21d"] = sub["fwd_21d"].mean()
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"Cluster profiles saved → {out_csv}")

    print("=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
