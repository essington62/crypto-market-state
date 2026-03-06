"""
HMM Diagnostic Script — Fase 1A
Analisa onde o modelo erra nos splits 1 e 2.
Saída: data/08_reporting/hmm_diagnostic/
"""

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import timedelta
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ── Constants ──────────────────────────────────────────────────────────────

plt.style.use("dark_background")
OUTPUT_DIR = Path("data/08_reporting/hmm_diagnostic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_COLORS = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}
STATE_LABELS = {0: "Bear", 1: "Lateral", 2: "Bull"}

# ── Load parameters ────────────────────────────────────────────────────────

with open("conf/base/parameters.yml") as f:
    params = yaml.safe_load(f)

wf = params["walkforward"]
hmm_cfg = params["hmm"]
mod_cfg = params["modeling"]["regime_hmm"]

features_cfg: list[str] = hmm_cfg["features"]
n_states: int = mod_cfg["n_states"]
covariance_type: str = mod_cfg["covariance_type"]
n_iter: int = mod_cfg["n_iter"]
random_state: int = mod_cfg["random_state"]
horizon_days: int = wf["horizon_days"]
embargo_days: int = wf["embargo_days"]
purge_total: int = horizon_days + embargo_days

# ── Load data ──────────────────────────────────────────────────────────────

df_raw = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
df_raw = df_raw.sort_index()

features = [f for f in features_cfg if f in df_raw.columns]
df_raw = df_raw.dropna(subset=features)

# Forward 5d return: soma dos próximos horizon_days log returns
df_raw["_forward_ret"] = sum(
    df_raw["log_return"].shift(-k) for k in range(1, horizon_days + 1)
)
# Close proxy para efficiency ratio
df_raw["_close_proxy"] = np.exp(df_raw["log_return"].cumsum())

# ── HMM helpers ────────────────────────────────────────────────────────────

def ordenar_por_drawdown(model: GaussianHMM, features: list[str]) -> dict[int, int]:
    if "drawdown" in features:
        idx = features.index("drawdown")
        vals = {s: model.means_[s][idx] for s in range(model.n_components)}
        ordered = sorted(vals, key=vals.get)
    else:
        idx = features.index("log_return") if "log_return" in features else 0
        ordered = sorted(range(model.n_components), key=lambda s: model.means_[s][idx])
    return {orig: new for new, orig in enumerate(ordered)}


def train_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X_train)
    mapa = ordenar_por_drawdown(model, features)
    return np.array([mapa[s] for s in model.predict(X_test)])

# ── Rolling stats helpers ──────────────────────────────────────────────────

def rolling_efficiency_ratio(close: pd.Series, window: int = 10) -> pd.Series:
    """ER = |close[-1] - close[0]| / sum(|diff(close)|) over rolling window."""
    direction = close.diff(window).abs()
    path = close.diff().abs().rolling(window).sum()
    return (direction / path).where(path > 0)


def _hurst_rs(x: np.ndarray) -> float:
    if len(x) < 10:
        return np.nan
    mean = x.mean()
    devs = x - mean
    cumdev = np.cumsum(devs)
    R = cumdev.max() - cumdev.min()
    S = x.std()
    if S == 0 or R == 0:
        return np.nan
    return np.log(R / S) / np.log(len(x))


def rolling_hurst(series: pd.Series, window: int = 30) -> pd.Series:
    return series.rolling(window).apply(_hurst_rs, raw=True)


def rolling_skew(series: pd.Series, window: int = 30) -> pd.Series:
    return series.rolling(window).skew()


def rolling_kurt(series: pd.Series, window: int = 30) -> pd.Series:
    return series.rolling(window).kurt()

# ── Overlap score (Bhattacharyya approximation) ────────────────────────────

def overlap_score(a: pd.Series, b: pd.Series, n_bins: int = 60) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    return float(np.minimum(ha, hb).sum() * (bins[1] - bins[0]))

# ── State background shading ───────────────────────────────────────────────

def shade_states(ax: plt.Axes, index: pd.DatetimeIndex, states: np.ndarray) -> None:
    """Shade background of ax by HMM state with low alpha."""
    prev_s, start = states[0], index[0]
    for i in range(1, len(states)):
        if states[i] != prev_s or i == len(states) - 1:
            end = index[i]
            ax.axvspan(start, end, color=STATE_COLORS[prev_s], alpha=0.12, linewidth=0)
            prev_s = states[i]
            start = index[i]

# ── Style helper ───────────────────────────────────────────────────────────

def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")

# ── Main diagnostic loop ───────────────────────────────────────────────────

target_splits = {"split_1", "split_2"}

for split_cfg in wf["splits"]:
    name: str = split_cfg["name"]
    if name not in target_splits:
        continue

    train_start = pd.Timestamp(str(split_cfg["train_start"]), tz="UTC")
    train_end   = pd.Timestamp(str(split_cfg["train_end"]),   tz="UTC")
    test_start  = pd.Timestamp(str(split_cfg["test_start"]),  tz="UTC")
    test_end_raw = split_cfg["test_end"]
    test_end = (
        df_raw.index.max()
        if test_end_raw is None
        else pd.Timestamp(str(test_end_raw), tz="UTC")
    )

    purged_train_end = min(train_end, test_start - timedelta(days=purge_total))
    train_df = df_raw.loc[train_start:purged_train_end]
    test_df  = df_raw.loc[test_start:test_end].copy()

    states = train_predict(train_df, test_df)
    test_df["state"] = states
    fwd = test_df["_forward_ret"]
    close_proxy = test_df["_close_proxy"]

    # ── Terminal summary ─────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  {name.upper()} | {test_start.date()} → {test_end.date()}"
          f"  |  n_train={len(train_df)}  n_test={len(test_df)}")
    print(f"{'='*62}")
    print(f"  {'State':<10} {'days':>5} {'%time':>6} {'mean_5d':>8} {'std_5d':>8} {'skew_5d':>8}")
    print(f"  {'-'*50}")
    for s, label in STATE_LABELS.items():
        mask = test_df["state"] == s
        fwd_s = fwd[mask].dropna()
        pct = mask.sum() / len(test_df)
        print(f"  {label:<10} {mask.sum():>5} {pct:>6.1%} "
              f"{fwd_s.mean():>8.4f} {fwd_s.std():>8.4f} {fwd_s.skew():>8.2f}")

    bull_fwd = fwd[test_df["state"] == 2]
    bear_fwd = fwd[test_df["state"] == 0]
    ov = overlap_score(bull_fwd, bear_fwd)
    delta = bull_fwd.mean() - bear_fwd.mean()
    print(f"\n  delta_5d (Bull - Bear) = {delta:.4f}")
    print(f"  Overlap Bull/Bear fwd_5d = {ov:.4f}")

    # ── ANÁLISE 1: Feature distributions by state (boxplot) ─────────────
    ncols = len(features)
    fig, axes = plt.subplots(1, ncols, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    if ncols == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        groups = [test_df.loc[test_df["state"] == s, feat].dropna().values
                  for s in range(n_states)]
        bp = ax.boxplot(groups, patch_artist=True, notch=False,
                        medianprops=dict(color="white", linewidth=2),
                        whiskerprops=dict(color="#aaa"),
                        capprops=dict(color="#aaa"),
                        flierprops=dict(marker="o", color="#aaa", markersize=2, alpha=0.4))
        for patch, s in zip(bp["boxes"], range(n_states)):
            patch.set_facecolor(STATE_COLORS[s])
            patch.set_alpha(0.75)
        ax.set_title(feat, color="white", fontsize=9)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Bear", "Lat", "Bull"], color="white", fontsize=8)
        style_ax(ax)

    fig.suptitle(f"Feature Distributions by State — {name} ({test_start.year})",
                 color="white", fontsize=12)
    plt.tight_layout()
    out = OUTPUT_DIR / f"features_by_state_{name}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out.name}")

    # ── ANÁLISE 2: Rolling statistics + state shading ────────────────────
    lr = test_df["log_return"]
    r_skew  = rolling_skew(lr, 30)
    r_kurt  = rolling_kurt(lr, 30)
    r_er    = rolling_efficiency_ratio(close_proxy, 10)
    r_hurst = rolling_hurst(lr, 30)

    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        style_ax(ax)
        shade_states(ax, test_df.index, states)

    # State sequence
    axes[0].step(test_df.index, test_df["state"], color="white", linewidth=0.8, where="post")
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(["Bear", "Lat", "Bull"], color="white", fontsize=8)
    axes[0].set_ylabel("State")

    axes[1].plot(r_skew.index, r_skew, color="#3498db", linewidth=1)
    axes[1].axhline(0, color="#666", linestyle="--", linewidth=0.7)
    axes[1].set_ylabel("Skew(30d)")

    axes[2].plot(r_kurt.index, r_kurt, color="#9b59b6", linewidth=1)
    axes[2].axhline(3, color="#666", linestyle="--", linewidth=0.7)
    axes[2].set_ylabel("Kurt(30d)")

    axes[3].plot(r_er.index, r_er, color="#e67e22", linewidth=1)
    axes[3].axhline(0.5, color="#666", linestyle="--", linewidth=0.7)
    axes[3].set_ylabel("Eff.Ratio(10d)")

    axes[4].plot(r_hurst.index, r_hurst, color="#1abc9c", linewidth=1)
    axes[4].axhline(0.5, color="#666", linestyle="--", linewidth=0.7)
    axes[4].set_ylabel("Hurst(30d)")

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    fig.suptitle(f"Rolling Stats + HMM States — {name} ({test_start.year})",
                 color="white", fontsize=12)
    plt.tight_layout()
    out = OUTPUT_DIR / f"rolling_stats_{name}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out.name}")

    # ── ANÁLISE 3: Scatter plots in feature space ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        style_ax(ax)

    for s in range(n_states):
        mask = test_df["state"] == s
        sub  = test_df[mask]
        label = STATE_LABELS[s]
        c = STATE_COLORS[s]

        if "drawdown" in sub.columns and "vol_short" in sub.columns:
            axes[0].scatter(sub["drawdown"], sub["vol_short"],
                            c=c, label=label, alpha=0.55, s=18, edgecolors="none")

        if "log_return" in sub.columns and "vol_ratio" in sub.columns:
            axes[1].scatter(sub["log_return"], sub["vol_ratio"],
                            c=c, label=label, alpha=0.55, s=18, edgecolors="none")

    axes[0].set_xlabel("drawdown")
    axes[0].set_ylabel("vol_short")
    axes[0].set_title("Drawdown vs Vol Short", color="white")
    axes[0].legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    axes[1].set_xlabel("log_return")
    axes[1].set_ylabel("vol_ratio")
    axes[1].set_title("Log Return vs Vol Ratio", color="white")
    axes[1].legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    fig.suptitle(f"Feature Space — {name} ({test_start.year})",
                 color="white", fontsize=12)
    plt.tight_layout()
    out = OUTPUT_DIR / f"scatter_{name}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out.name}")

    # ── ANÁLISE 4: Forward return by state over time ──────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    style_ax(ax)
    shade_states(ax, test_df.index, states)

    for s in range(n_states):
        mask = test_df["state"] == s
        ax.scatter(test_df.index[mask], fwd[mask],
                   c=STATE_COLORS[s], label=STATE_LABELS[s],
                   alpha=0.7, s=22, edgecolors="none", zorder=3)

    ax.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Forward Return 5d (log)")
    ax.set_title(f"Forward 5d Return by State — {name} ({test_start.year})",
                 color="white", fontsize=12)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    plt.tight_layout()
    out = OUTPUT_DIR / f"forward_return_by_state_{name}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out.name}")

print(f"\n✓ Diagnóstico completo. Plots em: {OUTPUT_DIR.resolve()}")
