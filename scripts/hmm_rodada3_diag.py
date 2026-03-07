"""
HMM Rodada 3: covariance_type="diag" vs "full".
Script exploratório — NAO modifica pipeline nem sobrescreve btc_states.parquet.
"""
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

BASE = Path("data/01_raw/derivatives/coinglass")

# Carregar dados
btc = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
fr  = pd.read_parquet(BASE / "funding/BTCUSDT_oi_weighted.parquet")
ls  = pd.read_parquet(BASE / "long_short_ratio/BTCUSDT_top_positions.parquet")

# Calcular features derivadas
fr["funding_zscore_30d"] = (
    (fr["funding_rate_oi_weighted"] -
     fr["funding_rate_oi_weighted"].rolling(30).mean()) /
     fr["funding_rate_oi_weighted"].rolling(30).std()
)
ratio_col = [c for c in ls.columns if "ratio" in c.lower()][0]
ls = ls.rename(columns={ratio_col: "top_position_ratio"})

df = btc.join(fr[["funding_zscore_30d"]], how="left")
df = df.join(ls[["top_position_ratio"]], how="left")

FEATURES_BASE = ["log_return", "vol_short", "vol_ratio",
                 "drawdown", "volume_z", "slope_21d"]
FEATURES_F    = FEATURES_BASE + ["funding_zscore_30d"]
FEATURES_FT   = FEATURES_BASE + ["funding_zscore_30d", "top_position_ratio"]

SPLITS = [
    {"train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start":  "2024-07-01", "test_end":  "2024-12-31"},
    {"train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  None},
]


def run_walkforward(df, features, cov_type="full", label=""):
    results = []
    for i, sp in enumerate(SPLITS):
        train    = df[sp["train_start"]:sp["train_end"]][features].dropna()
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        test     = df[sp["test_start"]:test_end][features].dropna()

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(train)
        X_test  = scaler.transform(test)

        model = GaussianHMM(n_components=2, covariance_type=cov_type,
                            n_iter=1000, random_state=42)
        model.fit(X_train)

        feat_idx  = features.index("drawdown")
        order     = np.argsort(model.means_[:, feat_idx])
        state_map = {order[j]: j for j in range(2)}
        states    = np.array([state_map[s] for s in model.predict(X_test)])

        test_ret = df[sp["test_start"]:test_end]["log_return"].reindex(test.index)
        ret_5d   = test_ret.rolling(5).sum().shift(-5)

        bull_ret = ret_5d[states == 1].mean()
        bear_ret = ret_5d[states == 0].mean()
        delta    = bull_ret - bear_ret

        results.append({
            "split":    f"split_{i+1}",
            "n_train":  len(train),
            "n_test":   len(test),
            "delta_5d": delta,
        })
        print(f"  {label} split_{i+1}: "
              f"n_train={len(train):3d}  n_test={len(test):3d}  "
              f"delta={delta:+.4f}")
    return pd.DataFrame(results)


# ── Rodar 4 configurações ─────────────────────────────────────
configs = [
    (FEATURES_BASE, "full", "A — baseline full (6f)     "),
    (FEATURES_BASE, "diag", "B — baseline diag (6f)     "),
    (FEATURES_F,    "diag", "C — diag + funding (7f)    "),
    (FEATURES_FT,   "diag", "D — diag + funding+top (8f)"),
]

all_results = {}
for features, cov, label in configs:
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    try:
        all_results[label.strip()] = run_walkforward(
            df, features, cov_type=cov, label=label[:4]
        )
    except Exception as e:
        print(f"  ERRO: {e}")
        all_results[label.strip()] = None

# ── Tabela comparativa ────────────────────────────────────────
valid_labels = [c[2].strip() for c in configs if all_results.get(c[2].strip()) is not None]

print("\n" + "=" * 70)
print("TABELA COMPARATIVA — 4 CONFIGURACOES")
print("=" * 70)

header = f"{'Split':<10}" + "".join(f"{l[:22]:>24}" for l in valid_labels)
print(header)
print("-" * 70)

for i in range(3):
    row = f"split_{i+1:<5}"
    for label in valid_labels:
        d = all_results[label]["delta_5d"].iloc[i]
        row += f"{d:>+24.4f}"
    print(row)

print("-" * 70)
row = f"{'Media':<10}"
means = {}
for label in valid_labels:
    m = all_results[label]["delta_5d"].mean()
    means[label] = m
    row += f"{m:>+24.4f}"
print(row)

# ── Veredicto ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VEREDICTO")
print("=" * 70)
base_label = valid_labels[0]
base_mean  = means[base_label]
for label in valid_labels[1:]:
    diff      = means[label] - base_mean
    splits_ok = (all_results[label]["delta_5d"].values >
                 all_results[base_label]["delta_5d"].values).sum()
    status = "APROVADO" if splits_ok >= 2 and diff > 0 else "REJEITADO"
    print(f"  {label[:35]}: diff={diff:+.4f}  splits={splits_ok}/3  {status}")

best = max(means, key=means.get)
print(f"\n  Melhor configuracao: {best}")
print(f"  Delta medio:         {means[best]:+.4f}")
