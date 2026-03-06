"""
HMM Ablation Study — Fase 1A
Testa combinações de features ortogonais e reporta delta_5d por split.
Não modifica nenhum arquivo de pipeline, catalog ou dataset.
"""

import numpy as np
import pandas as pd
import yaml
from datetime import timedelta
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ── Load parameters ────────────────────────────────────────────────────────

with open("conf/base/parameters.yml") as f:
    params = yaml.safe_load(f)

wf = params["walkforward"]
mod = params["modeling"]["regime_hmm"]

N_STATES: int = mod["n_states"]
COV_TYPE: str = mod["covariance_type"]
N_ITER: int = mod["n_iter"]
RANDOM_STATE: int = mod["random_state"]
HORIZON: int = wf["horizon_days"]
EMBARGO: int = wf["embargo_days"]
PURGE_TOTAL: int = HORIZON + EMBARGO

# ── Load data ──────────────────────────────────────────────────────────────

df_raw = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
df_raw = df_raw.sort_index()
df_raw["_fwd"] = sum(df_raw["log_return"].shift(-k) for k in range(1, HORIZON + 1))

# ── Feature sets ───────────────────────────────────────────────────────────

BASE = ["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z"]

FEATURE_SETS = {
    "BASE":    BASE,
    "A":       BASE + ["skew_30d"],
    "B":       BASE + ["kurt_30d"],
    "C":       BASE + ["hurst_30d"],
    "D":       BASE + ["er_10d"],
    "E":       BASE + ["skew_30d", "hurst_30d"],
    "F":       BASE + ["hurst_30d", "er_10d"],
    "G":       BASE + ["skew_30d", "kurt_30d", "hurst_30d", "er_10d"],
}

# ── HMM helpers ────────────────────────────────────────────────────────────

def _order_by_drawdown(model: GaussianHMM, features: list[str]) -> dict[int, int]:
    n = model.n_components
    if "drawdown" in features:
        idx = features.index("drawdown")
        vals = {s: model.means_[s][idx] for s in range(n)}
        ordered = sorted(vals, key=vals.get)
    else:
        idx = features.index("log_return") if "log_return" in features else 0
        ordered = sorted(range(n), key=lambda s: model.means_[s][idx])
    return {orig: new for new, orig in enumerate(ordered)}


def run_split(df: pd.DataFrame, features: list[str],
              train_start: pd.Timestamp, purged_train_end: pd.Timestamp,
              test_start: pd.Timestamp, test_end: pd.Timestamp) -> float:
    """Train HMM on train slice, predict on test slice, return delta_5d."""
    train_df = df.loc[train_start:purged_train_end].dropna(subset=features)
    test_df  = df.loc[test_start:test_end].dropna(subset=features)

    if len(train_df) < 30 or len(test_df) < 5:
        return np.nan

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test  = scaler.transform(test_df[features])

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type=COV_TYPE,
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train)

    mapa = _order_by_drawdown(model, features)
    states = np.array([mapa[s] for s in model.predict(X_test)])

    fwd = test_df["_fwd"]
    E_bull = fwd[states == 2].mean()
    E_bear = fwd[states == 0].mean()
    return float(E_bull - E_bear)


def run_walkforward(features: list[str]) -> dict[str, float]:
    """Run all 3 splits for a given feature set. Returns {split_name: delta}."""
    results = {}
    for cfg in wf["splits"]:
        train_start = pd.Timestamp(str(cfg["train_start"]), tz="UTC")
        train_end   = pd.Timestamp(str(cfg["train_end"]),   tz="UTC")
        test_start  = pd.Timestamp(str(cfg["test_start"]),  tz="UTC")
        test_end_raw = cfg["test_end"]
        test_end = (
            df_raw.index.max()
            if test_end_raw is None
            else pd.Timestamp(str(test_end_raw), tz="UTC")
        )
        purged_train_end = min(train_end, test_start - timedelta(days=PURGE_TOTAL))
        results[cfg["name"]] = run_split(
            df_raw, features,
            train_start, purged_train_end,
            test_start, test_end,
        )
    return results

# ── Run ablation ───────────────────────────────────────────────────────────

rows = []
for name, feats in FEATURE_SETS.items():
    available = [f for f in feats if f in df_raw.columns]
    missing   = [f for f in feats if f not in df_raw.columns]
    if missing:
        print(f"[SKIP] {name}: features não disponíveis: {missing}")
        continue

    deltas = run_walkforward(available)
    d1 = deltas.get("split_1", np.nan)
    d2 = deltas.get("split_2", np.nan)
    d3 = deltas.get("split_3", np.nan)
    mean_d = float(np.nanmean([d1, d2, d3]))
    n_pos  = sum(1 for d in [d1, d2, d3] if not np.isnan(d) and d > 0.01)

    rows.append({
        "conjunto":         name,
        "features":         "+".join(f for f in available if f not in BASE) or "—",
        "split_1_delta":    d1,
        "split_2_delta":    d2,
        "split_3_delta":    d3,
        "media_delta":      mean_d,
        "splits_positivos": n_pos,
    })

# ── Report ─────────────────────────────────────────────────────────────────

results_df = (
    pd.DataFrame(rows)
    .sort_values("media_delta", ascending=False)
    .reset_index(drop=True)
)

pd.set_option("display.float_format", "{:+.4f}".format)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 40)

print("\n" + "=" * 100)
print("  HMM ABLATION STUDY — delta_5d por conjunto de features (ordenado por média)")
print("=" * 100)
print(results_df.to_string(index=False))
print("=" * 100)

best = results_df.iloc[0]
print(f"\n  MELHOR CONJUNTO: {best['conjunto']}  (extra: {best['features']})")
print(f"  media_delta = {best['media_delta']:+.4f}  |  splits_positivos = {int(best['splits_positivos'])}/3")
print()
