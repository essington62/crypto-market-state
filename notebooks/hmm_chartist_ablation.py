"""
HMM Chartist Ablation Study
Testa adição individual e combinada das 5 features grafistas ao BASE.
n_states=2, ordenação por drawdown, walk-forward idêntico ao pipeline.
"""

import numpy as np
import pandas as pd
import yaml
from datetime import timedelta
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ── Parameters ─────────────────────────────────────────────────────────────

with open("conf/base/parameters.yml") as f:
    params = yaml.safe_load(f)

wf  = params["walkforward"]
mod = params["modeling"]["regime_hmm"]

N_STATES:     int = 2
COV_TYPE:     str = mod["covariance_type"]
N_ITER:       int = mod["n_iter"]
RANDOM_STATE: int = mod["random_state"]
HORIZON:      int = wf["horizon_days"]
EMBARGO:      int = wf["embargo_days"]
PURGE_TOTAL:  int = HORIZON + EMBARGO

# ── Data ───────────────────────────────────────────────────────────────────

df_raw = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
df_raw = df_raw.sort_index()
df_raw["_fwd"] = sum(df_raw["log_return"].shift(-k) for k in range(1, HORIZON + 1))

# ── Feature sets ───────────────────────────────────────────────────────────

BASE = ["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z"]

FEATURE_SETS = {
    "BASE": BASE,
    "A":    BASE + ["dist_to_ma_200d"],
    "B":    BASE + ["ma_50_200_ratio"],
    "C":    BASE + ["high_52w_dist"],
    "D":    BASE + ["slope_21d"],
    "E":    BASE + ["bb_width_20d"],
    "F":    BASE + ["dist_to_ma_200d", "ma_50_200_ratio"],
    "G":    BASE + ["dist_to_ma_200d", "slope_21d"],
    "H":    BASE + ["dist_to_ma_200d", "ma_50_200_ratio", "slope_21d"],
}

# ── Ordering ───────────────────────────────────────────────────────────────

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

# ── Walk-forward runner ────────────────────────────────────────────────────

def run_split(
    df: pd.DataFrame,
    features: list[str],
    train_start: pd.Timestamp,
    purged_train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> tuple[float, float]:
    """Return (delta_5d, bull_duration) for one split."""
    train_df = df.loc[train_start:purged_train_end].dropna(subset=features)
    test_df  = df.loc[test_start:test_end].dropna(subset=features)

    if len(train_df) < 30 or len(test_df) < 5:
        return np.nan, np.nan

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test  = scaler.transform(test_df[features])

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type=COV_TYPE,
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train)

    mapa   = _order_by_drawdown(model, features)
    states = np.array([mapa[s] for s in model.predict(X_test)])

    bull_state = N_STATES - 1
    bear_state = 0

    fwd    = test_df["_fwd"]
    E_bull = fwd[states == bull_state].mean()
    E_bear = fwd[states == bear_state].mean()
    delta  = float(E_bull - E_bear)

    # bull_duration: mean length of consecutive bull runs
    durations, count = [], 0
    for s in states:
        if s == bull_state:
            count += 1
        else:
            if count > 0:
                durations.append(count)
                count = 0
    if count > 0:
        durations.append(count)
    bull_dur = float(np.mean(durations)) if durations else 0.0

    return delta, bull_dur


def run_walkforward(features: list[str]) -> dict[str, tuple[float, float]]:
    results = {}
    for cfg in wf["splits"]:
        train_start  = pd.Timestamp(str(cfg["train_start"]), tz="UTC")
        train_end    = pd.Timestamp(str(cfg["train_end"]),   tz="UTC")
        test_start   = pd.Timestamp(str(cfg["test_start"]),  tz="UTC")
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
    missing = [f for f in feats if f not in df_raw.columns]
    if missing:
        print(f"[SKIP] {name}: features não disponíveis: {missing}")
        continue

    wf_results = run_walkforward(feats)

    d1, bd1 = wf_results.get("split_1", (np.nan, np.nan))
    d2, bd2 = wf_results.get("split_2", (np.nan, np.nan))
    d3, bd3 = wf_results.get("split_3", (np.nan, np.nan))

    deltas   = [d1, d2, d3]
    mean_d   = float(np.nanmean(deltas))
    n_pos    = sum(1 for d in deltas if not np.isnan(d) and d > 0.01)
    bull_durs = [bd1, bd2, bd3]
    n_bull_ok = sum(1 for bd in bull_durs if not np.isnan(bd) and bd > 15)

    extras = "+".join(f for f in feats if f not in BASE) or "—"

    rows.append({
        "id":            name,
        "features_extras": extras,
        "split_1":       d1,
        "split_2":       d2,
        "split_3":       d3,
        "media_delta":   mean_d,
        "splits>1%":     n_pos,
        "bull_dur_s1":   bd1,
        "bull_dur_s2":   bd2,
        "bull_dur_s3":   bd3,
        "bull_dur_ok":   n_bull_ok,   # splits with bull_dur > 15d
    })

# ── Report ─────────────────────────────────────────────────────────────────

results_df = (
    pd.DataFrame(rows)
    .sort_values("media_delta", ascending=False)
    .reset_index(drop=True)
)

pd.set_option("display.float_format", "{:+.4f}".format)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 50)

print("\n" + "=" * 130)
print("  HMM CHARTIST ABLATION — delta_5d + bull_duration por split (n_states=2, ordem drawdown)")
print("=" * 130)

display_cols = [
    "id", "features_extras",
    "split_1", "split_2", "split_3", "media_delta", "splits>1%",
    "bull_dur_s1", "bull_dur_s2", "bull_dur_s3", "bull_dur_ok",
]
print(results_df[display_cols].to_string(index=False))
print("=" * 130)

best = results_df.iloc[0]
print(
    f"\n  MELHOR: {best['id']}  extras={best['features_extras']}"
    f"  media_delta={best['media_delta']:+.4f}"
    f"  splits>1%={int(best['splits>1%'])}/3"
    f"  bull_dur_ok={int(best['bull_dur_ok'])}/3"
)
print()
