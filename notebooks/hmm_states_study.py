"""
HMM States Study — Ablation de n_states e critério de ordenação.
Não modifica nenhum pipeline, catalog ou dataset.
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

COV_TYPE:     str = mod["covariance_type"]
N_ITER:       int = mod["n_iter"]
RANDOM_STATE: int = mod["random_state"]
HORIZON:      int = wf["horizon_days"]
EMBARGO:      int = wf["embargo_days"]
PURGE_TOTAL:  int = HORIZON + EMBARGO

BASE = ["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z"]

# ── Data ───────────────────────────────────────────────────────────────────

df_raw = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
df_raw = df_raw.sort_index()
df_raw["_fwd"] = sum(df_raw["log_return"].shift(-k) for k in range(1, HORIZON + 1))

# ── Ordering functions ─────────────────────────────────────────────────────

def _order_by_feature(model: GaussianHMM, features: list[str],
                      key: str, reverse: bool = False) -> dict[int, int]:
    """
    Order states by the mean value of `key` across model.means_.
    reverse=False → ascending → index 0 = smallest mean (Bear).
    reverse=True  → descending → index 0 = largest mean (Bear).
    """
    n = model.n_components
    if key in features:
        fidx = features.index(key)
        vals = {s: model.means_[s][fidx] for s in range(n)}
    else:
        vals = {s: float(s) for s in range(n)}
    ordered = sorted(vals, key=vals.get, reverse=reverse)
    return {orig: new for new, orig in enumerate(ordered)}


ORDERINGS = {
    # (label, key_feature, reverse)
    # For drawdown: smaller (more negative) = Bear → ascending
    "drawdown":   ("drawdown",   False),
    # For ret_short: smaller return = Bear → ascending
    "ret_short":  ("ret_short",  False),
    # For log_return: smaller return = Bear → ascending
    "log_return": ("log_return", False),
    # For vol_short: larger vol = Bear → descending
    "vol_short":  ("vol_short",  True),
}

# ── Experiment definitions ─────────────────────────────────────────────────

EXPERIMENTS = [
    {"id": 1, "n_states": 3, "order_by": "drawdown"},
    {"id": 2, "n_states": 3, "order_by": "ret_short"},
    {"id": 3, "n_states": 3, "order_by": "log_return"},
    {"id": 4, "n_states": 2, "order_by": "drawdown"},
    {"id": 5, "n_states": 2, "order_by": "vol_short"},
    {"id": 6, "n_states": 2, "order_by": "log_return"},
]

# ── Walk-forward runner ─────────────────────────────────────────────────────

def run_experiment(n_states: int, order_by: str) -> dict[str, float]:
    key, reverse = ORDERINGS[order_by]
    bull_state = n_states - 1   # 1 for n=2, 2 for n=3
    bear_state = 0

    split_deltas = {}

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

        train_df = df_raw.loc[train_start:purged_train_end].dropna(subset=BASE)
        test_df  = df_raw.loc[test_start:test_end].dropna(subset=BASE)

        if len(train_df) < 30 or len(test_df) < 5:
            split_deltas[cfg["name"]] = np.nan
            continue

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(train_df[BASE])
        X_test  = scaler.transform(test_df[BASE])

        model = GaussianHMM(
            n_components=n_states,
            covariance_type=COV_TYPE,
            n_iter=N_ITER,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train)

        mapa   = _order_by_feature(model, BASE, key, reverse)
        states = np.array([mapa[s] for s in model.predict(X_test)])

        fwd    = test_df["_fwd"]
        E_bull = fwd[states == bull_state].mean()
        E_bear = fwd[states == bear_state].mean()
        split_deltas[cfg["name"]] = float(E_bull - E_bear)

    return split_deltas

# ── Run all experiments ─────────────────────────────────────────────────────

rows = []
for exp in EXPERIMENTS:
    deltas = run_experiment(exp["n_states"], exp["order_by"])
    d1 = deltas.get("split_1", np.nan)
    d2 = deltas.get("split_2", np.nan)
    d3 = deltas.get("split_3", np.nan)
    mean_d = float(np.nanmean([d1, d2, d3]))
    n_pos  = sum(1 for d in [d1, d2, d3] if not np.isnan(d) and d > 0.01)
    rows.append({
        "id":          exp["id"],
        "n_states":    exp["n_states"],
        "ordenação":   exp["order_by"],
        "split_1":     d1,
        "split_2":     d2,
        "split_3":     d3,
        "media_delta": mean_d,
        "splits>1%":   n_pos,
    })

# ── Report ──────────────────────────────────────────────────────────────────

results = (
    pd.DataFrame(rows)
    .sort_values("media_delta", ascending=False)
    .reset_index(drop=True)
)

pd.set_option("display.float_format", "{:+.4f}".format)
pd.set_option("display.width", 120)

print("\n" + "=" * 90)
print("  HMM STATES STUDY — n_states × ordenação (BASE features, ordenado por média)")
print("=" * 90)
print(results.to_string(index=False))
print("=" * 90)

best = results.iloc[0]
print(
    f"\n  MELHOR: id={int(best['id'])}  n_states={int(best['n_states'])}"
    f"  ordenação={best['ordenação']}"
    f"  media_delta={best['media_delta']:+.4f}"
    f"  splits>1%={int(best['splits>1%'])}/3"
)
print()
