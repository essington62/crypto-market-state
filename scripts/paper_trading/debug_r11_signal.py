"""
Debug R11 Signal — diagnostic only, no fixes.

Prints:
  1. Raw feature matrix for the last 30 rows of the OHLCV buffer
  2. predict_proba() output for each row
  3. model.means_ and model.covars_
  4. model.startprob_ and model.transmat_
  5. mapa / _inv_mapa (state label assignments)

Run:
    python scripts/paper_trading/debug_r11_signal.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
PT_DIR    = Path(__file__).parent
STATE_DIR = PT_DIR / "state"
MODEL_PKL = STATE_DIR / "r11_hmm_model.pkl"
BUFFER_PKL = STATE_DIR / "ohlcv_buffer.parquet"

FEATURES = ["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d"]

SEP  = "=" * 72
SEP2 = "-" * 72


# ── Feature computation (mirrors r11_paper_trader exactly) ─────────────────

def _slope_normalized(arr: np.ndarray) -> float:
    n = len(arr)
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, arr, 1)[0])
    last  = arr[-1]
    return slope / last if last != 0.0 else 0.0


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["close"]
    volume = df["volume"]

    log_ret   = np.log(close / close.shift(1))
    vol_short = log_ret.rolling(7).std()
    vol_long  = log_ret.rolling(30).std()
    vol_ratio = vol_short / vol_long.replace(0, np.nan)

    roll_max  = close.rolling(90, min_periods=1).max()
    drawdown  = close / roll_max - 1.0

    vol_mean  = volume.rolling(30).mean()
    vol_std   = volume.rolling(30).std()
    volume_z  = (volume - vol_mean) / vol_std.replace(0, np.nan)

    slope_21d = close.rolling(21).apply(_slope_normalized, raw=True)

    out = df.copy()
    out["log_return"] = log_ret
    out["vol_short"]  = vol_short
    out["vol_ratio"]  = vol_ratio
    out["drawdown"]   = drawdown
    out["volume_z"]   = volume_z
    out["slope_21d"]  = slope_21d
    return out


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Load model
    if not MODEL_PKL.exists():
        print(f"ERROR: model not found at {MODEL_PKL}")
        sys.exit(1)

    with open(MODEL_PKL, "rb") as fh:
        state = pickle.load(fh)

    model    = state["model"]
    scaler   = state["scaler"]
    mapa     = state["mapa"]        # {original_state → ordered (0=Bear, 1=Bull)}
    features = state["features"]
    meta     = state["meta"]
    inv_mapa = {v: k for k, v in mapa.items()}  # {ordered → original}

    # 2. Load OHLCV buffer
    if not BUFFER_PKL.exists():
        print(f"ERROR: buffer not found at {BUFFER_PKL}")
        sys.exit(1)

    buf = pd.read_parquet(BUFFER_PKL)
    if buf.index.tz is None:
        buf.index = buf.index.tz_localize("UTC")
    buf_feat = _compute_features(buf)
    window   = buf_feat.dropna(subset=FEATURES).tail(30)

    # ── Section 0: Model metadata ──────────────────────────────────────────
    print(SEP)
    print("MODEL METADATA")
    print(SEP)
    for k, v in meta.items():
        print(f"  {k:<20}: {v}")
    print(f"  {'features':<20}: {features}")

    # ── Section 1: state label mapping ────────────────────────────────────
    print()
    print(SEP)
    print("STATE LABEL MAPPING")
    print(SEP)
    print(f"  mapa (original → ordered):   {mapa}")
    print(f"  inv_mapa (ordered → original): {inv_mapa}")
    print(f"  Bear = ordered 0 → original model state {inv_mapa[0]}")
    print(f"  Bull = ordered 1 → original model state {inv_mapa[1]}")

    # ── Section 2: model internals ─────────────────────────────────────────
    print()
    print(SEP)
    print("MODEL INTERNALS")
    print(SEP)

    print("\n  startprob_ (initial state distribution):")
    for i, p in enumerate(model.startprob_):
        label = "Bear" if mapa[i] == 0 else "Bull"
        print(f"    state {i} ({label}): {p:.6f}")

    print("\n  transmat_ (transition matrix):")
    header = "         " + "".join(
        f"  →s{i}({('Bear' if mapa[i]==0 else 'Bull')})" for i in range(model.n_components)
    )
    print(header)
    for i, row in enumerate(model.transmat_):
        label = "Bear" if mapa[i] == 0 else "Bull"
        vals  = "  ".join(f"{p:.6f}" for p in row)
        print(f"    s{i}({label}):  {vals}")

    print("\n  means_ (one row per state, columns = features):")
    feat_hdr = "  ".join(f"{f:>12}" for f in features)
    print(f"    {'state':<10}  {feat_hdr}")
    for i, m in enumerate(model.means_):
        label = "Bear" if mapa[i] == 0 else "Bull"
        vals  = "  ".join(f"{v:>12.6f}" for v in m)
        print(f"    s{i}({label})     {vals}")

    print("\n  covars_ (diagonal variances, one row per state):")
    print(f"    {'state':<10}  {feat_hdr}")
    for i, cov in enumerate(model.covars_):
        label  = "Bear" if mapa[i] == 0 else "Bull"
        diag   = cov if cov.ndim == 1 else np.diag(cov)
        vals   = "  ".join(f"{v:>12.6f}" for v in diag)
        print(f"    s{i}({label})     {vals}")

    # ── Section 3: raw feature matrix ─────────────────────────────────────
    print()
    print(SEP)
    print(f"RAW FEATURE MATRIX — last {len(window)} rows (unscaled)")
    print(SEP)
    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.max_columns", 10,
                           "display.width", 120):
        print(window[features].to_string())

    # ── Section 4: scaled feature matrix ──────────────────────────────────
    X_raw    = window[features].values.astype(float)
    X_scaled = scaler.transform(X_raw)

    print()
    print(SEP)
    print("SCALED FEATURE MATRIX (after StandardScaler)")
    print(SEP)
    scaled_df = pd.DataFrame(X_scaled, index=window.index, columns=features)
    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.max_columns", 10,
                           "display.width", 120):
        print(scaled_df.to_string())

    # ── Section 5: production code path via SignalGenerator.predict() ────────
    import importlib.util, sys as _sys
    _spec = importlib.util.spec_from_file_location(
        "r11_signal_generator", PT_DIR / "r11_signal_generator.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _sys.modules["r11_signal_generator"] = _mod
    _spec.loader.exec_module(_mod)
    SignalGenerator = _mod.SignalGenerator

    gen       = SignalGenerator()
    last10    = buf_feat.dropna(subset=FEATURES).tail(10)

    print()
    print(SEP)
    print("PRODUCTION CODE PATH — generator.predict() for last 10 rows")
    print("(uses _build_context_X internally: 10-row sequence via buffer)")
    print(SEP)
    print(f"  {'date':<12}  {'regime':>6}  {'p_bull':>8}  {'p_bear':>8}")
    print(SEP2)

    for idx, row in last10.iterrows():
        sig = gen.predict(row, signal_date=str(idx.date()))
        print(f"  {sig['date']:<12}  {sig['regime']:>6}  "
              f"{sig['p_bull']:>8.4f}  {sig['p_bear']:>8.4f}")

    # ── Section 6: batch predict_proba (full sequence) ────────────────────
    print()
    print(SEP)
    print("predict_proba() — FULL SEQUENCE (all 30 rows as one sequence)")
    print(SEP)
    proba_seq = model.predict_proba(X_scaled)   # shape (30, n_states)
    pred_seq  = model.predict(X_scaled)

    print(f"  {'date':<12}  {'raw_s0':>8}  {'raw_s1':>8}  "
          f"{'p_bear(ord0)':>14}  {'p_bull(ord1)':>14}  "
          f"{'predict':>8}  {'regime':>6}")
    print(SEP2)

    for i, idx in enumerate(window.index):
        proba        = proba_seq[i]
        raw_pred     = int(pred_seq[i])
        ordered_pred = mapa[raw_pred]

        p_bear_raw = proba[inv_mapa[0]]
        p_bull_raw = proba[inv_mapa[1]]
        regime     = "Bull" if ordered_pred == 1 else "Bear"
        date_s     = str(idx.date())

        print(f"  {date_s:<12}  {proba[0]:>8.4f}  {proba[1]:>8.4f}  "
              f"{p_bear_raw:>14.6f}  {p_bull_raw:>14.6f}  "
              f"{raw_pred:>8}  {regime:>6}")

    # ── Section 7: scaler stats ────────────────────────────────────────────
    print()
    print(SEP)
    print("SCALER STATS (mean_ and scale_ from training)")
    print(SEP)
    print(f"  {'feature':<14}  {'mean':>12}  {'scale (std)':>14}")
    print(SEP2)
    for f, m, s in zip(features, scaler.mean_, scaler.scale_):
        print(f"  {f:<14}  {m:>12.6f}  {s:>14.6f}")

    print()
    print(SEP)
    print("DIAGNOSIS HINTS")
    print(SEP)
    if all(model.startprob_[inv_mapa[1]] > 0.9 for _ in [None]):
        print(f"  startprob_[Bull-state={inv_mapa[1]}] = {model.startprob_[inv_mapa[1]]:.4f}"
              f"  ← HIGH (bias suppressed by sequence context in Section 5)")
    print("  Compare Section 5 (production predict) vs Section 6 (raw sequence) p_bull.")
    print("  Both should now vary meaningfully. If Section 5 still ≈ 1.0 → check")
    print("  _build_context_X is loading the buffer and context has >= 2 rows.")
    print("  If both ≈ 1.0 → features are not discriminating (check means_/covars_).")
    print(SEP)


if __name__ == "__main__":
    main()
