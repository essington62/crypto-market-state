# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Arquitetura Atual (v2 — Pipeline Integrado, desde 2026-03-31)

### Produção: Pipeline Integrado (1 portfolio)
- Layer 1: HMM R11 (daily) — regime gate
- Layer 2: Specialist LightGBM (4h) — decisão de entrada + convicção
- Layer 3: Timing Layer (1h) — execução condicionada (RSI + BB + volume)
- Monitor 1h: stops + timing execution
- Cron: specialist :05 (4h), monitor :02 (1h)

### Benchmark: XGBoost Solo (congelado)
- Desativado em 2026-03-31
- Portfolio congelado para comparação
- Não executa trades

### Desativado: R11 Paper Trader Solo
- Desativado em 2026-03-31
- Incorporado como Layer 1 do Pipeline
- Dados históricos preservados em state/portfolio.json e state/equity_curve.csv

## Project Overview

A **Kedro**-based ML pipeline for crypto market state modeling. The system ingests raw market data (Binance, Yahoo Finance, FRED), normalizes it through a layered architecture, engineers features, and trains Hidden Markov Models (HMMs) to detect market regimes (bull, bear, transition).

## Environment Setup

```bash
conda env create -f environment.yml
conda activate crypto_market_state
pip install -e .
```

Key dependencies: `kedro==1.2.0`, `kedro-datasets==9.1.1`, `hmmlearn`, `fredapi`, `python-binance`, `yfinance`, `hmmlearn`, `optuna`.

## Common Commands

```bash
# Run a specific pipeline
kedro run --pipeline ingestion.binance.spot
kedro run --pipeline normalization.spot
kedro run --pipeline primary.spot.crypto
kedro run --pipeline modeling.regime_hmm

# Run the full HMM pipeline (L3 + modeling)
kedro run --pipeline modeling.regime_hmm_full

# New 4h and regime context pipelines
kedro run --pipeline ingestion.binance.spot_4h
kedro run --pipeline normalization.spot_intraday_4h
kedro run --pipeline ingestion.coinglass.derivatives_4h
kedro run --pipeline normalization.derivatives_4h
kedro run --pipeline primary.spot_4h
kedro run --pipeline primary.derivatives_4h
kedro run --pipeline primary.regime_context
kedro run --pipeline primary.model_features_4h

# Order book pipelines
kedro run --pipeline ingestion.coinglass.orderbook_4h
kedro run --pipeline normalization.orderbook_4h
kedro run --pipeline primary.orderbook_4h

# List all registered pipelines
kedro registry list

# Launch Jupyter
kedro jupyter lab

# Validate pipeline graph (no run)
kedro pipeline describe --pipeline <name>
```

## Layered Architecture

All data flows strictly downward through layers. **No layer may read from or write to a layer above it.**

| Layer | Kedro Stage | Data Path | Role |
|-------|------------|-----------|------|
| **L1** | `ingestion.*` | `data/01_raw/` | Raw API mirror — no transformations |
| **L2** | `normalization.*` | `data/02_intermediate/` | Schema normalization only |
| **L3** | `primary.*` | `data/03_primary/` | Per-asset feature engineering |
| **L4** | `primary.regime_context` | `data/04_model_input/regime_context/` | Regime context — daily consolidation of R11 + derivatives |
| **L5** | `primary.model_features_4h` | `data/05_model_features/` | Final model input — 4h supervised dataset |
| **Modeling** | `modeling.*` | `data/05_models/`, `data/06_reports/` | Model training & validation |

### Data Sources

- **Binance** (`ingestion.binance.spot`): Crypto spot OHLCV (BTC, ETH, BNB, SOL, ADA, AVAX, LINK, XRP) — 24/7 continuous calendar
- **Binance 4h** (`ingestion.binance.spot_4h`): BTC spot OHLCV at 4h interval — 24/7 continuous calendar
- **Yahoo Finance** (`ingestion.yfinance.*`): VIX, DXY, SP500, NASDAQ, Gold, Oil WTI, Oil Brent, Natural Gas, MOVE Index, OVX, Defense ETF (ITA), HYG, LQD, TIP — business day calendar
- **FRED** (`ingestion.fred`): FEDFUNDS, DGS2, DGS10, CPI, UNRATE, WALCL, INDPRO, PAYEMS, STLFSI4, RRPONTSYD, TEDRATE — business day calendar
- **CoinGlass** (`ingestion.coinglass.derivatives_4h`): BTC derivatives 4h — open interest, funding rates (OI-weighted, vol-weighted). Column format: `{endpoint_name}__{canonical_field_name}` (double underscore — L1 contract, never change)
- **CoinGlass Order Book** (`ingestion.coinglass.orderbook_4h`): BTC futures order book bid/ask volumes at 4h interval. 3 depth ranges: 0.5%, 1%, 2% of price. Partitioned by range: BTCUSDT_r05, BTCUSDT_r1, BTCUSDT_r2. ~166 days of history (API limit: 1000 records per range). No start_time/end_time params supported — single request only.

### Normalization (L2)

L2 modules are **asset-type oriented** (not exchange/API oriented). All pipelines produce:
- `timestamp` as the universal DatetimeIndex (UTC) — always renamed from `open_time` or `date`
- Unmodified values from L1 — no fills, no resampling, no feature engineering

Active L2 pipelines:
- `normalization.spot` → `spot_daily_clean` (Binance 24/7)
- `normalization.spot_business_day` → `spot_business_day_clean` (aligned to business days)
- `normalization.macro_daily/weekly/monthly` → `macro_*_clean`
- `normalization.spot_intraday_4h` → `spot_crypto_4h_normalized` (Binance 4h)
- `normalization.derivatives_4h` → `btc_coinglass_4h_clean` (CoinGlass derivatives)

### Feature Engineering (L3)

Per-asset only. No cross-asset aggregation. Features: log returns (1d/7d/21d/63d), volatility (rolling std, price range, realized vol), volume z-scores, buy pressure, price momentum (MA, slope, position), candle ratios, Hurst exponent, and autocorrelation.

The 44 primary features are added to each asset DataFrame, which preserves all L2 columns.

### Modeling

`modeling.regime_hmm`: Walk-forward HMM validation across 3 time splits (2023, 2024, 2025). Uses BTC L3 features (`log_return`, `vol_short`, `vol_ratio`). Input: `btc_spot_crypto_model_input`. Output: `hmm_walkforward_metrics_l4`.

## Multi-Horizon Model Stack

The system uses a two-layer decision architecture:

### R11 — Daily Regime Model (frozen, in production)
- HMM, 2 states (Bear/Bull), `covariance_type=diag`
- Features: `log_return`, `vol_short`, `vol_ratio`, `drawdown`, `volume_z`, `slope_21d`
- Train period: 2023-01-01 → 2024-12-31 (frozen)
- Role: define macro regime only — Bull or Bear
- Output signals used in L4: `r11_regime`, `r11_prob_bull`, `r11_prob_bear`, `r11_entropy`, `regime_age_days`, `regime_age_log`, `regime_is_new`, `regime_prob_change_3d`, `regime_prob_change_7d`
- Frozen model: `scripts/paper_trading/state/r11_hmm_model.pkl`
- R11 does NOT make trade decisions anymore

### Model 4h — Specialist AB ✓ APPROVED (2026-03-24)
- Algorithm: LightGBM binary classifier (SET_B_split_4_recent.pkl)
- Target: directional (price up in N candles)
- Features: 9 total (Group A technical + Group B R5C regime)
- Sharpe s4: +3.872 (best of all tested configurations)
- Paper trading: ACTIVE since 2026-03-24

**Group A — 4h Technical (6 features):**
  `returns_4h`, `returns_12h`, `volatility_24h`, `volume_zscore`, `buy_pressure`, `price_range_4h`

**Group B — R5C Regime (3 features):**
  `r11_prob_bull` (allocation_r5c), `r11_entropy`, `regime_age_log` (days_in_state)

**Position sizing:**
  `allocation_final > 0.5 → exposure = (allocation - 0.5) × 2`, else cash (0%)
  Min delta to trade: 0.05 (avoid overtrading at 4h frequency)
  Fees: 0.04% per trade (Binance taker)

### Ablation 4h — Complete Results (2026-03-24)

Walk-forward: 4 splits, split_4 covers 2026 regime (~244 candles × 4h)

| Model | Label                  | Sharpe μ | Sharpe s4 | AUC μ |
|-------|------------------------|----------|-----------|-------|
| A     | 4h Technical           | +1.169   | +3.323    | 0.507 |
| AB    | + R5C regime           | +0.920   | +3.872    | 0.520 | ← APPROVED
| ABC   | + CoinGlass deriv      | -0.905   | +2.783    | 0.522 |
| ABCD  | + OrderBook            | -1.430   | +1.599    | 0.534 |
| ABCDE | + MacroExt             | +1.958   | +2.116    | 0.534 |
| AR2   | + Deriv Normalized     | +0.570   | +1.212    | 0.538 |
| AR2B  | + funding_z+oi_div     | +1.013   | +1.374    | 0.523 |
| ABR4  | + Regime Interactions  | +0.513   | +2.657    | 0.505 |

**Key findings:**
- AB (técnico + R5C) is the best model for 2026 regime
- Every feature addition degrades Sharpe s4 — complexity is poison under concept drift
- Derivatives have signal (funding_z = #2 importance in AR2B) but degrade Sharpe when added as features
- Derivatives will be used as POST-MODEL GATES instead (control layer architecture)
- Raw derivatives (Group C) worse than normalized (AR2), but both worse than AB
- Interaction terms (ABR4) capture oi_regime/funding_regime (#7/#8 importance) but net negative
- MacroExt DESTRUCTIVE in 2026 4h: s4 drops from +1.099 to -0.437

### Decision Flow — Current Architecture
```
R11 HMM (daily) → regime features (prob_bull, entropy, regime_age)
       ↓
Specialist 4h (AB) → allocation_raw
       ↓
[ Control Layer ] (pass-through for now)
  - macro_gate (MOVE-based, planned)
  - derivatives_gate (funding/OI-based, planned)
       ↓
allocation_final → position sizing
```

## Pipeline Registry

All pipelines must be registered in `src/crypto_mkt_state/pipeline_registry.py`. The naming convention is dot-separated: `ingestion.binance.spot`, `normalization.spot`, `primary.spot.crypto`, `modeling.regime_hmm`.

**`__default__`** is set to `modeling.regime_hmm` (not the full flow).

## Catalog Structure

Each layer has a dedicated catalog file:
- `conf/base/catalog_l1.yml` — raw PartitionedDatasets
- `conf/base/catalog_l2.yml` — intermediate PartitionedDatasets
- `conf/base/catalog_l3.yml` — primary features + L3 model inputs
- `conf/base/catalog_l4.yml` — regime context (L4 daily)
- `conf/base/catalog_l5.yml` — final model features (L5 4h)

`settings.py` configures the catalog loader to auto-discover all `catalog*.yml` files.

## Parameters

All pipeline parameters live in `conf/base/parameters.yml`. The global `start_date` (`params:global.start_date`) is the single source of truth for all L1 pipelines — **never define `start_date` locally in a module**.

Additional model parameters (HMM states, walk-forward splits) are in `conf/base/parameters/hmm_2states.yml`.

## Critical Contracts

### L1 Contract
- Mirror of API — forbidden: feature engineering, aggregations, rolling windows, z-scores, interpolation
- All timestamps must be `datetime64[ns, UTC]`
- Binance pagination must use **forward** cursor strategy (`startTime`), with `max_iter` guard and `RATE_LIMIT_SLEEP_SEC = 0.4`
- `drop_duplicates(open_time)` is mandatory at the end of each ingestion node
- Cast numerics with `pd.to_numeric(..., errors="coerce").astype("float64")` — never raw `.astype()`

### L2 Contract
- Only allowed: column renaming, index setting, dtype enforcement, structural validation
- Forbidden: rolling, z-score, returns, fill, resample, interpolation, cross-asset harmonization
- `timestamp` (UTC DatetimeIndex) is the universal canonical name for all time columns
- Raise `ValueError` on integrity violations — never fill missing values

### L3 Contract
- Per-asset features only; no cross-asset operations
- All windows are strictly backward-looking (no lookahead)
- L2 columns are always preserved unchanged
- NaN is permitted at the start of a series (window burn-in), not filled

### L4 Contract — Regime Context Layer
- Purpose: consolidate R11 daily signals + derivatives aggregations into a single regime context dataset (daily frequency)
- Runs at DAILY frequency — no 4h data in this layer
- R11 HMM inference uses sequence context (proba_window=10 rows). Never use single-observation predict_proba
- `r11_model_path` must come from `parameters.yml` — never hardcoded
- Aggregation of 4h derivatives → daily uses last 6 × 4h obs (24h), strictly backward-looking
- Inner join on date — only dates where all inputs have data
- NaN permitted at start (burn-in) — never dropna
- Output: `data/04_model_input/regime_context/daily/BTCUSDT.parquet`
- 22 columns: regime state, stability, transition, derivatives daily, stress flags, volatility regime

### L5 Contract — Model Features Layer
- Purpose: final supervised dataset for 4h entry/exit/sizing model
- Runs at 4H frequency
- Joins: L3 spot 4h + L3 derivatives 4h + L4 regime context (daily) + L3 order book 4h
- L4 daily regime forward-filled into 4h grid with **mandatory day shift**: regime of day D fills candles from D+1 00:00 UTC onward. Never use same-day regime for same-day 4h candles (lookahead)
- `target_return_12h = close.shift(-3) / close - 1` — INTENTIONAL LOOKAHEAD for supervised training only. Never use target in production inference
- Output: `data/05_model_features/4h/BTCUSDT.parquet`
- 63 columns:
    - 10 OHLCV L2 preserved
    - 5  market structure (returns_4h, volatility_24h, volume_zscore, buy_pressure, price_range_4h)
    - 10 derivatives raw (oi_*, funding_*)
    - 18 regime context (r11_*, regime_*, stress_*, vol_*)
    - 15 order book (book_imbalance_*, bid_ask_ratio_*, book_depth_*, depth_gradient_*, imbalance_*, depth_zscore_24h, total_depth)
    - 1  target (target_return_12h — intentional lookahead)
- Range: 2025-09-14 → 2026-03-12 (1070 rows at 4h)
- DAG: 4 nodes — join_spot_and_derivatives → join_regime_context → join_orderbook → add_target

### Order Book Contract (L1 → L3)
- L1: 3 partitions per asset (r05=0.5%, r1=1%, r2=2%). Column format: flat (bids_usd, asks_usd, bids_quantity, asks_quantity, range_pct, timestamp). No double-underscore convention — different from derivatives
- L2: schema validation only — timestamp as DatetimeIndex UTC, float64 enforcement, no negative values allowed
- L3: features computed per range then joined on timestamp:
  - `book_imbalance_{r}` = (bids − asks) / (bids + asks), clipped to [−1, 1]
  - `bid_ask_ratio_{r}`  = bids / asks
  - `book_depth_{r}`     = bids + asks
  - Cross-range: `depth_gradient_near`, `depth_gradient_far`
  - Rolling (6 periods = 24h): `imbalance_ma_6`, `imbalance_std_6`, `depth_zscore_24h`
- Order book is microstructure/liquidity — NOT regime context
- Does NOT pass through L4 — goes directly to L5
- NaN at start expected (~104 rows burn-in in L5)

### L4 (Cross-Asset) Contract
- Pure function nodes — no IO, no Kedro internals
- Join only on `date` (inner join — intersection of dates)
- No resampling, forward-fill, or interpolation
- Select assets/series by `asset`/`category` metadata, not by raw tickers

## Module Structure for New Pipelines

**L1 ingestion:**
```
src/crypto_mkt_state/pipelines/ingestion/<exchange>/<market>/<module>/
  __init__.py  # empty
  nodes.py     # extraction logic
  pipeline.py  # Kedro Pipeline definition
```

**L2 normalization:**
```
src/crypto_mkt_state/pipelines/normalization/<asset_type>/
```

After creating any module: update `pipeline_registry.py`, add datasets to the appropriate `catalog_l*.yml`, add parameters to `parameters.yml`, and verify with `kedro registry list`.

## Utilities

- `src/crypto_mkt_state/utils/utils_temporal.py` — temporal helpers (used only in L1)
- `src/crypto_mkt_state/utils/utils_l3_semantic.py` — L3 feature computation helpers
- `src/crypto_mkt_state/clients/` — API clients for Binance, YFinance, and FRED

L2 nodes must **not** call `utils_temporal` — temporal handling belongs to L1 only.

## Validated Feature Decisions (from analysis)

Sources: `scripts/analysis/fed_macro_correlation.py`, `scripts/analysis/regime_deep_analysis.py`

### Macro correlation analysis findings

#### What predicts BTC 5d forward return

SIGNIFICANT (p < 0.10):

| Feature | Corr | Lag |
|---|---|---|
| `cpi_vs_target` | +0.120 | 0d |
| `rate_expectation_ch30` | +0.108 | 0d |
| `yield_curve_2y10y` | −0.117 | 4d |

NOT SIGNIFICANT — exclude:

| Feature | Reason |
|---|---|
| `vix_zscore_30d` | p=0.99 — not predictive |
| `dxy_return_5d` | p=0.46 — not predictive |
| `sp500_return_5d` | p=0.65 — not predictive |
| `dgs2_change_1d` | Unstable — 39 sign flips across periods |

#### What moves WITH BTC (regime context)

SP500 and VIX are contemporaneous, not predictive.
Use as regime state features, not directional signals.

#### Structural break — post-ETF (Jan 2024)

Pre-ETF correlations (DXY, DGS2) largely disappeared post-2024-01-10.
BTC behaves more like "digital gold" post-ETF.
**Train only on 2023+ data** — pre-2023 patterns are obsolete.

---

### Macro regime features for HMM v2

#### Predictive macro (enter as HMM features)

```python
cpi_vs_target         = CPIAUCSL.pct_change(12) * 100 - 2.0
rate_expectation_ch30 = (DGS3MO - FEDFUNDS).diff(30)
yield_curve_2y10y     = DGS10 - DGS2
```

#### Regime context (use as state proxy — not raw)

```python
# SP500 regime (trend state)
spx_trend  = SP500.pct_change(21)
spx_regime = np.where(spx_trend > 0.02,  1,
             np.where(spx_trend < -0.02, -1, 0))
# 1=bull, 0=sideways, -1=bear

# VIX regime (stress state)
vix_trend = VIX / VIX.shift(5) - 1
# rising VIX = stress increasing
# falling VIX = stress decreasing

features_context = ['spx_regime', 'vix_trend']
# Removes noise, preserves latent market state
```

#### Discarded features — do not retest without new evidence

| Feature | Reason |
|---|---|
| `dgs2_change_1d` | 39 sign flips, highly unstable |
| `dxy_return_5d` | Correlation disappeared post-ETF |
| `vix_zscore_30d` | Contemporaneous only, not predictive |
| `sp500_return_5d` | Contemporaneous only, not predictive |
| `days_to_fomc` | FOMC calendar does not increase vol (p=0.22) |
| `liquidity_proxy` | Redundant with `vol_ratio` (corr=0.72) |

---

### Liquidity analysis findings

Source: `scripts/analysis/liquidity_analysis.py`

`walcl_zscore_52w` is the strongest macro signal found (corr=−0.39, p≈0) — stronger than all previously tested macro variables.

**Key insight — 72-day lag:** Fed balance sheet changes take ~3 months to affect BTC price. Market partially anticipates but 72 trading days is the statistically optimal lag.

**Counter-intuitive finding:** DROUGHT regime outperforms FLOOD (+5.74% vs +1.64%, p=0.0006). BTC anticipates policy reversal during tightening periods. Use as lagged feature, not contemporaneous.

**Sign instability:** Raw signal flipped post-2022. Solution: `walcl_zscore_52w.shift(72)` stabilises the signal across periods.

| Signal | Corr (21d) | Best lag | Stable |
|---|---|---|---|
| `walcl_zscore_52w` | −0.39 | 72d | No (use lagged) |
| `walcl_change_12w` | −0.10 | 60d | No |
| `liquidity_score`  | −0.11 | 27d | No |

**Include in HMM v2:**
```python
walcl_zscore_52w_lag72 = walcl_zscore_52w.shift(72)
```

---

### Positioning analysis findings

Source: `scripts/analysis/positioning_analysis.py`

Window: 2025-09-13 → 2026-03-12 (~180 days, 1080 × 4h bars). CoinGlass Hobbyist plan only.

**INCLUDE:** `funding_zscore_14d` — corr=-0.135 (5d), best_lag=14d, regime separation=+0.44 (Bull mean=+0.42 vs Bear mean=-0.08), stable across window. Extreme LONG events (z≥+2) produce avg -2.45% BTC over 5d (contrarian signal confirmed).

**EXCLUDE:** All raw OI pct_change signals (corr<0.1 with raw transform — superseded by OI deep analysis below), leverage_stress (p=0.159), funding_zscore_30d, funding_divergence.

**CoinGlass upgrade:** WAIT — positioning script best corr=0.135 < 0.15. See OI analysis below for updated upgrade recommendation.

| Signal | Corr (5d) | Best lag | Separation | Verdict |
|---|---|---|---|---|
| `funding_zscore_14d` | −0.135 | 14d | +0.44 | **INCLUDE** |
| `oi_change_3d` (raw) | −0.056 | 25d | +0.79 | EXCLUDE (raw — use normalized) |
| `leverage_stress`    | −0.109 | 26d | +0.25 | EXCLUDE (p=0.159) |
| `liquidation_*`      | N/A    | N/A | N/A   | NOT AVAILABLE (Hobbyist plan) |

---

### OI deep analysis findings

Source: `scripts/analysis/oi_analysis.py`

**Critical insight:** Raw OI pct_change (tested in positioning_analysis.py) was the wrong transform. Normalized/stationary OI transforms dramatically outperform raw changes.

**Best signals (criteria: |corr_5d| ≥ 0.10 AND separation ≥ 0.30):**

| Signal | Corr (5d) | Best lag | Separation | Note |
|---|---|---|---|---|
| `oi_zscore_30d`    | −0.291 | 10d | 2.12 | Strongest OI signal |
| `oi_price_ratio_z` | −0.272 | 1d  | 0.34 | Removes price trend bias |
| `oi_change_7d`     | −0.269 | 21d | 1.39 | Weekly momentum |
| `oi_zscore_14d`    | −0.243 | 21d | 1.74 | Short z-score |
| `oi_vol_ratio`     | −0.217 | 19d | 1.19 | OI volatility analog |
| `oi_extreme`       | −0.220 | 20d | 1.69 | Discrete HMM signal |
| `oi_change_3d`     | −0.113 | 6d  | 0.97 | Borderline (corr≥0.10) |

**Reversal detection:** `oi_vol_ratio` is elevated before regime flips (At_flip=0.55 vs Stable=0.41, p=0.040) — confirmed leading indicator of regime transitions.

**Price×OI divergence (corrected sign):**
- `price_oi_div` > Q75 (price↑ OI↓ = short covering): avg BTC 5d = −0.42%
- `price_oi_div` < Q25 (price↑ OI↑ = fragile longs): avg BTC 5d = −2.23%
- Signal directionally correct but separation below threshold (sep=0.26) — EXCLUDE for now.

**Note on `funding_zscore_14d`:** In OI analysis (regime via 21d momentum) sep=0.21 < 0.30 — EXCLUDE. In positioning_analysis (regime via R11 HMM) sep=0.44 — INCLUDE. The R11 regime context is the right conditioning variable. Keep `funding_zscore_14d` in the feature set, conditioned on R11 regime.

**CoinGlass upgrade:** CONSIDER — best corr=0.291 > 0.15 threshold. Evaluate Standard plan ROI. Liquidation data could add `liq_cascade_risk` feature. Organic data hits 12 months ≈2026-09.

**Recommended OI signals to add to HMM v2:**
```python
oi_zscore_30d    # corr=-0.291, lag=10d — normalize to remove trend
oi_price_ratio_z # corr=-0.272, lag=1d  — price-adjusted OI
oi_vol_ratio     # corr=-0.217, lag=19d — leading indicator of reversals
```

---

### Final HMM v2 feature set

```python
# BTC technical (6 — validated in R11)
log_return, vol_short, vol_ratio,
drawdown, volume_z, slope_21d

# Macro predictive (4 — validated in analysis)
cpi_vs_target           # corr=+0.120, lag=0d
rate_expectation_ch30   # corr=+0.108, lag=0d
yield_curve_2y10y       # corr=-0.117, lag=4d
walcl_zscore_52w_lag72  # corr=-0.390, lag=72d  ← strongest signal

# Macro regime context (2 — contemporaneous, not predictive)
spx_regime   # SP500.pct_change(21) → -1/0/1
vix_trend    # VIX / VIX.shift(5) - 1

# Positioning (1 — validated)
funding_zscore_14d      # corr=-0.135, lag=14d  (contrarian at extremes)

# OI (3 — validated in oi_analysis.py)
oi_zscore_30d    # corr=-0.291, lag=10d  ← strongest OI signal
oi_price_ratio_z # corr=-0.272, lag=1d   (price-trend adjusted)
oi_vol_ratio     # corr=-0.217, lag=19d  (reversal leading indicator)
```

**Total: 16 features confirmed**

Discarded: raw OI pct_change (wrong transform), `leverage_stress` (p>0.15), `price_oi_div` (sep<0.30), `oi_exchange_div` (sep<0.30), `liquidation_intensity` (Hobbyist plan — not available).

---

## Model Architecture — v3 (Current, updated 2026-03-24)

### Overview

Two-stage architecture: HMM as feature generator → LightGBM as decision maker.

```
BTC daily → HMM R5C → regime features (prob_bull, entropy, regime_age)
                              ↓
BTC 4h technical → LightGBM (Specialist AB, 9 features) → allocation_raw
                              ↓
                     [ Control Layer ] → allocation_final → position
```

Key insight from ablation: **simplicity wins under concept drift.** Adding macro, derivatives, order book, or interaction terms as model features all degrade 2026 performance. The optimal model uses only 4h technical + daily R5C regime context.

Derivatives (funding, OI) have strong signal but work better as **post-model gates** than as model inputs.

---

### HMM Final Design — R5C (best unsupervised detector)

- `n_states`: 3 (Bull / Sideways / Bear)
- `covariance_type`: full
- Features: 6 stationary BTC features (`log_return`, `vol_short`, `vol_ratio`, `drawdown`, `volume_z`, `slope_21d`)
- Allocation rule: `max(0, prob_bull − prob_bear)` (continuous, not binary)

| Metric | Value |
|---|---|
| Bull mean return | +0.87% |
| Sharpe | +0.639 |
| Splits passing | 3/4 |

**Conclusion:** R5C is the best unsupervised regime detector found, but insufficient alone → used as feature input to LightGBM.

---

### LightGBM Specialist AB (approved model)

- **Target:** directional (binary)
- **Position sizing:** `allocation > 0.5 → exposure = (allocation - 0.5) × 2`, else cash
- **Input:** 6 technical 4h + 3 R5C regime = 9 features total
- **Model file:** `SET_B_split_4_recent.pkl`

| Metric | Value |
|---|---|
| Sharpe s4 (2026) | +3.872 |
| AUC μ (all splits) | 0.520 |
| Sharpe μ (all splits) | +0.920 |

**Key insight:** Adding ANY feature group beyond AB degrades 2026 Sharpe. Model works because R5C provides regime context while technical features capture 4h momentum.

---

### Feature Groups — 4h Model (tested universe)

| Group | Features | Sharpe s4 when added | Verdict |
|---|---|---|---|
| A — 4h Technical | returns_4h, returns_12h, volatility_24h, volume_zscore, buy_pressure, price_range_4h | +3.323 (baseline) | ✓ INCLUDE |
| B — R5C Regime | r11_prob_bull, r11_entropy, regime_age_log | +3.872 (+0.549) | ✓ INCLUDE |
| C — CoinGlass raw | funding_*, oi_* (raw values) | +2.783 (-1.089) | ✗ EXCLUDE |
| D — Order Book | book_imbalance_*, bid_ask_ratio_*, depth_* | +1.599 (-2.273) | ✗ EXCLUDE |
| E — MacroExt | MOVE, OVX, HYG, oil z-scores | +2.116 (-1.756) | ✗ EXCLUDE |
| R2 — Deriv Normalized | funding_z, oi_price_div, volume_z | +1.212 (-2.660) | ✗ EXCLUDE (as feature) |
| R4 — Regime Interactions | squeeze scores, oi_regime, funding_regime | +2.657 (-1.215) | ✗ EXCLUDE |

---

### Concept Drift — Critical Finding (2026)

**Observed regime inversion:**

| Period | Macro signal | BTC direction |
|---|---|---|
| 2023–2025 | Bearish (tightening) | Bullish |
| 2026 | Bearish (uncertainty) | Bearish |

**Consequence:** A model trained on 2023–2025 patterns produces AUC < 0.5 on 2026 data — predictions are systematically inverted.

**Definition:** *Concept drift / regime inversion* — the relationship between features and target reverses sign across market regimes. The feature is still informative; the direction is not stable.

**Implication:** Do not assume feature usefulness is stable across policy regimes. Contextualization (hawkish/dovish flag) is required.

---

### Ablation 2026 Strategy ✓ COMPLETE

**Window:** Train Sep/2025 → Dec/2025 | Test Jan/2026 → Mar/2026

**Result:** AB (technical + R5C) is optimal. See "Ablation 4h — Complete Results" in Multi-Horizon Model Stack for full table.

**Principle confirmed:** Feature usefulness is NOT stable across regimes. Complexity degrades under concept drift.

---

### Derivatives — Signal vs Feature Analysis (2026-03-24)

Derivatives normalization tested (funding_z, oi_price_div, volume_z):
- `funding_z` = #2 feature by importance (404) when added to technical model
- But net effect on Sharpe s4 is NEGATIVE (-1.949 vs baseline)
- Cause: model uses funding signal at wrong timing (mean-reversion vs trend)
- Conclusion: derivatives have signal but should be used as POST-MODEL GATES

Interaction terms tested (long_squeeze_score, short_squeeze_score, oi_regime, funding_regime):
- oi_regime (#7) and funding_regime (#8) have reasonable importance
- But net Sharpe s4 = +2.657 vs AB +3.872 → negative contribution
- Squeeze scores marginal (events too rare in 77-day test window)

---

### Current Hypothesis (updated 2026-03-24)

- ~~Macro features require contextualization by policy regime~~ → Macro is destructive in 4h timeframe
- ~~CoinGlass likely critical for post-2025 regime~~ → Derivatives degrade as features, promising as gates
- HMM R5C provides the right regime context for 4h decisions (AB > A)
- Simplicity wins: 9 features > 14 > 29 in 2026 regime
- Control layer (post-model gates) is the right architecture for derivatives signal

---

### Next Steps (as of 2026-03-24)

1. ✓ ~~Complete ablation study~~ → AB approved
2. ✓ ~~Paper trading setup~~ → specialist_4h active
3. Paper trading validation (30 days, target: Apr 2026)
4. Implement control layer:
   a. macro_gate (MOVE-based) over AB 4h
   b. derivatives_gate (funding/OI-based) over AB 4h
5. Compare specialist AB vs R11 baseline over same period
6. If validated → production (Binance Testnet → live)

---

## Current Status

### Fase 1A — R11 HMM Walk-Forward ✓ COMPLETE
Δ5d > 1% in 2/3 splits out-of-sample achieved.
R11 in paper trading on Binance Testnet since 2026-03-10.
Signal generator fix applied 2026-03-16: sequence context
(proba_window=10) replaces single-observation predict_proba.

| split   | n_train | n_test | delta_5d | bull_dur | bear_dur |
|---------|---------|--------|----------|----------|----------|
| split_1 | 359     | 182    | +0.0143  | 17.6d    | 8.4d     |
| split_2 | 541     | 184    | +0.0065  | 12.0d    | 9.8d     |
| split_3 | 725     | 423    | +0.0081  | 11.6d    | 48.9d    |

### Paper Trading — R11 (active)
- Script: `scripts/paper_trading/r11_paper_trader.py`
- Cron: 08:00 UTC daily
- Dashboard: `streamlit run apps/r11_dashboard.py`
- State: `scripts/paper_trading/state/`
- Testnet portfolio: ~$83,431 USDT (as of 2026-03-16)
- Current signal: Bear — 100% cash

### Paper Trading — Specialist 4h AB (active since 2026-03-24)
- Script: `scripts/paper_trading/specialist_4h_paper_trader.py`
- Cron: `0 0,4,8,12,16,20 * * *` (every 4h aligned to candles)
- Dashboard: `streamlit run apps/specialist_4h_dashboard.py`
- State: `scripts/paper_trading/state/specialist_4h/`
  - `config.json` — model path, capital, fees, feature list
  - `portfolio.json` — simulated portfolio state
  - `signals.csv` — incremental log per 4h candle (20 columns)
- Model: `SET_B_split_4_recent.pkl` (LGBMClassifier, 9 features)
- Initial capital: $100,000 USDT (simulated, no testnet)
- First signal: 2026-03-24 16:00 UTC, allocation_raw=0.476 → HOLD (cash)
- Data flow: Binance public API → ohlcv_4h_buffer.parquet → features → model → signal
- R5C regime: read from R11 HMM on daily buffer, day-shift enforced
- Control layer: `apply_control_layer()` stub ready for macro/derivatives gates

### Fase 2 — Model 4h ✓ ABLATION COMPLETE, PAPER TRADING ACTIVE
- Ablation complete: 8 model configurations tested (A through ABR4)
- Winner: AB (técnico 4h + R5C regime), Sharpe s4=+3.872
- Key insight: simplicity wins under concept drift — every addition degrades s4
- Paper trading active since 2026-03-24 (see above)
- Next: 30 days paper trading → evaluate → control layer implementation

### Control Layer — Planned Architecture (not yet implemented)
```
specialist_4h (AB) → allocation_raw → [gates] → allocation_final
```
Gates planned:
1. **macro_gate** — MOVE-based continuous gate (already exists for daily specialist)
   - Formula: `multiplier = (1 - stress_factor × bearish_strength × max_cut).clip(0,1)`
   - Parameters: move_stress_min=1.0, move_stress_max=3.0, max_cut=0.70
   - Daily result: delta_sharpe=+0.014 (marginal improvement)
2. **derivatives_gate** — funding/OI-based (new, to be designed)
   - Rationale: funding_z has strong signal (#2 importance) but degrades Sharpe as feature
   - Hypothesis: better as post-model filter than as model input
   - Design pending — requires paper trading data to calibrate

Implementation approach: gates are independent, can be enabled/disabled without retraining.
Evaluation: wait for paper trading baseline (30 days), then add gates incrementally.

### Constraints de código
- Sem hardcode de datas, paths ou features
- Index sempre DatetimeIndex UTC
- Código completo nos arquivos afetados — sem snippets parciais
- Sem prints ou logs customizados
## Claude Code Operating Rules

Claude must respect the project’s layered data architecture.  
All code generated must follow these rules strictly.

### Layer Access Rules

Data flows strictly downward:

L1 → L2 → L3 → L4 → Modeling

A layer may only read from the layer immediately below it.

Forbidden access patterns:

L3 reading L1  
L4 reading L2 directly  
Modeling reading L2 or L1

Always respect the declared data paths in the catalog.

---

### L1 — Raw Ingestion

Purpose: mirror external APIs.

Allowed:
- API calls
- timestamp parsing
- dtype normalization
- deduplication

Forbidden:
- feature engineering
- rolling windows
- aggregations
- resampling
- forward fill
- interpolation

L1 output must remain as close as possible to the original API schema.

---

### L2 — Normalization

Purpose: schema harmonization.

Allowed:
- column renaming
- dtype enforcement
- timezone normalization
- schema validation

Forbidden:
- feature engineering
- rolling windows
- resampling
- forward fill
- cross-asset joins

L2 outputs clean but **untransformed** time series.

---

### L3 — Feature Engineering

Purpose: compute market features.

Allowed:
- rolling windows
- resampling to daily frequency
- forward fill (when economically justified)
- cross-asset feature construction
- derived indicators

Forbidden:
- labels for models
- future leakage
- lookahead calculations

All windows must be strictly backward-looking.

---

### L4 — Model Input

Purpose: prepare datasets for models.

Allowed:
- final feature joins
- feature selection
- label creation
- removal of NaN burn-in

Forbidden:
- feature engineering
- resampling
- forward fill

L4 must only assemble previously computed features.

---

### Modeling Rules

Models may only read from:

data/04_model_input/

Model outputs must go to:

data/05_models/
data/06_reports/

Never read raw or intermediate layers inside modeling pipelines.

---

### Code Generation Constraints

Claude must follow these additional rules when generating code:

- Never hardcode dates or asset lists
- Always use parameters.yml for configuration
- Always use DatetimeIndex UTC
- Prefer pure functions in nodes
- Avoid side effects in Kedro nodes
- Never introduce lookahead bias

----

## Arquitetura de Código

### Scripts exploratórios e de validação
Permitidos em scripts/ e notebooks/
Uso: prototipagem, validação pontual, extração ad-hoc
NÃO referenciar no pipeline_registry
NÃO usar como fonte de dados para modelos

### Pipeline Kedro (fonte da verdade)
Toda extração que vai para produção DEVE ter
um node correspondente em pipelines/ingestion/
Os scripts/ são o rascunho — o Kedro é o contrato

### Fluxo de trabalho aprovado
1. Prototipar em script/ ou notebook
2. Validar resultado
3. Migrar para node Kedro quando estável
4. Script original pode ser mantido como referência
   mas o Kedro passa a ser a fonte oficial

### Regra prática para o Claude Code
- Exploração/validação pontual -> scripts/
- Dado que entra no modelo -> obrigatoriamente Kedro
- Nunca ler de scripts/ dentro de um pipeline node