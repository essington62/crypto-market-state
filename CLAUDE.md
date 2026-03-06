# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
| **Modeling** | `modeling.*` | `data/04_model_input/`, `data/05_models/` | HMM training & walk-forward validation |

### Data Sources

- **Binance** (`ingestion.binance.spot`): Crypto spot OHLCV (BTC, ETH, BNB, SOL, ADA, AVAX, LINK, XRP) — 24/7 continuous calendar
- **Yahoo Finance** (`ingestion.yfinance.*`): VIX, DXY, SP500, NASDAQ, Gold — business day calendar
- **FRED** (`ingestion.fred`): FEDFUNDS, DGS2, DGS10, CPI, UNRATE, WALCL, INDPRO, PAYEMS, STLFSI4, RRPONTSYD, TEDRATE — business day calendar

### Normalization (L2)

L2 modules are **asset-type oriented** (not exchange/API oriented). All pipelines produce:
- `timestamp` as the universal DatetimeIndex (UTC) — always renamed from `open_time` or `date`
- Unmodified values from L1 — no fills, no resampling, no feature engineering

Active L2 pipelines:
- `normalization.spot` → `spot_daily_clean` (Binance 24/7)
- `normalization.spot_business_day` → `spot_business_day_clean` (aligned to business days)
- `normalization.macro_daily/weekly/monthly` → `macro_*_clean`

### Feature Engineering (L3)

Per-asset only. No cross-asset aggregation. Features: log returns (1d/7d/21d/63d), volatility (rolling std, price range, realized vol), volume z-scores, buy pressure, price momentum (MA, slope, position), candle ratios, Hurst exponent, and autocorrelation.

The 44 primary features are added to each asset DataFrame, which preserves all L2 columns.

### Modeling

`modeling.regime_hmm`: Walk-forward HMM validation across 3 time splits (2023, 2024, 2025). Uses BTC L3 features (`log_return`, `vol_short`, `vol_ratio`). Input: `btc_spot_crypto_model_input`. Output: `hmm_walkforward_metrics_l4`.

## Pipeline Registry

All pipelines must be registered in `src/crypto_mkt_state/pipeline_registry.py`. The naming convention is dot-separated: `ingestion.binance.spot`, `normalization.spot`, `primary.spot.crypto`, `modeling.regime_hmm`.

**`__default__`** is set to `modeling.regime_hmm` (not the full flow).

## Catalog Structure

Each layer has a dedicated catalog file:
- `conf/base/catalog_l1.yml` — raw PartitionedDatasets
- `conf/base/catalog_l2.yml` — intermediate PartitionedDatasets
- `conf/base/catalog_l3.yml` — primary features + model inputs
- `conf/base/catalog_l4.yml` — model outputs and reports

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

## Current Focus — Fase 1A: HMM Walk-Forward

### Objetivo
Δ5d = E[R5d|Bull] − E[R5d|Bear] > 1% em 2/3 splits out-of-sample.

### Dataset de entrada correto
data/04_model_input/spot/daily/BTCUSDT.parquet
19 colunas (14 core + 5 chartist) | 1946 linhas | 2020-10-31 → 2026-02-27 | UTC
Chartist cols (200d burn-in, NaN permitido): dist_to_ma_200d, ma_50_200_ratio,
high_52w_dist, slope_21d, bb_width_20d

### Decisão de design — período de treino
Modelo treinado apenas com dados pós jan/2023.
Justificativa: aprovação ETF spot (jan/2024) e presença institucional tornaram
o mercado pré-2023 estruturalmente diferente. Crash 2022, covid 2020 = padrões obsoletos.

### Configuração atual
- n_states: 2 (Bear/Bull)
- Ordenação: drawdown das means_ (menor = Bear, maior = Bull)
- Features: log_return, vol_short, vol_ratio, drawdown, volume_z, slope_21d
- Treino: 2023-01-01 em diante
- Walk-forward: 3 splits semestrais/anuais pós-2023
  - split_1: train 2023, test H1-2024
  - split_2: train 2023–H1-2024, test H2-2024
  - split_3: train 2023–2024, test 2025→now

### Resultados baseline (2026-03-04)
| split   | n_train | n_test | delta_5d | bull_dur | bear_dur |
|---------|---------|--------|----------|----------|----------|
| split_1 | 359     | 182    | +0.0143  | 17.6d    | 8.4d     |
| split_2 | 541     | 184    | +0.0065  | 12.0d    | 9.8d     |
| split_3 | 725     | 423    | +0.0081  | 11.6d    | 48.9d    |
3/3 splits com delta positivo. Objetivo: delta > 1% em 2/3 ✓

### Constraints de código
- Sem hardcode de datas, paths ou features
- Index sempre DatetimeIndex UTC
- Código completo nos arquivos afetados — sem snippets parciais
- Sem prints ou logs customizados