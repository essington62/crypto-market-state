#!/bin/bash
# ─────────────────────────────────────────────
# Daily data update — crypto-market-state v1
# Runs at 07:00 AM — updates all data sources
# ─────────────────────────────────────────────

PROJECT=/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state
CONDA_ENV=crypto_market_state
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/${CONDA_ENV}/bin/python
KEDRO=/opt/homebrew/Caskroom/miniforge/base/envs/${CONDA_ENV}/bin/kedro
LOG_DIR=${PROJECT}/logs
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "========================================"
echo "Daily Update Started: ${DATE}"
echo "========================================"

cd ${PROJECT}

# ── Credentials ───────────────────────────────
SECRETS_FILE=${PROJECT}/conf/local/secrets.yml

if [ ! -f "${SECRETS_FILE}" ]; then
  echo "[ERROR] secrets.yml not found: ${SECRETS_FILE}"
  exit 1
fi

export COINGLASS_API_KEY=$(${PYTHON} -c "
import yaml
c = yaml.safe_load(open('${SECRETS_FILE}'))
print(c['api_keys']['coinglass'])
")

export FRED_API_KEY=$(${PYTHON} -c "
import yaml
c = yaml.safe_load(open('${SECRETS_FILE}'))
print(c['fred'])
")

export BINANCE_API_KEY=$(${PYTHON} -c "
import yaml
c = yaml.safe_load(open('${SECRETS_FILE}'))
print(c['binance_testnet']['api_key'])
")

export BINANCE_API_SECRET=$(${PYTHON} -c "
import yaml
c = yaml.safe_load(open('${SECRETS_FILE}'))
print(c['binance_testnet']['api_secret'])
")

if [ -z "${COINGLASS_API_KEY}" ]; then
  echo "[ERROR] COINGLASS_API_KEY not loaded — check secrets.yml"
  exit 1
fi
if [ -z "${FRED_API_KEY}" ]; then
  echo "[ERROR] FRED_API_KEY not loaded — check secrets.yml"
  exit 1
fi

echo "[$(date '+%H:%M:%S')] Credentials loaded ✓"

# ── BTC Spot ──────────────────────────────────
echo "[$(date '+%H:%M:%S')] Running ingestion.binance.spot_incremental..."
${KEDRO} run --pipeline ingestion.binance.spot_incremental
if [ $? -ne 0 ]; then
    echo "[ERROR] ingestion.binance.spot_incremental FAILED"
    exit 1
fi

echo "[$(date '+%H:%M:%S')] Running normalization.spot..."
${KEDRO} run --pipeline normalization.spot
if [ $? -ne 0 ]; then
    echo "[ERROR] normalization.spot FAILED"
    exit 1
fi

echo "[$(date '+%H:%M:%S')] Running primary.spot.crypto..."
${KEDRO} run --pipeline primary.spot.crypto
if [ $? -ne 0 ]; then
    echo "[ERROR] primary.spot.crypto FAILED"
    exit 1
fi

# ── BTC Spot 4h ───────────────────────────────
echo "[$(date '+%H:%M:%S')] Running ingestion.binance.spot_4h..."
${KEDRO} run --pipeline ingestion.binance.spot_4h
if [ $? -ne 0 ]; then
    echo "[WARN] ingestion.binance.spot_4h FAILED — continuing"
fi

echo "[$(date '+%H:%M:%S')] Running normalization.spot_intraday_4h..."
${KEDRO} run --pipeline normalization.spot_intraday_4h
if [ $? -ne 0 ]; then
    echo "[WARN] normalization.spot_intraday_4h FAILED — continuing"
fi

# ── Macro FRED ────────────────────────────────
echo "[$(date '+%H:%M:%S')] Running ingestion.fred.incremental..."
${KEDRO} run --pipeline ingestion.fred.incremental
if [ $? -ne 0 ]; then
    echo "[WARN] ingestion.fred.incremental FAILED — continuing"
fi

echo "[$(date '+%H:%M:%S')] Running normalization.macro_daily..."
${KEDRO} run --pipeline normalization.macro_daily
echo "[$(date '+%H:%M:%S')] Running normalization.macro_weekly..."
${KEDRO} run --pipeline normalization.macro_weekly
echo "[$(date '+%H:%M:%S')] Running normalization.macro_monthly..."
${KEDRO} run --pipeline normalization.macro_monthly

# ── Yahoo Finance ─────────────────────────────
echo "[$(date '+%H:%M:%S')] Running ingestion.yfinance.incremental..."
${KEDRO} run --pipeline ingestion.yfinance.incremental
if [ $? -ne 0 ]; then
    echo "[WARN] ingestion.yfinance.incremental FAILED — continuing"
fi

echo "[$(date '+%H:%M:%S')] Running normalization.spot_business_day..."
${KEDRO} run --pipeline normalization.spot_business_day

# ── CoinGlass 4h ──────────────────────────────
echo "[$(date '+%H:%M:%S')] Running ingestion.coinglass.derivatives_4h..."
${KEDRO} run --pipeline ingestion.coinglass.derivatives_4h
if [ $? -ne 0 ]; then
    echo "[WARN] ingestion.coinglass.derivatives_4h FAILED — continuing"
fi

echo "[$(date '+%H:%M:%S')] Running normalization.derivatives_4h..."
${KEDRO} run --pipeline normalization.derivatives_4h

# ── CoinGlass Order Book 4h ───────────────────
echo "[$(date '+%H:%M:%S')] Running ingestion.coinglass.orderbook_4h..."
${KEDRO} run --pipeline ingestion.coinglass.orderbook_4h
if [ $? -ne 0 ]; then
    echo "[WARN] ingestion.coinglass.orderbook_4h FAILED — continuing"
fi

echo "[$(date '+%H:%M:%S')] Running normalization.orderbook_4h..."
${KEDRO} run --pipeline normalization.orderbook_4h
if [ $? -ne 0 ]; then
    echo "[WARN] normalization.orderbook_4h FAILED — continuing"
fi

# ── CoinGlass supplemental (funding OI, OI agg, fear/greed) ──
echo "[$(date '+%H:%M:%S')] Running download_funding_rate_oi_weighted..."
${PYTHON} ${PROJECT}/scripts/download_funding_rate_oi_weighted.py
if [ $? -ne 0 ]; then
    echo "[WARN] download_funding_rate_oi_weighted FAILED — continuing"
fi

echo "[$(date '+%H:%M:%S')] Running download_oi_aggregated..."
${PYTHON} ${PROJECT}/scripts/download_oi_aggregated.py
if [ $? -ne 0 ]; then
    echo "[WARN] download_oi_aggregated FAILED — continuing"
fi

echo "[$(date '+%H:%M:%S')] Running download_fear_greed..."
${PYTHON} ${PROJECT}/scripts/download_fear_greed.py
if [ $? -ne 0 ]; then
    echo "[WARN] download_fear_greed FAILED — continuing"
fi

# ── CoinGlass Indices (daily) ────────────────
echo "[$(date '+%H:%M:%S')] Running download_ahr999..."
${PYTHON} ${PROJECT}/scripts/download_ahr999.py
if [ $? -ne 0 ]; then
    echo "[WARN] download_ahr999 FAILED — continuing"
fi

echo "[$(date '+%H:%M:%S')] Running download_bubble_index..."
${PYTHON} ${PROJECT}/scripts/download_bubble_index.py
if [ $? -ne 0 ]; then
    echo "[WARN] download_bubble_index FAILED — continuing"
fi

echo "[$(date '+%H:%M:%S')] Running download_puell_multiple..."
${PYTHON} ${PROJECT}/scripts/download_puell_multiple.py
if [ $? -ne 0 ]; then
    echo "[WARN] download_puell_multiple FAILED — continuing"
fi

echo "[$(date '+%H:%M:%S')] Running download_stablecoin_mcap..."
${PYTHON} ${PROJECT}/scripts/download_stablecoin_mcap.py
if [ $? -ne 0 ]; then
    echo "[WARN] download_stablecoin_mcap FAILED — continuing"
fi

echo "[$(date '+%H:%M:%S')] Running download_cdri_index..."
${PYTHON} ${PROJECT}/scripts/download_cdri_index.py
if [ $? -ne 0 ]; then
    echo "[WARN] download_cdri_index FAILED — continuing"
fi

echo "[$(date '+%H:%M:%S')] Running download_cgdi_index..."
${PYTHON} ${PROJECT}/scripts/download_cgdi_index.py
if [ $? -ne 0 ]; then
    echo "[WARN] download_cgdi_index FAILED — continuing"
fi

# ── HMM Regime ────────────────────────────────
echo "[$(date '+%H:%M:%S')] Running modeling.regime_hmm..."
${KEDRO} run --pipeline modeling.regime_hmm
if [ $? -ne 0 ]; then
    echo "[ERROR] modeling.regime_hmm FAILED"
    exit 1
fi

# ── Data Quality Report ───────────────────────
echo "[$(date '+%H:%M:%S')] Running DQ display..."
${PYTHON} ${PROJECT}/scripts/cron/dq_daily_update.py \
    2>&1 | tee -a ${LOG_DIR}/daily_update.log

echo "[$(date '+%H:%M:%S')] Daily Update COMPLETE"

# last_update_status.json is written by dq_daily_update.py

