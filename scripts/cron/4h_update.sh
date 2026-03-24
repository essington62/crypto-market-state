#!/bin/bash
# ─────────────────────────────────────────────────────────────
# 4h intraday update — crypto-market-state v1
# Runs at 00:15, 04:15, 08:15, 12:15, 16:15, 20:15 UTC
# Crontab: 15 0,4,8,12,16,20 * * * /path/to/4h_update.sh >> /path/to/logs/4h_update.log 2>&1
# ─────────────────────────────────────────────────────────────

PROJECT=/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state
CONDA_ENV=crypto_market_state
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/${CONDA_ENV}/bin/python
KEDRO=/opt/homebrew/Caskroom/miniforge/base/envs/${CONDA_ENV}/bin/kedro
LOG_DIR=${PROJECT}/logs
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "========================================"
echo "4h Update Started: ${DATE}"
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

echo "[$(date '+%H:%M:%S')] Credentials loaded ✓"

# ── BTC Spot 4h ───────────────────────────────
echo "[$(date '+%H:%M:%S')] Running ingestion.binance.spot_4h..."
${KEDRO} run --pipeline ingestion.binance.spot_4h
if [ $? -ne 0 ]; then
    echo "[ERROR] ingestion.binance.spot_4h FAILED"
    exit 1
fi

echo "[$(date '+%H:%M:%S')] Running normalization.spot_intraday_4h..."
${KEDRO} run --pipeline normalization.spot_intraday_4h
if [ $? -ne 0 ]; then
    echo "[ERROR] normalization.spot_intraday_4h FAILED"
    exit 1
fi

echo "[$(date '+%H:%M:%S')] Running primary.spot_4h..."
${KEDRO} run --pipeline primary.spot_4h
if [ $? -ne 0 ]; then
    echo "[ERROR] primary.spot_4h FAILED"
    exit 1
fi

# ── CoinGlass Derivatives 4h ──────────────────
echo "[$(date '+%H:%M:%S')] Running ingestion.coinglass.derivatives_4h..."
${KEDRO} run --pipeline ingestion.coinglass.derivatives_4h
if [ $? -ne 0 ]; then
    echo "[WARN] ingestion.coinglass.derivatives_4h FAILED — continuing"
fi

echo "[$(date '+%H:%M:%S')] Running normalization.derivatives_4h..."
${KEDRO} run --pipeline normalization.derivatives_4h
if [ $? -ne 0 ]; then
    echo "[WARN] normalization.derivatives_4h FAILED — continuing"
fi

echo "[$(date '+%H:%M:%S')] Running primary.derivatives_4h..."
${KEDRO} run --pipeline primary.derivatives_4h
if [ $? -ne 0 ]; then
    echo "[WARN] primary.derivatives_4h FAILED — continuing"
fi

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

echo "[$(date '+%H:%M:%S')] Running primary.orderbook_4h..."
${KEDRO} run --pipeline primary.orderbook_4h
if [ $? -ne 0 ]; then
    echo "[WARN] primary.orderbook_4h FAILED — continuing"
fi

# ── L5 Model Features ─────────────────────────
echo "[$(date '+%H:%M:%S')] Running primary.model_features_4h..."
${KEDRO} run --pipeline primary.model_features_4h
if [ $? -ne 0 ]; then
    echo "[WARN] primary.model_features_4h FAILED — continuing"
fi

# ── Lightweight DQ check ──────────────────────
echo "[$(date '+%H:%M:%S')] Running 4h DQ check..."
${PYTHON} - << 'PYEOF'
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

BASE = "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state"
NOW  = datetime.now(timezone.utc)
THRESHOLD_HOURS = 6

FILES = [
    ("BTC spot 4h",    f"{BASE}/data/01_raw/spot/crypto/4h/BTCUSDT.parquet"),
    ("CG deriv 4h",    f"{BASE}/data/01_raw/derivatives/coinglass/4h/BTCUSDT.parquet"),
    ("CG OB r1",       f"{BASE}/data/01_raw/orderbook/coinglass/4h/BTCUSDT_r1.parquet"),
    ("CG OB r2",       f"{BASE}/data/01_raw/orderbook/coinglass/4h/BTCUSDT_r2.parquet"),
]

issues = []
for name, path in FILES:
    try:
        df = pd.read_parquet(path)
        if df.empty:
            issues.append(f"{name}: EMPTY")
            continue
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(pd.to_datetime(df.index))
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        last = idx.max()
        hours_ago = (NOW - last).total_seconds() / 3600
        status = "FRESH" if hours_ago <= THRESHOLD_HOURS else "STALE"
        if status == "STALE":
            issues.append(f"{name}: {status} ({hours_ago:.1f}h ago)")
    except Exception as e:
        issues.append(f"{name}: ERROR ({e})")

if issues:
    print(f"[4h DQ] ⚠ {len(issues)} issue(s): " + " | ".join(issues))
    sys.exit(0)  # never break the cron
else:
    fresh_count = len(FILES)
    print(f"[4h DQ] ✓ {fresh_count}/{fresh_count} FRESH (threshold: {THRESHOLD_HOURS}h)")
PYEOF

echo "[$(date '+%H:%M:%S')] 4h Update COMPLETE"
