"""
AÇÃO 0 — Validar endpoints Coinglass antes de implementar pipeline.

Run:
    export COINGLASS_API_KEY=<your_key>
    conda run -n crypto_market_state python notebooks/validate_coinglass_endpoints.py
"""

import requests
import json
import os
import time

key = os.environ["COINGLASS_API_KEY"]
base = "https://open-api-v4.coinglass.com"
headers = {"CG-API-KEY": key}

endpoints = {
    "funding_rate":    "/api/futures/fundingRate/oi-weight-ohlc-history?symbol=BTC&interval=1d&limit=5",
    "open_interest":   "/api/futures/openInterest/ohlc-aggregated-history?symbol=BTC&interval=1d&limit=5",
    "liquidations":    "/api/futures/liquidation/history?symbol=BTC&interval=1d&limit=5",
    "longshort_ratio": "/api/futures/globalLongShortAccountRatio/history?symbol=BTC&interval=1d&limit=5",
}

for name, path in endpoints.items():
    r = requests.get(base + path, headers=headers, timeout=30)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Status: {r.status_code}")
    try:
        data = r.json()
        print(json.dumps(data, indent=2)[:3000])
    except Exception as e:
        print(f"Parse error: {e}")
        print(r.text[:400])
    time.sleep(2.5)

print("\nDone.")
