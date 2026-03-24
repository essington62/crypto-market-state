Task:
- Fix all remaining DQ issues: fear greed path mismatch, CoinGlass deriv 4h NaN, stale thresholds, and add missing pipelines to daily cron.

Constraints:
- Kedro project; respect CLAUDE.md layer contracts
- Do NOT alter yfinance nodes (already fixed)
- Do NOT alter download_funding_rate_oi_weighted.py or download_oi_aggregated.py (already working)
- L1 contract: no feature engineering, no rolling windows

Context:
- Issue 1 — CoinGlass fear greed (15d stale):
  - scripts/download_fear_greed.py saves to: data/01_raw/derivatives/coinglass/indices/fear_greed.parquet
  - scripts/cron/dq_deep_l1.py reads from: data/01_raw/derivatives/coinglass/fear_greed.parquet
  - Path mismatch — either update the script output path OR the DQ check path to match
  - Prefer updating dq_deep_l1.py FILES_TO_CHECK to point to the correct indices/ subfolder path

- Issue 2 — CoinGlass deriv 4h (4 NaN in last 5 rows):
  - Investigate: load data/01_raw/derivatives/coinglass/4h/BTCUSDT.parquet, check df.tail(5).isna().sum() per column
  - If NaN only in latest open candle → expected, no fix needed
  - If NaN in closed candles → fix numeric casting in ingestion node (pd.to_numeric errors="coerce")

- Issue 3 — Stale thresholds in dq_deep_l1.py FILES_TO_CHECK producing false positives:
  - FEDFUNDS: stale_days 2 → 45 (monthly series, ~1 month publication delay)
  - CPIAUCSL: stale_days 35 → 60 (monthly CPI, ~2 month delay)
  - WALCL weekly: stale_days 7 → 21 (weekly, ~2 week FRED delay)
  - DGS2: stale_days 2 → 5 (daily with weekends + FRED delay)
  - DGS10: stale_days 2 → 5 (same as DGS2)
  - MOVE Index: stale_days 4 → 6 (Cboe delay + weekends)
  - BTC spot 4h: stale_days 2 → 4 (pipeline not yet in daily cron)
  - CoinGlass OB r1: stale_days 2 → 4 (pipeline not yet in daily cron)
  - CoinGlass OB r2: stale_days 2 → 4 (pipeline not yet in daily cron)

- Issue 4 — Missing pipelines in scripts/cron/daily_update.sh:
  - Add ingestion.binance.spot_4h + normalization.spot_intraday_4h (after spot block)
  - Add ingestion.coinglass.orderbook_4h + normalization.orderbook_4h (after derivatives block)

Output Format:
- Modified: scripts/cron/dq_deep_l1.py (fear greed path + stale thresholds)
- Modified: scripts/cron/daily_update.sh (add spot_4h and orderbook_4h pipelines)
- Diagnostic: deriv 4h NaN column report
- No explanations
