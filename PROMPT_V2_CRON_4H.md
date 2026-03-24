Task:
- Create a separate 4h cron script that runs every 4 hours for intraday data ingestion, normalization, and L3 features. Keep the existing daily_update.sh for macro/regime data.

Constraints:
- Kedro project; respect CLAUDE.md layer contracts
- Same credential loading pattern as daily_update.sh (secrets.yml)
- Same error handling: warn-and-continue for non-critical, exit-on-error for BTC spot
- Cron schedule: 00:15, 04:15, 08:15, 12:15, 16:15, 20:15 UTC (15min offset to let candles close)
- Log to logs/4h_update.log (separate from daily)
- Idempotent: safe to re-run without duplicating data

Context:
- New script: scripts/cron/4h_update.sh
- Pipelines to run every 4h (all already registered in pipeline_registry.py):
  1. ingestion.binance.spot_4h → normalization.spot_intraday_4h → primary.spot_4h
  2. ingestion.coinglass.derivatives_4h → normalization.derivatives_4h → primary.derivatives_4h
  3. ingestion.coinglass.orderbook_4h → normalization.orderbook_4h → primary.orderbook_4h
  4. primary.model_features_4h (L5 join — runs after all L3 sources are fresh)
- Do NOT include in 4h cron (stays in daily only):
  - ingestion.binance.spot (daily 24x7)
  - ingestion.fred.incremental
  - ingestion.yfinance.incremental
  - normalization.macro_daily/weekly/monthly
  - normalization.spot_business_day
  - primary.spot.crypto (daily L3)
  - primary.regime_context (daily L4)
  - modeling.regime_hmm (R11 daily)
  - CoinGlass supplemental scripts (funding OI, OI agg, fear greed — daily)
  - DQ report (stays in daily)
- Update daily_update.sh: remove spot_4h, derivatives_4h, orderbook_4h blocks (they move to 4h cron). Keep normalization.derivatives_4h in daily for regime_context dependency.
- Provide crontab entry for both scripts:
  - daily_update.sh: 0 7 * * * (existing)
  - 4h_update.sh: 15 0,4,8,12,16,20 * * *
- Add a lightweight DQ check at the end of 4h_update.sh: verify BTC spot 4h, deriv 4h, OB r1/r2 last dates are within 6h. Print one-line status, no full report.

Output Format:
- New: scripts/cron/4h_update.sh
- Modified: scripts/cron/daily_update.sh (remove 4h blocks that moved)
- Crontab entries for both
- No explanations
