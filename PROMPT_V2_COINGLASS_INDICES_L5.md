Task:
- Integrate CoinGlass daily indices (fear_greed, ahr999, puell_multiple, bubble_index, stablecoin_mcap, cdri_index, cgdi_index) into the L5 4h model features pipeline.

Constraints:
- Kedro project; respect CLAUDE.md layer contracts strictly
- L1→L2: no feature engineering. L3: per-asset features only, backward-looking windows
- Daily indices forward-filled into 4h grid with mandatory D+1 day shift (same anti-lookahead pattern as regime context in join_regime_context node)
- NaN at start of series (burn-in) is permitted — never dropna
- Do not alter existing L5 nodes (join_spot_and_derivatives, join_regime_context, join_orderbook, add_target)
- Add new node(s) in the L5 DAG — insert between join_orderbook and add_target
- All parameters from parameters.yml — no hardcoded paths or column names

Context:
- Source files already exist in data/01_raw/derivatives/coinglass/indices/:
  - fear_greed.parquet (daily, col: fear_greed)
  - ahr999.parquet (daily)
  - puell_multiple.parquet (daily)
  - bubble_index.parquet (daily)
  - stablecoin_mcap.parquet (daily)
  - cdri_index.parquet (daily)
  - cgdi_index.parquet (daily)
- These files have NO catalog entry, NO L2 normalization, NO L3 features yet
- Pipeline path needed:
  1. catalog_l1.yml: add PartitionedDataset for data/01_raw/derivatives/coinglass/indices/
  2. L2 normalization: new pipeline normalization.coinglass_indices — schema validation only (timestamp as DatetimeIndex UTC, float64 enforcement, no transforms)
  3. L3 features: new pipeline primary.coinglass_indices — compute per-index features:
     - {index}_zscore_30d = (value - rolling_mean_30) / rolling_std_30
     - {index}_change_7d = value.pct_change(7)
     - {index}_ma_ratio = value / value.rolling(30).mean()
  4. L5 join: new node join_coinglass_indices between join_orderbook and add_target — same forward-fill + D+1 shift pattern as join_regime_context
- Update L5 DAG: join_spot_and_derivatives → join_regime_context → join_orderbook → join_coinglass_indices → add_target
- Register all new pipelines in pipeline_registry.py
- Add datasets to catalog_l1.yml, catalog_l2.yml, catalog_l3.yml, catalog_l5.yml

Output Format:
- New: pipelines/normalization/coinglass_indices/ (nodes.py, pipeline.py)
- New: pipelines/primary/coinglass_indices/ (nodes.py, pipeline.py)
- Modified: pipelines/primary/model_features_4h/nodes.py (add join_coinglass_indices)
- Modified: pipelines/primary/model_features_4h/pipeline.py (update DAG)
- Modified: pipeline_registry.py, catalog_l1-l5.yml, parameters.yml
- No explanations
