# Data Lake Audit — 2026-03-24

## Resumo

Auditoria completa do data lake L1. Identificados 31 arquivos fantasma (~1.7 MB) não declarados em nenhum catalog, 6 duplicatas em paths diferentes, e 3 entradas de catalog sem arquivo correspondente no disco.

Nenhum arquivo deletado — mantidos para potencial uso em ablation study e feature discovery.

---

## Arquivos Fantasma (no disco, fora do catalog)

### CoinGlass ETF Flows — 9 arquivos (284 KB)

Path: `data/01_raw/derivatives/coinglass/etf/`

| Arquivo | Tamanho | Data |
|---------|---------|------|
| BTC_ARKB.parquet | 24K | 2026-03-09 |
| BTC_BITB.parquet | 27K | 2026-03-09 |
| BTC_FBTC.parquet | 24K | 2026-03-09 |
| BTC_GBTC.parquet | 27K | 2026-03-09 |
| BTC_IBIT.parquet | 29K | 2026-03-09 |
| BTC_holdings_consolidated.parquet | 29K | 2026-03-09 |
| BTC_netassets_consolidated.parquet | 31K | 2026-03-09 |
| BTC_flows_by_ticker.parquet | 33K | 2026-03-09 |
| BTC_flows_total.parquet | 11K | 2026-03-09 |

Potencial: ETF flows são feature group confirmado no ablation (Set D). Dados por ticker permitem construir `etf_flow_zscore`, `etf_cumsum`, `etf_trend`.

### CoinGlass Indices — 7 arquivos (663 KB)

Path: `data/01_raw/derivatives/coinglass/indices/`

| Arquivo | Tamanho | Data | Nota |
|---------|---------|------|------|
| ahr999.parquet | 103K | 2026-03-09 | Indicador de acumulação BTC |
| bubble_index.parquet | 187K | 2026-03-09 | Índice de bolha CoinGlass |
| cdri_index.parquet | 16K | 2026-03-09 | Crypto Derivatives Risk Index |
| cgdi_index.parquet | 16K | 2026-03-09 | CoinGlass Derivatives Index |
| fear_greed.parquet | 31K | 2026-03-24 | Ativo no cron |
| puell_multiple.parquet | 106K | 2026-03-09 | Múltiplo de Puell (mineração) |
| stablecoin_mcap.parquet | 154K | 2026-03-09 | Market cap stablecoins |

Potencial: `stablecoin_mcap` pode ser proxy de liquidez cripto. `ahr999` e `puell_multiple` são on-chain signals não testados. `cdri_index` e `cgdi_index` são derivativos compostos — possível feature para regime detection.

### CoinGlass Open Interest — 4 arquivos (328 KB)

Path: `data/01_raw/derivatives/coinglass/open_interest/`

| Arquivo | Tamanho | Data | Nota |
|---------|---------|------|------|
| BTCUSDT.parquet | 85K | 2026-03-09 | OI por exchange |
| BTCUSDT_aggregated.parquet | 86K | 2026-03-24 | Ativo no cron |
| BTCUSDT_coin_margin.parquet | 82K | 2026-03-09 | OI coin-margined |
| BTCUSDT_stablecoin.parquet | 75K | 2026-03-09 | OI stablecoin-margined |

Potencial: split coin-margin vs stablecoin pode revelar positioning de diferentes tipos de trader. Ratio `coin_margin / stablecoin` como signal de convicção.

### CoinGlass Options — 2 arquivos (231 KB)

Path: `data/01_raw/derivatives/coinglass/options/`

| Arquivo | Tamanho | Data |
|---------|---------|------|
| BTC_oi_by_exchange.parquet | 117K | 2026-03-09 |
| BTC_volume_by_exchange.parquet | 114K | 2026-03-09 |

Potencial: options OI/volume por exchange não testado. Put/call ratio implícito se os dados contêm breakdown por tipo.

### CoinGlass Taker — 1 arquivo (57 KB)

Path: `data/01_raw/derivatives/coinglass/taker/`

| Arquivo | Tamanho | Data |
|---------|---------|------|
| spot_BTC_aggregated.parquet | 57K | 2026-03-09 |

Potencial: taker buy/sell ratio é signal de agressão direcional. Não testado no ablation.

### CoinGlass Spot — 1 arquivo (94 KB)

Path: `data/01_raw/spot/coinglass/`

| Arquivo | Tamanho | Data |
|---------|---------|------|
| BTCUSDT_taker_buy_sell.parquet | 94K | 2026-03-09 |

Potencial: mesma lógica do taker acima, possivelmente com granularidade diferente.

---

## Duplicatas (mesmo dado em paths diferentes)

| Arquivo | Path A (raiz) | Path B (indices/) | Diff |
|---------|---------------|-------------------|------|
| fear_greed.parquet | coinglass/fear_greed.parquet (34K, 03-09) | coinglass/indices/fear_greed.parquet (31K, 03-24) | B é canônico (cron atualiza B) |
| ahr999.parquet | coinglass/ahr999.parquet (55K) | coinglass/indices/ahr999.parquet (103K) | B maior — provável histórico mais longo |
| bubble_index.parquet | coinglass/bubble_index.parquet (80K) | coinglass/indices/bubble_index.parquet (187K) | B 2.3× maior |
| puell_multiple.parquet | coinglass/puell_multiple.parquet (55K) | coinglass/indices/puell_multiple.parquet (106K) | B ~2× maior |
| vix.parquet | spot/business_day/vix.parquet (52K) | macro/daily/vix.parquet (5.4K) | Ambos atualizados 03-24. macro/daily é canônico |
| dxy.parquet | spot/business_day/dxy.parquet (52K) | macro/daily/dxy.parquet (5.3K) | Ambos atualizados 03-24. macro/daily é canônico |

Decisão: manter ambos por enquanto. Path canônico documentado acima.

---

## Entradas de Catalog sem Arquivo no Disco

| Dataset | Catalog | Path esperado | Status |
|---------|---------|---------------|--------|
| btc_regime_model | catalog_l4.yml | data/05_models/regime_hmm/btc_hmm.pkl | Ausente |
| btc_spot_daily_model_input_v2 | catalog_l3.yml | data/04_model_input/spot/daily/BTCUSDT_v2.parquet | Ausente (experimental) |
| btc_daily_signal | catalog_l4.yml | data/07_output/daily_signal/btc_signal.parquet | Layer incorreto |

---

## Features Não Testadas no Ablation

Baseado nos arquivos fantasma, estes dados existem mas não foram avaliados:

| Grupo | Dados disponíveis | Feature potencial | Prioridade |
|-------|-------------------|-------------------|------------|
| ETF Flows | 5 tickers + consolidado | etf_flow_zscore, etf_cumsum, etf_trend | Alta (Set D do ablation) |
| Stablecoin Mcap | mcap histórico | liquidez cripto proxy, variação semanal | Média |
| OI coin vs stablecoin | OI por tipo de margem | coin_margin_ratio como conviction signal | Média |
| Options OI/Volume | por exchange | put_call_ratio implícito, options_oi_change | Média |
| Taker Buy/Sell | spot aggregated | taker_ratio como aggression signal | Alta |
| On-chain (ahr999, puell) | índices compostos | ciclo de mineração, acumulação | Baixa (on-chain noisy) |
| Derivatives Indices (CDRI, CGDI) | índices compostos CoinGlass | risk/derivatives composite | Média |

---

## Ação

Nenhum arquivo deletado. Todos mantidos para avaliação futura no ablation study (Sets E-H).

Próximos passos sugeridos:
1. Avaliar ETF flows e taker ratio como candidatos prioritários para Set D/E
2. Decidir se indices/ passa a ser o path canônico para CoinGlass indices (substituindo raiz)
3. Limpar entradas de catalog órfãs (v2, btc_hmm.pkl)
