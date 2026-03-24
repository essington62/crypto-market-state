# PROMPT V2 — Data Lake: Ingestão dos novos tickers até L2
## Projeto: crypto-market-state (data lake)
## Env: crypto_market_state
## Escopo: APENAS este projeto — não tocar no crypto_v3

---

Leia o CLAUDE.md deste projeto antes de qualquer ação:
`/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/CLAUDE.md`

---

## Contexto

Estamos expandindo o data lake com novos índices para enriquecer o modelo de regime
de BTC no projeto v3 (crypto-regime-trader). O modelo v3 está sofrendo concept drift
em 2026: BTC está se comportando como risk asset (correlacionado com crédito e energia),
o que não era o padrão em 2023-2024.

Os novos índices cobrem:
- **Crédito**: HYG (high yield), LQD (investment grade) → stress de crédito
- **Bond market**: MOVE Index → incerteza sobre política do Fed
- **Energia/Geopolítico**: Oil WTI, Oil Brent, Natural Gas, OVX → pressão inflacionária
- **Inflação**: TIP (TIPS ETF) → expectativas de breakeven
- **Geopolítico**: ITA (Defense ETF) → apetite por risco em tensão

---

## O que já foi feito

Os tickers já foram adicionados ao `conf/base/parameters.yml` (seção yfinance.assets):
- CL=F → oil_wti, category: commodity
- BZ=F → oil_brent, category: commodity
- NG=F → natural_gas, category: commodity
- ^MOVE → move_index, category: volatility
- OVX → oil_volatility, category: volatility
- ITA → defense_etf, category: geopolitical
- HYG → high_yield_bonds, category: credit
- LQD → investment_grade_bonds, category: credit
- TIP → tips_etf, category: inflation

O pipeline `normalization.spot_business_day` já processa todos os tickers em
`yfinance.assets` automaticamente — não precisa criar nenhum novo pipeline.

---

## Tarefas

### Tarefa 1 — Verificar configuração

Confirmar que os tickers estão corretamente no parameters.yml:
```bash
grep -A3 "oil_wti\|move_index\|high_yield\|tips_etf\|defense_etf" \
  conf/base/parameters.yml | head -60
```

### Tarefa 2 — Rodar ingestão yfinance

```bash
kedro run --pipeline ingestion.yfinance.incremental
```

Se algum ticker falhar (ex: `^MOVE` pode ter nome diferente no yfinance), verificar
o erro, corrigir o ticker no parameters.yml se necessário, e rodar novamente.

**Atenção:** O `^MOVE` (MOVE Index) pode não estar disponível no yfinance com esse
ticker. Testar manualmente se necessário:
```python
import yfinance as yf
df = yf.download("^MOVE", start="2023-01-01")
print(df.head())
```
Se não disponível, remover do parameters.yml e registrar no CLAUDE.md como
"não disponível via yfinance — buscar fonte alternativa".

### Tarefa 3 — Rodar normalização L2

```bash
kedro run --pipeline normalization.spot_business_day
```

### Tarefa 4 — Verificar os arquivos L2 gerados

```bash
ls -la data/02_intermediate/spot/business_day/
```

Para cada arquivo novo, verificar cobertura e qualidade:
```python
import pandas as pd, os

files = {
    "oil_wti":              "data/02_intermediate/spot/business_day/oil_wti.parquet",
    "oil_brent":            "data/02_intermediate/spot/business_day/oil_brent.parquet",
    "natural_gas":          "data/02_intermediate/spot/business_day/natural_gas.parquet",
    "move_index":           "data/02_intermediate/spot/business_day/move_index.parquet",
    "oil_volatility":       "data/02_intermediate/spot/business_day/oil_volatility.parquet",
    "defense_etf":          "data/02_intermediate/spot/business_day/defense_etf.parquet",
    "high_yield_bonds":     "data/02_intermediate/spot/business_day/high_yield_bonds.parquet",
    "investment_grade_bonds":"data/02_intermediate/spot/business_day/investment_grade_bonds.parquet",
    "tips_etf":             "data/02_intermediate/spot/business_day/tips_etf.parquet",
}

for name, path in files.items():
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"{name}: {len(df)} rows | {df.index.min().date()} → {df.index.max().date()} | "
              f"close NaN: {df['close'].isna().sum()}")
    else:
        print(f"{name}: *** ARQUIVO NÃO GERADO ***")
```

### Tarefa 5 — Atualizar dq_daily_update.py

Verificar se as entradas de monitoramento dos novos tickers já foram adicionadas
ao script `scripts/cron/dq_daily_update.py`. Se não estiverem, adicionar à lista
FILES do script:

```python
{"name": "Oil WTI",       "path": "data/02_intermediate/spot/business_day/oil_wti.parquet",               "stale_days": 2},
{"name": "Oil Brent",     "path": "data/02_intermediate/spot/business_day/oil_brent.parquet",             "stale_days": 2},
{"name": "Natural Gas",   "path": "data/02_intermediate/spot/business_day/natural_gas.parquet",           "stale_days": 2},
{"name": "MOVE Index",    "path": "data/02_intermediate/spot/business_day/move_index.parquet",            "stale_days": 2},
{"name": "OVX",           "path": "data/02_intermediate/spot/business_day/oil_volatility.parquet",        "stale_days": 2},
{"name": "Defense ETF",   "path": "data/02_intermediate/spot/business_day/defense_etf.parquet",           "stale_days": 2},
{"name": "HYG",           "path": "data/02_intermediate/spot/business_day/high_yield_bonds.parquet",      "stale_days": 2},
{"name": "LQD",           "path": "data/02_intermediate/spot/business_day/investment_grade_bonds.parquet","stale_days": 2},
{"name": "TIP",           "path": "data/02_intermediate/spot/business_day/tips_etf.parquet",              "stale_days": 2},
```

### Tarefa 6 — Rodar o DQ check completo

```bash
python scripts/cron/dq_daily_update.py
```

Verificar que os novos tickers aparecem no relatório sem alertas de staleness.

---

## Constraints

- **Não tocar no crypto_v3** — escopo restrito a este projeto
- **Não criar novos pipelines** — usar os existentes (ingestion.yfinance.incremental
  e normalization.spot_business_day)
- **Não alterar parâmetros de tickers existentes** — apenas adicionar
- Se um ticker falhar, registrar no CLAUDE.md como "ticker indisponível" e continuar
- Código completo nos arquivos afetados

## Deliverables

1. ✅ L1: `data/01_raw/spot/business_day/{ticker}.parquet` para cada ticker
2. ✅ L2: `data/02_intermediate/spot/business_day/{ticker}.parquet` para cada ticker
3. ✅ `scripts/cron/dq_daily_update.py` atualizado com novos tickers
4. ✅ CLAUDE.md atualizado: registrar quais tickers foram ingeridos com sucesso
   e quais (se algum) não estavam disponíveis no yfinance
