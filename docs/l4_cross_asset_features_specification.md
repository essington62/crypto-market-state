## L4 (04_cross_asset) - Cross-Asset Features Specification

### Objetivo

Criar features **cross-asset** que descrevem o **estado global de mercado** em cada data, combinando:
- Séries macroeconômicas do FRED (taxas, inflação, crescimento, liquidez)
- Séries financeiras do Yahoo Finance (índices de equity, VIX, DXY, ouro)

O dataset resultante deve permitir:
- **Detecção de regimes** (risk-on / risk-off, alta / baixa volatilidade)
- **Controle de risco** (stress de mercado, liquidez, dólar forte/fraco)
- **Condicionamento de modelos preditivos** (features de contexto macro / cross-asset)

---

### Princípios de Design

1. **Camada puramente de features**  
   - Nenhum modelo (ML, estatístico ou econométrico) é treinado aqui  
   - Nenhum label/target é criado

2. **Sem lookahead bias**  
   - Todas as features usam apenas dados até a data de referência  
   - Janelas rolling são estritamente backward-looking

3. **Cross-asset explícito e interpretável**  
   - Combinações claras: equity vs VIX, dólar vs inflação, curva de juros, ouro vs equity  
   - Nenhuma transformação opaca (PCA, clustering, etc.)

4. **Sem transformação temporal agressiva**  
   - **Sem resampling**, **sem forward-fill**, **sem interpolação**  
   - Datas válidas são apenas interseções (inner join em `date`)

5. **Separação de responsabilidades**  
   - L1–L3 cuidam de ingestão, normalização e features per-asset  
   - L4 apenas consome `*_primary` e gera features globais por data

---

### Inputs do L4

#### 1) `fred_macro_primary`

- **Tipo**: `PartitionedDataset` (uma partição por série FRED)  
- **Chave de partição**: `series_id`  
- **Esquema (por DataFrame)**:
  - `date` (datetime64[ns, UTC])
  - `value` (float64)
  - `series_id` (string)
  - `asset` (string) — identificador amigável (ex: `us_10y_yield`, `cpi`)
  - `category` (string) — ex: `yield_curve`, `inflation`, `monetary_policy`, `liquidity`, `growth`
  - `source` (string) — `"fred"`
  - `interval` (string) — `"1d"`
  - `ingestion_ts` (datetime64[ns, UTC])
  - Features primárias (L3): deltas, pct_changes, rolling_means/std, z-scores, percentiles, etc.

#### 2) `yfinance_macro_primary`

- **Tipo**: `PartitionedDataset` (uma partição por ativo / symbol)  
- **Chave de partição**: `symbol`  
- **Esquema (por DataFrame)**:
  - `date` (datetime64[ns, UTC])
  - `open`, `high`, `low`, `close` (float64)
  - `volume` (float64)
  - `symbol` (string) — ex: `^GSPC`, `^VIX`, `DX-Y.NYB`, `GC=F`
  - `asset` (string) — ex: `sp500`, `vix`, `dxy`, `gold`
  - `category` (string) — ex: `equity_index`, `volatility`, `fx`, `commodity`, `rates`
  - `source` (string) — `"yfinance"`
  - `interval` (string) — `"1d"`
  - `ingestion_ts` (datetime64[ns, UTC])
  - Features primárias (L3): returns, log_returns, rolling_means/std, z-scores, realized_vol, percentiles, etc.

#### Garantias herdadas de L3

- Dados por asset/série:
  - Ordenados por `date` (crescente)
  - Sem duplicatas de data
  - Timezone `UTC` preservado
- Features L3 já calculadas (z-scores, rolling stats, percentiles)

---

### Output do L4

- **Dataset**: `cross_asset_features`
- **Tipo**: **não-particionado** (`pd.DataFrame` único)
- **Index**: opcional (padrão: coluna `date` normal, sem setar index no contrato)
- **Grão**: **uma linha por `date`** (interseção de datas relevantes)
- **Formato**: definido no `catalog.yml` (sugestão: Parquet, daily-level)

#### Esquema esperado (mínimo)

Colunas principais (sujeito a extensão futura, mas com estes nomes mínimos):

- `date`
- `vix_level`
- `vix_zscore_63`
- `equity_vol_risk_index`
- `dxy_zscore_252`
- `real_rate_proxy`
- `yield_curve_slope`
- `growth_vs_inflation_score`
- `gold_vs_equity_ratio`
- `equity_vol_63`
- `vol_regime_flag`

#### Semântica

- Uma linha por data representa o **estado global de mercado** naquele dia.
- Valores podem ser `NaN`:
  - Início de janelas rolling (burn-in)
  - Datas onde algum ativo/série ainda não possuía observações

---

### Node L4 - Contrato

#### Assinatura

```python
def build_cross_asset_features(
    fred: Dict[str, Callable[[], pd.DataFrame]],
    yfinance: Dict[str, Callable[[], pd.DataFrame]],
) -> pd.DataFrame:
    ...
```

#### Regras de Design (obrigatórias)

- **Função pura**:
  - Sem efeitos colaterais
  - Não grava/em leitura de disco
  - Não acessa `kedro.io` ou `Dataset` dentro do node

- **Entrada compatível com PartitionedDataset**:
  - Aceitar tanto `Callable[[], DataFrame]` quanto `DataFrame` direto
  - Sempre usar `df.copy()` antes de manipular

- **Integridade temporal**:
  - `sort_values("date")` explícito em todos os DataFrames intermediários
  - `drop_duplicates(subset=["date"], keep="last")` explícito
  - **Nenhum resample**
  - **Nenhum forward-fill**
  - **Nenhuma mudança de timezone** (assumir UTC já garantido por L2/L3)

- **Join**:
  - Sempre **inner join** em `date` (interseção de datas)
  - Não juntar por symbol/series_id, apenas por `date`

- **NaNs**:
  - Permitidos e esperados em:
    - Início de janelas rolling
    - Períodos em que algum ativo/série ainda não existe
    - Features condicionais em que insumos estão ausentes (ex: falta de série de crescimento)

---

### Seleção de Ativos (sem hardcoding de tickers)

Ativos e séries são identificados por **metadados**, não por tickers fixos.

- **Yahoo Finance (`yfinance_macro_primary`)**  
  Seleção via coluna `asset` / `category`:
  - `asset = "sp500"` (categoria `equity_index`) → índice de ações principal
  - `asset = "vix"` (categoria `volatility`) → índice de volatilidade implícita
  - `asset = "dxy"` (categoria `fx`) → índice do dólar
  - `asset = "gold"` (categoria `commodity`) → hedge de inflação / risk-off

- **FRED (`fred_macro_primary`)**  
  Seleção via `category` + `asset`:
  - `category = "yield_curve"`:
    - `asset` contendo `"10y"` / `"30y"` → long rate (ex: `us_10y_yield`)
    - `asset` contendo `"2y"` / `"3m"` → short rate (curto prazo)
  - `category = "inflation"`:
    - ex: `asset = "cpi"`
  - `category = "growth"`:
    - ex: séries de crescimento (PIB, produção industrial) se/quando existirem

> Importante: o código deve se basear em `asset`/`category` e não em valores brutos de `ticker` ou `series_id`.

---

### Blocos de Features do L4

#### 1. RISK / STRESS

**Objetivo**: medir aversão a risco e estresse de mercado a partir de volatilidade implícita e retorno de equity.

Features mínimas:

- `vix_level`  
  - Fonte: `yfinance_macro_primary` (`asset = "vix"`)  
  - Cálculo: `vix_level = close` (nível do índice VIX)

- `vix_zscore_63`  
  - Fonte: mesma partição (`vix`)  
  - Cálculo: `zscore_63` da série `close` (rolling 63 dias) herdado do L3

- `equity_vol_risk_index`  
  - Combina risco (volatilidade) com retorno de equity:
  - Componentes:
    - `vix_zscore_63` (stress / volatilidade implícita)
    - `sp500_return_21d` (`asset = "sp500"`, feature L3 `return_21d`)
  - Cálculo conceitual:
    - \[
    \text{equity_vol_risk_index} = \text{vix_zscore_63} - \text{sp500_return_21d}
    \]
  - Interpretação:
    - Valores altos: VIX alto e/ou equity fraco → regime de stress
    - Valores baixos: VIX baixo e equity forte → regime benigno

#### 2. LIQUIDITY / DOLLAR

**Objetivo**: capturar força do dólar e um proxy de juros reais.

- `dxy_zscore_252`  
  - Fonte: `yfinance_macro_primary` (`asset = "dxy"`)  
  - Cálculo: `zscore_252` da série `close` (herdado do L3)

- `real_rate_proxy`  
  - Combina taxa de juros longa nominal com inflação:
  - Componentes:
    - `long_rate`: série FRED com `category = "yield_curve"` e `asset` tipo `us_10y_yield` (ou similar)
    - `inflation_zscore_252`: `zscore_252` da série de inflação (ex: `cpi`, `category = "inflation"`)
  - Cálculo conceitual:
    - \[
    \text{real_rate_proxy} = \text{long_rate} - \text{inflation\_zscore\_252}
    \]
  - Observação:
    - Se alguma das séries não existir, a coluna é preenchida com `NaN`.

#### 3. MACRO REGIME

**Objetivo**: caracterizar o regime macro em termos de curva de juros e balanço crescimento vs inflação.

- `yield_curve_slope`  
  - Componentes:
    - `long_rate`: FRED `category = "yield_curve"` (10y / 30y)
    - `short_rate`: FRED `category = "yield_curve"` (2y / 3m)
  - Cálculo:
    - \[
    \text{yield\_curve\_slope} = \text{long\_rate} - \text{short\_rate}
    \]
  - Interpretação:
    - > 0: curva normal (expectativa de crescimento/inflacao futura)
    - < 0: curva invertida (sinal clássico de stress/recessão)

- `growth_vs_inflation_score`  
  - Componentes:
    - `growth_zscore_252`: z-score de alguma série de crescimento (ex: PIB, produção industrial), `category = "growth"`
    - `inflation_zscore_252`: z-score de inflação, `category = "inflation"`
  - Cálculo:
    - \[
    \text{growth\_vs\_inflation\_score} = \text{growth\_zscore\_252} - \text{inflation\_zscore\_252}
    \]
  - Interpretação:
    - > 0: crescimento relativamente forte vs inflação
    - < 0: inflação relativamente forte vs crescimento

Se qualquer uma das séries faltar, `growth_vs_inflation_score` fica `NaN`.

#### 4. CROSS-ASSET RELATIONSHIPS

**Objetivo**: quantificar relações entre classes de ativos (equity, ouro, cripto, etc.).

- `gold_vs_equity_ratio`  
  - Componentes:
    - `gold_close`: `asset = "gold"` (ouro)
    - `sp500_close`: `asset = "sp500"` (equity)
  - Cálculo:
    - \[
    \text{gold\_vs\_equity\_ratio} = \frac{\text{gold\_close}}{\text{sp500\_close}}
    \]
  - Interpretação:
    - Valores altos: ouro caro vs ações (regime defensivo / risk-off)
    - Valores baixos: ouro barato vs ações (risk-on)

- `equity_vs_crypto_momentum`  
  - Feature reservada para comparar momentum de equity vs cripto.
  - No contrato atual do L4:
    - O node de L4 **não recebe** L3 de cripto, então o valor pode ser deixado como `NaN` até que o input seja estendido.

#### 5. VOLATILITY REGIME

**Objetivo**: identificar regimes de alta/baixa volatilidade estrutural em equity.

- `equity_vol_63`  
  - Fonte: `yfinance_macro_primary` (`asset = "sp500"`)  
  - Cálculo: `rolling_std_63` da série `close` (herdado do L3)

- `vol_regime_flag`  
  - Usa z-score de volatilidade/preço de equity:
  - Exemplo de regra:
    - \[
    \text{vol\_regime\_flag} = (\text{sp500\_zscore\_63} > 1.5)
    \]
  - Tipo: booleano (True/False), podendo ser `NaN` quando o z-score não existe.

---

### Pipeline Kedro (L4)

Definição do pipeline:

```python
from kedro.pipeline import Pipeline, node
from crypto_mkt_state.pipelines.cross_asset.nodes import build_cross_asset_features


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=build_cross_asset_features,
                inputs={
                    "fred": "fred_macro_primary",
                    "yfinance": "yfinance_macro_primary",
                },
                outputs="cross_asset_features",
                name="build_cross_asset_features",
            ),
        ]
    )
```

---

### Invariantes do L4

- **Integridade Temporal**
  - ✅ `date` preservado de L3 (UTC)
  - ✅ Ordenação crescente por `date`
  - ✅ Sem duplicatas de `date` no dataset final
  - ✅ Inner join em `date` entre blocos intermediários

- **Integridade de Features**
  - ✅ Nenhum uso de informações futuras (apenas histórico)
  - ✅ Janelas rolling respeitam o burn-in (NaNs no início da série)
  - ✅ Features são combinações simples de:
    - valores (`value`, `close`)
    - z-scores
    - diferenças entre séries
    - razões entre séries

- **Integridade de Dados Originais**
  - ✅ Nenhuma alteração nos datasets de origem (`fred_macro_primary`, `yfinance_macro_primary`)
  - ✅ L4 trabalha apenas sobre cópias (`df.copy()`)

---

### O que ENTRA no L4

✅ **Features cross-asset**:
- Relações equity–volatilidade (SP500 vs VIX)
- Dólar vs outras classes (DXY)
- Curva de juros (long vs short rate)
- Crescimento vs inflação (growth vs inflation z-scores)
- Ouro vs equity (gold vs SP500)
- Regimes de volatilidade

✅ **Cálculos permitidos**:
- Operações aritméticas simples (`+`, `-`, `*`, `/`)
- Combinações lineares de z-scores e retornos
- Ratios entre preços de ativos
- Joins em `date`

---

### O que NÃO ENTRA no L4

❌ **Modelos / ML / Estatística avançada**:
- Nenhum modelo de regressão, árvore, rede neural, etc.
- Nenhum clustering (k-means, HMM, etc.)
- Nenhuma redução de dimensionalidade (PCA, autoencoders, etc.)

❌ **Labels / Targets**:
- Nenhuma variável de resposta ou label (ex: retorno futuro, drawdown futuro)

❌ **Heurísticas econômicas escondidas**:
- Nenhum if/else com lógica econômica subjetiva (ex: "se taxa > 3% então…")
- Apenas combinações diretas e transparentes de séries e z-scores

❌ **Transformações temporais adicionais**:
- Nenhum resampling de frequência
- Nenhum forward-fill de gaps
- Nenhuma interpolação

❌ **Acesso a IO ou Kedro dentro do node**:
- Nenhum acesso a disco
- Nenhuma chamada explícita a `kedro.io.Dataset` ou similares

---

### Extensibilidade Futura

O L4 foi desenhado para ser facilmente estendido:

1. **Novos ativos/séries**  
   - Adicionar novas séries em `parameters.yml` (FRED/YFinance)  
   - L1–L3 passam a produzi-las automaticamente  
   - L4 pode incorporar novas relações (ex: crédito, commodities adicionais)

2. **Novos blocos de features**  
   - Novos grupos de features podem ser adicionados (ex: spreads de crédito, correlações rolling)  
   - Desde que:
     - Respeitem as regras de pureza e integridade temporal
     - Sejam interpretáveis e documentados

3. **Integração com crypto (L3)**  
   - Futuras versões podem estender o node para aceitar `crypto_ohlcv_daily_primary`  
   - Permite implementar de fato `equity_vs_crypto_momentum`

---

### Resumo de Design

| Camada | Grão | Particionamento | Foco |
|--------|------|-----------------|------|
| L1 | tick-by / daily | Por fonte/asset | Ingestão bruta |
| L2 | daily | Por asset/série | Normalização de schema + metadata |
| L3 | daily | Por asset/série | Features estatísticas per-asset |
| **L4** | **daily** | **Nenhum** (dataset único) | **Features cross-asset e estado global de mercado** |

L4 é a camada onde a visão fragmentada (por asset/série) é consolidada em um **vetor de estado único por data**, ainda puramente descritivo e adequado para ser insumo de modelos de regime/risk-control.

