# L3 (03_primary) - Primary Features Specification

## Objetivo

Criar features primárias **por ativo** (per-asset) que caracterizem:
- **Regime**: Estado de mercado predominante (tendência, volatilidade estrutural)
- **Stress**: Pressões e tensões no mercado (compressão, expansão, anomalias)
- **Predictability**: Grau de previsibilidade/estrutura nos movimentos

## Princípios de Design

1. **Sem lookahead bias**: Apenas dados históricos passados
2. **Rolling windows explícitas**: Janelas temporais claras e documentadas
3. **Robustez**: Features estáveis, não sinais frágeis ou altamente parametrizados
4. **Interpretabilidade**: Cada feature tem significado claro e direto
5. **Per-asset apenas**: Nenhuma agregação cross-asset no L3
6. **Estatísticas puras**: Foco em estatísticas descritivas, não modelos

---

## Input do L3

### Dataset de Entrada
- **Fonte**: `crypto_ohlcv_daily_intermediate` (L2)
- **Estrutura**: PartitionedDataset (um DataFrame por asset)
- **Index**: `timestamp` (DatetimeIndex, UTC)
- **Colunas disponíveis**:
  - `open`, `high`, `low`, `close` (float64)
  - `volume`, `quote_volume` (float64)
  - `close_time` (datetime64[ns, UTC])
  - `trades` (int64)
  - `taker_buy_base_volume`, `taker_buy_quote_volume` (float64)

### Garantias do L2
- Dados ordenados por timestamp (crescente)
- Sem duplicatas temporais
- Valores originais preservados
- Index temporal UTC

---

## Output do L3

### Estrutura
- **Dataset**: `crypto_ohlcv_daily_primary` (PartitionedDataset)
- **Index**: `timestamp` (DatetimeIndex, UTC) - **preservado do L2**
- **Particionamento**: Por asset (mantém estrutura do L2)
- **Formato**: Parquet (compression: snappy)

### Colunas do L3

O L3 **preserva todas as colunas do L2** e **adiciona** features primárias.

**Colunas herdadas do L2** (preservadas):
- `open`, `high`, `low`, `close`
- `volume`, `quote_volume`
- `close_time`
- `trades`
- `taker_buy_base_volume`, `taker_buy_quote_volume`

**Features primárias adicionadas** (novas colunas):

---

## Tabela de Features Primárias

| Feature | Tipo | Janela | Categoria | Descrição |
|---------|------|--------|-----------|-----------|
| `log_return_1d` | float64 | 1 dia | Returns | Log return diário: `ln(close / close.shift(1))` |
| `log_return_7d` | float64 | 7 dias | Returns | Log return acumulado 7 dias |
| `log_return_21d` | float64 | 21 dias | Returns | Log return acumulado 21 dias |
| `log_return_63d` | float64 | 63 dias | Returns | Log return acumulado 63 dias |
| `volatility_7d` | float64 | 7 dias | Volatilidade | Desvio padrão de log returns (rolling 7d) |
| `volatility_21d` | float64 | 21 dias | Volatilidade | Desvio padrão de log returns (rolling 21d) |
| `volatility_63d` | float64 | 63 dias | Volatilidade | Desvio padrão de log returns (rolling 63d) |
| `price_range_7d` | float64 | 7 dias | Volatilidade | `(high.max() - low.min()) / close.mean()` (normalizado) |
| `price_range_21d` | float64 | 21 dias | Volatilidade | `(high.max() - low.min()) / close.mean()` (normalizado) |
| `realized_volatility_7d` | float64 | 7 dias | Volatilidade | Soma dos log returns absolutos (7d) |
| `realized_volatility_21d` | float64 | 21 dias | Volatilidade | Soma dos log returns absolutos (21d) |
| `volume_ma_7d` | float64 | 7 dias | Liquidez | Média móvel de volume (7 dias) |
| `volume_ma_21d` | float64 | 21 dias | Liquidez | Média móvel de volume (21 dias) |
| `volume_zscore_7d` | float64 | 7 dias | Liquidez | Z-score de volume (vs média 7d) |
| `volume_zscore_21d` | float64 | 21 dias | Liquidez | Z-score de volume (vs média 21d) |
| `volume_change_7d` | float64 | 7 dias | Liquidez | Variação percentual de volume (7d) |
| `quote_volume_ma_7d` | float64 | 7 dias | Liquidez | Média móvel de quote_volume (7 dias) |
| `quote_volume_ma_21d` | float64 | 21 dias | Liquidez | Média móvel de quote_volume (21 dias) |
| `trades_ma_7d` | float64 | 7 dias | Liquidez | Média móvel de trades (7 dias) |
| `trades_ma_21d` | float64 | 21 dias | Liquidez | Média móvel de trades (21 dias) |
| `trades_zscore_7d` | float64 | 7 dias | Liquidez | Z-score de trades (vs média 7d) |
| `buy_pressure_7d` | float64 | 7 dias | Liquidez | `taker_buy_quote_volume / quote_volume` (média 7d) |
| `buy_pressure_21d` | float64 | 21 dias | Liquidez | `taker_buy_quote_volume / quote_volume` (média 21d) |
| `price_ma_7d` | float64 | 7 dias | Tendência | Média móvel simples de close (7 dias) |
| `price_ma_21d` | float64 | 21 dias | Tendência | Média móvel simples de close (21 dias) |
| `price_ma_63d` | float64 | 63 dias | Tendência | Média móvel simples de close (63 dias) |
| `price_slope_7d` | float64 | 7 dias | Tendência | Coeficiente angular de regressão linear (7d) |
| `price_slope_21d` | float64 | 21 dias | Tendência | Coeficiente angular de regressão linear (21d) |
| `price_slope_63d` | float64 | 63 dias | Tendência | Coeficiente angular de regressão linear (63d) |
| `price_position_7d` | float64 | 7 dias | Tendência | `(close - low.min()) / (high.max() - low.min())` (7d) |
| `price_position_21d` | float64 | 21 dias | Tendência | `(close - low.min()) / (high.max() - low.min())` (21d) |
| `candle_body_ratio` | float64 | 1 dia | Compressão | `abs(close - open) / (high - low)` |
| `candle_body_ratio_7d` | float64 | 7 dias | Compressão | Média de `candle_body_ratio` (7d) |
| `candle_body_ratio_21d` | float64 | 21 dias | Compressão | Média de `candle_body_ratio` (21d) |
| `upper_shadow_ratio` | float64 | 1 dia | Compressão | `(high - max(open, close)) / (high - low)` |
| `lower_shadow_ratio` | float64 | 1 dia | Compressão | `(low - min(open, close)) / (high - low)` |
| `shadow_ratio_7d` | float64 | 7 dias | Compressão | Média de `(upper_shadow + lower_shadow)` (7d) |
| `shadow_ratio_21d` | float64 | 21 dias | Compressão | Média de `(upper_shadow + lower_shadow)` (21d) |
| `high_low_ratio_7d` | float64 | 7 dias | Compressão | `high.max() / low.min()` (7d) |
| `high_low_ratio_21d` | float64 | 21 dias | Compressão | `high.max() / low.min()` (21d) |
| `autocorr_return_1d` | float64 | 21 dias | Predictability | Autocorrelação de log returns (lag 1) |
| `autocorr_return_7d` | float64 | 63 dias | Predictability | Autocorrelação de log returns (lag 7) |
| `hurst_exponent_21d` | float64 | 21 dias | Predictability | Expoente de Hurst (aproximado via R/S) |
| `hurst_exponent_63d` | float64 | 63 dias | Predictability | Expoente de Hurst (aproximado via R/S) |

**Total**: 44 features primárias + 11 colunas herdadas do L2 = **55 colunas no L3**

---

## Justificativa de Features por Categoria

### 1. Returns (Log Returns)

**Features**: `log_return_1d`, `log_return_7d`, `log_return_21d`, `log_return_63d`

**Cálculo**:
- `log_return_1d = ln(close / close.shift(1))`
- `log_return_Nd = ln(close / close.shift(N))` (acumulado)

**Justificativa**:
- **Regime**: Retornos acumulados indicam tendência de longo prazo (bull/bear)
- **Stress**: Retornos extremos (positivos ou negativos) indicam eventos de stress
- **Predictability**: Retornos consistentes sugerem regimes mais previsíveis

**Janelas**: 1d (curto), 7d (semanal), 21d (mensal), 63d (trimestral)

---

### 2. Volatilidade

**Features**: `volatility_7d`, `volatility_21d`, `volatility_63d`, `price_range_7d`, `price_range_21d`, `realized_volatility_7d`, `realized_volatility_21d`

**Cálculo**:
- `volatility_Nd = std(log_return_1d).rolling(N).std() * sqrt(N)` (anualizado aproximado)
- `price_range_Nd = (high.rolling(N).max() - low.rolling(N).min()) / close.rolling(N).mean()`
- `realized_volatility_Nd = abs(log_return_1d).rolling(N).sum()`

**Justificativa**:
- **Regime**: Volatilidade estrutural define regimes (alta volatilidade = regime volátil)
- **Stress**: Picos de volatilidade indicam stress de mercado
- **Predictability**: Baixa volatilidade sugere maior previsibilidade

**Janelas**: 7d (curto), 21d (médio), 63d (longo)

---

### 3. Liquidez / Volume

**Features**: `volume_ma_7d`, `volume_ma_21d`, `volume_zscore_7d`, `volume_zscore_21d`, `volume_change_7d`, `quote_volume_ma_7d`, `quote_volume_ma_21d`, `trades_ma_7d`, `trades_ma_21d`, `trades_zscore_7d`, `buy_pressure_7d`, `buy_pressure_21d`

**Cálculo**:
- `volume_ma_Nd = volume.rolling(N).mean()`
- `volume_zscore_Nd = (volume - volume.rolling(N).mean()) / volume.rolling(N).std()`
- `volume_change_7d = (volume - volume.shift(7)) / volume.shift(7)`
- `buy_pressure_Nd = (taker_buy_quote_volume / quote_volume).rolling(N).mean()`

**Justificativa**:
- **Regime**: Volumes consistentes indicam regimes estáveis
- **Stress**: Volumes anômalos (z-scores extremos) indicam eventos de stress
- **Predictability**: Alta liquidez (volumes/trades) facilita previsibilidade

**Janelas**: 7d (curto), 21d (médio)

---

### 4. Tendência

**Features**: `price_ma_7d`, `price_ma_21d`, `price_ma_63d`, `price_slope_7d`, `price_slope_21d`, `price_slope_63d`, `price_position_7d`, `price_position_21d`

**Cálculo**:
- `price_ma_Nd = close.rolling(N).mean()`
- `price_slope_Nd`: Coeficiente angular de regressão linear de `close` vs `range(0, N)` (rolling)
- `price_position_Nd = (close - low.rolling(N).min()) / (high.rolling(N).max() - low.rolling(N).min())`

**Justificativa**:
- **Regime**: Tendências definem regimes (alta, baixa, lateral)
- **Stress**: Mudanças bruscas de tendência indicam stress
- **Predictability**: Tendências consistentes são mais previsíveis

**Janelas**: 7d (curto), 21d (médio), 63d (longo)

---

### 5. Compressão / Expansão de Preço

**Features**: `candle_body_ratio`, `candle_body_ratio_7d`, `candle_body_ratio_21d`, `upper_shadow_ratio`, `lower_shadow_ratio`, `shadow_ratio_7d`, `shadow_ratio_21d`, `high_low_ratio_7d`, `high_low_ratio_21d`

**Cálculo**:
- `candle_body_ratio = abs(close - open) / (high - low)` (quando high != low)
- `upper_shadow_ratio = (high - max(open, close)) / (high - low)`
- `lower_shadow_ratio = (min(open, close) - low) / (high - low)`
- `shadow_ratio_Nd = (upper_shadow_ratio + lower_shadow_ratio).rolling(N).mean()`
- `high_low_ratio_Nd = high.rolling(N).max() / low.rolling(N).min()`

**Justificativa**:
- **Regime**: Compressão indica consolidação (regime lateral)
- **Stress**: Expansão extrema indica eventos de stress
- **Predictability**: Compressão precede movimentos grandes (maior previsibilidade potencial)

**Janelas**: 1d (instantâneo), 7d (curto), 21d (médio)

---

### 6. Predictability

**Features**: `autocorr_return_1d`, `autocorr_return_7d`, `hurst_exponent_21d`, `hurst_exponent_63d`

**Cálculo**:
- `autocorr_return_1d`: Autocorrelação de `log_return_1d` com lag 1 (rolling 21d)
- `autocorr_return_7d`: Autocorrelação de `log_return_1d` com lag 7 (rolling 63d)
- `hurst_exponent_Nd`: Expoente de Hurst aproximado via análise R/S (rolling)

**Justificativa**:
- **Regime**: Autocorrelação indica persistência de tendência (regime)
- **Stress**: Hurst próximo de 0.5 indica aleatoriedade (stress/incerteza)
- **Predictability**: Hurst > 0.5 indica tendência (previsível), < 0.5 indica reversão

**Janelas**: 21d (médio), 63d (longo)

---

## Invariantes do L3

### 3.1 Integridade Temporal
- ✅ `timestamp` preservado do L2 (DatetimeIndex, UTC)
- ✅ Ordenação crescente por `timestamp`
- ✅ Sem duplicatas temporais
- ✅ Todas as features calculadas apenas com dados passados (sem lookahead)

### 3.2 Integridade de Features
- ✅ Features de janela N requerem pelo menos N observações históricas
- ✅ Valores NaN permitidos apenas no início (janelas incompletas)
- ✅ Features de retorno: podem ser negativas, zero ou positivas
- ✅ Features de volatilidade: sempre não-negativas
- ✅ Features de volume: sempre não-negativas
- ✅ Features de ratio: sempre entre 0 e 1 (quando aplicável)
- ✅ Features de z-score: podem ser negativas, zero ou positivas

### 3.3 Integridade de Dados Originais
- ✅ Todas as colunas do L2 são preservadas sem modificação
- ✅ Valores originais (open, high, low, close, volumes) permanecem idênticos ao L2
- ✅ Nenhuma transformação é aplicada aos dados originais

### 3.4 Estrutura do DataFrame
- ✅ Index = `timestamp` (DatetimeIndex, UTC)
- ✅ Ordenação crescente por `timestamp`
- ✅ Particionamento por asset mantido
- ✅ Tipos de dados explícitos e consistentes

---

## O que ENTRA no L3

✅ **Features primárias per-asset**:
- Returns (log returns em múltiplas janelas)
- Volatilidade (múltiplas medidas e janelas)
- Liquidez (volumes, trades, buy pressure)
- Tendência (médias móveis, slopes, posição relativa)
- Compressão/expansão (ratios de candles, shadows)
- Predictability (autocorrelação, Hurst)

✅ **Colunas herdadas do L2** (preservadas):
- Todas as colunas originais do L2

✅ **Cálculos permitidos**:
- Rolling windows (backward-looking apenas)
- Estatísticas descritivas (média, std, min, max, sum)
- Transformações matemáticas simples (log, ratio, z-score)
- Regressão linear simples (para slopes)
- Autocorrelação e Hurst (estatísticas de séries temporais)

---

## O que NÃO ENTRA no L3

❌ **Cross-asset features**:
- Nenhuma comparação entre assets
- Nenhuma agregação cross-asset
- Nenhuma correlação entre assets

❌ **Labels ou targets**:
- Nenhum label futuro (ex: "preço em 7 dias")
- Nenhum target de classificação
- Nenhuma variável dependente

❌ **Modelos ou predições**:
- Nenhum modelo ML
- Nenhuma predição
- Nenhum score de modelo

❌ **Features derivadas de modelos**:
- Nenhum residual de modelo
- Nenhum componente de decomposição (ex: STL, ETS)
- Nenhum indicador técnico complexo (ex: MACD, RSI - exceto os simples acima)

❌ **Transformações temporais**:
- Nenhum preenchimento de gaps
- Nenhuma interpolação
- Nenhum resampling

❌ **Features de mercado externo**:
- Nenhum dado de outras exchanges
- Nenhum dado on-chain
- Nenhum dado macroeconômico

❌ **Features altamente parametrizadas**:
- Nenhum indicador com muitos parâmetros livres
- Nenhum sinal frágil ou instável

---

## Janelas Temporais Utilizadas

| Janela | Dias | Uso Principal | Justificativa |
|--------|------|---------------|---------------|
| 1d | 1 | Returns instantâneos, ratios de candle | Movimento diário |
| 7d | 7 | Volatilidade curta, volume, tendência curta | Semana de trading |
| 21d | 21 | Volatilidade média, volume, tendência média, autocorr | Mês de trading (~1 mês calendário) |
| 63d | 63 | Volatilidade longa, tendência longa, Hurst | Trimestre (~3 meses) |

**Observação**: Todas as janelas são **backward-looking** (rolling windows). Nenhuma feature usa dados futuros.

---

## Tratamento de Valores NaN

### Regras
1. **Início da série**: Features de janela N terão NaN nas primeiras N-1 observações
2. **Dados faltantes no L2**: Se o L2 tiver NaN, as features derivadas também terão NaN
3. **Divisão por zero**: Features de ratio tratam divisão por zero retornando NaN
4. **Preservação**: NaN do L2 é preservado (não interpolado)

### Exemplo
- Feature `volatility_21d` requer 21 observações
- Primeiras 20 linhas terão `volatility_21d = NaN`
- Linha 21+ terão valores calculados

---

## Extensibilidade Futura

O L3 é projetado para ser extensível:

1. **Novas features**: Podem ser adicionadas sem quebrar o contrato
2. **Novas janelas**: Podem ser adicionadas para features existentes
3. **Novos assets**: Estrutura particionada suporta novos assets automaticamente
4. **Compatibilidade**: Features do L3 podem ser consumidas pelo L4 (cross-asset)

---

## Resumo de Design

| Aspecto | L2 | L3 | Mudança |
|---------|----|----|---------|
| **Colunas originais** | 11 | 11 | Preservadas |
| **Features adicionadas** | 0 | 44 | Novas |
| **Total de colunas** | 11 | 55 | +44 features |
| **Index** | timestamp | timestamp | Preservado |
| **Particionamento** | Por asset | Por asset | Preservado |
| **Valores originais** | Originais | Originais | Preservados |
| **Escopo** | Schema | Features per-asset | Expansão |

---

## Confirmação de Aderência às Regras

✅ **Sem lookahead bias**: Todas as features usam apenas dados passados
✅ **Rolling windows explícitas**: Todas as janelas documentadas
✅ **Features robustas**: Estatísticas descritivas, não sinais frágeis
✅ **Per-asset apenas**: Nenhuma agregação cross-asset
✅ **Interpretabilidade**: Cada feature tem significado claro
✅ **Imutabilidade**: L3 é imutável após gerado
✅ **Preservação do L2**: Valores originais mantidos
✅ **UTC timestamps**: Index temporal UTC preservado
✅ **Parquet**: Formato Parquet com compressão
✅ **Particionamento**: Estrutura particionada mantida

---

## Versão

**Versão**: 1.0  
**Data**: 2024  
**Status**: Proposta de Contrato (aguardando aprovação)

---

## Notas de Implementação Futura

Quando implementar o L3:

1. **Node único**: Um node que processa cada partição (asset) independentemente
2. **Função pura**: Node deve ser função pura (sem side effects)
3. **Validação**: Validar invariantes após cálculo
4. **Performance**: Usar pandas vectorizado quando possível
5. **Documentação**: Cada feature deve ter docstring explicando cálculo e justificativa
