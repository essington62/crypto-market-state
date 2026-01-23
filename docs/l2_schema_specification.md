Proposta do Schema Canônico do L2 (02_intermediate) para OHLCV Daily
1. COLUNAS FINAIS E TIPOS
Coluna	Tipo	Obrigatória	Origem L1	Observação
timestamp	datetime64[ns, UTC]	✅	open_time	RENAME: mais genérico que "open_time"
open	float64	✅	open	Mantido
high	float64	✅	high	Mantido
low	float64	✅	low	Mantido
close	float64	✅	close	Mantido
volume	float64	✅	volume	Volume em base asset
quote_volume	float64	✅	quote_volume	Volume em quote asset
close_time	datetime64[ns, UTC]	✅	close_time	Mantido
trades	int64	✅	trades	Mantido
taker_buy_base_volume	float64	✅	taker_buy_base_volume	Mantido
taker_buy_quote_volume	float64	✅	taker_buy_quote_volume	Mantido
2. INDEX
Index: timestamp (DatetimeIndex, timezone-aware UTC)

Ordenação: crescente por timestamp

Sem duplicatas no index

3. RENOMEAÇÕES NECESSÁRIAS
Apenas uma renomeação:

open_time → timestamp

Justificativa:

timestamp é mais genérico e comum em outras exchanges

Mantém a semântica (início do período do candle)

Facilita integração futura com outras fontes

Nota: Todas as outras colunas permanecem idênticas ao L1.

4. VALORES IDÊNTICOS AO L1
Confirmado: todos os valores observados permanecem idênticos:

timestamp = open_time (sem transformação)

open, high, low, close = valores originais

volume, quote_volume = valores originais

close_time = valor original (sem normalização)

trades = contagem original

taker_buy_base_volume, taker_buy_quote_volume = valores originais

✅ Nenhum valor numérico ou temporal é alterado.

5. INVARIANTES DO L2
5.1 Integridade Temporal
timestamp está em UTC (timezone-aware)

close_time está em UTC (timezone-aware)

close_time >= timestamp (cada candle)

timestamp é único (sem duplicatas)

5.2 Integridade OHLC
high >= max(open, close) (high cobre open/close)

low <= min(open, close) (low cobre open/close)

high >= low (high >= low)

open > 0, high > 0, low > 0, close > 0 (preços positivos)

5.3 Integridade de Volumes
volume >= 0 (não negativo)

quote_volume >= 0 (não negativo)

taker_buy_base_volume >= 0 (não negativo)

taker_buy_quote_volume >= 0 (não negativo)

taker_buy_base_volume <= volume (taker buy ≤ total)

taker_buy_quote_volume <= quote_volume (taker buy ≤ total)

5.4 Integridade de Trades
trades >= 0 (não negativo)

trades é inteiro

5.5 Estrutura do DataFrame
Index = timestamp (DatetimeIndex)

Ordenação crescente por timestamp

Sem valores NaN em colunas obrigatórias

Sem duplicatas no index

6. EXTENSIBILIDADE PARA OUTRAS EXCHANGES
O schema é extensível:

timestamp é genérico (não específico da Binance)

Colunas específicas da Binance (taker_buy_*) são mantidas como opcionais futuras

Outras exchanges podem mapear seus campos para este schema

Campos adicionais podem ser adicionados como colunas opcionais sem quebrar o contrato

7. RESUMO DE MUDANÇAS EM RELAÇÃO AO L1
Aspecto	L1	L2	Mudança
Nome da coluna temporal	open_time	timestamp	Renomeação
Index	Sem index	timestamp (DatetimeIndex)	Adicionado
Ordenação	Por open_time	Por timestamp (index)	Mantida
Valores	Originais Binance	Originais Binance	Idênticos
Tipos	Definidos	Definidos	Idênticos
8. OBSERVAÇÕES IMPORTANTES
✅ Nenhuma transformação de valores: todos os valores numéricos e temporais permanecem exatamente como no L1

✅ Nenhuma normalização temporal: timestamp e close_time são preservados exatamente como recebidos

✅ Nenhuma feature engineering: L2 é apenas normalização de schema

📋 Contrato de dados: o schema define o que é esperado, não transforma dados

Este schema estabelece o contrato canônico para OHLCV daily no projeto, permitindo integração futura com outras exchanges mantendo a imutabilidade dos dados observados.

Versão: 1.0
Última atualização: {{DATA_ATUAL}}
Responsável: {{RESPONSÁVEL}}