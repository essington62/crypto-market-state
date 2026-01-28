L1 – Binance (Crypto Market Data)

Purpose

The Binance L1 layer captures raw crypto market data designed to represent the state of the crypto ecosystem, not individual trading opportunities.

The selected assets act as market sensors for:
- Price discovery
- Liquidity
- Leverage
- Market stress
- Directional flow

All assets are configured via parameters.yml, ensuring full reproducibility and extensibility.

# Ativos Monitorados

Esta tabela descreve os principais ativos e métricas utilizados para análise de mercado, seus significados e relevância.

| Ativo                  | Tipo              | O que representa                    | Por que é relevante                                                             |
| ---------------------- | ----------------- | ----------------------------------- | ------------------------------------------------------------------------------- |
| **BTCUSDT**            | Spot              | Preço e liquidez do Bitcoin         | Principal proxy de risco do mercado cripto. Lidera ciclos, drawdowns e regimes. |
| **ETHUSDT**            | Spot              | Atividade econômica on-chain / DeFi | Complementa BTC com maior sensibilidade a inovação, staking e ciclos de risco.  |
| **BTCUSDT Perp**       | Perpetual Futures | Alavancagem e funding               | Proxy direto de posicionamento especulativo e stress via funding rates.         |
| **ETHUSDT Perp**       | Perpetual Futures | Alavancagem em ETH                  | Importante para capturar risco sistêmico em DeFi e Layer-1s.                    |
| **BNBUSDT**            | Spot              | Risco de exchange / infra           | Sensível a eventos de exchange, liquidez e confiança sistêmica.                 |
| **TOTAL VOLUME (BTC)** | Derivado          | Liquidez agregada                   | Mede entrada/saída de capital e intensidade de participação.                    |
| **TRADE COUNT**        | Derivado          | Atividade de mercado                | Proxy de micro-estrutura e stress intradiário.                                  |

Nota: stablecoins não são usadas como sinal direcional, mas podem entrar futuramente como proxy de liquidez off-risk.

# Asset Selection Philosophy

Asset selection in Binance L1 follows three core principles:

1.Systemic Representativeness

BTC and ETH capture the majority of risk and capital concentration in crypto markets.

2.Spot vs. Leverage Separation
- Spot markets → clean price discovery
- Perpetual futures → leverage, positioning, and stress

This separation is critical to detect:
- Speculative excess
- Long/short squeezes
- Regime transitions before price moves

3.Stress Sensitivity

Perpetuals, volume, and trade count typically react earlier than price during stress events, making them leading indicators.

## L1 Binance Schema

The L1 layer stores raw OHLCV daily data with minimal metadata.
No statistical transformations are applied at this stage.

open_time
open
high
low
close
volume
close_time
quote_volume
trades
taker_buy_base_volume
taker_buy_quote_volume
asset
symbol
source = "binance"
interval = "1d"
ingestion_ts

