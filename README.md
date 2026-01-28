# Data Ingestion Layers

## L1 – FRED (Macroeconomic Data)

### Purpose
The FRED L1 layer ingests raw macroeconomic time series that represent the structural state of the global and U.S. economy.

These variables provide slow-moving, regime-defining signals that anchor financial markets across:
- Growth vs. recession cycles
- Inflationary vs. disinflationary regimes
- Monetary policy stance
- Long-term risk premia

All series are configured via `parameters.yml`, ensuring transparency and reproducibility.

### 📦 Macroeconomic Series Ingested (FRED)

| Series | Category | What It Represents | Why It Matters |
|--------|----------|-------------------|----------------|
| **DGS10** | Rates | 10Y US Treasury yield | Benchmark for discount rates, equity valuation, and macro risk regimes. |
| **DGS2** | Rates | 2Y US Treasury yield | Proxy for monetary policy expectations and short-term rates. |
| **T10Y2Y** | Rates | Yield curve slope | Classic recession indicator and macro regime classifier. |
| **CPIAUCSL** | Inflation | Consumer Price Index | Measures inflation pressure and policy constraint. |
| **PCEPI** | Inflation | PCE inflation | Fed's preferred inflation gauge. |
| **UNRATE** | Labor | Unemployment rate | Late-cycle stress and recession confirmation signal. |
| **INDPRO** | Growth | Industrial production | Real economic activity and cyclical momentum. |
| **PAYEMS** | Labor | Non-farm payrolls | Labor market strength and economic expansion proxy. |

> ⚠️ **Note:** All FRED series are ingested as-is, without resampling, forward-filling, or interpolation.

### 🧠 Series Selection Philosophy
FRED series are selected to represent macro regime drivers, not short-term trading signals.

#### 1️⃣ Structural, Not Tactical
FRED data moves slowly but defines the boundary conditions under which markets operate.

#### 2️⃣ Policy & Cycle Awareness
Rates, inflation, and labor indicators jointly describe:
- Policy flexibility
- Risk-free rate dynamics
- Economic overheating or contraction

#### 3️⃣ Cross-Asset Relevance
These variables influence:
- Equity valuation
- Bond duration risk
- FX differentials
- Crypto liquidity via global financial conditions

### 🧱 L1 FRED Schema
The L1 FRED layer preserves the raw FRED schema, enriched only with metadata.

| Column | Description |
|--------|-------------|
| `date` | Observation date |
| `value` | Raw series value |
| `series_id` | FRED series identifier |
| `asset` | Asset name |
| `category` | Category (Rates, Inflation, etc.) |
| `source` | Always `"fred"` |
| `interval` | Always `"1d"` |
| `ingestion_ts` | Timestamp of ingestion |

### 🔄 Role in Higher Layers

| Layer | Transformation |
|-------|---------------|
| **L2 (Normalization)** | Adds consistent metadata and cleans time indices |
| **L3 (Primary Features)** | Generates statistical features (changes, rolling z-scores, percentiles) |
| **L4 (Cross-Asset Layer)** | Combines macro regimes with market-based signals to:<br>• Detect risk-on / risk-off environments<br>• Filter false signals in high-volatility periods<br>• Condition model behavior on macro states |

---

## L1 – Yahoo Finance (Market & Risk Proxies)

### Purpose
The Yahoo Finance L1 layer ingests cross-asset market data that acts as a bridge between macro conditions and crypto markets.

These assets represent global risk sentiment, capital flows, and financial stress, providing context that pure crypto data cannot capture alone.

All assets are parameterized via `parameters.yml`.

### 📦 Assets Ingested (Yahoo Finance)

| Asset | Category | What It Represents | Why It Matters |
|-------|----------|-------------------|----------------|
| **S&P 500 (^GSPC)** | Equity Index | Global risk-on benchmark | Defines global equity regime and risk appetite |
| **NASDAQ (^IXIC)** | Equity Index | Growth & liquidity sensitivity | Highly sensitive to rates, tech, and liquidity cycles |
| **US 10Y Yield (^TNX)** | Rates | Market-based long-term rates | Forward-looking discount rate proxy |
| **DXY (DX-Y.NYB)** | FX | US dollar strength | Global liquidity conditions and risk-off pressure |
| **Gold (GC=F)** | Commodity | Inflation & stress hedge | Safe-haven demand and inflation expectations |
| **VIX (^VIX)** | Volatility | Implied equity volatility | Canonical global risk & stress indicator |

### 🧠 Asset Selection Philosophy
Yahoo Finance assets are chosen as global state variables, not alpha signals.

#### 1️⃣ Risk-On / Risk-Off Decomposition
Equities, volatility, FX, and commodities jointly describe the market's risk posture.

#### 2️⃣ Liquidity & Stress Sensitivity
- Rising DXY + VIX → tightening financial conditions
- Falling equities + rising volatility → risk-off regime

#### 3️⃣ Crypto Spillover Channels
Global markets often lead crypto during:
- Liquidity shocks
- Macro-driven selloffs
- Policy regime changes

### 🧱 L1 Yahoo Finance Schema
Yahoo Finance data is ingested as daily OHLCV series with metadata.

| Column | Description |
|--------|-------------|
| `date` | Trading date |
| `open` | Opening price |
| `high` | High price |
| `low` | Low price |
| `close` | Closing price |
| `volume` | Trading volume |
| `symbol` | Yahoo Finance symbol |
| `asset` | Asset name |
| `category` | Category (Equity, Rates, etc.) |
| `source` | Always `"yfinance"` |
| `interval` | Always `"1d"` |
| `ingestion_ts` | Timestamp of ingestion |

### 🔄 Role in Higher Layers

| Layer | Transformation |
|-------|---------------|
| **L2 (Normalization)** | Ensures consistent schema across all market assets |
| **L3 (Primary Features)** | Computes returns, volatility, rolling statistics, and relative state metrics |
| **L4 (Cross-Asset Layer)** | Enables construction of:<br>• Global Risk Index<br>• Volatility-adjusted positioning signals<br>• Regime classifiers combining macro + markets + crypto |

---

## 📌 Architectural Summary

| Layer | Role |
|-------|------|
| **L1** | Raw, source-faithful data ingestion |
| **L2** | Schema normalization & metadata alignment |
| **L3** | Per-asset statistical feature engineering |
| **L4** | Cross-asset, regime-aware features |


## L1 – Binance (Crypto Market Data)

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

## Planned Extensions (Not Implemented Yet)

The architecture supports future extensions without breaking existing pipelines:
- Open Interest (OI)
- Explicit Funding Rates
- Long/Short Ratios
- Stablecoin supply / dominance
- Exchange inflow / outflow (on-chain)

These signals would enter L1 or L2 only, never directly into L3/L4.

## Connection to Higher Layers

- L2 (Normalization)
Standardizes schema, cleans timestamps, and attaches metadata.

- L3 (Primary Features)
Generates per-asset statistical features (volatility, z-scores, momentum).

- L4 (Cross-Asset Layer)
Combines Binance, Macro (FRED), and Market (Yahoo Finance) data to model:

   - Risk-On / Risk-Off regimes
   - Systemic stress
   - Strategy gating and allocation logic
