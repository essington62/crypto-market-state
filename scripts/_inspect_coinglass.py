import pandas as pd
from pathlib import Path

BASE = Path('data/01_raw/derivatives/coinglass')

files = {
    'OI aggregated':        BASE / 'open_interest/BTCUSDT_aggregated.parquet',
    'OI coin_margin':       BASE / 'open_interest/BTCUSDT_coin_margin.parquet',
    'OI stablecoin':        BASE / 'open_interest/BTCUSDT_stablecoin.parquet',
    'OI raw (Binance)':     BASE / 'open_interest/BTCUSDT.parquet',
    'Funding oi_weighted':  BASE / 'funding/BTCUSDT_oi_weighted.parquet',
    'Funding vol_weighted': BASE / 'funding/BTCUSDT_vol_weighted.parquet',
    'Funding raw':          BASE / 'funding/BTCUSDT_raw.parquet',
    'LS global':            BASE / 'long_short_ratio/BTCUSDT.parquet',
    'LS top_accounts':      BASE / 'long_short_ratio/BTCUSDT_top_accounts.parquet',
    'LS top_positions':     BASE / 'long_short_ratio/BTCUSDT_top_positions.parquet',
    'Taker spot':           BASE / 'taker/spot_BTC_aggregated.parquet',
    'Options OI':           BASE / 'options/BTC_oi_by_exchange.parquet',
    'Options volume':       BASE / 'options/BTC_volume_by_exchange.parquet',
    'ETF flows_total':      BASE / 'etf/BTC_flows_total.parquet',
    'ETF flows_by_ticker':  BASE / 'etf/BTC_flows_by_ticker.parquet',
    'ETF holdings_consol':  BASE / 'etf/BTC_holdings_consolidated.parquet',
    'ETF netassets_consol': BASE / 'etf/BTC_netassets_consolidated.parquet',
    'ETF GBTC':             BASE / 'etf/BTC_GBTC.parquet',
    'ETF IBIT':             BASE / 'etf/BTC_IBIT.parquet',
    'Fear & Greed':         BASE / 'indices/fear_greed.parquet',
    'AHR999':               BASE / 'indices/ahr999.parquet',
    'Puell Multiple':       BASE / 'indices/puell_multiple.parquet',
    'Bubble Index':         BASE / 'indices/bubble_index.parquet',
    'Stablecoin MCAP':      BASE / 'indices/stablecoin_mcap.parquet',
    'CDRI Index':           BASE / 'indices/cdri_index.parquet',
    'CGDI Index':           BASE / 'indices/cgdi_index.parquet',
}

hdr = "{:<28} {:<55} {:<12} {:<12} {}".format('Arquivo', 'Colunas', 'Inicio', 'Fim', 'n')
print(hdr)
print('-' * 115)
for name, path in files.items():
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    cols = ', '.join(df.columns[:4].tolist())
    if len(df.columns) > 4:
        cols += ' (+{})'.format(len(df.columns) - 4)
    start = str(df.index.min().date())
    end   = str(df.index.max().date())
    n     = len(df)
    print("{:<28} {:<55} {:<12} {:<12} {}".format(name, cols, start, end, n))
