"""API clients for external data sources."""

from .binance_client import fetch_spot_daily_klines

__all__ = ["fetch_spot_daily_klines"]
