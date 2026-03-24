"""
FRED incremental ingestion nodes.

Imports FRED economic data with incremental update capability.
Maintains L1 contract: append-only, raw data, no transformations.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, List, Union, Any
import json

import pandas as pd
import numpy as np

from crypto_mkt_state.clients.fred_client import fetch_fred_series


def parse_fred_series(params_fred: Union[dict, str, Any]) -> list[dict]:
    """
    Parse FRED series configuration from parameters.yml.
    
    Handles multiple input types to be robust to Kedro parameter passing.
    
    Parameters
    ----------
    params_fred
        Dictionary containing the 'series' key, or a string that should be parsed,
        or any other type that can be converted.
    
    Returns
    -------
    list[dict]
        Validated series configurations with keys: series_id, name, category.
        Returns empty list if no valid series found.
    """
    # Debug: print what we received
    print(f"[FRED DEBUG] parse_fred_series received type: {type(params_fred)}")
    print(f"[FRED DEBUG] parse_fred_series received value: {params_fred}")
    
    # Handle different input types
    if params_fred is None:
        print("[FRED WARN] params_fred is None")
        return []
    
    # If it's already a dict, use it directly
    if isinstance(params_fred, dict):
        series = params_fred.get("series", [])
    # If it's a string, try to parse it as JSON or YAML-like structure
    elif isinstance(params_fred, str):
        print(f"[FRED DEBUG] Parsing string: {params_fred[:200]}...")  # First 200 chars
        
        # Try to parse as JSON
        try:
            params_fred = json.loads(params_fred)
            if isinstance(params_fred, dict):
                series = params_fred.get("series", [])
            else:
                print(f"[FRED ERROR] JSON parsed to {type(params_fred)}, expected dict")
                return []
        except json.JSONDecodeError:
            # Try to evaluate as Python literal (for dict strings)
            try:
                import ast
                params_fred = ast.literal_eval(params_fred)
                if isinstance(params_fred, dict):
                    series = params_fred.get("series", [])
                else:
                    print(f"[FRED ERROR] ast.literal_eval parsed to {type(params_fred)}, expected dict")
                    return []
            except (ValueError, SyntaxError) as e:
                print(f"[FRED ERROR] Failed to parse string: {e}")
                return []
    else:
        # Unknown type
        print(f"[FRED ERROR] Unexpected params_fred type: {type(params_fred)}")
        return []
    
    # Now we should have a series list
    if not isinstance(series, list):
        print(f"[FRED ERROR] series is not a list: {type(series)}")
        return []
    
    if not series:
        print("[FRED WARN] No series found in params:fred")
        return []
    
    parsed = []
    for s in series:
        if not isinstance(s, dict):
            print(f"[FRED WARN] Skipping non-dict series config: {s}")
            continue
        
        series_id = s.get("id")
        if not series_id:
            print(f"[FRED WARN] Skipping series config missing 'id': {s}")
            continue
        
        parsed.append({
            "series_id": str(series_id),
            "name": str(s.get("name", series_id)),
            "category": str(s.get("category", "unknown")),
        })
    
    if not parsed:
        print("[FRED WARN] No valid FRED series after parsing")
    else:
        print(f"[FRED INFO] Parsed {len(parsed)} series: {[p['series_id'] for p in parsed]}")
    
    return parsed


def update_fred_incremental(
    macro_daily_raw: Dict[str, Callable[[], pd.DataFrame]],
    params_fred: Union[dict, str, Any],
    global_start_date: str,
) -> Dict[str, pd.DataFrame]:
    """
    Incrementally update FRED daily macro series (L1).
    
    Implements append-only ingestion with deduplication and UTC timezone safety.
    Never modifies historical data - only appends new observations.
    
    Parameters
    ----------
    macro_daily_raw
        PartitionedDataset mapping series_id -> loader callable.
        Contains existing historical data for each FRED series.
    params_fred
        Dictionary containing FRED configuration from params:fred,
        or string that should be parsed.
        Expected structure: {'series': [{'id': 'FEDFUNDS', ...}, ...]}
    global_start_date
        ISO date string used as initial start_date when a partition does not exist.
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Updated partitions for macro_daily_incremental.
        Each DataFrame has columns:
        - date: datetime64[ns, UTC] (timestamp)
        - value: float64 (observation value)
        Returns empty dict if no valid series configured.
    """
    print(f"[FRED DEBUG] update_fred_incremental called")
    print(f"[FRED DEBUG] params_fred type: {type(params_fred)}")
    
    # Parse series configuration from params_fred
    series_config = parse_fred_series(params_fred)
    
    if not series_config:
        print("[FRED WARN] No valid series to process, returning empty result")
        return {}
    
    # Convert global_start_date to UTC date
    start_date_utc = pd.to_datetime(global_start_date, utc=True).normalize()
    today_utc = datetime.now(timezone.utc).date()
    today_str = today_utc.isoformat()
    
    # Track processing statistics
    total_series = len(series_config)
    processed_series = 0
    updated_series = 0
    
    updated: Dict[str, pd.DataFrame] = {}
    
    print(f"[FRED INFO] Processing {total_series} series")
    
    for idx, cfg in enumerate(series_config, 1):
        series_id = cfg["series_id"]
        series_name = cfg["name"]
        category = cfg["category"]
        
        print(f"[FRED INFO] [{idx}/{total_series}] {series_id} ({series_name}) - {category}")
        
        # --- Load existing partition if present
        existing_df = None
        if series_id in macro_daily_raw:
            try:
                existing_df = macro_daily_raw[series_id]()
                print(f"[FRED INFO]   Loaded existing data: {len(existing_df) if existing_df is not None else 0} rows")
            except Exception as e:
                print(f"[FRED ERROR]   Failed to load existing partition: {e}")
                # Continue with empty DataFrame
                existing_df = None
        
        # Initialize empty DataFrame if none exists
        if existing_df is None or existing_df.empty:
            existing_df = pd.DataFrame(columns=["date", "value"])
            print(f"[FRED INFO]   No existing data, starting fresh")
        
        # Make a copy to avoid modifying original
        existing_df = existing_df.copy()
        
        # --- Ensure date column is proper UTC datetime
        if not existing_df.empty:
            if "date" not in existing_df.columns:
                print(f"[FRED ERROR]   Existing DataFrame missing 'date' column")
                continue
            
            # Convert to UTC datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(existing_df["date"]):
                existing_df["date"] = pd.to_datetime(existing_df["date"], utc=True)
            elif existing_df["date"].dt.tz is None:
                # Localize naive datetime to UTC
                existing_df["date"] = existing_df["date"].dt.tz_localize("UTC")
            elif str(existing_df["date"].dt.tz) != "UTC":
                # Convert to UTC
                existing_df["date"] = existing_df["date"].dt.tz_convert("UTC")
        
        # --- Determine next start date for incremental fetch
        if not existing_df.empty and "date" in existing_df.columns:
            # Get the most recent date (ensuring UTC)
            last_date = existing_df["date"].max().date()
            next_start_date = (last_date + timedelta(days=1)).isoformat()
        else:
            next_start_date = start_date_utc.date().isoformat()
        
        print(f"[FRED INFO]   Next start date: {next_start_date}")
        
        # --- Check if already up-to-date
        if next_start_date > today_str:
            print(f"[FRED INFO]   Already up-to-date (last date: {last_date if 'last_date' in locals() else 'N/A'})")
            # Ensure contract compliance before returning
            if not existing_df.empty:
                existing_df = existing_df.sort_values("date").reset_index(drop=True)
                existing_df["value"] = existing_df["value"].astype("float64")
            updated[series_id] = existing_df
            processed_series += 1
            continue
        
        # --- Fetch new observations from FRED
        try:
            print(f"[FRED INFO]   Fetching from {next_start_date} to {today_str}")
            new_df = fetch_fred_series(
                series_id=series_id,
                start_date=next_start_date,
                end_date=today_str,
            )
            
            if new_df.empty:
                print(f"[FRED INFO]   No new data available")
                # Keep existing data
                if not existing_df.empty:
                    existing_df = existing_df.sort_values("date").reset_index(drop=True)
                    existing_df["value"] = existing_df["value"].astype("float64")
                updated[series_id] = existing_df
                processed_series += 1
                continue
            
            print(f"[FRED INFO]   Fetched {len(new_df)} new rows")
            
        except Exception as e:
            print(f"[FRED ERROR]   Failed to fetch {series_id}: {e}")
            # Keep existing data if fetch fails
            if not existing_df.empty:
                existing_df = existing_df.sort_values("date").reset_index(drop=True)
                existing_df["value"] = existing_df["value"].astype("float64")
                updated[series_id] = existing_df
            else:
                # Return empty DataFrame with correct schema
                empty_df = pd.DataFrame(columns=["date", "value"])
                empty_df["date"] = pd.to_datetime(empty_df["date"], utc=True)
                empty_df["value"] = empty_df["value"].astype("float64")
                updated[series_id] = empty_df
            processed_series += 1
            continue
        
        # --- Merge existing + new data
        if not existing_df.empty and not new_df.empty:
            merged = pd.concat([existing_df, new_df], axis=0, ignore_index=True)
            print(f"[FRED INFO]   Merged: {len(existing_df)} existing + {len(new_df)} new = {len(merged)} rows")
        elif existing_df.empty and not new_df.empty:
            merged = new_df
            print(f"[FRED INFO]   Using only new data: {len(merged)} rows")
        elif not existing_df.empty and new_df.empty:
            merged = existing_df
            print(f"[FRED INFO]   No new data, keeping existing: {len(merged)} rows")
        else:
            # Both empty - create empty DataFrame
            merged = pd.DataFrame(columns=["date", "value"])
            print(f"[FRED INFO]   No data available")
        
        if merged.empty:
            empty_df = pd.DataFrame(columns=["date", "value"])
            empty_df["date"] = pd.to_datetime(empty_df["date"], utc=True)
            empty_df["value"] = empty_df["value"].astype("float64")
            updated[series_id] = empty_df
            processed_series += 1
            continue
        
        # --- L1 Final Cleaning (contract compliance)
        # 1. Ensure date is UTC datetime
        merged["date"] = pd.to_datetime(merged["date"], utc=True)
        
        # 2. Convert value to float64, coerce errors to NaN
        merged["value"] = pd.to_numeric(merged["value"], errors="coerce").astype("float64")
        
        # 3. Drop rows with NaN values (invalid FRED observations)
        initial_rows = len(merged)
        nan_count = merged["value"].isna().sum()
        if nan_count > 0:
            print(f"[FRED WARN]   Dropping {nan_count} rows with NaN values")
            merged = merged.dropna(subset=["value"])
        
        # 4. Sort by date ascending
        merged = merged.sort_values("date")
        
        # 5. Deduplicate by date (keep last observation for each date)
        duplicate_count = merged.duplicated(subset=["date"]).sum()
        if duplicate_count > 0:
            print(f"[FRED WARN]   Removing {duplicate_count} duplicate rows")
            merged = merged.drop_duplicates(subset=["date"], keep="last")
        
        # 6. Reset index
        merged = merged.reset_index(drop=True)
        
        # --- Final validation
        if merged.empty:
            print(f"[FRED WARN]   No valid data after cleaning")
            empty_df = pd.DataFrame(columns=["date", "value"])
            empty_df["date"] = pd.to_datetime(empty_df["date"], utc=True)
            empty_df["value"] = empty_df["value"].astype("float64")
            updated[series_id] = empty_df
            processed_series += 1
            continue
        
        # Verify no NaNs remain
        if merged["value"].isna().any():
            print(f"[FRED ERROR]   Still have NaN values after cleaning")
            # Force convert any remaining NaNs to 0? No - better to drop
            merged = merged.dropna(subset=["value"])
        
        # Ensure date range is correct
        date_range = f"{merged['date'].min().date()} → {merged['date'].max().date()}"
        
        print(f"[FRED INFO]   Final: {len(merged)} rows, {date_range}")
        updated[series_id] = merged
        processed_series += 1
        updated_series += 1 if len(new_df) > 0 else 0
    
    # --- Summary
    print(f"[FRED INFO] ========== SUMMARY ==========")
    print(f"[FRED INFO] Processed: {processed_series}/{total_series} series")
    print(f"[FRED INFO] Updated: {updated_series} series with new data")
    
    return updated