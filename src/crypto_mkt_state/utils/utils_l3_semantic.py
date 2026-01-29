"""
L3 semantic feature resolution helpers.

This module resolves which features are allowed for each asset based on
explicit rules defined in parameters.yml under `l3_semantic.assets` and
`l3_semantic.indices`.
"""

from __future__ import annotations

from typing import Dict, List


def normalize_asset_name(asset_name: str) -> str:
    """
    Normalize asset name for lookup in semantic config.

    Rules:
    - Convert to lowercase
    - Remove 'usdt' suffix if present (e.g. 'BTCUSDT' -> 'btc')
    - Strip whitespace

    Args:
        asset_name:
            Raw asset name (e.g. 'BTCUSDT', 'SP500', 'VIX').

    Returns:
        Normalized asset name (e.g. 'btc', 'sp500', 'vix').
    """
    normalized = asset_name.lower().strip()

    # Remove 'usdt' suffix for crypto pairs
    if normalized.endswith("usdt"):
        normalized = normalized[:-4]

    return normalized


def get_semantic_features(
    asset_name: str,
    semantic_config: Dict,
) -> List[str]:
    """
    Get the list of allowed features for an asset from semantic config.

    Search order:
    1. l3_semantic.assets[normalized_name]
    2. l3_semantic.indices[normalized_name]

    Args:
        asset_name:
            Asset name (e.g. 'BTCUSDT', 'sp500', 'VIX', 'cpi').
            Will be normalized before lookup.
        semantic_config:
            The `l3_semantic` section from parameters.yml.

    Returns:
        List of feature names allowed for this asset.

    Raises:
        ValueError:
            If the asset is not found in either assets or indices sections.
    """
    normalized = normalize_asset_name(asset_name)

    # Search in assets first
    assets_config = semantic_config.get("assets", {})
    if normalized in assets_config:
        asset_config = assets_config[normalized]
        features = asset_config.get("features", [])
        if isinstance(features, list):
            return features
        return []

    # Search in indices
    indices_config = semantic_config.get("indices", {})
    if normalized in indices_config:
        index_config = indices_config[normalized]
        features = index_config.get("features", [])
        if isinstance(features, list):
            return features
        return []

    # Asset not found
    raise ValueError(
        f"Asset '{asset_name}' (normalized: '{normalized}') not found in "
        "l3_semantic.assets or l3_semantic.indices. "
        "Please add it to parameters.yml under l3_semantic."
    )
