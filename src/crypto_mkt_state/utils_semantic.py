"""
Semantic feature resolution helpers for L3.

This module resolves which semantic family an asset belongs to and
which features are allowed for that family, based on the configuration
in parameters.yml under `l3_semantic`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def resolve_semantic_family(
    source: str,
    category: str,
    semantic_config: Dict,
) -> Optional[str]:
    """
    Resolve the semantic family for an asset based on source and category.

    Args:
        source:
            Data source (e.g. 'binance', 'fred', 'yfinance').
        category:
            Asset category (e.g. 'crypto', 'equity_index', 'inflation').
        semantic_config:
            The `l3_semantic` section from parameters.yml.

    Returns:
        Family name (e.g. 'assets', 'indices') or None if no match.

    Raises:
        ValueError:
            If the asset matches multiple families (should not happen
            with proper config) or if required config structure is missing.
    """
    families = semantic_config.get("families", {})

    matching_families = []

    for family_name, family_config in families.items():
        applies_to = family_config.get("applies_to", {})
        allowed_sources = applies_to.get("sources", [])
        allowed_categories = applies_to.get("categories", [])

        if source in allowed_sources and category in allowed_categories:
            matching_families.append(family_name)

    if len(matching_families) > 1:
        raise ValueError(
            f"Asset (source={source}, category={category}) matches "
            f"multiple families: {matching_families}"
        )

    if len(matching_families) == 0:
        return None

    return matching_families[0]


def get_allowed_features_for_family(
    family_name: str,
    semantic_config: Dict,
) -> List[str]:
    """
    Get the flat list of allowed feature names for a semantic family.

    Args:
        family_name:
            Semantic family name (e.g. 'assets', 'indices').
        semantic_config:
            The `l3_semantic` section from parameters.yml.

    Returns:
        Flat list of feature names allowed for this family.

    Raises:
        KeyError:
            If the family does not exist in the config.
    """
    families = semantic_config.get("families", {})

    if family_name not in families:
        raise KeyError(f"Family '{family_name}' not found in semantic config")

    family_config = families[family_name]
    features_config = family_config.get("features", {})

    # Flatten nested feature groups into a single list
    allowed_features = []
    for feature_group in features_config.values():
        if isinstance(feature_group, list):
            allowed_features.extend(feature_group)
        elif isinstance(feature_group, str):
            allowed_features.append(feature_group)

    return allowed_features


def should_compute_feature(
    feature_name: str,
    allowed_features: List[str],
) -> bool:
    """
    Check if a feature should be computed based on the allowed list.

    Args:
        feature_name:
            Name of the feature to check.
        allowed_features:
            List of allowed feature names for the asset's family.

    Returns:
        True if the feature should be computed, False otherwise.
    """
    return feature_name in allowed_features


def get_asset_metadata(
    df: pd.DataFrame,
) -> tuple[str, str]:
    """
    Extract source and category from a DataFrame's metadata columns.

    Args:
        df:
            DataFrame with metadata columns (source, category).

    Returns:
        Tuple of (source, category).

    Raises:
        KeyError:
            If required metadata columns are missing.
    """
    if "source" not in df.columns or "category" not in df.columns:
        raise KeyError(
            "DataFrame must contain 'source' and 'category' columns "
            "for semantic feature resolution"
        )

    source = str(df["source"].iloc[0])
    category = str(df["category"].iloc[0])

    return source, category
