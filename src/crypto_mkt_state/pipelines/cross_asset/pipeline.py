"""
L4 cross-asset features pipeline.

Builds a global market state representation by:
- assembling L3 regime features (BTC + macro),
- estimating latent regimes via HMM,
- enriching regime states with regime_entropy,
- building cross-asset regime features,
- summarizing regimes ex-post,
- inferring the current market regime.

Contract:
- No overwriting datasets in-place (immutability respected)
- regime_entropy is computed once and propagated forward
- All joins are date-based, UTC
"""

from kedro.pipeline import Pipeline, node

from crypto_mkt_state.pipelines.cross_asset.nodes import (
    add_regime_entropy_to_states,
    assemble_l3_regime_features,
    build_cross_asset_features,
    compute_viterbi_diagnostics,
    estimate_market_regime_hmm,
    infer_current_market_regime,
    summarize_hmm_regimes,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create L4 cross-asset pipeline.

    Flow:
    L3 assemble -> HMM -> enrich entropy -> cross-asset features
                 -> regime stats -> current regime
    """
    return Pipeline(
        [
            # ------------------------------------------------------------------
            # L3: Assemble regime features for HMM
            # ------------------------------------------------------------------
            node(
                func=assemble_l3_regime_features,
                inputs={
                    "crypto_primary": "crypto_ohlcv_daily_primary",
                    "yfinance_indices_primary": "yfinance_indices_primary",
                },
                outputs="l3_regime_features",
                name="assemble_l3_regime_features",
            ),

            # ------------------------------------------------------------------
            # L4: Estimate latent market regimes via HMM
            # ------------------------------------------------------------------
            node(
                func=estimate_market_regime_hmm,
                inputs={
                    "l3_regime_features": "l3_regime_features",
                    "l4_config": "params:l4",
                },
                outputs=[
                    "l4_regime_states",
                    "l4_regime_transition_matrix",
                ],
                name="estimate_market_regime_hmm",
            ),

            # ------------------------------------------------------------------
            # L3 semantics: add regime entropy (UNCERTAINTY METRIC)
            # ------------------------------------------------------------------
            node(
                func=add_regime_entropy_to_states,
                inputs="l4_regime_states",
                outputs="l4_regime_states_enriched",
                name="add_regime_entropy_to_states",
            ),

            # ------------------------------------------------------------------
            # L4: Build cross-asset regime features (macro + entropy)
            # ------------------------------------------------------------------
            node(
                func=build_cross_asset_features,
                inputs={
                    "fred": "fred_macro_primary",
                    "yfinance_assets": "yfinance_assets_primary",
                    "yfinance_indices": "yfinance_indices_primary",
                    "l4_config": "params:l4",
                    "l4_regime_states": "l4_regime_states_enriched",
                },
                outputs="cross_asset_features",
                name="build_cross_asset_features",
            ),


            # ------------------------------------------------------------------
            # L4: Ex-post regime characterization (interpretability only)
            # ------------------------------------------------------------------
            node(
                func=summarize_hmm_regimes,
                inputs={
                    "l4_regime_states": "l4_regime_states_enriched",
                    "l3_regime_features": "l3_regime_features",
                },
                outputs="l4_regime_state_stats",
                name="summarize_hmm_regimes",
            ),

            # ------------------------------------------------------------------
            # L4: Infer current market regime (latest observation)
            # ------------------------------------------------------------------
            node(
                func=infer_current_market_regime,
                inputs={
                    "l4_regime_states": "l4_regime_states_enriched",
                    "l4_regime_transition_matrix": "l4_regime_transition_matrix",
                },
                outputs="l4_current_market_regime",
                name="infer_current_market_regime",
            ),

            # ------------------------------------------------------------------
            # L4: Viterbi rolling (k=5) diagnostics (entropy gate; no HMM change)
            # ------------------------------------------------------------------
            node(
                func=compute_viterbi_diagnostics,
                inputs={
                    "l4_regime_states_enriched": "l4_regime_states_enriched",
                    "l4_regime_transition_matrix": "l4_regime_transition_matrix",
                    "l4_config": "params:l4",
                },
                outputs="l4_viterbi_diagnostics",
                name="compute_viterbi_diagnostics",
            ),
        ]
    )

