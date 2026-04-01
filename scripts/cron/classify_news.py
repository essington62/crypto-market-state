"""
classify_news.py — DeepSeek classifier para notícias BTC + Macro.

Roda após ambos os ingests (crypto :55, macro :56) às :57.
Batch unificado [CRYPTO]/[MACRO] → prompt único com contexto macro.
Métricas usam EXCLUSIVAMENTE ds_score — nunca sentiment_raw.

Pipeline:
  1. Filtro de relevância macro (ANTES do DeepSeek — economiza tokens)
  2. Notícias irrelevantes → ds_classified=True, ds_score=0, ds_event_type=NOISE
  3. Notícias relevantes → DeepSeek batch com [CRYPTO]/[MACRO] tags
  4. Agregação pós-classificação com source weighting

Inputs:
    data/01_raw/news/cryptocompare/btc_news.parquet
    data/01_raw/news/cryptocompare/macro_news.parquet
    conf/macro_keywords.json
    .streamlit/secrets.toml

Outputs:
    data/01_raw/news/cryptocompare/btc_news.parquet     — + colunas ds_*
    data/01_raw/news/cryptocompare/macro_news.parquet   — + colunas ds_*
    data/01_raw/news/cryptocompare/sentiment_metrics.json
"""

import json
import logging
import os
import re
import hashlib
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT        = Path(__file__).resolve().parents[2]
DATA_DIR       = PROJECT / "data" / "01_raw" / "news" / "cryptocompare"
CRYPTO_PARQUET = DATA_DIR / "btc_news.parquet"
MACRO_PARQUET  = DATA_DIR / "macro_news.parquet"
METRICS_PATH   = DATA_DIR / "sentiment_metrics.json"
KEYWORDS_CFG   = PROJECT / "conf" / "macro_keywords.json"
SECRETS_PATH   = PROJECT / ".streamlit" / "secrets.toml"

DEEPSEEK_URL    = "https://api.deepseek.com/chat/completions"
CLASSIFY_WINDOW = 6    # horas — janela de notícias a classificar
MAX_ITEMS       = 30   # max por chamada de API

DS_COLS = [
    "ds_classified", "ds_topic", "ds_event_type",
    "ds_impact", "ds_score", "ds_direction",
    "ds_reason", "ds_classified_at",
]

# ── Correção 5: Source weighting ──────────────────────────────────────────────
SOURCE_WEIGHTS: dict[str, float] = {
    "reuters":          1.0,
    "bloomberg":        1.0,
    "wall street journal": 0.9,
    "wsj":              0.9,
    "financial times":  0.9,
    "ft":               0.9,
    "new york times":   0.9,
    "nyt":              0.9,
    "associated press": 0.9,
    "ap news":          0.9,
    "cnbc":             0.8,
    "bbc":              0.8,
    "al jazeera":       0.8,
    "cnn":              0.7,
    "yahoo finance":    0.6,
    "yahoo":            0.5,
    "the motley fool":  0.4,
    "substack":         0.5,
    "cryptocompare":    0.6,
    "coindesk":         0.7,
    "cointelegraph":    0.6,
}
DEFAULT_SOURCE_WEIGHT = 0.5

# ── Correção 3: Relevance filter keywords ─────────────────────────────────────
MACRO_RELEVANCE_KEYWORDS = [
    "oil", "crude", "petroleum", "fuel", "energy",
    "inflation", "cpi", "consumer price", "prices",
    "fed", "federal reserve", "fomc", "powell", "rate hike", "rate cut",
    "monetary policy", "interest rate",
    "war", "conflict", "iran", "military", "sanctions", "hormuz", "missile",
    "china", "tariff", "trade war", "trade deal",
    "economy", "recession", "gdp", "unemployment", "jobs report",
    "yield", "bond", "treasury", "debt crisis",
    "bank failure", "market crash", "systemic",
    "opec", "saudi", "gas price",
]

# ── Prompt unificado [CRYPTO]/[MACRO] ─────────────────────────────────────────
CLASSIFY_PROMPT = """You are a senior risk analyst for Bitcoin trading.

MAIN RULE: Classify by the FINAL RESULT of the event on Bitcoin price, NOT by individual words.
Examples:
- "Trump signals end of Iran war" -> DEESCALATION -> score +7 (end of war = risk-on = BTC rises)
- "Trump threatens ground invasion of Iran" -> ESCALATION -> score -8 (war escalates = risk-off)
- "Fed signals rate cut" -> POLICY_SHIFT -> score +6 (rates fall = liquidity = BTC rises)
- "Investors pull $414M from crypto funds" -> CAPITAL_FLOW -> score -5 (outflow = selling pressure)
- "Google warns quantum threatens crypto" -> NOISE -> score 0 (no immediate price impact)

IMPACT RULE:
- HIGH: any event that shifts macro narrative (war, peace, Fed decision, tariff, national ban).
  ALWAYS HIGH if involves Trump+war, Fed+rate, or national regulation.
- MEDIUM: significant but does not shift narrative (ETF flow, miner sell-off, whale move)
- LOW: informative, analysis, forecast, opinion

ATTENTION for news tagged [MACRO]:
These do not mention BTC directly. Classify by INDIRECT impact via causal chain:
- Oil rises -> inflation -> higher rates -> risk-off -> BTC falls
- War escalates -> risk-off -> flight to safety -> BTC falls (short term)
- Fed dovish -> liquidity -> risk-on -> BTC rises
- Recession -> context-dependent: risk-off (BTC falls) or digital gold (BTC rises)
Same score scale applies to both [CRYPTO] and [MACRO].

Classify each news item. Reply ONLY in valid JSON, no markdown, no explanation:
[
  {{
    "index": 0,
    "topic": "geopolitical_war|trump_policy|fed_monetary|institutional_btc|oil_energy|regulatory|market_stress|mining|noise",
    "event_type": "ESCALATION|DEESCALATION|POLICY_SHIFT|CAPITAL_FLOW|MARKET_SHOCK|DATA_RELEASE|INSTITUTIONAL_MOVE|NOISE",
    "impact": "HIGH|MEDIUM|LOW",
    "score": integer from -10 to +10,
    "reason": "max 8 words, ASCII only"
  }}
]

Score scale:
+7 to +10: shifts entire narrative bullish
+3 to +6: significant upside pressure
+1 to +2: slight positive
0: neutral/irrelevant
-1 to -2: slight negative
-3 to -6: significant downside pressure
-7 to -10: shifts entire narrative bearish

NEWS (classify ALL):
{news_list}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_title(title: str) -> str:
    """Normalizar título: remove pontuação, lowercase, strip. (Correção 4)"""
    return re.sub(r'[^a-z0-9 ]', '', title.lower()).strip()


def generate_title_hash(title: str) -> str:
    """SHA1 do título normalizado — mais robusto contra variações de fonte. (Correção 4)"""
    return hashlib.sha1(normalize_title(title).encode()).hexdigest()


def _get_source_weight(source: str) -> float:
    """Lookup source weight case-insensitively. (Correção 5)"""
    src_lower = source.lower()
    for key, weight in SOURCE_WEIGHTS.items():
        if key in src_lower:
            return weight
    return DEFAULT_SOURCE_WEIGHT


def macro_relevance_filter(title: str) -> bool:
    """Retorna True se o título é relevante para macro BTC. (Correção 3)"""
    title_lower = title.lower()
    return any(kw in title_lower for kw in MACRO_RELEVANCE_KEYWORDS)


def _derive_direction(score: float) -> str:
    if score >= 2:
        return "BULLISH"
    if score <= -2:
        return "BEARISH"
    return "NEUTRAL"


def _load_api_key() -> str | None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    if SECRETS_PATH.exists():
        try:
            with open(SECRETS_PATH, "rb") as f:
                secrets = tomllib.load(f)
            return secrets.get("DEEPSEEK_API_KEY")
        except Exception as e:
            log.warning("Failed to read secrets.toml: %s", e)
    return None


def _load_config() -> dict:
    if KEYWORDS_CFG.exists():
        with open(KEYWORDS_CFG) as f:
            return json.load(f)
    return {"combine_weights": {"macro_weight": 0.60, "crypto_weight": 0.40}}


def _load_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "published_at" in df.columns and df["published_at"].dt.tz is None:
        df["published_at"] = df["published_at"].dt.tz_localize("UTC")
    for col in DS_COLS:
        if col not in df.columns:
            df[col] = None if col != "ds_classified" else False
    # Normalise title_hash to SHA1 if not already (backward compat with MD5 rows)
    if "title_hash" not in df.columns:
        df["title_hash"] = df["title"].apply(generate_title_hash)
    return df


# ── DeepSeek ──────────────────────────────────────────────────────────────────

def _call_deepseek(news_items: list[dict], api_key: str) -> list[dict]:
    """
    news_items: list of {index, title, news_type}
    Returns: list of classification dicts.
    """
    news_list = "\n".join(
        f"[{n['index']}] [{'MACRO' if n['news_type'] == 'macro' else 'CRYPTO'}] {n['title']}"
        for n in news_items
    )
    prompt = CLASSIFY_PROMPT.format(news_list=news_list)

    payload = {
        "model": "deepseek-chat",
        "max_tokens": 2000,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"]
    content = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()

    result = _parse_json_safe(content)
    if result is None:
        log.error("JSON parse failed.\nContent: %s", content[:800])
        raise ValueError("Could not parse DeepSeek response as JSON")
    return result


def _parse_json_safe(content: str) -> list[dict] | None:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\[.*\]', content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    items = []
    for m in re.finditer(r'\{[^{}]+\}', content, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if "index" in obj and "score" in obj:
                items.append(obj)
        except json.JSONDecodeError:
            continue
    if items:
        log.warning("JSON recovery: extracted %d items via object-level parsing.", len(items))
        return items
    return None


# ── Metrics (APENAS ds_score — Correção 1) ────────────────────────────────────

def _filter_classified(df: pd.DataFrame | None, hours: int) -> pd.DataFrame:
    """Return rows classified by DeepSeek within time window."""
    if df is None or df.empty or "ds_score" not in df.columns:
        return pd.DataFrame()
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    mask = (
        (df["published_at"] >= cutoff)
        & (df["ds_classified"] == True)  # noqa: E712
        & df["ds_score"].notna()
    )
    return df[mask].copy()


def _get_top_stories(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """Top N notícias por abs(ds_score) — as mais impactantes."""
    ranked = df.iloc[df["ds_score"].abs().argsort()[::-1]].head(n)
    return [
        {
            "title":       str(row["title"])[:120],
            "score":       float(row["ds_score"]),
            "event_type":  str(row.get("ds_event_type", "")),
            "group":       str(row.get("search_group", row.get("ds_topic", ""))),
            "source":      str(row.get("source", "")),
            "news_type":   str(row.get("news_type", "")),
        }
        for _, row in ranked.iterrows()
    ]


def _compute_crypto_impact(df: pd.DataFrame | None, hours: int) -> dict:
    """DeepSeek impact metrics for crypto news (source-weighted)."""
    recent = _filter_classified(df, hours)
    if recent.empty:
        return {
            "news_count": 0, "crypto_score": 0.0,
            "high_impact_count": 0, "escalation_count": 0, "deescalation_count": 0,
            "dominant_topic": None, "dominant_event_type": None, "top_stories": [],
        }

    # Source weighting
    recent = recent.copy()
    recent["source_weight"] = recent.get("source", pd.Series("", index=recent.index)).apply(
        lambda s: _get_source_weight(str(s))
    )
    recent["weighted_score"] = recent["ds_score"] * recent["source_weight"]

    crypto_score       = float(recent["weighted_score"].mean())
    high               = recent[recent["ds_impact"] == "HIGH"]
    escalation_count   = int((recent["ds_event_type"] == "ESCALATION").sum())
    deescalation_count = int((recent["ds_event_type"] == "DEESCALATION").sum())

    dominant_topic = (
        recent["ds_topic"].value_counts().index[0]
        if "ds_topic" in recent.columns and not recent["ds_topic"].isna().all() else None
    )
    dominant_event_type = (
        recent["ds_event_type"].value_counts().index[0]
        if not recent["ds_event_type"].isna().all() else None
    )

    return {
        "news_count":          int(len(recent)),
        "crypto_score":        crypto_score,
        "high_impact_count":   int(len(high)),
        "escalation_count":    escalation_count,
        "deescalation_count":  deescalation_count,
        "dominant_topic":      dominant_topic,
        "dominant_event_type": dominant_event_type,
        "top_stories":         _get_top_stories(recent),
    }


def _compute_macro_impact(macro_df: pd.DataFrame | None, config: dict, hours: int) -> dict:
    """Macro metrics: source-weighted score por grupo. (Correções 1, 5)"""
    recent = _filter_classified(macro_df, hours)
    if recent.empty:
        return {
            "news_count": 0, "macro_score": 0.0,
            "escalation_count": 0, "deescalation_count": 0,
            "dominant_group": None, "by_group": {}, "top_stories": [],
        }

    recent = recent.copy()
    recent["source_weight"] = recent.get("source", pd.Series("", index=recent.index)).apply(
        lambda s: _get_source_weight(str(s))
    )
    recent["weighted_score"] = recent["ds_score"] * recent["source_weight"]

    macro_score        = float(recent["weighted_score"].mean())
    escalation_count   = int((recent["ds_event_type"] == "ESCALATION").sum())
    deescalation_count = int((recent["ds_event_type"] == "DEESCALATION").sum())

    # By group
    groups   = config.get("search_groups", {})
    by_group = {}
    for gname in groups:
        gdf = recent[recent["search_group"] == gname]
        if gdf.empty:
            by_group[gname] = {"count": 0, "score": 0.0}
        else:
            by_group[gname] = {
                "count": int(len(gdf)),
                "score": float(gdf["weighted_score"].mean()),
            }

    # Dominant group: largest abs weighted score with ≥1 article
    by_group_filtered = {k: v for k, v in by_group.items() if v["count"] > 0}
    dominant_group = None
    if by_group_filtered:
        dominant_group = max(by_group_filtered, key=lambda k: abs(by_group_filtered[k]["score"]))

    return {
        "news_count":        int(len(recent)),
        "macro_score":       macro_score,
        "escalation_count":  escalation_count,
        "deescalation_count": deescalation_count,
        "dominant_group":    dominant_group,
        "by_group":          by_group,
        "top_stories":       _get_top_stories(recent),
    }


def _compute_combined(crypto_score: float, macro_score: float, weights: dict) -> dict:
    """Combined score com pesos explícitos e auditáveis."""
    mw = float(weights.get("macro_weight",  0.60))
    cw = float(weights.get("crypto_weight", 0.40))
    combined = macro_score * mw + crypto_score * cw
    return {
        "crypto_score":    crypto_score,
        "macro_score":     macro_score,
        "combined_score":  float(combined),
        "weights":         {"macro_weight": mw, "crypto_weight": cw},
        "dominant_driver": "macro" if abs(macro_score) > abs(crypto_score) else "crypto",
    }


def _compute_high_stress(combined: dict) -> bool:
    """High stress = combined muito negativo OU macro+crypto ambos negativos. (Ajuste adicional)"""
    c4h = combined.get("4h", {})
    return (
        c4h.get("combined_score", 0) < -3.0
        or (c4h.get("macro_score", 0) < -4.0 and c4h.get("crypto_score", 0) < -0.3)
    )


def _compute_news_zscore(crypto_df: pd.DataFrame | None) -> float:
    """Z-score do volume de notícias na última hora vs média histórica."""
    if crypto_df is None or crypto_df.empty:
        return 0.0
    try:
        now    = pd.Timestamp.now(tz="UTC")
        count_1h  = int((crypto_df["published_at"] >= now - pd.Timedelta(hours=1)).sum())
        count_24h = int((crypto_df["published_at"] >= now - pd.Timedelta(hours=24)).sum())
        hourly_avg = count_24h / 24.0
        hourly_std = max(hourly_avg ** 0.5, 1.0)
        return float((count_1h - hourly_avg) / hourly_std)
    except Exception:
        return 0.0


def _update_metrics(
    crypto_df: pd.DataFrame | None,
    macro_df:  pd.DataFrame | None,
    config:    dict,
) -> None:
    """
    Update sentiment_metrics.json after classification.
    Uses ONLY ds_score (source-weighted) — never sentiment_raw. (Correção 1)
    """
    if not METRICS_PATH.exists():
        log.warning("sentiment_metrics.json not found — skipping.")
        return

    try:
        metrics = json.loads(METRICS_PATH.read_text())
    except Exception as e:
        log.warning("Failed to read metrics: %s", e)
        metrics = {}

    weights = config.get("combine_weights", {"macro_weight": 0.60, "crypto_weight": 0.40})

    combined_all: dict = {}
    for hours in [1, 4, 24]:
        key = f"{hours}h"

        crypto_imp = _compute_crypto_impact(crypto_df, hours)
        macro_imp  = _compute_macro_impact(macro_df, config, hours)
        combined   = _compute_combined(
            crypto_imp.get("crypto_score", 0.0),
            macro_imp.get("macro_score", 0.0),
            weights,
        )
        combined_all[key] = combined

        metrics.setdefault("impact", {})[key] = {
            "crypto_score":        crypto_imp.get("crypto_score", 0.0),
            "high_impact_count":   crypto_imp.get("high_impact_count", 0),
            "escalation_count":    crypto_imp.get("escalation_count", 0),
            "deescalation_count":  crypto_imp.get("deescalation_count", 0),
            "dominant_topic":      crypto_imp.get("dominant_topic"),
            "dominant_event_type": crypto_imp.get("dominant_event_type"),
            "top_stories":         crypto_imp.get("top_stories", []),
        }

        metrics.setdefault("macro", {})[key] = {
            "macro_score":        macro_imp.get("macro_score", 0.0),
            "news_count":         macro_imp.get("news_count", 0),
            "escalation_count":   macro_imp.get("escalation_count", 0),
            "deescalation_count": macro_imp.get("deescalation_count", 0),
            "dominant_group":     macro_imp.get("dominant_group"),
            "by_group":           macro_imp.get("by_group", {}),
            "top_stories":        macro_imp.get("top_stories", []),
        }

        metrics.setdefault("combined", {})[key] = combined

    metrics["high_stress"]      = _compute_high_stress(combined_all)
    metrics["news_zscore_1h"]   = _compute_news_zscore(crypto_df)
    metrics["impact_updated_at"] = datetime.now(timezone.utc).isoformat()

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    c4h = combined_all.get("4h", {})
    log.info(
        "Metrics updated — combined 4h: %.3f (crypto=%.3f × %.1f + macro=%.3f × %.1f) "
        "driver=%s | high_stress=%s",
        c4h.get("combined_score", 0),
        c4h.get("crypto_score", 0),  weights.get("crypto_weight", 0.4),
        c4h.get("macro_score", 0),   weights.get("macro_weight", 0.6),
        c4h.get("dominant_driver", "—"),
        metrics["high_stress"],
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = _load_api_key()
    if not api_key:
        log.error("DEEPSEEK_API_KEY not found in env or .streamlit/secrets.toml — aborting.")
        return

    config = _load_config()

    # ── Load both parquets ────────────────────────────────────────────────────
    crypto_df = _load_parquet(CRYPTO_PARQUET)
    macro_df  = _load_parquet(MACRO_PARQUET)

    if crypto_df is None:
        log.warning("btc_news.parquet not found — run cryptocompare_news_ingest.py first.")
        return

    # Global dedup: remove from macro what already exists in crypto
    if macro_df is not None and not macro_df.empty:
        crypto_hashes = set(crypto_df["title_hash"])
        before = len(macro_df)
        macro_df = macro_df[~macro_df["title_hash"].isin(crypto_hashes)].copy()
        dupes = before - len(macro_df)
        if dupes:
            log.info("Dedup: removed %d macro articles already in crypto parquet.", dupes)

    # ── Correção 3: Pre-classify irrelevant macro as NOISE (no API cost) ─────
    classified_at = datetime.now(timezone.utc).isoformat()
    noise_count = 0
    if macro_df is not None and not macro_df.empty:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=CLASSIFY_WINDOW)
        unclassified_mask = (
            (macro_df["published_at"] >= cutoff)
            & (macro_df["ds_classified"] != True)  # noqa: E712
        )
        unclassified_idx = macro_df[unclassified_mask].index
        for idx in unclassified_idx:
            title = str(macro_df.at[idx, "title"])
            if not macro_relevance_filter(title):
                macro_df.at[idx, "ds_classified"]  = True
                macro_df.at[idx, "ds_score"]        = 0.0
                macro_df.at[idx, "ds_event_type"]   = "NOISE"
                macro_df.at[idx, "ds_impact"]       = "LOW"
                macro_df.at[idx, "ds_topic"]        = "noise"
                macro_df.at[idx, "ds_direction"]    = "NEUTRAL"
                macro_df.at[idx, "ds_reason"]       = "irrelevant to macro BTC"
                macro_df.at[idx, "ds_classified_at"] = classified_at
                noise_count += 1
        if noise_count:
            log.info("Pre-classified %d macro articles as NOISE (relevance filter).", noise_count)

    # ── Collect unclassified within window ────────────────────────────────────
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=CLASSIFY_WINDOW)

    def _pending(df: pd.DataFrame | None, news_type: str) -> list[dict]:
        if df is None or df.empty:
            return []
        mask = (df["published_at"] >= cutoff) & (df["ds_classified"] != True)  # noqa: E712
        return [
            {"index": -1, "id": row["id"], "title": row["title"], "news_type": news_type}
            for _, row in df[mask].iterrows()
        ]

    pending = _pending(crypto_df, "crypto") + _pending(macro_df, "macro")

    if not pending:
        log.info("Nothing to classify in last %dh.", CLASSIFY_WINDOW)
        _update_metrics(crypto_df, macro_df, config)
        return

    if len(pending) > MAX_ITEMS:
        pending = pending[:MAX_ITEMS]
        log.info("Capped to %d items.", MAX_ITEMS)

    for i, item in enumerate(pending):
        item["index"] = i

    log.info(
        "Classifying %d items (%d crypto, %d macro) via DeepSeek...",
        len(pending),
        sum(1 for p in pending if p["news_type"] == "crypto"),
        sum(1 for p in pending if p["news_type"] == "macro"),
    )

    # ── Call DeepSeek ─────────────────────────────────────────────────────────
    try:
        classifications = _call_deepseek(pending, api_key)
    except requests.HTTPError as e:
        log.error("DeepSeek API error: %s", e)
        _update_metrics(crypto_df, macro_df, config)
        return
    except Exception as e:
        log.error("Classification failed: %s", e)
        return

    log.info("Received %d classifications.", len(classifications))

    cls_map: dict[int, dict] = {
        int(c.get("index", -1)): c for c in classifications if "index" in c
    }

    def _apply_classifications(df: pd.DataFrame, news_type: str) -> pd.DataFrame:
        pending_type = [p for p in pending if p["news_type"] == news_type]
        updated = 0
        for item in pending_type:
            cls = cls_map.get(item["index"])
            if cls is None:
                continue
            mask = df["id"] == item["id"]
            if not mask.any():
                continue
            score = float(cls.get("score", 0))
            df.loc[mask, "ds_classified"]   = True
            df.loc[mask, "ds_topic"]        = str(cls.get("topic", "noise"))
            df.loc[mask, "ds_event_type"]   = str(cls.get("event_type", "NOISE"))
            df.loc[mask, "ds_impact"]       = str(cls.get("impact", "LOW"))
            df.loc[mask, "ds_score"]        = score
            df.loc[mask, "ds_direction"]    = _derive_direction(score)
            df.loc[mask, "ds_reason"]       = str(cls.get("reason", ""))[:120]
            df.loc[mask, "ds_classified_at"] = classified_at
            updated += 1
        log.info("[%s] Updated %d rows.", news_type, updated)
        return df

    # ── Apply + save both parquets ────────────────────────────────────────────
    crypto_df = _apply_classifications(crypto_df, "crypto")
    crypto_df.to_parquet(CRYPTO_PARQUET, index=False)
    log.info("Saved crypto parquet → %s", CRYPTO_PARQUET)

    if macro_df is not None and not macro_df.empty:
        macro_df = _apply_classifications(macro_df, "macro")
        macro_df.to_parquet(MACRO_PARQUET, index=False)
        log.info("Saved macro parquet → %s", MACRO_PARQUET)

    # Aggregate metrics only after classification (Correção 2)
    _update_metrics(crypto_df, macro_df, config)


if __name__ == "__main__":
    main()
