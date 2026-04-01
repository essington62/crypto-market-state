"""
Macro News Ingest — busca notícias macro exógenas via Google News RSS.
APENAS ingestão — classificação e métricas são responsabilidade do classify_news.py.

Keywords configuráveis em conf/macro_keywords.json.
Rotação de queries: 1 query por grupo por hora → 5 requests/hora (rate-limit friendly).

Output:
    data/01_raw/news/cryptocompare/macro_news.parquet
"""

import hashlib
import json
import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT      = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT / "data" / "01_raw" / "news" / "cryptocompare"
PARQUET_PATH = DATA_DIR / "macro_news.parquet"
KEYWORDS_CFG = PROJECT / "conf" / "macro_keywords.json"
RETENTION_DAYS = 90

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


def _normalize_title(title: str) -> str:
    """Remove pontuação, lowercase, strip — robusto contra variações entre fontes."""
    return re.sub(r'[^a-z0-9 ]', '', title.lower()).strip()


def title_hash(title: str) -> str:
    """SHA1 do título normalizado — robusto contra variações de fonte."""
    return hashlib.sha1(_normalize_title(title).encode()).hexdigest()


def _fetch_rss(query: str, max_results: int = 10) -> list[dict]:
    """Buscar notícias via Google News RSS."""
    url = f"{GOOGLE_NEWS_RSS}?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        req  = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        root = ET.fromstring(resp.read())
    except Exception as e:
        log.warning("RSS failed for '%s': %s", query, e)
        return []

    items = []
    for item in root.findall(".//item")[:max_results]:
        title_el  = item.find("title")
        link_el   = item.find("link")
        pub_el    = item.find("pubDate")
        source_el = item.find("source")

        title  = title_el.text  if title_el  is not None else ""
        link   = link_el.text   if link_el   is not None else ""
        pub_date = pub_el.text  if pub_el    is not None else ""
        source = source_el.text if source_el is not None else ""

        try:
            published_at = parsedate_to_datetime(pub_date)
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)
        except Exception:
            published_at = datetime.now(timezone.utc)

        items.append({
            "title":        title,
            "url":          link,
            "source":       source,
            "published_at": published_at,
            "title_hash":   title_hash(title),
        })

    return items


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not KEYWORDS_CFG.exists():
        log.error("macro_keywords.json not found at %s", KEYWORDS_CFG)
        return

    with open(KEYWORDS_CFG) as f:
        config = json.load(f)

    groups      = config.get("search_groups", {})
    max_results = int(config.get("max_results_per_query", 10))
    hour        = datetime.now(timezone.utc).hour

    # ── Load existing ─────────────────────────────────────────────────────────
    if PARQUET_PATH.exists():
        existing = pd.read_parquet(PARQUET_PATH)
        if "published_at" in existing.columns and existing["published_at"].dt.tz is None:
            existing["published_at"] = existing["published_at"].dt.tz_localize("UTC")
        existing_hashes: set = set(existing["title_hash"].astype(str))
    else:
        existing        = pd.DataFrame()
        existing_hashes = set()

    # ── Fetch ─────────────────────────────────────────────────────────────────
    new_rows: list[dict] = []
    for group_name, group_cfg in groups.items():
        if not group_cfg.get("enabled", True):
            log.info("Group %s disabled — skipping.", group_name)
            continue

        queries = group_cfg["queries"]
        query   = queries[hour % len(queries)]
        log.info("Group %s → query: '%s'", group_name, query)

        try:
            raw = _fetch_rss(query, max_results=max_results)
        except Exception as e:
            log.warning("Fetch failed for '%s': %s — skipping.", query, e)
            continue

        for a in raw:
            thash = a["title_hash"]
            if thash in existing_hashes:
                continue
            new_rows.append({
                "id":           thash,          # URL ou hash como ID único
                "title":        a["title"],
                "title_hash":   thash,
                "body":         "",
                "url":          a["url"],
                "published_at": pd.Timestamp(a["published_at"]).tz_convert("UTC"),
                "sentiment_raw": "NEUTRAL",     # RSS não tem sentimento — campo mantido por schema
                "source":       a["source"],
                "search_group": group_name,
                "search_query": query,
                "news_type":    "macro",
                "ingested_at":  datetime.now(timezone.utc).isoformat(),
                # ds_* iniciam vazios — preenchidos pelo classify_news.py
                "ds_classified": False,
                "ds_topic":      None,
                "ds_event_type": None,
                "ds_impact":     None,
                "ds_score":      None,
                "ds_direction":  None,
                "ds_reason":     None,
            })
            existing_hashes.add(thash)

    log.info("Fetched %d new macro articles.", len(new_rows))

    # ── Merge + dedup + save ──────────────────────────────────────────────────
    if new_rows:
        new_df   = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    else:
        combined = existing

    if combined.empty:
        log.info("No macro news data yet.")
        return

    combined = combined.drop_duplicates(subset=["title_hash"], keep="last")

    cutoff   = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=RETENTION_DAYS)
    combined = combined[combined["published_at"] >= cutoff]
    combined = combined.sort_values("published_at", ascending=False).reset_index(drop=True)

    combined.to_parquet(PARQUET_PATH, index=False)
    log.info("Saved %d total macro rows → %s", len(combined), PARQUET_PATH)


if __name__ == "__main__":
    main()
