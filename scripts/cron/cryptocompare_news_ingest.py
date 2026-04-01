"""
CryptoCompare News Ingest — roda a cada hora via cron.
Busca notícias BTC com sentimento nativo, salva em parquet incremental.
1 request por execução (respeita 50 calls/hora do free tier).

Outputs:
    data/01_raw/news/cryptocompare/btc_news.parquet       — event-level history
    data/01_raw/news/cryptocompare/sentiment_metrics.json  — aggregated metrics
"""

import json
import logging
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "data" / "01_raw" / "news" / "cryptocompare"
PARQUET_PATH = DATA_DIR / "btc_news.parquet"
METRICS_PATH = DATA_DIR / "sentiment_metrics.json"
BASE_URL = "https://data-api.cryptocompare.com/news/v1/article/list"
RETENTION_DAYS = 90


def fetch_articles(limit: int = 50) -> list[dict]:
    url = f"{BASE_URL}?lang=EN&categories=BTC&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "crypto-market-state/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    return data.get("Data", [])


def parse_article(r: dict) -> dict:
    return {
        "id": str(r["ID"]),
        "guid": r.get("GUID", ""),
        "title": r.get("TITLE", ""),
        "body": r.get("BODY", "")[:500],
        "url": r.get("URL", ""),
        "published_at": pd.Timestamp(r["PUBLISHED_ON"], unit="s", tz="UTC"),
        "sentiment": r.get("SENTIMENT", "NEUTRAL"),
        "upvotes": int(r.get("UPVOTES", 0) or 0),
        "downvotes": int(r.get("DOWNVOTES", 0) or 0),
        "score": float(r.get("SCORE", 0) or 0),
        "keywords": r.get("KEYWORDS", ""),
        "source": (r.get("SOURCE_DATA") or {}).get("NAME", ""),
        "categories": str(r.get("CATEGORY_DATA", [])),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def load_existing() -> pd.DataFrame:
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
        if "published_at" in df.columns and df["published_at"].dt.tz is None:
            df["published_at"] = df["published_at"].dt.tz_localize("UTC")
        return df
    return pd.DataFrame()


def compute_metrics(df: pd.DataFrame, hours: int) -> dict:
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    recent = df[df["published_at"] >= cutoff]

    if len(recent) == 0:
        return {
            "news_count": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
            "sentiment_score": 0.0,
            "weighted_sentiment": 0.0,
            "total_score": 0.0,
        }

    n = len(recent)
    pos = int((recent["sentiment"] == "POSITIVE").sum())
    neg = int((recent["sentiment"] == "NEGATIVE").sum())
    sentiment_score = (pos - neg) / n

    def _weight(row):
        sign = 1 if row["sentiment"] == "POSITIVE" else -1 if row["sentiment"] == "NEGATIVE" else 0
        return sign * (1 + row["upvotes"])

    weighted = recent.apply(_weight, axis=1)

    return {
        "news_count": int(n),
        "positive_pct": float(pos / n * 100),
        "negative_pct": float(neg / n * 100),
        "neutral_pct": float((n - pos - neg) / n * 100),
        "sentiment_score": float(sentiment_score),
        "weighted_sentiment": float(weighted.sum() / n),
        "total_score": float(recent["score"].sum()),
    }


def compute_news_zscore(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    series = df.set_index("published_at").resample("1h").size()
    if len(series) < 2:
        return 0.0
    mean = series.rolling(24, min_periods=2).mean()
    std = series.rolling(24, min_periods=2).std().replace(0, np.nan)
    z = (series - mean) / std
    last = z.iloc[-1]
    return float(last) if pd.notna(last) else 0.0


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Fetching CryptoCompare BTC news...")
    try:
        raw = fetch_articles(limit=50)
    except Exception as e:
        log.warning("API fetch failed: %s — skipping", e)
        return

    new_rows = [parse_article(r) for r in raw]
    new_df = pd.DataFrame(new_rows)
    log.info("Fetched %d articles", len(new_df))

    existing = load_existing()
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["id"], keep="last")

    cutoff_90d = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=RETENTION_DAYS)
    combined = combined[combined["published_at"] >= cutoff_90d]
    combined = combined.sort_values("published_at", ascending=False).reset_index(drop=True)

    combined.to_parquet(PARQUET_PATH, index=False)
    log.info("Saved %d total rows → %s", len(combined), PARQUET_PATH)

    m1h = compute_metrics(combined, 1)
    m4h = compute_metrics(combined, 4)
    m24h = compute_metrics(combined, 24)
    zscore = compute_news_zscore(combined)

    high_stress = bool(zscore > 2.0 and m1h["sentiment_score"] < -0.3)

    metrics = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "news_zscore_24h": float(zscore),
        "high_stress": high_stress,
        "1h": m1h,
        "4h": m4h,
        "24h": m24h,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(
        "Metrics saved — 1h: %d news, score=%.2f | 4h: %d news, score=%.2f | stress=%s",
        m1h["news_count"], m1h["sentiment_score"],
        m4h["news_count"], m4h["sentiment_score"],
        high_stress,
    )


if __name__ == "__main__":
    main()
