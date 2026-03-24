"""
audit_start_dates.py
====================
Auditoria de cobertura temporal do data lake (L1 + L2).
Agrupa por classe de ativo e sinaliza descompasso vs target start_date.

Uso:
    cd /Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state
    conda activate crypto_market_state
    python scripts/analysis/audit_start_dates.py

Saída:
    - tabela no terminal por classe de ativo
    - CSV em scripts/analysis/output/audit_start_dates.csv
"""

import os
import pandas as pd
from pathlib import Path
from datetime import date

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]   # raiz do projeto
TARGET_DEFAULT = pd.Timestamp("2020-10-01", tz="UTC")

# Start-date target por classe — alinhado com parameters.yml
# Regras:
#   daily / weekly / monthly  →  2020-10-01  (default)
#   crypto spot 4h            →  2025-09-22  (janela móvel ≈ 6 meses, alinhado CoinGlass)
#   coinglass derivatives 4h  →  2025-09-24  (início real do acúmulo via cron)
#   coinglass orderbook 4h    →  2025-10-05  (início real do acúmulo via cron)
#   ETFs BTC (IBIT etc.)      →  2024-01-26  (lançamento dos ETFs spot)
#   ETF flows                 →  2024-01-11  (início dos dados de flow)
#   cgdi_index                →  2024-01-01  (dado genuinamente ausente antes)
#   cdri_index                →  2022-03-05  (dado genuinamente ausente antes)
TARGET_BY_CLASS = {
    "Crypto Spot 4h (L2)":          pd.Timestamp("2025-09-22", tz="UTC"),
    "Crypto Spot 4h (L1)":          pd.Timestamp("2025-09-22", tz="UTC"),
    "CoinGlass 4h (derivatives)":   pd.Timestamp("2025-09-24", tz="UTC"),
    "CoinGlass 4h (orderbook)":     pd.Timestamp("2025-10-05", tz="UTC"),
}

# ─────────────────────────────────────────────
# MAPA DE ARQUIVOS POR CLASSE
# ─────────────────────────────────────────────
ASSET_MAP = {
    # ── Crypto spot ──────────────────────────────────────────────────────
    "Crypto Spot Daily (L2)": {
        "layer": "L2",
        "freq": "daily (24x7)",
        "files": sorted(BASE.glob("data/02_intermediate/spot/daily/*.parquet")),
    },
    "Crypto Spot 4h (L2)": {
        "layer": "L2",
        "freq": "4h",
        "files": sorted(BASE.glob("data/02_intermediate/spot/crypto/4h/*.parquet")),
    },
    "Crypto Spot Daily (L1)": {
        "layer": "L1",
        "freq": "daily (24x7)",
        "files": sorted(BASE.glob("data/01_raw/spot/crypto/daily_24x7/*.parquet")),
    },
    "Crypto Spot 4h (L1)": {
        "layer": "L1",
        "freq": "4h",
        "files": sorted(BASE.glob("data/01_raw/spot/crypto/4h/*.parquet")),
    },
    # ── yFinance / mercado ────────────────────────────────────────────────
    "yFinance Business Day (L2)": {
        "layer": "L2",
        "freq": "business day",
        "files": sorted(BASE.glob("data/02_intermediate/spot/business_day/*.parquet")),
    },
    "yFinance Business Day (L1)": {
        "layer": "L1",
        "freq": "business day",
        "files": sorted(BASE.glob("data/01_raw/spot/business_day/*.parquet")),
    },
    # ── FRED Macro ────────────────────────────────────────────────────────
    "FRED Daily (L2)": {
        "layer": "L2",
        "freq": "daily",
        "files": sorted(BASE.glob("data/02_intermediate/macro/daily/*.parquet")),
    },
    "FRED Weekly (L2)": {
        "layer": "L2",
        "freq": "weekly",
        "files": sorted(BASE.glob("data/02_intermediate/macro/weekly/*.parquet")),
    },
    "FRED Monthly (L2)": {
        "layer": "L2",
        "freq": "monthly",
        "files": sorted(BASE.glob("data/02_intermediate/macro/monthly/*.parquet")),
    },
    # ── CoinGlass Derivatives ─────────────────────────────────────────────
    "CoinGlass 4h (derivatives)": {
        "layer": "L2",
        "freq": "4h",
        "files": sorted(BASE.glob("data/02_intermediate/derivatives/coinglass/*.parquet")),
    },
    "CoinGlass 4h (orderbook)": {
        "layer": "L2",
        "freq": "4h",
        "files": sorted(BASE.glob("data/02_intermediate/orderbook/coinglass/4h/*.parquet")),
    },
    # ── CoinGlass Indices / sinais (L1 apenas) ────────────────────────────
    "CoinGlass Indices/Daily (L1)": {
        "layer": "L1",
        "freq": "daily",
        "files": sorted(BASE.glob("data/01_raw/derivatives/coinglass/indices/*.parquet"))
               + sorted(BASE.glob("data/01_raw/derivatives/coinglass/*.parquet"))
               + sorted(BASE.glob("data/01_raw/derivatives/coinglass/funding/*.parquet"))
               + sorted(BASE.glob("data/01_raw/derivatives/coinglass/open_interest/*.parquet"))
               + sorted(BASE.glob("data/01_raw/derivatives/coinglass/long_short_ratio/*.parquet"))
               + sorted(BASE.glob("data/01_raw/derivatives/coinglass/options/*.parquet"))
               + sorted(BASE.glob("data/01_raw/derivatives/coinglass/taker/*.parquet"))
               + sorted(BASE.glob("data/01_raw/derivatives/coinglass/etf/*.parquet")),
    },
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _normalize_tz(ts):
    """Garante que o timestamp seja UTC para comparação."""
    if ts is None or pd.isna(ts):
        return pd.NaT
    if hasattr(ts, "tzinfo") and ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def inspect_file(path: Path) -> dict:
    """Lê um parquet e retorna metadados de cobertura."""
    try:
        df = pd.read_parquet(path)

        # Pega o index ou a primeira coluna de data
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
        else:
            date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            if date_cols:
                idx = pd.DatetimeIndex(pd.to_datetime(df[date_cols[0]]))
            else:
                idx = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0]))

        start = _normalize_tz(idx.min())
        end   = _normalize_tz(idx.max())
        rows  = len(df)

        return {"start": start, "end": end, "rows": rows, "error": None}
    except Exception as e:
        return {"start": pd.NaT, "end": pd.NaT, "rows": 0, "error": str(e)[:60]}


# Ativos que têm start_date natural posterior ao default (dado genuinamente
# ausente antes disso — não é gap de ingestion, é limitação da fonte).
NATURAL_START = {
    "BTC_ARKB":                   pd.Timestamp("2024-01-26", tz="UTC"),
    "BTC_BITB":                   pd.Timestamp("2024-01-26", tz="UTC"),
    "BTC_FBTC":                   pd.Timestamp("2024-01-26", tz="UTC"),
    "BTC_GBTC":                   pd.Timestamp("2024-01-26", tz="UTC"),
    "BTC_IBIT":                   pd.Timestamp("2024-01-26", tz="UTC"),
    "BTC_holdings_consolidated":  pd.Timestamp("2024-01-26", tz="UTC"),
    "BTC_netassets_consolidated": pd.Timestamp("2024-01-26", tz="UTC"),
    "BTC_flows_by_ticker":        pd.Timestamp("2024-01-11", tz="UTC"),
    "BTC_flows_total":            pd.Timestamp("2024-01-11", tz="UTC"),
    "cgdi_index":                 pd.Timestamp("2024-01-01", tz="UTC"),
    "cdri_index":                 pd.Timestamp("2022-03-05", tz="UTC"),
    "BTCUSDT_vol_weighted":       pd.Timestamp("2023-04-23", tz="UTC"),
    "WALCL":                      pd.Timestamp("2020-10-07", tz="UTC"),  # FRED genuíno
}


def flag(start, target, asset_name=""):
    """Sinaliza descompasso vs target, respeitando NATURAL_START."""
    if pd.isna(start):
        return "❌ ERRO"
    # Se o ativo tem uma data natural posterior ao default, usa ela como referência
    effective_target = NATURAL_START.get(asset_name, target)
    diff_days = (start - effective_target).days
    if diff_days <= 5:
        return "✓"
    elif diff_days <= 90:
        return f"⚠  +{diff_days}d"
    else:
        return f"🔴 +{diff_days}d"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    TARGET_STR = TARGET_DEFAULT.strftime("%Y-%m-%d")
    print(f"\n{'=' * 85}")
    print(f"  DATA LAKE — AUDITORIA DE START DATES   |   target default: {TARGET_STR}")
    print(f"{'=' * 85}\n")

    records = []
    all_issues = []

    for class_name, cfg in ASSET_MAP.items():
        files = cfg["files"]
        if not files:
            continue

        target = TARGET_BY_CLASS.get(class_name, TARGET_DEFAULT)
        target_str = target.strftime("%Y-%m-%d")

        print(f"── {class_name}  [{cfg['layer']} | {cfg['freq']} | target: {target_str}]")
        print(f"   {'arquivo':<35} {'start':>12}  {'end':>12}  {'rows':>6}  {'status':>12}")
        print(f"   {'-'*35} {'-'*12}  {'-'*12}  {'-'*6}  {'-'*12}")

        for path in files:
            meta  = inspect_file(path)
            name  = path.stem
            start = meta["start"]
            end   = meta["end"]
            rows  = meta["rows"]
            err   = meta["error"]

            start_str = start.strftime("%Y-%m-%d") if not pd.isna(start) else "N/A"
            end_str   = end.strftime("%Y-%m-%d")   if not pd.isna(end)   else "N/A"
            status    = flag(start, target, name) if not err else f"❌ {err[:20]}"

            print(f"   {name:<35} {start_str:>12}  {end_str:>12}  {rows:>6}  {status:>12}")

            records.append({
                "class":        class_name,
                "layer":        cfg["layer"],
                "freq":         cfg["freq"],
                "asset":        name,
                "start_date":   start_str,
                "end_date":     end_str,
                "rows":         rows,
                "target":       target_str,
                "status":       status.replace("✓", "ok"),
                "error":        err or "",
            })

            effective_target = NATURAL_START.get(name, target)
            if err or (not pd.isna(start) and (start - effective_target).days > 5):
                all_issues.append((class_name, name, start_str, target_str, status))

        print()

    # ── Sumário de problemas ──────────────────────────────────────────────
    print(f"{'=' * 85}")
    print("  RESUMO — ativos com start_date posterior ao target")
    print(f"{'=' * 85}")
    if all_issues:
        print(f"   {'classe':<35} {'ativo':<35} {'start':>12}  {'target':>12}  {'delta':>12}")
        print(f"   {'-'*35} {'-'*35} {'-'*12}  {'-'*12}  {'-'*12}")
        for c, a, s, t, st in all_issues:
            print(f"   {c:<35} {a:<35} {s:>12}  {t:>12}  {st:>12}")
    else:
        print("   Nenhum problema encontrado — todos os ativos atingem o target ✓")
    print()

    # ── Salvar CSV ────────────────────────────────────────────────────────
    out_dir = BASE / "scripts" / "analysis" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "audit_start_dates.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"  CSV salvo em: {out_path.relative_to(BASE)}\n")


if __name__ == "__main__":
    main()
