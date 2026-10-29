"""Liste les coins pump.fun qui ont perce (graduation sur PumpSwap, market cap > seuil) via GeckoTerminal.

Usage : .venv\\Scripts\\python scripts\\discover_breakouts.py [--min-mcap 200000] [--days 120] [--pages 10]
Ecrit data/breakouts.json (adresse, nom, ticker, market cap, date de creation du pool, volume).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.geckoterminal.com/api/v2/networks/solana/dexes/pumpswap/pools"


def get(url: str) -> dict:
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "pumpml/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:  # limite gratuite : ~30 requetes/min
            time.sleep(3 + attempt * 4)
    raise RuntimeError(url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mcap", type=float, default=200_000)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--pages", type=int, default=10)
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    out: dict[str, dict] = {}
    for sort in ("h24_volume_usd_desc", "h24_tx_count_desc"):
        for page in range(1, a.pages + 1):
            j = get(f"{API}?include=base_token&sort={sort}&page={page}")
            toks = {t["id"]: t["attributes"] for t in j.get("included", []) if t.get("type") == "token"}
            for p in j.get("data", []):
                at = p["attributes"]
                base = p["relationships"]["base_token"]["data"]["id"]  # "solana_<mint>"
                mint = base.split("_", 1)[1]
                if not mint.endswith("pump"):
                    continue  # pas un coin pump.fun
                mcap = float(at.get("market_cap_usd") or at.get("fdv_usd") or 0)
                created = at.get("pool_created_at")
                age_d = (now - datetime.fromisoformat(created.replace("Z", "+00:00"))).days if created else None
                if mcap < a.min_mcap or (age_d is not None and age_d > a.days):
                    continue
                t = toks.get(base, {})
                out[mint] = {"mint": mint, "name": t.get("name") or at.get("name"), "symbol": t.get("symbol"),
                             "mcap_usd": round(mcap), "pool_created": created, "age_days": age_d,
                             "vol24_usd": round(float((at.get("volume_usd") or {}).get("h24") or 0))}
            time.sleep(2.2)
            print(f"{sort} page {page} : {len(out)} coins", file=sys.stderr)
    res = sorted(out.values(), key=lambda x: -x["mcap_usd"])
    path = ROOT / "data" / "breakouts.json"
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(res)} coins pump.fun > {a.min_mcap:.0f} $ (pools de moins de {a.days} j) -> {path}")


if __name__ == "__main__":
    main()
