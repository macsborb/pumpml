"""Importe comme exemples les coins qui ont perce (data/breakouts.json + liste connue), via le dashboard.

Filtre les valorisations artificielles (market cap enorme pour tres peu de volume) et verifie chaque
adresse sur DexScreener (nom, market cap) avant l'import.
Usage : .venv\\Scripts\\python scripts\\import_breakouts.py
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASH = "http://127.0.0.1:8787"

# Grands viraux pump.fun connus (adresses verifiees sur DexScreener avant import).
KNOWN = {
    "9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump": "Fartcoin",
    "CzLSujWBLFsSjncfkh59rUFqvafWcY5tzedWJSuypump": "Goatseus Maximus",
    "Df6yfrKC8kZE3KNkrHERKzAetSxbrWeniQfyJY4Jpump": "Just a chill guy",
    "ED5nyyWEzpPPiWimP8vYm7sD7TD3LAt3Q3gRTWHzPJBY": "Moo Deng",
    "2qEHjDLDLbuBgRYvsxhc5D6uDWAivNFZGan56P1tpump": "Peanut the Squirrel",
}


def http(url: str, data: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None,
                                 headers={"Content-Type": "application/json", "User-Agent": "pumpml/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def dexscreener(mint: str) -> dict | None:
    try:
        j = http(f"https://api.dexscreener.com/latest/dex/tokens/{mint}")
    except Exception:
        return None
    pairs = [p for p in (j.get("pairs") or []) if p.get("baseToken", {}).get("address") == mint]
    if not pairs:
        return None
    p = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd") or 0)
    return {"name": p["baseToken"].get("name"), "symbol": p["baseToken"].get("symbol"),
            "mcap": p.get("marketCap") or p.get("fdv") or 0, "liq": (p.get("liquidity") or {}).get("usd") or 0,
            "vol24": (p.get("volume") or {}).get("h24") or 0}


def plausible(d: dict) -> bool:
    """Ecarte les valorisations artificielles : il faut de la liquidite et du volume en rapport."""
    return d["liq"] >= 15_000 and (d["mcap"] < 20_000_000 or d["vol24"] >= d["mcap"] / 200)


def main():
    cands = {x["mint"]: x.get("name") for x in json.loads((ROOT / "data" / "breakouts.json").read_text("utf-8"))}
    cands.update(KNOWN)
    done = {e["mint"] for e in http(f"{DASH}/api/examples")}
    report = []
    for mint, name in cands.items():
        if mint in done:
            continue
        d = dexscreener(mint)
        time.sleep(1.1)
        if d is None:
            report.append((mint, name, "introuvable sur DexScreener"))
            continue
        if mint not in KNOWN and not plausible(d):
            report.append((mint, d["name"], f"ecarte (mcap {d['mcap']:,.0f} $, liquidite {d['liq']:,.0f} $, vol {d['vol24']:,.0f} $)"))
            continue
        r = http(f"{DASH}/api/import", {"q": mint})
        report.append((mint, d["name"], f"import demande (mcap {d['mcap']:,.0f} $)" if r.get("ok") else "deja en cours"))
    for m, n, s in report:
        print(f"{m[:10]}  {str(n)[:30]:30}  {s}")


if __name__ == "__main__":
    main()
