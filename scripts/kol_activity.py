"""Tri des KOL : ne garder que les wallets actifs ces dernieres semaines.

Pour chaque wallet de la liste complete (``data/external/kol/kols.json``), on lit ses signatures recentes sur la
chaine (Helius getSignaturesForAddress, ~1 credit par page de 1000) : nombre de transactions sur ``--days``
jours et date de la derniere. On y ajoute les trades pump.fun vus sur nos coins suivis.
Garde : derniere transaction il y a moins de ``--max-idle-days`` jours ET (au moins ``--min-tx`` transactions sur
la periode, echouees comprises, OU au moins 5 trades vus sur nos coins). Ecrit ``data/external/kol/kols_actifs.json`` (meme format) et un rapport CSV de tous les wallets.

Usage : .venv\\Scripts\\python scripts\\kol_activity.py [--days 30] [--min-tx 100] [--max-idle-days 7]
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pumpml.config import Config  # noqa: E402

log = logging.getLogger("kol_activity")
SRC = ROOT / "data" / "external" / "kol" / "kols.json"
OUT = ROOT / "data" / "external" / "kol" / "kols_actifs.json"
REPORT = ROOT / "data" / "external" / "kol" / "activite.csv"


async def activity(s: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore, wallet: str, since_s: int,
                   max_pages: int, credits: list) -> tuple[int, int | None, bool]:
    """(transactions depuis since_s, reussies ou non, heure de la derniere, plus que max_pages pages)."""
    n, last, before = 0, None, None
    async with sem:
        for _ in range(max_pages):
            p = {"limit": 1000}
            if before:
                p["before"] = before
            for attempt in range(6):
                credits[0] += 1
                try:
                    async with s.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress",
                                                 "params": [wallet, p]}, timeout=aiohttp.ClientTimeout(total=40)) as r:
                        if r.status == 429 or r.status >= 500:
                            raise aiohttp.ClientError(str(r.status))
                        page = (await r.json()).get("result") or []
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.sleep(1 + 2 * attempt)
            else:
                return n, last, False
            if not page:
                return n, last, False
            if last is None:
                last = page[0].get("blockTime")
            recent = [x for x in page if (x.get("blockTime") or 0) >= since_s]
            n += len(recent)  # echouees comprises : les robots des gros traders en envoient beaucoup
            if len(recent) < len(page) or len(page) < 1000:
                return n, last, False
            before = page[-1]["signature"]
    return n, last, True


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30)
    ap.add_argument("--min-tx", type=int, default=100)
    ap.add_argument("--max-idle-days", type=float, default=7)
    ap.add_argument("--max-pages", type=int, default=3)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = Config()
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    url = f"{cfg['collector']['helius_rpc']}?api-key={cfg.secrets.helius_key}"
    now = int(time.time())
    since = int(now - a.days * 86_400)
    # trades pump.fun vus sur nos coins suivis (complement, pas obligatoire)
    from pumpml import kol as K
    h = K.history(cfg)
    seen = h.groupby("wallet").size().to_dict() if len(h) else {}
    sem, credits = asyncio.Semaphore(8), [0]
    async with aiohttp.ClientSession() as s:
        res = await asyncio.gather(*(activity(s, url, sem, r["address"], since, a.max_pages, credits) for r in rows))
    keep, rep = [], []
    for r, (n, last, more) in zip(rows, res):
        idle = (now - last) / 86_400 if last else None
        ok = last is not None and idle <= a.max_idle_days and (n >= a.min_tx or seen.get(r["address"], 0) >= 5)
        rep.append({"name": r.get("name"), "address": r["address"], "tx_periode": f"{n}{'+' if more else ''}",
                    "derniere_tx_jours": None if idle is None else round(idle, 1), "trades_vus_chez_nous": seen.get(r["address"], 0),
                    "garde": int(ok)})
        if ok:
            keep.append(r)
    OUT.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(REPORT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rep[0]))
        w.writeheader()
        w.writerows(sorted(rep, key=lambda x: (-x["garde"], x["derniere_tx_jours"] if x["derniere_tx_jours"] is not None else 1e9)))
    people = {r.get("name") for r in keep}
    log.info("KOL actifs : %d wallets / %d personnes garde(s) sur %d wallets ; credits Helius ~%d",
             len(keep), len(people), len(rows), credits[0])


if __name__ == "__main__":
    asyncio.run(main())
