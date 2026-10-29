"""Historique des achats pump.fun des gros KOL (top 50 Kolscan) sur ``--days`` jours, lu sur la chaine.

1. ``--count`` : compte seulement les transactions reussies de chaque wallet (getSignaturesForAddress, ~1 credit
   par 1000) pour estimer le cout avant de lancer l'import.
2. Import : lit chaque transaction reussie (getTransaction, 1 credit), garde les ACHATS pump.fun (courbe), et
   enregistre (wallet, mint, heure, SOL) dans ``data/kol_history.db``. Reprend la ou il s'etait arrete.
   ``--max-credits`` arrete l'import au-dela d'un budget.
Ensuite ``--meta`` complete chaque coin achete : nom, ticker, description, image, liens, createur (Helius DAS +
JSON IPFS, ~10 credits par lot de 1000 coins), pour la memoire « ce que les gros KOL achetent ».

Usage : .venv\\Scripts\\python scripts\\kol_backfill.py --count
        .venv\\Scripts\\python scripts\\kol_backfill.py --days 90 --max-credits 150000
        .venv\\Scripts\\python scripts\\kol_backfill.py --meta
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pumpml import pump  # noqa: E402
from pumpml.config import Config  # noqa: E402

log = logging.getLogger("kol_backfill")
TOP = ROOT / "data" / "external" / "kol" / "kolscan_mensuel.json"
SCHEMA = """
CREATE TABLE IF NOT EXISTS sigs(sig TEXT PRIMARY KEY, wallet TEXT, block_ms INTEGER, done INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS sigs_todo ON sigs(done, wallet);
CREATE TABLE IF NOT EXISTS walk(wallet TEXT PRIMARY KEY, oldest_sig TEXT, oldest_ms INTEGER, complete INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS buys(sig TEXT, wallet TEXT, mint TEXT, block_ms INTEGER, sol REAL, is_buy INTEGER,
  PRIMARY KEY(sig, mint, is_buy));
CREATE INDEX IF NOT EXISTS buys_mint ON buys(mint);
CREATE TABLE IF NOT EXISTS spend(day TEXT PRIMARY KEY, credits INTEGER);
CREATE TABLE IF NOT EXISTS coins(mint TEXT PRIMARY KEY, name TEXT, symbol TEXT, description TEXT, image_url TEXT,
  image_path TEXT, json_uri TEXT, twitter TEXT, website TEXT, telegram TEXT, creator TEXT, created_ms INTEGER, step INTEGER DEFAULT 0);
"""


class Rpc:
    def __init__(self, url: str, s: aiohttp.ClientSession, conc: int):
        self.url, self.s, self.sem, self.credits = url, s, asyncio.Semaphore(conc), 0

    async def __call__(self, method: str, params, cost: int = 1):
        async with self.sem:
            for attempt in range(6):
                self.credits += cost
                try:
                    async with self.s.post(self.url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                                           timeout=aiohttp.ClientTimeout(total=40)) as r:
                        if r.status == 429 or r.status >= 500:
                            raise aiohttp.ClientError(f"HTTP {r.status}")
                        j = await r.json()
                    if "error" in j:
                        raise aiohttp.ClientError(str(j["error"]))
                    return j["result"]
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.sleep(1 + attempt * 2)
        return None


async def list_sigs(rpc: Rpc, db: sqlite3.Connection, wallet: str, since_ms: int, cap: int) -> int:
    """Signatures reussies du wallet jusqu'a since_ms, au plus ``cap`` (les plus recentes). Certains gros KOL sont
    des robots (150 000 transactions en 3 mois) : tout lire couterait des millions de credits."""
    row = db.execute("SELECT oldest_sig, complete FROM walk WHERE wallet=?", (wallet,)).fetchone()
    if row and row[1]:
        return db.execute("SELECT COUNT(*) FROM sigs WHERE wallet=?", (wallet,)).fetchone()[0]
    before = row[0] if row else None
    while True:
        p = {"limit": 1000}
        if before:
            p["before"] = before
        page = await rpc("getSignaturesForAddress", [wallet, p])
        if not page:
            break
        ok = [(x["signature"], wallet, (x.get("blockTime") or 0) * 1000) for x in page
              if x.get("err") is None and (x.get("blockTime") or 0) * 1000 >= since_ms]
        db.executemany("INSERT OR IGNORE INTO sigs(sig, wallet, block_ms) VALUES(?,?,?)", ok)
        before = page[-1]["signature"]
        t = (page[-1].get("blockTime") or 0) * 1000
        db.execute("INSERT OR REPLACE INTO walk VALUES(?,?,?,?)", (wallet, before, t, int(t < since_ms or len(page) < 1000)))
        db.commit()
        if t < since_ms or len(page) < 1000:
            break
        if db.execute("SELECT COUNT(*) FROM sigs WHERE wallet=?", (wallet,)).fetchone()[0] >= cap:
            db.execute("UPDATE walk SET complete=1 WHERE wallet=?", (wallet,))
            db.commit()
            break
    return db.execute("SELECT COUNT(*) FROM sigs WHERE wallet=?", (wallet,)).fetchone()[0]


async def read_tx(rpc: Rpc, db: sqlite3.Connection, sig: str, wallet: str, block_ms: int):
    tx = await rpc("getTransaction", [sig, {"encoding": "json", "maxSupportedTransactionVersion": 1,
                                            "commitment": "confirmed"}])
    if tx is None:
        return
    rows = []
    for ev in pump.iter_events(((tx.get("meta") or {}).get("logMessages")) or []):
        if isinstance(ev, pump.Trade) and ev.user == wallet:
            rows.append((sig, wallet, ev.mint, block_ms, ev.sol_amount / 1e9, int(ev.is_buy)))
    db.executemany("INSERT OR IGNORE INTO buys VALUES(?,?,?,?,?,?)", rows)
    db.execute("UPDATE sigs SET done=1 WHERE sig=?", (sig,))


async def meta(rpc: Rpc, db: sqlite3.Connection, s: aiohttp.ClientSession, cfg):
    """Nom, ticker, description, image, liens et createur de chaque coin achete par un gros KOL."""
    from pumpml import metadata as md
    from pumpml.backfill import bonding_curve_pda
    import base58
    import base64
    db.execute("INSERT OR IGNORE INTO coins(mint) SELECT DISTINCT mint FROM buys WHERE is_buy=1")
    db.commit()
    todo = [r[0] for r in db.execute("SELECT mint FROM coins WHERE step=0")]
    log.info("coins a decrire : %d", len(todo))
    for i in range(0, len(todo), 1000):
        part = todo[i:i + 1000]
        res = await rpc("getAssetBatch", {"ids": part}, cost=10) or []
        for m, a in zip(part, res):
            c = (a or {}).get("content") or {}
            mt = c.get("metadata") or {}
            files = c.get("files") or [{}]
            img = (files[0] or {}).get("cdn_uri") or (c.get("links") or {}).get("image")
            db.execute("UPDATE coins SET name=?, symbol=?, description=?, image_url=?, json_uri=?, step=1 WHERE mint=?",
                       (mt.get("name"), mt.get("symbol"), (mt.get("description") or "")[:1000], img, c.get("json_uri"), m))
        db.commit()
        log.info("DAS : %d / %d", min(i + 1000, len(todo)), len(todo))
    # createur (compte de la courbe) et liens (JSON IPFS)
    m_cfg = cfg["metadata"]
    todo = db.execute("SELECT mint, json_uri FROM coins WHERE step=1").fetchall()
    sem = asyncio.Semaphore(16)

    async def one(mint, uri):
        async with sem:
            j = await md.fetch_json(s, uri, m_cfg["ipfs_gateways"], 15) if uri else None
            p = md.parse_metadata(j) if j else {}
            rel = f"images/kolbuys/{mint}.jpg"
            row = db.execute("SELECT image_url FROM coins WHERE mint=?", (mint,)).fetchone()
            ok = row and row[0] and await md.fetch_image(s, row[0], m_cfg["ipfs_gateways"], m_cfg["timeout_seconds"],
                                                         m_cfg["max_image_bytes"], cfg.data / rel)
            db.execute("UPDATE coins SET twitter=?, website=?, telegram=?, image_path=?, step=2 WHERE mint=?",
                       (p.get("twitter"), p.get("website"), p.get("telegram"), rel if ok else None, mint))

    for i in range(0, len(todo), 400):
        await asyncio.gather(*(one(m, u) for m, u in todo[i:i + 400]))
        db.commit()
        log.info("liens / images : %d / %d", min(i + 400, len(todo)), len(todo))
    todo = [r[0] for r in db.execute("SELECT mint FROM coins WHERE creator IS NULL AND step=2")]
    for i in range(0, len(todo), 100):
        part = todo[i:i + 100]
        accs = (await rpc("getMultipleAccounts", [[bonding_curve_pda(m) for m in part], {"encoding": "base64"}]) or {}).get("value") or []
        for m, acc in zip(part, accs):
            if acc:
                b = base64.b64decode(acc["data"][0])
                if len(b) >= 81:
                    db.execute("UPDATE coins SET creator=? WHERE mint=?", (base58.b58encode(b[49:81]).decode(), m))
        db.commit()
    log.info("description des coins terminee (credits ~%d)", rpc.credits)


def spent(db: sqlite3.Connection, day: str) -> int:
    r = db.execute("SELECT credits FROM spend WHERE day=?", (day,)).fetchone()
    return r[0] if r else 0


async def loop(a, rpc: Rpc, db: sqlite3.Connection, s: aiohttp.ClientSession, cfg):
    """Budget propre de ``--daily`` credits les jours ``--dates`` ; s'arrete apres le dernier jour ou quand tout
    est lu, puis decrit les coins (--meta)."""
    dates = sorted(x for x in a.dates.split(",") if x)
    while True:
        day = time.strftime("%Y-%m-%d")
        todo = db.execute("SELECT sig, wallet, block_ms FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY wallet ORDER BY"
                          " block_ms DESC) k FROM sigs) WHERE done=0 AND k<=? ORDER BY k, wallet", (a.per_wallet,)).fetchall()
        if not todo or (dates and day > dates[-1]):
            log.info("import termine (%d transactions restantes)", len(todo))
            break
        room = a.daily - spent(db, day) if day in dates else 0
        if room <= 1000:
            log.info("budget du %s epuise ou jour non autorise : attente du lendemain", day)
            await asyncio.sleep(max(60, 86_400 - (time.time() - time.mktime(time.strptime(day, "%Y-%m-%d")))) + 120)
            continue
        c0 = rpc.credits
        for i in range(0, len(todo), 300):
            if spent(db, day) >= a.daily - 1000 or time.strftime("%Y-%m-%d") != day:
                break
            await asyncio.gather(*(read_tx(rpc, db, *x) for x in todo[i:i + 300]))
            db.execute("INSERT OR REPLACE INTO spend VALUES(?,?)", (day, spent(db, day) + rpc.credits - c0))
            c0 = rpc.credits
            db.commit()
            n = db.execute("SELECT COUNT(*), COUNT(DISTINCT mint) FROM buys WHERE is_buy=1").fetchone()
            log.info("lues %d / %d · achats pump.fun %d sur %d coins · credits du jour %d / %d", min(i + 300, len(todo)),
                     len(todo), n[0], n[1], spent(db, day), a.daily)
    await meta(rpc, db, s, cfg)


def used_today(cfg) -> int:
    """Credits Helius deja consommes aujourd'hui par le collecteur (son heartbeat)."""
    try:
        h = sqlite3.connect(f"file:{(cfg.data / 'hot.db').as_posix()}?mode=ro", uri=True, timeout=30)
        r = h.execute("SELECT info FROM heartbeat WHERE name='collector'").fetchone()
        h.close()
        return int(json.loads(r[0]).get("credits_jour", 0)) if r else 0
    except Exception:
        return 0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=90)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--meta", action="store_true")
    ap.add_argument("--max-credits", type=int, default=150_000)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--conc", type=int, default=12)
    ap.add_argument("--per-wallet", type=int, default=2000, help="transactions max lues par wallet (les plus recentes)")
    ap.add_argument("--keep-free", type=int, default=30_000,
                    help="credits du jour laisses au collecteur (budget [collector] moins sa conso du jour)")
    ap.add_argument("--daily", type=int, default=0,
                    help="budget propre de l'import par jour (en plus de celui du collecteur), 0 = reste du collecteur")
    ap.add_argument("--dates", default="", help="jours autorises pour --daily (AAAA-MM-JJ,AAAA-MM-JJ)")
    ap.add_argument("--loop", action="store_true", help="continue jour apres jour jusqu'a la fin, puis --meta")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = Config()
    db = sqlite3.connect(cfg.data / "kol_history.db", timeout=120)
    db.executescript(SCHEMA)
    top = json.loads(TOP.read_text(encoding="utf-8"))["top"][:a.top]
    since = int((time.time() - a.days * 86_400) * 1000)
    url = f"{cfg['collector']['helius_rpc']}?api-key={cfg.secrets.helius_key}"
    async with aiohttp.ClientSession() as s:
        rpc = Rpc(url, s, a.conc)
        if a.meta:
            await meta(rpc, db, s, cfg)
            return
        tot = 0
        for r in top:
            n = await list_sigs(rpc, db, r["address"], since, a.per_wallet)
            tot += n
            log.info("%-20s %7d transactions reussies sur %d jours", r["name"], n, a.days)
        log.info("TOTAL %d transactions (credits de listage ~%d)", tot, rpc.credits)
        if a.count:
            return
        if a.loop:
            await loop(a, rpc, db, s, cfg)
            return
        # import : les plus recentes d'abord, wallet par wallet en alternance
        # par wallet, ses ``per_wallet`` transactions les plus recentes ; puis on alterne entre wallets
        todo = db.execute("SELECT sig, wallet, block_ms FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY wallet ORDER BY"
                          " block_ms DESC) k FROM sigs) WHERE done=0 AND k<=? ORDER BY k, wallet", (a.per_wallet,)).fetchall()
        log.info("%d transactions a lire", len(todo))
        for i in range(0, len(todo), 300):
            if rpc.credits >= a.max_credits or rpc.credits + used_today(cfg) > cfg["collector"]["helius_daily_credit_budget"] - a.keep_free:
                log.info("budget atteint (import %d credits, collecteur %d) : on reprendra plus tard", rpc.credits,
                         used_today(cfg))
                break
            await asyncio.gather(*(read_tx(rpc, db, *x) for x in todo[i:i + 300]))
            db.commit()
            n = db.execute("SELECT COUNT(*), COUNT(DISTINCT mint) FROM buys WHERE is_buy=1").fetchone()
            log.info("lues %d / %d · achats pump.fun %d sur %d coins · credits ~%d", min(i + 300, len(todo)), len(todo),
                     n[0], n[1], rpc.credits)


if __name__ == "__main__":
    asyncio.run(main())
