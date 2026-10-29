"""Historique des coins pump.fun qui ont perce : memoire longue du modele (pumpml/fame.py).

Relit sur la chaine TOUTES les migrations pump.fun -> PumpSwap des ``--days`` derniers jours, garde les
migrations naturelles (pas les lancements groupes du bloc 0) et recupere createur, nom, ticker,
description et image. Reprend la ou il s'etait arrete. Resultat : data/fame.db.

Cout Helius (estime) : ~2 credits par migration + 10 par lot de 1000 coins (DAS).
Usage : .venv\\Scripts\\python scripts\\import_fame.py [--days 90]
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import sqlite3
import sys
import time
from pathlib import Path

import aiohttp
import base58

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pumpml import amm, metadata as md, pump  # noqa: E402
from pumpml.backfill import bonding_curve_pda  # noqa: E402
from pumpml.config import Config  # noqa: E402

log = logging.getLogger("fame_import")
SCHEMA = """
CREATE TABLE IF NOT EXISTS sigs(sig TEXT PRIMARY KEY, slot INTEGER, block_ms INTEGER, done INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS fame(mint TEXT PRIMARY KEY, pool TEXT, migrated_ms INTEGER, creator TEXT,
  created_ms INTEGER, curve_tx INTEGER, curve_s REAL, block0_tx INTEGER, organic INTEGER, name TEXT, symbol TEXT,
  description TEXT, image_url TEXT, image_path TEXT, step INTEGER DEFAULT 0);
"""
# colonnes ajoutees apres coup
MIGRATIONS = ["ALTER TABLE fame ADD COLUMN supply REAL", "ALTER TABLE fame ADD COLUMN peak_mcap_usd REAL",
              "ALTER TABLE fame ADD COLUMN win_ms INTEGER", "ALTER TABLE fame ADD COLUMN peak_checked INTEGER",
              "ALTER TABLE fame ADD COLUMN json_uri TEXT", "ALTER TABLE fame ADD COLUMN source TEXT",
              "ALTER TABLE fame ADD COLUMN block0_share REAL", "ALTER TABLE fame ADD COLUMN peak15_usd REAL",
              "ALTER TABLE fame ADD COLUMN peak15_ms INTEGER", "ALTER TABLE fame ADD COLUMN hold_ratio REAL",
              "ALTER TABLE fame ADD COLUMN hold_checked INTEGER"]
GECKO = "https://api.geckoterminal.com/api/v2/networks/solana/pools/{}/ohlcv/day?limit=200&currency=usd&token=base"
GECKO15 = ("https://api.geckoterminal.com/api/v2/networks/solana/pools/{}/ohlcv/minute?aggregate=15&limit=1000"
           "&currency=usd&token=base&before_timestamp={}")


async def gecko(s: aiohttp.ClientSession, url: str) -> list | None:
    """Bougies GeckoTerminal triees. Limite de debit (429) : pause croissante puis on reessaie la MEME requete."""
    for attempt in range(8):
        async with s.get(url, headers={"Accept": "application/json"}, timeout=aiohttp.ClientTimeout(total=30)) as r:
            if r.status == 429:
                await asyncio.sleep(20 * (attempt + 1))
                continue
            j = await r.json() if r.status == 200 else {}
            return sorted(((j.get("data") or {}).get("attributes") or {}).get("ohlcv_list") or [])
    return None


def hold_of(bars: list, sup: float, mig_s: int, hold_s: int, now_s: int):
    """Plus haut sur bougies de 15 min (pas une meche de bougie journaliere) et tenue : market cap
    ``hold_s`` apres la fin de la bougie du plus haut / plus haut. Une bougie absente = pas de trade, on
    reprend la derniere cloture. None si la tenue n'est pas encore mesurable."""
    bars = [b for b in bars if b[0] >= mig_s - 900]
    if not bars:
        return None
    k = max(range(len(bars)), key=lambda i: bars[i][2])
    peak, t_pk = bars[k][2] * sup, bars[k][0] + 900
    at = t_pk + hold_s
    if at > now_s:
        return None
    close = bars[k][4]
    for b in bars[k + 1:]:
        if b[0] + 900 > at:
            break
        close = b[4]
    return peak, t_pk * 1000, close * sup / peak if peak else 0.0


class Rpc:
    def __init__(self, url: str, s: aiohttp.ClientSession, conc: int):
        self.url, self.s, self.sem, self.credits = url, s, asyncio.Semaphore(conc), 0

    async def __call__(self, method: str, params, cost: int = 1):
        async with self.sem:
            for attempt in range(6):
                self.credits += cost
                try:
                    async with self.s.post(self.url, json={"jsonrpc": "2.0", "id": 1, "method": method,
                                                           "params": params}, timeout=aiohttp.ClientTimeout(total=40)) as r:
                        if r.status == 429 or r.status >= 500:
                            raise aiohttp.ClientError(f"HTTP {r.status}")
                        j = await r.json()
                    if "error" in j:
                        raise aiohttp.ClientError(str(j["error"]))
                    return j["result"]
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.sleep(1 + attempt * 2)
        raise RuntimeError(method)


async def list_migrations(rpc: Rpc, db: sqlite3.Connection, since_ms: int):
    """Signatures du compte present dans chaque migration. D'abord les NOUVELLES (de maintenant jusqu'aux
    signatures deja connues), puis l'historique plus ancien que le plus vieux connu, jusqu'a ``since_ms``."""
    newest = db.execute("SELECT MAX(block_ms) FROM sigs").fetchone()[0] or 0
    oldest = db.execute("SELECT sig, block_ms FROM sigs ORDER BY block_ms LIMIT 1").fetchone()

    async def walk(before, stop_ms):
        while True:
            p = {"limit": 1000}
            if before:
                p["before"] = before
            page = await rpc("getSignaturesForAddress", [amm.MIGRATION_ACCOUNT, p])
            if not page:
                return
            rows = [(x["signature"], x["slot"], (x.get("blockTime") or 0) * 1000) for x in page if x.get("err") is None]
            db.executemany("INSERT OR IGNORE INTO sigs(sig, slot, block_ms) VALUES(?,?,?)", rows)
            db.commit()
            t = (page[-1].get("blockTime") or 0) * 1000
            log.info("migrations listees jusqu'au %s", time.strftime("%Y-%m-%d %H:%M", time.localtime(t / 1000)))
            if t <= stop_ms or len(page) < 1000:
                return
            before = page[-1]["signature"]

    await walk(None, max(newest, since_ms))            # nouvelles
    if oldest and oldest[1] > since_ms:
        await walk(oldest[0], since_ms)                 # suite de l'historique


async def resolve_tx(rpc: Rpc, db: sqlite3.Connection, sig: str, block_ms: int):
    tx = await rpc("getTransaction", [sig, {"encoding": "json", "maxSupportedTransactionVersion": 1,
                                            "commitment": "confirmed"}])
    got = amm.parse_migration(tx) if tx and (tx.get("meta") or {}).get("err") is None else None
    if got:
        mint, pool, _ = got
        db.execute("INSERT OR IGNORE INTO fame(mint, pool, migrated_ms) VALUES(?,?,?)", (mint, pool, block_ms))
    db.execute("UPDATE sigs SET done=1 WHERE sig=?", (sig,))


async def curve_info(rpc: Rpc, db: sqlite3.Connection, mints: list[str]):
    """Createur (compte de la courbe, verifie) + historique de la courbe : lancement groupe ou naturel."""
    curves = [bonding_curve_pda(m) for m in mints]
    accs = (await rpc("getMultipleAccounts", [curves, {"encoding": "base64"}]))["value"]
    for m, c, a in zip(mints, curves, accs):
        creator = None
        if a:
            b = base64.b64decode(a["data"][0])
            if len(b) >= 81:
                creator = base58.b58encode(b[49:81]).decode()
        sigs = await rpc("getSignaturesForAddress", [c, {"limit": 1000}])
        ok = [x for x in sigs if x.get("err") is None]
        if len(sigs) >= 1000:
            # plus de 1000 transactions sur la courbe : forcement un vrai marche (debut non visible)
            row = (creator, None, len(ok), None, None, 1)
        elif ok:
            s0 = min(x["slot"] for x in ok)
            t0 = min((x.get("blockTime") or 0) for x in ok) * 1000
            mig = db.execute("SELECT migrated_ms FROM fame WHERE mint=?", (m,)).fetchone()[0]
            curve_s = (mig - t0) / 1000
            b0 = sum(1 for x in ok if x["slot"] <= s0 + 2)
            # naturel : courbe remplie en plus d'1 min, et l'essentiel des transactions APRES le bloc 0
            organic = int(curve_s >= 60 and b0 <= 0.25 * len(ok))
            row = (creator, t0, len(ok), curve_s, b0, organic)
        else:
            row = (creator, None, 0, None, None, 0)
        db.execute("UPDATE fame SET creator=?, created_ms=?, curve_tx=?, curve_s=?, block0_tx=?, organic=?, step=1"
                   " WHERE mint=?", (*row, m))
    db.commit()


async def deepen(rpc: Rpc, db: sqlite3.Connection, thr: float, min_mig_ms: int):
    """Gros gagnants dont la courbe a plus de 1000 transactions : on relit TOUT l'historique de la courbe
    (1 credit / 1000 transactions) et on decode les achats des premiers slots, pour verifier que la montee
    est legitime (part de la courbe achetee au bloc 0, duree). Seulement les gagnants : ils sont rares."""
    rows = db.execute("SELECT mint, migrated_ms FROM fame WHERE organic=1 AND curve_s IS NULL AND peak_mcap_usd >= ?"
                      " AND migrated_ms >= ? AND source IS NULL", (thr, min_mig_ms)).fetchall()
    for mint, mig in rows:
        curve = bonding_curve_pda(mint)
        sigs, before = [], None
        for _ in range(60):
            p = {"limit": 1000}
            if before:
                p["before"] = before
            page = await rpc("getSignaturesForAddress", [curve, p])
            sigs += page
            if len(page) < 1000:
                break
            before = page[-1]["signature"]
        ok = [x for x in sigs if x.get("err") is None]
        if not ok or len(sigs) % 1000 == 0 and len(sigs) >= 60_000:
            continue
        s0 = min(x["slot"] for x in ok)
        t0 = min((x.get("blockTime") or 0) for x in ok if x["slot"] == s0) * 1000
        first = sorted((x for x in ok if x["slot"] <= s0 + 2), key=lambda x: x["slot"])[:40]
        bought = 0
        for x in first:
            tx = await rpc("getTransaction", [x["signature"], {"encoding": "json", "maxSupportedTransactionVersion": 1}])
            for ev in pump.iter_events(((tx or {}).get("meta") or {}).get("logMessages") or []):
                if isinstance(ev, pump.Trade) and ev.mint == mint and ev.is_buy:
                    bought += ev.token_amount
        share = bought / amm.CURVE_SUPPLY
        curve_s = (mig - t0) / 1000
        organic = int(curve_s >= 60 and share < 0.25)
        db.execute("UPDATE fame SET created_ms=?, curve_tx=?, curve_s=?, block0_tx=?, block0_share=?, organic=?"
                   " WHERE mint=?", (t0, len(ok), curve_s, len(first), share, organic, mint))
        db.commit()
        log.info("courbe complete %s : %d tx, %.1f min, bloc 0 %.0f %% -> %s", mint[:8], len(ok), curve_s / 60,
                 100 * share, "naturel" if organic else "groupe")


async def assets(rpc: Rpc, db: sqlite3.Connection, mints: list[str]):
    res = await rpc("getAssetBatch", {"ids": mints}, cost=10)
    for m, a in zip(mints, res):
        c = (a or {}).get("content") or {}
        meta = c.get("metadata") or {}
        files = c.get("files") or [{}]
        img = (files[0] or {}).get("cdn_uri") or (c.get("links") or {}).get("image")
        ti = (a or {}).get("token_info") or {}
        sup = (ti.get("supply") or 0) / 10 ** (ti.get("decimals") or 6) or 1e9
        db.execute("UPDATE fame SET name=?, symbol=?, description=?, image_url=?, supply=?, json_uri=?, step=2"
                   " WHERE mint=?", (meta.get("name"), meta.get("symbol"), (meta.get("description") or "")[:1000], img,
                                     sup, c.get("json_uri"), m))
    db.commit()


async def peaks(s: aiohttp.ClientSession, db: sqlite3.Connection, thr: float, min_mig_ms: int = 0):
    """Plus haut de market cap en dollars (bougies journalieres GeckoTerminal, ~28 requetes/min gratuites).
    ``win_ms`` = fin du premier jour ou le coin a depasse le seuil : instant ou son succes est CONNU."""
    # jamais mesures, ou migres il y a moins de 24 h (ils peuvent encore monter)
    todo = db.execute("SELECT mint, pool, COALESCE(supply, 1e9) FROM fame WHERE organic=1 AND pool IS NOT NULL"
                      " AND (peak_checked IS NULL OR migrated_ms > ?) AND migrated_ms >= ? ORDER BY migrated_ms DESC",
                      (int(time.time() * 1000) - 86_400_000, min_mig_ms)).fetchall()
    log.info("plus hauts a mesurer : %d coins (~%d min)", len(todo), len(todo) // 20)
    for k, (mint, pool, sup) in enumerate(todo):
        t0 = time.monotonic()
        try:
            bars = await gecko(s, GECKO.format(pool))
            if bars is None:
                continue
            peak, win = 0.0, None
            for ts, _o, hi, _l, _c, _v in bars:
                peak = max(peak, hi * sup)
                if win is None and hi * sup >= thr:
                    win = (ts + 86_400) * 1000
            db.execute("UPDATE fame SET peak_mcap_usd=?, win_ms=?, peak_checked=1 WHERE mint=?", (peak, win, mint))
            db.commit()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            pass
        if k % 200 == 0:
            n = db.execute("SELECT COUNT(*) FROM fame WHERE win_ms IS NOT NULL").fetchone()[0]
            log.info("plus hauts : %d / %d, gagnants (>= %.0f $) : %d", k, len(todo), thr, n)
        await asyncio.sleep(max(0.0, 3.0 - (time.monotonic() - t0)))  # ~20 requetes/min


async def holds(s: aiohttp.ClientSession, db: sqlite3.Connection, thr: float, hold_min: float, min_mig_ms: int = 0):
    """Tenue des coins dont le plus haut journalier depasse le seuil : un gagnant doit encore valoir une bonne
    part de son plus haut ``hold_min`` apres (sinon c'est un vidage). Une requete par coin (15 min x 1000)."""
    now = int(time.time())
    todo = db.execute("SELECT mint, pool, migrated_ms, COALESCE(supply, 1e9) FROM fame WHERE organic=1 AND source IS NULL"
                      " AND pool IS NOT NULL AND peak_mcap_usd >= ? AND migrated_ms >= ?"
                      " AND (hold_checked IS NULL OR migrated_ms > ?) ORDER BY migrated_ms DESC",
                      (thr, min_mig_ms, (now - 86_400) * 1000)).fetchall()
    log.info("tenues a mesurer : %d coins", len(todo))
    for mint, pool, mig, sup in todo:
        t0 = time.monotonic()
        try:
            bars = await gecko(s, GECKO15.format(pool, min(now, mig // 1000 + 1000 * 900)))
            if bars is not None:
                h = hold_of(bars, sup, mig // 1000, int(hold_min * 60), now)
                if h or not bars:
                    db.execute("UPDATE fame SET peak15_usd=?, peak15_ms=?, hold_ratio=?, hold_checked=1 WHERE mint=?",
                               (*(h or (None, None, None)), mint))
                    db.commit()
                if h:
                    log.info("tenue %s : plus haut 15 min %.0f k$, %.0f %% %d min apres", mint[:8], h[0] / 1000,
                             100 * h[2], hold_min)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            pass
        await asyncio.sleep(max(0.0, 3.0 - (time.monotonic() - t0)))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--conc", type=int, default=12)
    ap.add_argument("--peaks-hours", type=float, default=0,
                    help="ne mesurer les plus hauts que des migrations de ces dernieres heures (0 = toutes)")
    ap.add_argument("--holds-only", action="store_true", help="mesurer seulement les tenues des gros coins")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = Config()
    db = sqlite3.connect(cfg.data / "fame.db", timeout=120)  # import_melt peut ecrire en meme temps
    db.executescript(SCHEMA)
    for m in MIGRATIONS:
        try:
            db.execute(m)
        except sqlite3.OperationalError:
            pass
    fm = cfg["fame"]
    recent = int(time.time() * 1000 - a.peaks_hours * 3_600_000) if a.peaks_hours else 0
    if a.holds_only:
        async with aiohttp.ClientSession() as s:
            await holds(s, db, fm["min_peak_mcap_usd"], fm["hold_minutes"], recent)
        return
    url = f"{cfg['collector']['helius_rpc']}?api-key={cfg.secrets.helius_key}"
    since = int(time.time() * 1000) - a.days * 86_400_000
    async with aiohttp.ClientSession() as s:
        rpc = Rpc(url, s, a.conc)
        await list_migrations(rpc, db, since)
        todo = db.execute("SELECT sig, block_ms FROM sigs WHERE done=0 AND block_ms >= ?", (since,)).fetchall()
        log.info("%d migrations a lire", len(todo))
        for i in range(0, len(todo), 200):
            await asyncio.gather(*(resolve_tx(rpc, db, sg, t) for sg, t in todo[i:i + 200]))
            db.commit()
            log.info("transactions lues : %d / %d (credits ~%d)", min(i + 200, len(todo)), len(todo), rpc.credits)
        todo = [r[0] for r in db.execute("SELECT mint FROM fame WHERE step=0")]
        for i in range(0, len(todo), 100):
            await curve_info(rpc, db, todo[i:i + 100])
            log.info("courbes : %d / %d (credits ~%d)", min(i + 100, len(todo)), len(todo), rpc.credits)
        todo = [r[0] for r in db.execute("SELECT mint FROM fame WHERE (step=1 OR supply IS NULL) AND organic=1")]
        for i in range(0, len(todo), 1000):
            await assets(rpc, db, todo[i:i + 1000])
        # images (CDN Helius) des gagnants, pour la ressemblance visuelle
        m = cfg["metadata"]
        todo = db.execute("SELECT mint, image_url FROM fame WHERE organic=1 AND image_url IS NOT NULL"
                          " AND image_path IS NULL AND migrated_ms >= ?", (recent,)).fetchall()
        sem = asyncio.Semaphore(16)

        async def img(mint, u):
            async with sem:
                rel = f"images/fame/{mint}.jpg"
                if await md.fetch_image(s, u, m["ipfs_gateways"], m["timeout_seconds"], m["max_image_bytes"],
                                        cfg.data / rel):
                    db.execute("UPDATE fame SET image_path=? WHERE mint=?", (rel, mint))

        for i in range(0, len(todo), 400):
            await asyncio.gather(*(img(mm, u) for mm, u in todo[i:i + 400]))
            db.commit()
            log.info("images : %d / %d", min(i + 400, len(todo)), len(todo))
        await peaks(s, db, fm["min_peak_mcap_usd"], recent)
        await deepen(rpc, db, fm["min_peak_mcap_usd"], recent)
        await holds(s, db, fm["min_peak_mcap_usd"], fm["hold_minutes"], recent)
    n = db.execute("SELECT COUNT(*), SUM(organic) FROM fame").fetchone()
    log.info("termine : %d migrations, %d naturelles (gagnants), credits ~%d", n[0], n[1] or 0, rpc.credits)


if __name__ == "__main__":
    asyncio.run(main())
