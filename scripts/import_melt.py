"""Ajoute a la memoire des narratives (data/fame.db) les coins du dataset academique MemeTrans / MELT
(github.com/git-disl/MELT, licence CC BY-NC 4.0 : usage non commercial, citer l'article arXiv 2602.13480).

41 470 coins pump.fun migres entre decembre 2024 et mars 2025, dont 29 556 avec nom, ticker, description
et image. Pour chacun : resultat 20 min apres la migration (``return_ratio``) et detection de manipulation
(comptes lies / bundles). Market cap estimee = market cap de migration (~410 SOL au prix du SOL de l'heure)
x (1 + return_ratio). Gagnant si >= 100 000 $ (config [fame]) et pas manipule ; sinon c'est un flop.

Usage : .venv\\Scripts\\python scripts\\import_melt.py   (fichiers dans data/external/melt/)
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import sys
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pumpml import metadata as md, pump  # noqa: E402
from pumpml.config import Config  # noqa: E402

log = logging.getLogger("melt")
SRC = ROOT / "data" / "external" / "melt"


def migration_mcap_sol() -> float:
    k = pump.INITIAL_VSOL * pump.INITIAL_VTOK
    vt = pump.INITIAL_VTOK - 793_100_000_000_000
    return pump.mcap_sol(k / vt, vt)


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = Config()
    thr = cfg["fame"]["min_peak_mcap_usd"]
    lab = pd.read_csv(SRC / "label.csv")
    lst = pd.read_json(SRC / "memecoin_list.jsonl", lines=True, convert_dates=False).drop_duplicates("token_address")
    lst["timestamp"] = lst.timestamp.astype(np.int64)  # secondes epoch
    meta = pd.read_json(SRC / "metadata.jsonl", lines=True).drop_duplicates("address")
    sol = pd.read_csv(SRC / "sol_hourly.txt", sep="\t", header=None, names=["ts", "usd"])
    df = lab.merge(lst[["token_address", "timestamp"]], left_on="mint_address", right_on="token_address") \
        .merge(meta, left_on="mint_address", right_on="address")
    df["ts"] = df.timestamp.astype(np.int64) // 3600 * 3600
    df = df.merge(sol, on="ts", how="left")
    df["usd"] = df.usd.ffill().bfill()
    df["mcap_usd"] = migration_mcap_sol() * df.usd * (1 + df.return_ratio.clip(lower=-1))
    df["organic"] = (df.manipulated != "yes").astype(int)
    win = (df.mcap_usd >= thr) & (df.organic == 1)
    log.info("%d coins avec narrative ; gagnants (>= %.0f $, sans manipulation) : %d ; flops : %d",
             len(df), thr, win.sum(), (~win).sum())

    db = sqlite3.connect(cfg.data / "fame.db", timeout=120)  # import_fame peut ecrire en meme temps
    try:
        db.execute("DELETE FROM fame WHERE source='melt'")  # reimport complet (idempotent)
    except sqlite3.OperationalError:
        pass
    for m in ("ALTER TABLE fame ADD COLUMN supply REAL", "ALTER TABLE fame ADD COLUMN peak_mcap_usd REAL",
              "ALTER TABLE fame ADD COLUMN win_ms INTEGER", "ALTER TABLE fame ADD COLUMN peak_checked INTEGER",
              "ALTER TABLE fame ADD COLUMN source TEXT"):
        try:
            db.execute(m)
        except sqlite3.OperationalError:
            pass
    mig_ms = df.timestamp.astype(np.int64) * 1000
    rows = [(r.mint_address, int(t), int(t) - 3_600_000, r.organic, r.name, r.symbol,
             (r.description or "")[:1000] if isinstance(r.description, str) else None,
             r.imageUrl if isinstance(r.imageUrl, str) else None, float(r.mcap_usd), int(t) + 20 * 60_000)
            for r, t in zip(df.itertuples(index=False), mig_ms)]
    db.executemany("INSERT OR IGNORE INTO fame(mint, migrated_ms, created_ms, organic, name, symbol, description,"
                   " image_url, peak_mcap_usd, win_ms, peak_checked, step, source) VALUES(?,?,?,?,?,?,?,?,?,?,1,2,'melt')",
                   rows)
    db.commit()
    # images des gagnants seulement (IPFS, lent) : la ressemblance visuelle compte surtout pour eux
    # flops : pas d'image (inutile, et le script import_fame les telechargerait sinon)
    db.execute("UPDATE fame SET image_url=NULL WHERE source='melt' AND NOT (organic=1 AND peak_mcap_usd >= ?)", (thr,))
    db.commit()
    todo = db.execute("SELECT mint, image_url FROM fame WHERE source='melt' AND organic=1 AND peak_mcap_usd >= ?"
                      " AND image_url IS NOT NULL AND image_path IS NULL", (thr,)).fetchall()
    m = cfg["metadata"]
    sem = asyncio.Semaphore(24)
    async with aiohttp.ClientSession() as s:
        async def img(mint, u):
            async with sem:
                rel = f"images/fame/{mint}.jpg"
                if await md.fetch_image(s, u, m["ipfs_gateways"], m["timeout_seconds"], m["max_image_bytes"],
                                        cfg.data / rel):
                    db.execute("UPDATE fame SET image_path=? WHERE mint=?", (rel, mint))

        for i in range(0, len(todo), 300):
            await asyncio.gather(*(img(a, b) for a, b in todo[i:i + 300]))
            db.commit()
            log.info("images : %d / %d", min(i + 300, len(todo)), len(todo))
    log.info("termine")


if __name__ == "__main__":
    asyncio.run(main())
