"""Rapport periodique des gros gagnants legitimes (lance toutes les 4 h par une tache planifiee Claude).

1. Met a jour la memoire (scripts/import_fame.py en mode incremental) : nouvelles migrations, tri montee
   naturelle / lancement groupe, plus haut en dollars.
2. Liste les coins migres pendant la periode, de facon LEGITIME (courbe remplie en plus d'1 min, pas achetee
   au bloc 0 par un groupe de wallets) et dont la market cap a depasse le seuil (100 000 $), pas encore annotes.
3. Pour chacun : narrative (nom, description, liens X / site), montee de la courbe, createur, ce que notre
   IA en a pense en direct. Ecrit rapports/gagnants/<date>.md et .json.

Ensuite, Claude recherche chaque narrative et l'annote avec scripts/annotate_winner.py.
Usage : .venv\\Scripts\\python scripts\\winners_report.py [--hours 6] [--skip-import]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pumpml import fame as FM, metadata as md  # noqa: E402
from pumpml.config import Config  # noqa: E402

OUT = ROOT / "rapports" / "gagnants"


def ts(ms) -> str:
    return time.strftime("%d/%m %H:%M", time.localtime(ms / 1000)) if ms else "?"


async def links(rows: list[dict], gateways: list[str]):
    """Liens X / site / telegram lus dans le JSON de metadonnees (IPFS)."""
    async with aiohttp.ClientSession() as s:
        async def one(r):
            if not r.get("json_uri"):
                return
            j = await md.fetch_json(s, r["json_uri"], gateways, 15)
            if j:
                p = md.parse_metadata(j)
                r.update({k: p.get(k) for k in ("twitter", "website", "telegram") if p.get(k)})
                if not r.get("description"):
                    r["description"] = p.get("description")
        await asyncio.gather(*(one(r) for r in rows))


async def market(rows: list[dict]):
    """Tenue : market cap et volume actuels (DexScreener, gratuit, 30 coins par appel)."""
    async with aiohttp.ClientSession() as s:
        for i in range(0, len(rows), 30):
            part = rows[i:i + 30]
            url = "https://api.dexscreener.com/tokens/v1/solana/" + ",".join(r["mint"] for r in part)
            try:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=20)) as r:
                    pairs = await r.json() if r.status == 200 else []
            except (aiohttp.ClientError, asyncio.TimeoutError):
                continue
            best: dict[str, dict] = {}
            for p in pairs or []:
                m = (p.get("baseToken") or {}).get("address")
                liq = ((p.get("liquidity") or {}).get("usd") or 0)
                if m and liq >= ((best.get(m) or {}).get("liquidity") or {}).get("usd", -1):
                    best[m] = p
            for row in part:
                p = best.get(row["mint"])
                if p:
                    row["mcap_now"] = p.get("marketCap") or p.get("fdv")
                    row["vol24"] = (p.get("volume") or {}).get("h24")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6)
    ap.add_argument("--skip-import", action="store_true")
    a = ap.parse_args()
    cfg = Config()
    thr = cfg["fame"]["min_peak_mcap_usd"]
    if not a.skip_import:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "import_fame.py"), "--days", "2",
                        "--peaks-hours", str(max(30, a.hours + 24))], cwd=ROOT, check=False)
    since = int(time.time() * 1000 - a.hours * 3_600_000)
    db = sqlite3.connect(f"file:{(cfg.data / 'fame.db').as_posix()}?mode=ro", uri=True, timeout=60)
    db.row_factory = sqlite3.Row
    period = db.execute("SELECT COUNT(*) n, SUM(organic=1) org, SUM(organic=0) grp, SUM(organic=1 AND peak_mcap_usd >= ?) win"
                        " FROM fame WHERE migrated_ms >= ? AND source IS NULL", (thr, since)).fetchone()
    big = pd.read_sql("SELECT * FROM fame WHERE source IS NULL AND organic=1 AND peak_mcap_usd >= ? AND migrated_ms >= ?"
                      " ORDER BY peak_mcap_usd DESC", db, params=(thr, since))
    # regle de tenue : plus haut 15 min, valeur 1 h apres, foule sur la courbe (vidages et tenues pas encore mesurees a part)
    win, _ = FM.label(big, cfg)
    dumps = big[win == 0]
    pending = big[win.isna()]
    rows = [{k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
            for r in big[win == 1].to_dict("records")]
    # createur : autres gros gagnants et nombre de migrations
    for r in rows:
        if r.get("creator"):
            c = db.execute("SELECT COUNT(*) n, SUM(organic=1 AND peak_mcap_usd >= ?) w FROM fame WHERE creator=? AND mint != ?",
                           (thr, r["creator"], r["mint"])).fetchone()
            r["creator_other_migrations"], r["creator_other_winners"] = c["n"], c["w"] or 0
    db.close()
    j = sqlite3.connect(f"file:{(cfg.data / 'journal.db').as_posix()}?mode=ro", uri=True)
    j.row_factory = sqlite3.Row
    done = {x["mint"] for x in j.execute("SELECT mint FROM feedback")}
    rows = [r for r in rows if r["mint"] not in done]
    for r in rows:  # ce que notre IA en a pense en direct (si le coin etait suivi)
        d = j.execute("SELECT action, final, threshold, age_s, scores FROM decisions WHERE mint=?", (r["mint"],)).fetchone()
        r["ia"] = dict(d) if d else None
    j.close()
    hot = sqlite3.connect(f"file:{(cfg.data / 'hot.db').as_posix()}?mode=ro", uri=True)
    for r in rows:  # tweet lu en direct
        s = hot.execute("SELECT json FROM social WHERE mint=?", (r["mint"],)).fetchone()
        if s:
            so = json.loads(s[0])
            r["tweet"] = {k: so.get(k) for k in ("url", "handle", "followers", "verified", "tweet_text", "likes", "views",
                                                  "rel", "parent_handle", "parent_text") if so.get(k) is not None}
    hot.close()
    asyncio.run(links(rows, cfg["metadata"]["ipfs_gateways"]))
    asyncio.run(market(rows))

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%Hh%M")
    (OUT / f"{stamp}.json").write_text(json.dumps({"since_ms": since, "period": dict(period), "coins": rows},
                                                  ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    L = [f"# Gros gagnants legitimes — {stamp.replace('_', ' ')}", "",
         f"Periode : {a.hours:.0f} dernieres heures. Migrations : {period['n'] or 0}, montees naturelles : {period['org'] or 0}, "
         f"lancements groupes : {period['grp'] or 0}, plus haut journalier >= {thr:,.0f} $ : {period['win'] or 0}, vrais gros gagnants (tenue) : {int((win == 1).sum())}, "
         f"a analyser (pas encore annotes) : {len(rows)}.",
         f"Ecartes par la regle de tenue (plus haut 15 min < seuil, < {100 * cfg['fame']['min_hold_ratio']:.0f} % "
         f"{cfg['fame']['hold_minutes']} min apres, ou < {cfg['fame']['min_curve_tx']} tx sur la courbe) : {len(dumps)}"
         + (" — " + ", ".join(f"{n} ({(h or 0) * 100:.0f} %)" for n, h in zip(dumps.name, dumps.hold_ratio)) if len(dumps) else "")
         + f". Tenue pas encore mesuree : {len(pending)}.", ""]
    for r in rows:
        L += [f"## {r.get('name')} (${r.get('symbol')}) — plus haut {r['peak15_usd'] / 1000:,.0f} k$ (15 min), "
              f"{100 * r['hold_ratio']:.0f} % du plus haut {cfg['fame']['hold_minutes']} min apres",
              f"- Mint : `{r['mint']}` — https://dexscreener.com/solana/{r['mint']}",
              f"- Migre le {ts(r['migrated_ms'])}, courbe remplie en {r['curve_s'] / 60:.1f} min" if r.get("curve_s") else
              f"- Migre le {ts(r['migrated_ms'])} (plus de 1000 transactions sur la courbe)",
              f"- Transactions sur la courbe : {r.get('curve_tx')}, dont au bloc 0 : {r.get('block0_tx')}"
              + (f" (part de la courbe achetee au bloc 0 : {100 * r['block0_share']:.0f} %)" if r.get("block0_share") is not None else ""),
              f"- Tenue : market cap actuelle {r['mcap_now'] / 1000:,.0f} k$, volume 24 h {(r.get('vol24') or 0) / 1000:,.0f} k$"
              if r.get("mcap_now") else "- Tenue : inconnue (pas de paire DexScreener)",
              f"- Description : {(r.get('description') or '').strip()[:600]}",
              f"- X : {r.get('twitter') or '-'} | Site : {r.get('website') or '-'} | Telegram : {r.get('telegram') or '-'}",
              f"- Image : {r.get('image_path') or r.get('image_url') or '-'}",
              f"- Createur : `{r.get('creator')}` — autres migrations : {r.get('creator_other_migrations', '?')}, "
              f"autres gros gagnants : {r.get('creator_other_winners', '?')}"]
        if r.get("tweet"):
            t = r["tweet"]
            L.append(f"- Tweet lu en direct : @{t.get('handle')} ({t.get('followers')} abonnes) : "
                     f"\"{(t.get('tweet_text') or '')[:300]}\"" + (f" — en reponse a @{t.get('parent_handle')} : \"{(t.get('parent_text') or '')[:200]}\""
                                                                    if t.get("parent_handle") else ""))
        if r.get("ia"):
            i = r["ia"]
            L.append(f"- Notre IA : {i['action']} (score {i['final'] if i['final'] is not None else '-'} / seuil {i['threshold']})")
        else:
            L.append("- Notre IA : coin pas analyse en direct (pas dans l'echantillon suivi, ou sans lien)")
        L.append("")
    (OUT / f"{stamp}.md").write_text("\n".join(L), encoding="utf-8")
    print(OUT / f"{stamp}.md")
    print("\n".join(L[:3]))


if __name__ == "__main__":
    main()
