"""Revue des erreurs du modele (tache planifiee des gagnants, toutes les 4 h) : ses pires achats en papier
(PnL Jupiter reel) et les coins qui ont PERCE (~30 000 $ ou migres, pumpml/percees.py) qu'il n'a pas achetes,
avec ce qu'il en pensait (score, seuil, raisons) et si des KOL avaient achete.

Usage : .venv\\Scripts\\python scripts\\errors_report.py [--hours 24]
Ecrit rapports/erreurs/<date>.md et .json ; la lecture (pourquoi il s'est trompe) et les annotations se font
ensuite (skill pumpml-gagnants, section 5).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pumpml.config import Config  # noqa: E402
from pumpml import percees as PC  # noqa: E402


def dec_rows(db, mints):
    if not mints:
        return {}
    q = ",".join("?" * len(mints))
    return {r["mint"]: dict(r) for r in db.execute(
        f"SELECT mint, action, final, threshold, top_reasons, text, twitter, vamp, name, symbol, description"
        f" FROM decisions WHERE mint IN ({q})", list(mints))}


def reasons(r):
    try:
        return ", ".join(f"{k} {v:+.2f}" for k, v in json.loads(r.get("top_reasons") or "[]")[:5])
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24)
    a = ap.parse_args()
    cfg = Config()
    since = int((time.time() - a.hours * 3600) * 1000)
    db = sqlite3.connect(cfg.data / "journal.db", timeout=60)
    db.row_factory = sqlite3.Row
    pos = [dict(r) for r in db.execute(
        "SELECT mint, pnl, exit_reason, exit_ms FROM positions WHERE status='closed' AND pnl IS NOT NULL"
        " AND exit_ms >= ? ORDER BY pnl", (since,))]
    dec = dec_rows(db, [p["mint"] for p in pos])
    worst = [{**p, **dec.get(p["mint"], {})} for p in pos if p["pnl"] < 0][:15]
    best = [{**p, **dec.get(p["mint"], {})} for p in sorted(pos, key=lambda p: -p["pnl"]) if p["pnl"] > 0][:5]
    m = PC.memory(cfg)
    perc = m[(m.perc == 1) & (m.known_ms >= since)]
    pdec = dec_rows(db, list(perc.mint))
    bought = {p["mint"] for p in pos}
    missed = []
    for r in perc.itertuples(index=False):
        d = pdec.get(r.mint, {})
        if r.mint in bought or d.get("action") == "buy":
            continue
        missed.append({"mint": r.mint, "name": r.name, "symbol": r.symbol, "description": r.description if isinstance(r.description, str) else "",
                       "kol": bool(r.kol), "min_to_perc": round((r.known_ms - r.created_ms) / 60000, 1), **{
                           k: d.get(k) for k in ("action", "final", "threshold", "top_reasons", "text", "twitter",
                                                 "vamp")}})
    caught = [r.mint for r in perc.itertuples(index=False) if r.mint in bought]
    n_pos = len(pos)
    tot = sum(p["pnl"] for p in pos)
    day = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    out = ROOT / "rapports" / "erreurs"
    out.mkdir(parents=True, exist_ok=True)
    L = [f"# Revue des erreurs du modele ({a.hours:g} h, {day})", "",
         f"- Achats clos : {n_pos}, PnL reel {tot:+.3f} SOL ; perdants {sum(p['pnl'] < 0 for p in pos)}.",
         f"- Coins perces sur la periode : {len(perc)} ; achetes par le modele : {len(caught)} ; rates : {len(missed)}"
         f" (dont {sum(x['kol'] for x in missed)} achetes par un KOL avant leur percee).", "",
         "## Pires achats", ""]
    for w in worst:
        L.append(f"- **{w.get('name')} ${w.get('symbol')}** `{w['mint']}` : {w['pnl']:+.4f} SOL ({w['exit_reason']}),"
                 f" score {w.get('final') or 0:.3f} / seuil {w.get('threshold') or 0:.3f}. Raisons : {reasons(w)}."
                 f" X : {w.get('twitter') or '-'}. Texte : {(w.get('text') or '')[:200]}")
    L += ["", "## Meilleurs achats (pour comparer)", ""]
    for w in best:
        L.append(f"- **{w.get('name')} ${w.get('symbol')}** `{w['mint']}` : {w['pnl']:+.4f} SOL. Raisons : {reasons(w)}.")
    L += ["", "## Percees ratees", ""]
    for x in missed[:20]:
        why = ("jamais analyse (exclu ou pas suivi)" if not x.get("action") else
               f"{x['action']}, score {x.get('final') or 0:.3f} / seuil {x.get('threshold') or 0:.3f}")
        L.append(f"- **{x['name']} ${x['symbol']}** `{x['mint']}` : percee en {x['min_to_perc']} min"
                 f"{', KOL avant' if x['kol'] else ''}. Modele : {why}. Raisons : {reasons(x)}."
                 f" X : {x.get('twitter') or '-'}. Description : {(x.get('description') or '')[:200]}")
    (out / f"{day}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / f"{day}.json").write_text(json.dumps({"worst": worst, "best": best, "missed": missed, "caught": caught},
                                                ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    print(out / f"{day}.md")


if __name__ == "__main__":
    main()
