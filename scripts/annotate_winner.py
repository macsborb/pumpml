"""Annote un coin analyse (rapport periodique des gros gagnants) : l'IA l'apprend dans sa bibliotheque de
narratives (ressemblance a tes bons / mauvais exemples, categories).

Usage :
  .venv\\Scripts\\python scripts\\annotate_winner.py <mint> <verdict 1|-1> "<categories, separees par des virgules>" "<explication>"
Les categories reprennent celles du dashboard (animal, meme / humour, culture internet, news du jour, ia / tech,
celebrite / influenceur, politique, communaute / culte, crypto meta, tweet viral...) ; une nouvelle categorie est
acceptee. L'explication dit POURQUOI la narrative a marche (ou pas) : source, viralite, moment, clin d'oeil.
Chaque annotation est aussi gardee dans rapports/gagnants/annotations.jsonl.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    if len(sys.argv) != 5 or sys.argv[2] not in ("1", "-1"):
        print(__doc__)
        sys.exit(2)
    mint, verdict, tags, comment = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    body = {"mint": mint, "verdict": verdict, "tags": tags, "comment": comment, "after_outcome": True}
    req = urllib.request.Request("http://127.0.0.1:8787/api/feedback", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Origin": "http://127.0.0.1:8787"})
    urllib.request.urlopen(req, timeout=20).read()
    out = ROOT / "rapports" / "gagnants"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "annotations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": int(time.time() * 1000), **body}, ensure_ascii=False) + "\n")
    print("annote :", mint, verdict, tags)


if __name__ == "__main__":
    main()
