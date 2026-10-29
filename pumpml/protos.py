"""Bibliotheque de narratives : tes exemples notes (coins importes par CA ou vus en direct).

Chaque coin note 👍 ou 👎 devient un "prototype" (texte + image, et tes categories). Pour un nouveau
coin, on mesure a quel point il ressemble a tes bons et a tes mauvais exemples, et a chacune de
tes categories (ex. "news du jour", "animal mignon", "celebrite"). Ca marche des quelques exemples :
c'est le filtre de narrative que tu nourris en expliquant tes choix.

Pas de fuite du futur : un prototype ne sert qu'a partir du moment ou son resultat etait connu
(creation + duree d'un trade complet), et jamais pour le coin lui-meme.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from pathlib import Path

import numpy as np
import pandas as pd

from . import narrative as N

MAX_CATEGORIES = 12  # categories (tags) les plus utilisees, transformees en features


def slug(tag: str) -> str:
    t = unicodedata.normalize("NFKD", tag.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")[:24]


@dataclass
class Library:
    mints: np.ndarray        # (n,)
    known_ms: np.ndarray     # (n,) instant a partir duquel le prototype est utilisable
    verdict: np.ndarray      # (n,) +1 / -1
    T: np.ndarray            # (n, d) texte
    I: np.ndarray            # (n, d) image (zeros si pas d'image)
    has_img: np.ndarray      # (n,)
    tags: list[set[str]]     # categories par prototype
    categories: list[str]    # categories retenues comme features

    @property
    def size(self) -> int:
        return len(self.mints)

    def features(self, q_mint: np.ndarray, q_time: np.ndarray, q_T: np.ndarray, q_I: np.ndarray,
                 q_has_img: np.ndarray) -> dict[str, np.ndarray]:
        """Ressemblance aux bons / mauvais exemples et a chaque categorie, pour chaque requete."""
        n = len(q_mint)
        names = ["proto_bon_texte", "proto_bon_image", "proto_mauvais_texte", "proto_mauvais_image", "proto_bon_net"]
        names += [f"cat_{slug(c)}" for c in self.categories]
        out = {k: np.full(n, np.nan, np.float32) for k in names}
        if self.size == 0:
            return out
        st = q_T @ self.T.T                                   # (n, m)
        si = np.where(q_has_img[:, None] & self.has_img[None, :], q_I @ self.I.T, np.nan)
        usable = (self.known_ms[None, :] <= q_time[:, None]) & (self.mints[None, :] != q_mint[:, None])
        good, bad = usable & (self.verdict > 0)[None, :], usable & (self.verdict < 0)[None, :]

        def best(sim, mask):
            v = np.where(mask, sim, -np.inf)
            m = np.nanmax(np.where(np.isnan(v), -np.inf, v), axis=1) if v.size else np.full(n, -np.inf)
            return np.where(np.isfinite(m), m, np.nan).astype(np.float32)

        out["proto_bon_texte"], out["proto_bon_image"] = best(st, good), best(si, good)
        out["proto_mauvais_texte"], out["proto_mauvais_image"] = best(st, bad), best(si, bad)
        out["proto_bon_net"] = out["proto_bon_texte"] - out["proto_mauvais_texte"]
        for c in self.categories:
            in_cat = np.array([c in t for t in self.tags])
            out[f"cat_{slug(c)}"] = best(st, good & in_cat[None, :])
        return out


def load_feedback(journal_path) -> pd.DataFrame:
    from .feedback import Journal
    j = Journal(journal_path)  # cree les tables si le journal n'existe pas encore
    with j.lock:
        fb = pd.read_sql_query(
            "SELECT f.mint, f.verdict, f.tags, f.comment, f.ts, f.resolved_at_rating,"
            " COALESCE(d.created_ms, e.created_ms) created_ms,"
            " COALESCE(d.text, '') d_text, d.image_path d_img, e.name e_name, e.symbol e_symbol,"
            " e.description e_desc, e.image_path e_img FROM feedback f"
            " LEFT JOIN decisions d ON d.mint=f.mint LEFT JOIN examples e ON e.mint=f.mint"
            " WHERE f.verdict != 0", j.db)
    # coins annotes depuis la memoire des gagnants (analyses periodiques) : narrative lue dans data/fame.db
    fp = Path(journal_path).parent / "fame.db"
    miss = fb[fb.created_ms.isna() & (fb.d_text == "")]
    if fp.exists() and len(miss):
        import sqlite3
        db = sqlite3.connect(f"file:{fp.as_posix()}?mode=ro", uri=True, timeout=60)
        q = ",".join("?" * len(miss))
        fm = pd.read_sql(f"SELECT mint, name, symbol, description, image_path, COALESCE(created_ms, migrated_ms) c"
                         f" FROM fame WHERE mint IN ({q})", db, params=list(miss.mint))
        db.close()
        fm = fm.set_index("mint")
        for col, src in (("e_name", "name"), ("e_symbol", "symbol"), ("e_desc", "description"), ("e_img", "image_path"),
                         ("created_ms", "c")):
            fb[col] = fb[col].where(fb[col].notna(), fb.mint.map(fm[src]))
    return fb


def build(cfg, emb_store: N.EmbStore, embedder: N.Embedder, hold_ms: int) -> Library:
    """Construit la bibliotheque a partir de tous tes avis (coins en direct + exemples importes)."""
    fb = load_feedback(cfg.data / "journal.db")
    fb = fb[fb.created_ms.notna() & (fb.created_ms > 0)]
    items, tags = [], []
    for r in fb.itertuples(index=False):
        if r.d_text:
            text, img = r.d_text, r.d_img
        else:
            text = " | ".join(x for x in (r.e_name, r.e_symbol, r.e_desc) if isinstance(x, str) and x.strip())
            img = r.e_img
        img = img if isinstance(img, str) else None
        # Avis donne apres coup (resultat deja connu) sur un coin recent : il n'existe qu'a partir du moment ou
        # il a ete ecrit (sinon l'entrainement verrait une annotation du futur). Vieux exemples : date du coin.
        known = int(r.created_ms) + hold_ms
        if r.resolved_at_rating and r.ts and r.ts - r.created_ms < 7 * 86_400_000:
            known = max(known, int(r.ts))
        items.append({"mint": r.mint, "key": N.emb_key(r.mint, text, img), "recv_ms": int(r.created_ms),
                      "text": text, "image_path": img, "known": known})
        tags.append({t.strip().lower() for t in (r.tags or "").split(",") if t.strip()})
    if not items:
        z = np.zeros((0, 512), np.float32)
        return Library(np.array([], object), np.array([], np.int64), np.array([]), z, z, np.array([], bool), [], [])
    embs = N.ensure_embeddings(emb_store, embedder, items, cfg.data)
    good_tags = pd.Series([t for tg, v in zip(tags, fb.verdict) if v > 0 for t in tg])
    cats = good_tags.value_counts().index.tolist()[:MAX_CATEGORIES] if len(good_tags) else []
    return Library(
        mints=np.array([it["mint"] for it in items], object),
        known_ms=np.array([it["known"] for it in items], np.int64),
        verdict=np.sign(fb.verdict.to_numpy()).astype(np.int8),
        T=np.stack([embs[it["key"]][0] for it in items]), I=np.stack([embs[it["key"]][1] for it in items]),
        has_img=np.array([embs[it["key"]][2] for it in items]), tags=tags, categories=cats)
