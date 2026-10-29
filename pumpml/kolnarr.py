"""Memoire « ce que les gros KOL achetent » : narratives, images, createurs des coins achetes par le top 50
Kolscan (``[kol] top_path``), pour predire les prochains coins qu'ils vont acheter, AVANT eux.

Sources : ``data/kol_history.db`` (scripts/kol_backfill.py : historique on-chain de leurs achats pump.fun, nom,
description, image, liens et createur de chaque coin) + leurs achats vus en direct sur nos coins suivis.
Un coin entre en memoire a l'instant du PREMIER achat d'un gros KOL (c'est la qu'on sait qu'ils l'ont choisi) ;
« a migre » n'est connu qu'a l'heure de la migration. Aucun prix ni flux : seulement nom, ticker, description,
image, createur.

Signaux pour un nouveau coin, a l'instant de decision :
* ressemblance texte / image au coin le plus proche deja achete par un gros KOL, sur tout l'historique et sur
  les dernieres 48 h (la mode du moment chez eux) ; nombre de coins proches ;
* ressemblance aux coins achetes par plusieurs gros KOL a la fois (consensus) et a ceux qui ont ensuite migre ;
* le createur a-t-il deja lance des coins achetes par les gros KOL ? combien, il y a combien de temps ;
* meme nom / ticker que le coin le plus proche (copie) ;
* rang Kolscan du KOL qui a achete le coin le plus proche.
"""
from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd

from . import features as F
from . import narrative as N
from .fame import CLOSE, Bank, _norm

log = logging.getLogger("kolnarr")
FEATURES = ["kolnarr_text_sim", "kolnarr_img_sim", "kolnarr_n_close", "kolnarr_recent_sim", "kolnarr_recent_n",
            "kolnarr_multi_sim", "kolnarr_win_sim", "kolnarr_same_name", "kolnarr_best_rank", "kolnarr_age_h",
            "kolnarr_creator_n", "kolnarr_creator_last_d"]
RECENT_MS = 48 * 3_600_000


def memory(cfg) -> pd.DataFrame:
    """Un coin par ligne : mint, createur, nom, ticker, description, image, known_ms (1er achat d'un gros KOL),
    n_top (gros KOL differents), best_rank, migrated_ms."""
    from . import kol as K
    top = K.top(cfg)
    wallets, _ = K.kols(cfg)
    rank_of_wallet = {w: top[p] for w, p in wallets.items() if p in top}
    parts = []
    hp = cfg.data / "kol_history.db"
    if hp.exists():
        db = sqlite3.connect(f"file:{hp.as_posix()}?mode=ro", uri=True, timeout=60)
        b = pd.read_sql("SELECT wallet, mint, block_ms AS ms FROM buys WHERE is_buy=1", db)
        c = pd.read_sql("SELECT mint, name, symbol, description, image_path, creator FROM coins", db)
        db.close()
        parts.append((b, c))
    # achats vus en direct (coins suivis) : description depuis nos propres donnees
    h = K.history(cfg)
    if len(h):
        lb = h[h.is_buy & h.wallet.isin(rank_of_wallet)][["wallet", "mint", "recv_ms"]].rename(columns={"recv_ms": "ms"})
        if len(lb):
            from . import dataset as D
            days = D.raw_days(cfg.data)
            tok = D.dedupe_tokens(D.read(cfg.data, "tokens", days))
            meta = D.read(cfg.data, "metadata", days)
            tok = tok[tok.mint.isin(lb.mint)][["mint", "name", "symbol", "creator"]]
            if not meta.empty:
                meta = meta[meta.mint.isin(lb.mint)].drop_duplicates("mint")[["mint", "description", "image_path"]]
                tok = tok.merge(meta, on="mint", how="left")
            parts.append((lb, tok))
    if not parts:
        return pd.DataFrame(columns=["mint", "creator", "name", "symbol", "description", "image_path", "known_ms",
                                     "created_ms", "n_top", "best_rank", "migrated_ms"])
    buys = pd.concat([p[0] for p in parts], ignore_index=True)
    buys = buys[buys.wallet.isin(rank_of_wallet)]
    buys["rank"] = buys.wallet.map(rank_of_wallet)
    buys["person"] = buys.wallet.map(wallets)
    g = buys.sort_values("ms").groupby("mint").agg(known_ms=("ms", "min"), n_top=("person", "nunique"),
                                                   best_rank=("rank", "min"))
    coins = pd.concat([p[1] for p in parts], ignore_index=True).drop_duplicates("mint", keep="first")
    m = g.join(coins.set_index("mint"), how="inner").reset_index()
    # migration (connue a son heure) : fame.db couvre toutes les migrations des 90 derniers jours
    mig = {}
    fp = cfg.data / "fame.db"
    if fp.exists():
        db = sqlite3.connect(f"file:{fp.as_posix()}?mode=ro", uri=True, timeout=60)
        mig = dict(db.execute("SELECT mint, migrated_ms FROM fame WHERE migrated_ms IS NOT NULL"))
        db.close()
    m["migrated_ms"] = m.mint.map(mig)
    m["created_ms"] = m.known_ms
    return m


class KolNarr:
    def __init__(self, allb: Bank, recent_known: np.ndarray, multi: Bank, win: Bank, best_rank: np.ndarray):
        self.all, self.multi, self.win = allb, multi, win
        self.best_rank = best_rank
        by_creator: dict[str, list[int]] = {}
        for j, c in enumerate(allb.creator):
            if isinstance(c, str):
                by_creator.setdefault(c, []).append(j)
        self.by_creator = by_creator

    def features(self, mint, creator, name, symbol, T, I, has_img, at_ms) -> dict[str, np.ndarray]:
        n = len(mint)
        at_ms = np.asarray(at_ms, np.int64)
        mint = np.asarray(mint, object)
        out = {k: np.full(n, np.nan, np.float32) for k in FEATURES}
        a = self.all
        ts, ti, tc = a.nearest(T, at_ms, mint)
        out["kolnarr_text_sim"], out["kolnarr_n_close"] = ts, tc
        out["kolnarr_img_sim"], _, _ = a.nearest(I, at_ms, mint, img=True, has_q=has_img)
        rs, _, rc = a.nearest(T, at_ms, mint, min_ms=at_ms - RECENT_MS)
        out["kolnarr_recent_sim"], out["kolnarr_recent_n"] = rs, rc
        out["kolnarr_multi_sim"], _, _ = self.multi.nearest(T, at_ms, mint)
        out["kolnarr_win_sim"], _, _ = self.win.nearest(T, at_ms, mint)
        for k in np.where(ti >= 0)[0]:
            j = ti[k]
            nm, sy = _norm(name[k]), _norm(symbol[k])
            out["kolnarr_same_name"][k] = float(bool((nm and nm == a.name[j]) or (sy and sy == a.symbol[j])))
            out["kolnarr_best_rank"][k] = float(self.best_rank[j])
            out["kolnarr_age_h"][k] = (at_ms[k] - a.known_ms[j]) / 3_600_000
        for k in range(n):
            js = [j for j in self.by_creator.get(creator[k], []) if a.known_ms[j] < at_ms[k] and a.mint[j] != mint[k]]
            out["kolnarr_creator_n"][k] = float(len(js))
            if js:
                out["kolnarr_creator_last_d"][k] = (at_ms[k] - a.known_ms[max(js)]) / 86_400_000
        return out


def build(cfg, emb_store: N.EmbStore, embedder: N.Embedder) -> KolNarr:
    m = memory(cfg)
    s = lambda v: v if isinstance(v, str) else None  # noqa: E731
    items = []
    for r in m.itertuples(index=False):
        text = F.coin_text({"name": s(r.name), "symbol": s(r.symbol)}, {"description": s(r.description)})
        img = s(r.image_path)
        items.append({"mint": r.mint, "key": N.emb_key(r.mint, "kolnarr:" + text, img), "recv_ms": int(r.known_ms),
                      "text": text, "image_path": img})
    embs = N.ensure_embeddings(emb_store, embedder, items, cfg.data) if items else {}
    dim = len(embedder.text([""])[0])
    keys = np.array([it["key"] for it in items], object)
    allb = Bank.make(m.reset_index(drop=True), embs, list(keys), dim)
    order = np.argsort(m.known_ms.to_numpy(), kind="stable")
    best_rank = m.best_rank.to_numpy()[order]
    multi = (m.n_top >= 2).to_numpy()
    mu = m[multi].reset_index(drop=True)
    multib = Bank.make(mu, embs, list(keys[multi]), dim)
    w = m.migrated_ms.notna().to_numpy()
    wm = m[w].assign(known_ms=lambda x: np.maximum(x.known_ms, x.migrated_ms.astype(np.int64))).reset_index(drop=True)
    winb = Bank.make(wm, embs, list(keys[w]), dim)
    log.info("memoire des gros KOL : %d coins achetes, %d par plusieurs gros KOL, %d ont migre",
             len(allb), len(multib), len(winb))
    return KolNarr(allb, None, multib, winb, best_rank)
