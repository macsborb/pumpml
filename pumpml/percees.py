"""Memoire des coins qui ont PERCE pendant nos collectes : montes a ``[percees] seuil_sol`` de market cap
(~30 000 $, presque la migration) ou migres. Pourquoi eux et pas les autres ? Le modele compare chaque nouveau
coin a cette memoire par sa narrative seulement (nom, ticker, description, image, createur) ; le prix ne sert
qu'a dire qui a perce, jamais dans les poids.

Population : les coins suivis en entier (``sampled``, courbe standard), hors lancements groupes (bloc 0) et hors
coins pris dans une coupure de la collecte. Un coin percé entre en memoire a l'instant ou il franchit le seuil
(ou migre) ; un coin qui n'a rien fait, a la fin de son suivi. Les KOL : achat d'un KOL suivi AVANT la percee.

Signaux, connus avant l'instant de decision :
* ressemblance texte / image a la percee la plus proche ; sur les dernieres 24 h ; nombre de percees proches ;
* ressemblance aux percees achetees par des KOL avant leur envolee ;
* taux de percee (lisse) parmi les 30 coins suivis les plus proches : la narrative marche-t-elle en ce moment ?
* percees deja faites par le createur.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import dataset as D
from . import features as F
from . import narrative as N
from .fame import CLOSE, Bank

log = logging.getLogger("percees")
FEATURES = ["perc_text_sim", "perc_img_sim", "perc_n_close", "perc_recent_sim", "perc_kol_sim", "perc_rate",
            "perc_creator_n"]
RECENT_MS = 24 * 3_600_000
K_NEAR = 30
PRIOR = 10.0


def _outages(cfg, days) -> list[tuple[int, int]]:
    o = D.read(cfg.data, "outages", days)
    return [] if o.empty else list(zip(o.start_ms.astype("int64"), o.end_ms.astype("int64")))


def memory(cfg, days: list[str] | None = None) -> pd.DataFrame:
    """Un coin suivi par ligne : mint, creator, name, symbol, description, image_path, created_ms, known_ms,
    perc (1 percee / 0 rien), kol (un KOL a achete avant la percee)."""
    pc = cfg.raw.get("percees", {})
    thr = float(pc.get("seuil_sol", 300))
    track_ms = int(cfg["collector"]["track_minutes"]) * 60_000
    days = days or D.raw_days(cfg.data)[-(cfg["training"]["window_days"] + 1):]
    cols = ["mint", "creator", "name", "symbol", "description", "image_path", "created_ms", "known_ms", "perc", "kol"]
    tok = D.dedupe_tokens(D.read(cfg.data, "tokens", days))
    files = D._glob(cfg.data, "trades", days)
    if tok.empty or not files or "sampled" not in tok:
        return pd.DataFrame(columns=cols)
    tok = tok[tok.sampled.fillna(False).astype(bool)]
    import duckdb
    con = duckdb.connect()
    src = f"read_parquet({files!r}, union_by_name=true)"
    # plus haut, instant du franchissement du seuil, part du bloc 0, courbe standard (vsol - real_sol = 30 SOL)
    t = con.execute(f"""
        WITH t AS (SELECT * FROM {src} WHERE vtok > 0),
             s0 AS (SELECT mint, MIN(slot) s0 FROM t GROUP BY mint)
        SELECT t.mint, MAX(vsol * 1e6 / vtok) AS peak,
               MIN(CASE WHEN vsol * 1e6 / vtok >= {thr} THEN recv_ms END) AS cross_ms,
               SUM(CASE WHEN is_buy AND slot <= s0.s0 + 2 THEN token_amount ELSE 0 END) / {F.CURVE_SUPPLY} AS b0,
               MAX(ABS(vsol - real_sol - {30_000_000_000})) AS off
        FROM t JOIN s0 USING (mint) GROUP BY t.mint""").df()
    m = tok.merge(t, on="mint", how="inner")
    m = m[(m.off <= 0.02 * 30_000_000_000) & (m.b0 < cfg["strategy"].get("max_block0_share", 0.25))]
    m["created_ms"] = m["chain_ms"].fillna(m["recv_ms"] - 6_000).astype(np.int64)
    mig = D.read(cfg.data, "migrations", days)
    if not mig.empty:
        mig["ms"] = mig["chain_ms"].fillna(mig["recv_ms"]) if "chain_ms" in mig else mig["recv_ms"]
        m["mig_ms"] = m.mint.map(mig.groupby("mint").ms.min())
    else:
        m["mig_ms"] = np.nan
    m["cross_ms"] = m.cross_ms.astype("float64")
    m["mig_ms"] = m.mig_ms.astype("float64")
    m["perc"] = (m.cross_ms.notna() | m.mig_ms.notna()).astype(int)
    m["known_ms"] = np.where(m.perc == 1, np.fmin(m.cross_ms.fillna(np.inf), m.mig_ms.fillna(np.inf)),
                             m.created_ms + track_ms).astype(np.int64)
    # coupure de la collecte pendant le suivi : un « rien » n'est pas sur (la percee a pu etre manquee)
    out = _outages(cfg, days)
    if out:
        hit = np.zeros(len(m), bool)
        c, e = m.created_ms.to_numpy(), (m.created_ms + track_ms).to_numpy()
        for a, b in out:
            hit |= (c < b) & (e > a)
        m = m[~hit | (m.perc == 1).to_numpy()]
    meta = D.read(cfg.data, "metadata", days)
    if not meta.empty:
        meta = meta[meta.ok].sort_values("recv_ms").drop_duplicates("mint")[["mint", "description", "image_path"]]
        m = m.merge(meta, on="mint", how="left")
    else:
        m["description"], m["image_path"] = None, None
    m["kol"] = False
    if cfg.raw.get("kol", {}).get("enabled"):
        from . import kol as K
        h = K.history(cfg, days)
        if len(h):
            first_buy = h[h.is_buy].groupby("mint").recv_ms.min()
            fb = m.mint.map(first_buy)
            m["kol"] = (m.perc == 1) & fb.notna() & (fb < m.known_ms)
    return m[cols].reset_index(drop=True)


class Percees:
    def __init__(self, allb: Bank, label: np.ndarray, perc: Bank, kol: Bank):
        self.all, self.label, self.perc, self.kol = allb, label.astype(np.float32), perc, kol
        self.base = float(label.mean()) if len(label) else 0.0
        by_creator: dict[str, list[int]] = {}
        for j, c in enumerate(perc.creator):
            if isinstance(c, str):
                by_creator.setdefault(c, []).append(j)
        self.by_creator = by_creator

    def _rate(self, T: np.ndarray, at_ms: np.ndarray, mint: np.ndarray) -> np.ndarray:
        """Taux de percee lisse parmi les K_NEAR coins suivis les plus proches, connus avant ``at_ms``."""
        n, a = len(T), self.all
        out = np.full(n, np.nan, np.float32)
        if not n or len(a) < K_NEAR:
            return out
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float16 if dev == "cuda" else torch.float32
        W = torch.from_numpy(a.T).to(dev, dt)
        lab = torch.from_numpy(self.label).to(dev)
        ar = torch.arange(len(a), device=dev)
        b = np.searchsorted(a.known_ms, at_ms, side="left")
        pos = {m: j for j, m in enumerate(a.mint)}
        selfj = np.array([pos.get(m, -1) for m in mint], np.int64)
        for s in range(0, n, 1024):
            q = torch.from_numpy(np.ascontiguousarray(T[s:s + 1024], np.float32)).to(dev, dt)
            sim = (q @ W.T).float()
            valid = (ar[None, :] < torch.from_numpy(b[s:s + 1024]).to(dev)[:, None])
            valid &= ar[None, :] != torch.from_numpy(selfj[s:s + 1024]).to(dev)[:, None]
            sim = torch.where(valid, sim, torch.full_like(sim, -2.0))
            v, i = sim.topk(K_NEAR, dim=1)
            ok = v > -1.5
            hits = (lab[i] * ok).sum(1)
            cnt = ok.sum(1).float()
            r = ((hits + PRIOR * self.base) / (cnt + PRIOR)).cpu().numpy()
            enough = (cnt >= 5).cpu().numpy()
            out[s:s + 1024] = np.where(enough, r, np.nan)
        return out

    def features(self, mint, creator, T, I, has_img, at_ms) -> dict[str, np.ndarray]:
        n = len(mint)
        at_ms = np.asarray(at_ms, np.int64)
        mint = np.asarray(mint, object)
        out = {k: np.full(n, np.nan, np.float32) for k in FEATURES}
        p = self.perc
        out["perc_text_sim"], _, out["perc_n_close"] = p.nearest(T, at_ms, mint)
        out["perc_img_sim"], _, _ = p.nearest(I, at_ms, mint, img=True, has_q=has_img)
        out["perc_recent_sim"], _, _ = p.nearest(T, at_ms, mint, min_ms=at_ms - RECENT_MS)
        out["perc_kol_sim"], _, _ = self.kol.nearest(T, at_ms, mint)
        out["perc_rate"] = self._rate(T, at_ms, mint)
        for k in range(n):
            js = [j for j in self.by_creator.get(creator[k], []) if p.known_ms[j] < at_ms[k] and p.mint[j] != mint[k]]
            out["perc_creator_n"][k] = float(len(js))
        return out


def make(m: pd.DataFrame, embs: dict, keys: list[str], dim: int) -> Percees:
    keys = np.array(keys, object)
    allb = Bank.make(m, embs, list(keys), dim)
    label = m.perc.to_numpy()[np.argsort(m.known_ms.to_numpy(), kind="stable")]
    pm = (m.perc == 1).to_numpy()
    km = (m.kol.astype(bool)).to_numpy()
    return Percees(allb, label, Bank.make(m[pm].reset_index(drop=True), embs, list(keys[pm]), dim),
                   Bank.make(m[km].reset_index(drop=True), embs, list(keys[km]), dim))


def build(cfg, emb_store: N.EmbStore, embedder: N.Embedder) -> Percees:
    m = memory(cfg)
    s = lambda v: v if isinstance(v, str) else None  # noqa: E731
    items = []
    for r in m.itertuples(index=False):
        text = F.coin_text({"name": s(r.name), "symbol": s(r.symbol)}, {"description": s(r.description)})
        img = s(r.image_path)
        items.append({"mint": r.mint, "key": N.emb_key(r.mint, "perc:" + text, img), "recv_ms": int(r.created_ms),
                      "text": text, "image_path": img})
    embs = N.ensure_embeddings(emb_store, embedder, items, cfg.data) if items else {}
    dim = len(embedder.text([""])[0])
    p = make(m, embs, [it["key"] for it in items], dim)
    log.info("memoire des percees : %d coins suivis, %d percees (%.1f %%), %d achetees par un KOL avant",
             len(p.all), len(p.perc), 100 * p.base, len(p.kol))
    return p
