"""Memoire longue des narratives : ce qui a marche, et ce qui a rate.

* "Gros gagnant" : il a rempli sa courbe naturellement (pas un lancement groupe) ET sa market cap a depasse
  100 000 $ (config [fame]), ou c'est un de tes exemples 👍. Les petits x2 ne comptent pas.
* "Flop" : migre naturellement mais jamais 100 000 $, ou un de tes exemples 👎.
Sources : data/fame.db (scripts/import_fame.py : 90 jours relus sur la chaine ; scripts/import_melt.py :
dataset academique MemeTrans, ~27 000 coins de dec. 2024 a mars 2025) + tes exemples.

Pour chaque nouveau coin, au moment de decider (seuls les resultats DEJA connus a cet instant comptent) :
* le createur a-t-il deja fait un gros gagnant ? combien, il y a combien de temps ;
* ressemble-t-il (texte, image) a un gros gagnant ? a un flop ? lequel est le plus proche ;
* copie (meme nom / ticker) ou clin d'oeil (sens proche, nom different : la reprise detournee, souvent
  avec humour, que toute la communaute comprend).
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import features as F
from . import narrative as N

log = logging.getLogger("fame")

FEATURES = ["creator_wins", "creator_last_win_d", "fame_text_sim", "fame_img_sim", "fame_n_close",
            "fame_age_d", "fame_same_name", "fame_twist", "flop_text_sim", "flop_img_sim", "flop_n_close",
            "fame_vs_flop"]
CLOSE = 0.80  # similarite texte (CLIP) a partir de laquelle deux coins parlent de la meme chose


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower()) if isinstance(s, str) else ""


@dataclass
class Bank:
    """Un groupe de coins (gagnants ou flops), tries par instant ou leur resultat est connu."""
    mint: np.ndarray
    creator: np.ndarray
    known_ms: np.ndarray
    created_ms: np.ndarray
    name: np.ndarray
    symbol: np.ndarray
    T: np.ndarray
    I: np.ndarray
    has_img: np.ndarray

    def __len__(self):
        return len(self.mint)

    @staticmethod
    def make(rows: pd.DataFrame, embs: dict, keys: list[str], dim: int) -> "Bank":
        o = np.argsort(rows.known_ms.to_numpy(), kind="stable")
        rows = rows.iloc[o]
        keys = [keys[i] for i in o]
        T = np.stack([embs[k][0] for k in keys]) if keys else np.zeros((0, dim), np.float32)
        I = np.stack([embs[k][1] for k in keys]) if keys else np.zeros((0, dim), np.float32)
        return Bank(rows.mint.to_numpy(object), rows.creator.to_numpy(object), rows.known_ms.to_numpy(np.int64),
                    rows.created_ms.to_numpy(np.int64), np.array([_norm(x) for x in rows.name], object),
                    np.array([_norm(x) for x in rows.symbol], object), T.astype(np.float32), I.astype(np.float32),
                    np.array([embs[k][2] for k in keys], bool))

    def nearest(self, Q: np.ndarray, at_ms: np.ndarray, qmint: np.ndarray, img: bool = False, has_q=None,
                min_ms: np.ndarray | None = None):
        """Pour chaque requete : (similarite max, indice du plus proche, nb >= CLOSE) parmi les coins dont le
        resultat est connu avant ``at_ms`` (et apres ``min_ms`` si donne), autres que lui-meme. Par blocs, sur la
        carte graphique si possible."""
        n = len(Q)
        best = np.full(n, np.nan, np.float32)
        arg = np.full(n, -1, np.int64)
        close = np.full(n, np.nan, np.float32)
        if not n:
            return best, arg, close
        close[:] = 0.0
        if not len(self):
            return best, arg, close
        W = self.I if img else self.T
        ok = self.has_img if img else np.ones(len(self), bool)
        b = np.searchsorted(self.known_ms, at_ms, side="left")
        lo = np.searchsorted(self.known_ms, min_ms, side="left") if min_ms is not None else np.zeros(n, np.int64)
        pos = {m: j for j, m in enumerate(self.mint)}
        selfj = np.array([pos.get(m, -1) for m in qmint], np.int64)  # le coin lui-meme, s'il est en memoire
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float16 if dev == "cuda" else torch.float32
        Wt = torch.from_numpy(W).to(dev, dt)
        okt = torch.from_numpy(ok).to(dev)
        ar = torch.arange(len(self), device=dev)
        for a in range(0, n, 2048):
            q = torch.from_numpy(np.ascontiguousarray(Q[a:a + 2048], np.float32)).to(dev, dt)
            s = (q @ Wt.T).float()
            valid = (ar[None, :] < torch.from_numpy(b[a:a + 2048]).to(dev)[:, None]) & okt[None, :]
            valid &= ar[None, :] >= torch.from_numpy(lo[a:a + 2048]).to(dev)[:, None]
            valid &= ar[None, :] != torch.from_numpy(selfj[a:a + 2048]).to(dev)[:, None]
            s = torch.where(valid, s, torch.full_like(s, -2.0))
            v, i = s.max(1)
            best[a:a + 2048] = v.cpu().numpy()
            arg[a:a + 2048] = i.cpu().numpy()
            close[a:a + 2048] = (s >= CLOSE).sum(1).float().cpu().numpy()
        empty = best <= -1.5
        best[empty], arg[empty] = np.nan, -1
        if has_q is not None:
            hq = np.asarray(has_q, bool)
            best[~hq], arg[~hq], close[~hq] = np.nan, -1, np.nan
        return best, arg, close


@dataclass
class Fame:
    win: Bank
    flop: Bank

    def __len__(self):
        return len(self.win) + len(self.flop)

    def features(self, mint, creator, name, symbol, T, I, has_img, at_ms) -> dict[str, np.ndarray]:
        """Une ligne par requete (tableaux de meme longueur). Sans fuite : resultats connus avant ``at_ms``."""
        n = len(mint)
        at_ms = np.asarray(at_ms, np.int64)
        mint = np.asarray(mint, object)
        out = {k: np.full(n, np.nan, np.float32) for k in FEATURES}
        w = self.win
        ts, ti, tc = w.nearest(T, at_ms, mint)
        out["fame_text_sim"], out["fame_n_close"] = ts, tc
        out["fame_img_sim"], _, _ = w.nearest(I, at_ms, mint, img=True, has_q=has_img)
        fs, _, fc = self.flop.nearest(T, at_ms, mint)
        out["flop_text_sim"], out["flop_n_close"] = fs, fc
        out["flop_img_sim"], _, _ = self.flop.nearest(I, at_ms, mint, img=True, has_q=has_img)
        out["fame_vs_flop"] = ts - fs
        for k in np.where(ti >= 0)[0]:
            j = ti[k]
            out["fame_age_d"][k] = (at_ms[k] - w.created_ms[j]) / 86_400_000
            nm, sy = _norm(name[k]), _norm(symbol[k])
            same = bool((nm and nm == w.name[j]) or (sy and sy == w.symbol[j]))
            out["fame_same_name"][k] = float(same)
            out["fame_twist"][k] = float(ts[k] >= CLOSE and not same)
        by_creator: dict[str, list[int]] = {}
        for j, c in enumerate(w.creator):
            if isinstance(c, str):
                by_creator.setdefault(c, []).append(j)
        for k in range(n):
            js = [j for j in by_creator.get(creator[k], []) if w.known_ms[j] < at_ms[k] and w.mint[j] != mint[k]]
            out["creator_wins"][k] = float(len(js))
            if js:
                out["creator_last_win_d"][k] = (at_ms[k] - w.known_ms[max(js)]) / 86_400_000
        return out


def label(f: pd.DataFrame, cfg) -> tuple[pd.Series, pd.Series]:
    """(win, known_ms) d'une migration naturelle de fame.db. win = 1 gros gagnant, 0 flop (dont les vidages),
    NaN = pas encore mesure (le coin est laisse hors memoire). Regle "tenue" : plus haut sur bougies de 15 min
    >= seuil, encore >= min_hold_ratio du plus haut hold_minutes apres, et une vraie foule sur la courbe."""
    fc = cfg["fame"]
    thr, day = fc["min_peak_mcap_usd"], f.migrated_ms + 86_400_000
    big = f.peak_mcap_usd >= thr
    if fc.get("win_rule", "plus_haut") != "tenue" or "hold_ratio" not in f:
        return big.astype(float), pd.Series(np.where(big, f.win_ms.fillna(day), day), index=f.index)
    measured = f.peak15_usd.notna()
    crowd = f.curve_tx.isna() | (f.curve_tx >= fc["min_curve_tx"])  # NULL = plus de 1000 tx
    win = (big & measured & (f.peak15_usd >= thr) & (f.hold_ratio >= fc["min_hold_ratio"]) & crowd).astype(float)
    win[big & ~measured] = np.nan
    known = np.where(big & measured, f.peak15_ms + fc["hold_minutes"] * 60_000, day)
    return win, pd.Series(known, index=f.index)


def memory(cfg) -> pd.DataFrame:
    """Gagnants (win=1) et flops (win=0) de tout l'historique + tes exemples. ``known_ms`` : instant ou le
    resultat est connu (gagnant : fin du premier jour au-dessus du seuil ; flop : un jour apres la migration)."""
    cols = ["mint", "creator", "name", "symbol", "description", "image_path", "created_ms", "known_ms", "win"]
    parts = []
    fp = cfg.data / "fame.db"
    if fp.exists():
        db = sqlite3.connect(f"file:{fp.as_posix()}?mode=ro", uri=True, timeout=60)
        names = {r[1] for r in db.execute("PRAGMA table_info(fame)")}
        extra = ", source, curve_tx, peak15_usd, peak15_ms, hold_ratio" if "hold_ratio" in names else ""
        f = pd.read_sql("SELECT mint, creator, name, symbol, description, image_path, created_ms, migrated_ms,"
                        f" win_ms, peak_mcap_usd{extra} FROM fame WHERE organic=1 AND peak_mcap_usd IS NOT NULL", db)
        db.close()
        if extra:
            # MELT (pas de bougies 15 min) : ancienne regle, le jeu de donnees a deja ecarte les manipulations
            melt = f.source.eq("melt")
            w0, k0 = label(f[melt], {"fame": {**cfg["fame"], "win_rule": "plus_haut"}})
            w1, k1 = label(f[~melt], cfg)
            f["win"], f["known_ms"] = pd.concat([w0, w1]), pd.concat([k0, k1])
        else:
            f["win"], f["known_ms"] = label(f, cfg)
        f = f[f.win.notna()].copy()
        f["win"] = f.win.astype(int)
        parts.append(f)
    jp = cfg.data / "journal.db"
    if jp.exists():
        db = sqlite3.connect(f"file:{jp.as_posix()}?mode=ro", uri=True)
        ex = pd.read_sql("SELECT e.mint, e.name, e.symbol, e.created_ms, e.description, e.image_path, f.verdict"
                         " FROM examples e JOIN feedback f ON f.mint=e.mint WHERE f.verdict != 0 AND e.created_ms > 0", db)
        db.close()
        if not ex.empty:
            ex["creator"] = None
            ex["win"] = (ex.verdict > 0).astype(int)
            ex["known_ms"] = ex.created_ms + 86_400_000  # resultat connu de longue date
            parts.append(ex)
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat([p.reindex(columns=cols) for p in parts], ignore_index=True)
    out = out.drop_duplicates("mint", keep="last")  # tes exemples priment
    out["created_ms"] = out.created_ms.fillna(out.known_ms - 86_400_000).astype(np.int64)
    out["known_ms"] = out.known_ms.astype(np.int64)
    return out.reset_index(drop=True)


def build(cfg, emb_store: N.EmbStore, embedder: N.Embedder) -> Fame:
    m = memory(cfg)
    s = lambda v: v if isinstance(v, str) else None  # noqa: E731 (NaN pandas -> None)
    items = []
    for r in m.itertuples(index=False):
        text = F.coin_text({"name": s(r.name), "symbol": s(r.symbol)}, {"description": s(r.description)})
        img = s(r.image_path)
        items.append({"mint": r.mint, "key": N.emb_key(r.mint, "fame:" + text, img), "recv_ms": int(r.created_ms),
                      "text": text, "image_path": img})
    embs = N.ensure_embeddings(emb_store, embedder, items, cfg.data) if items else {}
    dim = len(embedder.text([""])[0])
    keys = np.array([it["key"] for it in items], object)
    win = (m.win == 1).to_numpy()
    f = Fame(Bank.make(m[win].reset_index(drop=True), embs, list(keys[win]), dim),
             Bank.make(m[~win].reset_index(drop=True), embs, list(keys[~win]), dim))
    log.info("memoire des narratives : %d gros gagnants, %d flops", len(f.win), len(f.flop))
    return f
