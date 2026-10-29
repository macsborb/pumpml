"""Detection des copies ("vamps") : un coin qui reprend le ticker, le nom, l'image ou le lien X
d'un coin sorti avant.

Une memoire de tous les coins vus (fenetre glissante) est croisee avec chaque nouveau coin.
Les features ne regardent que les coins crees AVANT lui, et les resultats de ces coins ne sont
utilises que s'ils etaient connus au moment de decider (pas de fuite du futur).
La meme classe sert a l'entrainement (rejouee dans l'ordre chronologique) et en live.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import social

WINDOW_MS = 7 * 86_400_000
IMG_MAX_DIST = 6  # bits differents sur 64 pour considerer deux images identiques
_NON_ALNUM = re.compile(r"[^0-9a-zÀ-￿]+")


def norm(s: str | None) -> str:
    return _NON_ALNUM.sub("", (s or "").lower()).lstrip("$")


def twitter_key(url: str | None) -> str | None:
    l = social.parse_link(url)
    if l["kind"] == "tweet" and l["tweet_id"]:
        return f"t:{l['tweet_id']}"
    if l["kind"] == "community" and l["community_id"]:
        return f"c:{l['community_id']}"
    if l["kind"] == "profile" and l["handle"]:
        return f"p:{l['handle']}"
    return None


def dhash(path: Path) -> int | None:
    """Empreinte visuelle 64 bits (difference hash) : robuste au redimensionnement et a la compression."""
    try:
        from PIL import Image
        img = Image.open(path).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        a = np.asarray(img, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        return int(np.packbits(bits).view(">u8")[0]) - (1 << 63)  # tient dans un int64 signe
    except Exception:
        return None


class HashCache:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.lock = threading.Lock()
        with self.db:
            self.db.execute("CREATE TABLE IF NOT EXISTS dhash(image_path TEXT PRIMARY KEY, h INTEGER)")

    def get(self, data_root: Path, rel: str | None) -> int | None:
        if not rel:
            return None
        with self.lock:
            r = self.db.execute("SELECT h FROM dhash WHERE image_path=?", (rel,)).fetchone()
        if r:
            return r[0]
        h = dhash(data_root / rel)
        if h is not None:
            with self.lock, self.db:
                self.db.execute("INSERT OR REPLACE INTO dhash VALUES(?,?)", (rel, h))
        return h


@dataclass
class Entry:
    mint: str
    created_ms: int
    name: str
    symbol: str
    meta_ms: int | None = None       # quand les metadonnees (image, lien X) sont arrivees
    tw: str | None = None
    h: int | None = None
    resolve_ms: int | None = None    # quand le resultat est connu
    max_mult: float | None = None
    pnl: float | None = None
    migrated: bool = False


def _popcount64(x: np.ndarray) -> np.ndarray:
    """Nombre de bits a 1 de chaque entier 64 bits."""
    return np.unpackbits(np.ascontiguousarray(x).view(np.uint8).reshape(-1, 8), axis=1).sum(1, dtype=np.int64)


@dataclass
class Memory:
    entries: dict[str, Entry] = field(default_factory=dict)
    by_sym: dict[str, list[str]] = field(default_factory=dict)
    by_name: dict[str, list[str]] = field(default_factory=dict)
    by_tw: dict[str, list[str]] = field(default_factory=dict)
    img_mints: list[str] = field(default_factory=list)
    img_hash: list[int] = field(default_factory=list)

    def add(self, mint: str, created_ms: int, name: str | None, symbol: str | None):
        if mint in self.entries:
            return
        e = Entry(mint, created_ms, norm(name), norm(symbol))
        self.entries[mint] = e
        if e.symbol:
            self.by_sym.setdefault(e.symbol, []).append(mint)
        if e.name and len(e.name) >= 3:
            self.by_name.setdefault(e.name, []).append(mint)

    def set_meta(self, mint: str, meta_ms: int, twitter: str | None, h: int | None):
        e = self.entries.get(mint)
        if e is None or e.meta_ms is not None:
            return
        e.meta_ms, e.tw, e.h = meta_ms, twitter_key(twitter), h
        if e.tw:
            self.by_tw.setdefault(e.tw, []).append(mint)
        if h is not None:
            self.img_mints.append(mint)
            self.img_hash.append(h)

    def set_outcome(self, mint: str, resolve_ms: int, max_mult: float | None, pnl: float | None, migrated: bool):
        e = self.entries.get(mint)
        if e is not None:
            e.resolve_ms, e.max_mult, e.pnl, e.migrated = resolve_ms, max_mult, pnl, bool(migrated)

    def prune(self, now_ms: int):
        cut = now_ms - WINDOW_MS
        old = {m for m, e in self.entries.items() if e.created_ms < cut}
        if not old:
            return
        for m in old:
            del self.entries[m]
        for d in (self.by_sym, self.by_name, self.by_tw):
            for k in list(d):
                d[k] = [m for m in d[k] if m not in old]
                if not d[k]:
                    del d[k]
        keep = [i for i, m in enumerate(self.img_mints) if m not in old]
        self.img_mints = [self.img_mints[i] for i in keep]
        self.img_hash = [self.img_hash[i] for i in keep]

    def features(self, mint: str, created_ms: int, name: str | None, symbol: str | None, twitter: str | None,
                 h: int | None, decision_ms: int) -> tuple[dict, dict | None]:
        """Features de copie + description de l'original le plus probable (pour le dashboard)."""
        sym, nm, tw = norm(symbol), norm(name), twitter_key(twitter)
        lo = created_ms - WINDOW_MS

        def prior(mints):
            return [self.entries[m] for m in mints if m != mint and m in self.entries
                    and lo <= self.entries[m].created_ms < created_ms]

        m_sym = prior(self.by_sym.get(sym, [])) if sym else []
        m_name = prior(self.by_name.get(nm, [])) if nm and len(nm) >= 3 else []
        # image et lien X d'un coin precedent : seulement si connus au moment de decider
        m_tw = [e for e in prior(self.by_tw.get(tw, [])) if e.meta_ms is not None and e.meta_ms <= decision_ms] if tw else []
        m_img: list[Entry] = []
        if h is not None and self.img_hash:
            hs = np.array(self.img_hash, dtype=np.int64)
            close = np.where(_popcount64(hs ^ np.int64(h)) <= IMG_MAX_DIST)[0]
            m_img = [e for e in prior([self.img_mints[i] for i in close])
                     if e.meta_ms is not None and e.meta_ms <= decision_ms]
        allm = {e.mint: e for e in m_sym + m_name + m_tw + m_img}
        names = {e.mint for e in m_name}
        strong = {e.mint: e for e in m_img + m_tw + [e for e in m_sym if e.mint in names]}
        f = {
            "dup_symbol": float(len(m_sym)), "dup_name": float(len(m_name)),
            "dup_twitter": float(len(m_tw)), "dup_image": float(len(m_img)),
            "dup_any": float(len(allm)), "is_first": float(not allm),
            "dup_first_age_min": np.nan, "dup_best_mult": np.nan, "dup_any_migrated": np.nan, "dup_best_pnl": np.nan,
            "dup_strong": float(bool(m_img or m_tw or (m_sym and m_name))),
        }
        if not allm:
            return f, None
        first = min(allm.values(), key=lambda e: e.created_ms)
        f["dup_first_age_min"] = (created_ms - first.created_ms) / 60_000
        known = [e for e in allm.values() if e.resolve_ms is not None and e.resolve_ms <= decision_ms]
        best = None
        if known:
            mults = [e.max_mult for e in known if e.max_mult is not None]
            f["dup_best_mult"] = float(max(mults)) if mults else np.nan
            f["dup_any_migrated"] = float(any(e.migrated for e in known))
            pnls = [e.pnl for e in known if e.pnl is not None]
            f["dup_best_pnl"] = float(max(pnls)) if pnls else np.nan
            best = max(known, key=lambda e: (e.migrated, e.max_mult or 0))
        ref = best or first
        how = [k for k, v in (("ticker", m_sym), ("nom", m_name), ("lien X", m_tw), ("image", m_img)) if ref in v]
        info = {"mint": ref.mint, "age_min": round((created_ms - ref.created_ms) / 60_000, 1),
                "max_mult": ref.max_mult if ref.resolve_ms and ref.resolve_ms <= decision_ms else None,
                "migrated": ref.migrated if ref.resolve_ms and ref.resolve_ms <= decision_ms else None,
                "how": how, "n": len(allm)}
        if strong:
            # original de la copie (regle block_copies) : le premier coin repris par image, lien X ou nom + ticker
            o = min(strong.values(), key=lambda e: e.created_ms)
            info["orig"] = {"mint": o.mint, "created_ms": o.created_ms, "tw": o.tw}
        return f, info
