"""KOL (gros traders suivis de Solana) : qui achete le coin, et a quel point ces gens ont du flair.

Liste : ``[kol] list_path`` (JSON ``[{address, name, emoji}]`` ; une personne peut avoir plusieurs wallets).
Sources de leurs trades :
- en direct, PumpPortal ``subscribeAccountTrade`` sur tous leurs wallets (gratuit, tous les coins pump.fun,
  meme ceux qu'on ne suit pas) -> table ``kol_trades`` de hot.db + Parquet ``data/raw/kol`` ;
- l'historique : les trades des coins suivis (``data/raw/trades``) dont le wallet est un KOL.

Score d'un KOL (sans fuite du futur) : part de ses achats sur des coins qui ont MIGRE dans les 24 h, parmi
ceux dont le resultat etait connu avant l'instant de decision (migration vue, ou 24 h ecoulees sans
migration), lissee vers le taux moyen. Signaux pour le modele (groupe « kol ») : nombre de KOL acheteurs avant la
decision, SOL achetes, age du premier achat KOL, ventes de KOL, meilleur / moyen score des acheteurs."""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("kol")
FEATURES = ["kol_n", "kol_buys", "kol_buy_sol", "kol_first_age_s", "kol_sells", "kol_holding_n",
            "kol_best_score", "kol_mean_score", "kol_good_n", "kol_known_n", "kol_top_n", "kol_top_best_rank"]
ZERO = ("kol_n", "kol_buys", "kol_buy_sol", "kol_sells", "kol_holding_n", "kol_good_n", "kol_known_n", "kol_top_n")
HORIZON_MS = 24 * 3_600_000
PRIOR = 5.0  # poids du taux moyen dans le score (achats fictifs)


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> tuple[dict, dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    wallets = {r["address"]: r.get("name") or r["address"][:6] for r in rows if r.get("address")}
    emoji = {r.get("name") or r["address"][:6]: r.get("emoji") or "" for r in rows if r.get("address")}
    return wallets, emoji


def kols(cfg) -> tuple[dict[str, str], dict[str, str]]:
    """(wallet -> nom, nom -> emoji). Vide si la liste n'existe pas."""
    from .config import ROOT
    p = Path(cfg["kol"]["list_path"])
    p = p if p.is_absolute() else ROOT / p
    if not p.exists():
        return {}, {}
    return _load(str(p), p.stat().st_mtime)


def top(cfg) -> dict[str, int]:
    """Gros noms : personne -> rang au classement mensuel Kolscan (``[kol] top_path``, gains reels en SOL)."""
    from .config import ROOT
    p = Path(cfg["kol"].get("top_path", ""))
    p = p if p.is_absolute() else ROOT / p
    if not cfg["kol"].get("top_path") or not p.exists():
        return {}
    wallets, _ = kols(cfg)
    out: dict[str, int] = {}
    for r in json.loads(p.read_text(encoding="utf-8"))["top"]:
        person = wallets.get(r["address"])
        if person:
            out[person] = min(out.get(person, 999), int(r["rank"]))
    return out


def history(cfg, days: list[str] | None = None) -> pd.DataFrame:
    """Tous les trades KOL connus (flux PumpPortal + trades des coins suivis), un par signature.
    Colonnes : signature, mint, recv_ms, wallet, person, is_buy, sol."""
    from . import dataset as D
    wallets, _ = kols(cfg)
    if not wallets:
        return pd.DataFrame(columns=["signature", "mint", "recv_ms", "wallet", "person", "is_buy", "sol"])
    days = days or D.raw_days(cfg.data)
    parts = []
    k = D.read(cfg.data, "kol", days)
    if not k.empty:
        parts.append(k[["signature", "mint", "recv_ms", "wallet", "is_buy", "sol"]])
    files = D._glob(cfg.data, "trades", days)
    if files:
        import duckdb
        con = duckdb.connect()
        con.register("w", pd.DataFrame({"wallet": list(wallets)}))
        t = con.execute(f"SELECT t.signature, t.mint, t.recv_ms, t.user AS wallet, t.is_buy, t.sol_amount AS sol"
                        f" FROM read_parquet({files!r}, union_by_name=true) t JOIN w ON t.user = w.wallet").df()
        if not t.empty:
            t["sol"] = t.sol / 1e9  # lamports -> SOL (le flux PumpPortal est deja en SOL)
            parts.append(t)
    if not parts:
        return pd.DataFrame(columns=["signature", "mint", "recv_ms", "wallet", "person", "is_buy", "sol"])
    h = pd.concat(parts, ignore_index=True).sort_values("recv_ms").drop_duplicates("signature")
    h["is_buy"] = h.is_buy.astype(bool)
    h["person"] = h.wallet.map(wallets)
    return h.reset_index(drop=True)


class Book:
    """Achats passes des KOL et leur resultat (migration du coin dans les 24 h), pour scorer chaque personne
    a un instant donne sans regarder le futur."""

    def __init__(self, hist: pd.DataFrame, migrated: dict[str, int], top_rank: dict[str, int] | None = None):
        self.hist = hist
        self.top = top_rank or {}
        self.by_mint = {m: g for m, g in hist.groupby("mint")} if not hist.empty else {}
        b = hist[hist.is_buy].drop_duplicates(["person", "mint"]) if not hist.empty else hist
        recs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        allk, alls = [], []
        for p, g in b.groupby("person") if not b.empty else []:
            buy = g.recv_ms.to_numpy(np.int64)
            mig = np.array([migrated.get(m, -1) for m in g.mint], np.int64)
            win = (mig >= 0) & (mig - buy <= HORIZON_MS)
            known = np.where(win, np.maximum(mig, buy), buy + HORIZON_MS)
            o = np.argsort(known)
            recs[p] = (known[o], win[o], g.mint.to_numpy(object)[o], buy[o])
            allk.append(known)
            alls.append(win)
        self.recs = recs
        k = np.concatenate(allk) if allk else np.zeros(0, np.int64)
        s = np.concatenate(alls) if alls else np.zeros(0, bool)
        o = np.argsort(k)
        self.all_known, self.all_cum = k[o], np.cumsum(s[o])

    def base_rate(self, at: int) -> float:
        n = int(np.searchsorted(self.all_known, at, side="right"))
        return float((self.all_cum[n - 1] + 1) / (n + 50)) if n else 0.02

    def score(self, person: str, at: int, exclude: str | None = None) -> tuple[float, int]:
        """(score lisse, achats dont le resultat est connu) de ``person`` a l'instant ``at``."""
        r = self.recs.get(person)
        p0 = self.base_rate(at)
        if r is None:
            return p0, 0
        n = int(np.searchsorted(r[0], at, side="right"))
        win, mints = r[1][:n], r[2][:n]
        if exclude is not None:
            keep = mints != exclude
            win = win[keep]
        return float((win.sum() + PRIOR * p0) / (len(win) + PRIOR)), int(len(win))

    def features(self, mint: str, created_ms: int, at: int, trades: pd.DataFrame | None = None) -> dict[str, float]:
        g = trades if trades is not None else self.by_mint.get(mint)
        out = {k: np.nan for k in FEATURES}
        out.update({k: 0.0 for k in ZERO})
        if g is None or g.empty:
            return out
        g = g[g.recv_ms <= at]
        if g.empty:
            return out
        buys, sells = g[g.is_buy], g[~g.is_buy]
        if buys.empty:
            out["kol_sells"] = float(len(sells))
            return out
        people = list(dict.fromkeys(buys.person))
        sold = set(sells.person)
        p0 = self.base_rate(at)
        sc = [self.score(p, at, exclude=mint) for p in people]
        out.update(kol_n=float(len(people)), kol_buys=float(len(buys)), kol_buy_sol=float(buys.sol.sum()),
                   kol_first_age_s=float((buys.recv_ms.min() - created_ms) / 1000), kol_sells=float(len(sells)),
                   kol_holding_n=float(sum(p not in sold for p in people)),
                   kol_best_score=max(s for s, _ in sc), kol_mean_score=float(np.mean([s for s, _ in sc])),
                   kol_good_n=float(sum(s >= 2 * p0 and n >= 3 for s, n in sc)),
                   kol_known_n=float(sum(n >= 3 for _, n in sc)))
        ranks = [self.top[p] for p in people if p in self.top]
        out["kol_top_n"] = float(len(ranks))
        if ranks:
            out["kol_top_best_rank"] = float(min(ranks))
        return out


def migrations(cfg, days: list[str] | None = None) -> dict[str, int]:
    from . import dataset as D
    days = days or D.raw_days(cfg.data)
    return D.migrations_chain(D.read(cfg.data, "migrations", days)).to_dict()


def build(cfg) -> Book:
    h = history(cfg)
    b = Book(h, migrations(cfg), top(cfg))
    log.info("KOL : %d trades connus, %d personnes actives, %d coins", len(h), h.person.nunique() if len(h) else 0,
             h.mint.nunique() if len(h) else 0)
    return b


def add(df: pd.DataFrame, book: Book, lat_ms: int) -> None:
    """Ajoute les signaux KOL a chaque ligne (coin x point d'analyse), connus a ``decision_ms + latence``."""
    q = (df.decision_ms.to_numpy() + lat_ms) if "decision_ms" in df else df.created_ms.to_numpy()
    cols = {k: np.full(len(df), np.nan, np.float32) for k in FEATURES}
    for i, (m, c, t) in enumerate(zip(df.mint.to_numpy(object), df.created_ms.to_numpy(np.int64), q)):
        if m in book.by_mint:
            for k, v in book.features(m, int(c), int(t)).items():
                cols[k][i] = v
        else:
            for k in ZERO:
                cols[k][i] = 0.0
    for k, v in cols.items():
        df[k] = v
