"""Construction du dataset depuis l'archive Parquet.

Etape 1 (par jour, mise en cache) : features + label de chaque token cree ce jour-la.
Etape 2 (a l'entrainement) : features qui dependent du passe des AUTRES tokens
(historique du createur, tendance de narrative), calculees en ordre chronologique.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import features as F
from .config import Config
from .storage import collection_broken

log = logging.getLogger(__name__)


def _glob(data: Path, table: str, days: list[str]) -> list[str]:
    out = []
    for d in days:
        p = data / "raw" / table / f"date={d}"
        if any(p.glob("*.parquet")):
            out.append((p / "*.parquet").as_posix())
    return out


def read(data: Path, table: str, days: list[str], where: str = "") -> pd.DataFrame:
    files = _glob(data, table, days)
    if not files:
        return pd.DataFrame()
    con = duckdb.connect()
    q = f"SELECT * FROM read_parquet({files!r}, union_by_name=true) {where}"
    return con.execute(q).df()


def raw_days(data: Path) -> list[str]:
    d = data / "raw" / "tokens"
    return sorted(p.name[5:] for p in d.glob("date=*")) if d.exists() else []


def dedupe_tokens(tok: pd.DataFrame) -> pd.DataFrame:
    """Un token = une ligne. Date de creation = premiere reception, toutes sources confondues."""
    if tok.empty:
        return tok
    tok = tok.sort_values("recv_ms")
    first = tok.groupby("mint", as_index=False).agg(recv_ms=("recv_ms", "min"))
    pp = tok[tok.source == "pumpportal"].drop_duplicates("mint")
    ch = tok[tok.source == "chain"].drop_duplicates("mint")
    cols = ["mint", "name", "symbol", "uri", "creator", "sol_amount", "initial_buy", "raw"]
    chc = ["mint", "name", "symbol", "uri", "creator"] + (["chain_ms"] if "chain_ms" in ch else [])
    out = first.merge(pp[cols], on="mint", how="left").merge(ch[chc], on="mint", how="left", suffixes=("", "_ch"))
    if "chain_ms" not in out:
        out["chain_ms"] = np.nan
    for c in ("name", "symbol", "uri"):
        out[c] = out[c].fillna(out[f"{c}_ch"])
    out["creator"] = out["creator_ch"].fillna(out["creator"])  # le createur on-chain fait foi
    out = out.drop(columns=[c for c in out.columns if c.endswith("_ch")])
    # chain_ms vient de la source "chain" (suffixe absent car pas de colonne homonyme cote PumpPortal)
    # Suivi integral (mode coin par coin) : vrai si une des sources l'a marque.
    if "sampled" in tok:
        smp = tok.assign(sampled=tok["sampled"].astype("boolean")).groupby("mint")["sampled"].max()
        out["sampled"] = out["mint"].map(smp)
        # coins coupes pour robots de volume (faux volume) : jamais dans l'entrainement
        spam = set(tok.loc[tok.source == "spam", "mint"])
        if spam:
            out.loc[out["mint"].isin(spam), "sampled"] = False
    else:
        out["sampled"] = pd.NA
    # PumpPortal publie aussi des pools hors pump.fun : on ne garde que pump (ou vu on-chain).
    pool = out["raw"].fillna("").str.extract(r'"pool":\s*"([^"]+)"')[0]
    on_chain = out["mint"].isin(ch["mint"])
    return out[on_chain | pool.isna() | (pool == "pump")].drop(columns=["raw"]).reset_index(drop=True)


def _nn(d: dict | None) -> dict | None:
    """NaN / NA pandas -> None, pour que les lignes ressemblent a celles du live."""
    if d is None:
        return None
    return {k: (None if (v is pd.NA or v is pd.NaT or (isinstance(v, float) and np.isnan(v))) else v)
            for k, v in d.items()}


def migrations_chain(mig: pd.DataFrame) -> pd.Series:
    """Heure on-chain de migration par mint (sinon reception moins le retard typique du flux)."""
    if mig.empty:
        return pd.Series(dtype=np.int64)
    t = mig["chain_ms"] if "chain_ms" in mig else pd.Series(np.nan, index=mig.index)
    t = t.fillna(mig["recv_ms"] - 6_000)
    return t.groupby(mig["mint"]).min().astype(np.int64)


def gap_minutes(trades: pd.DataFrame) -> set[int]:
    """Minutes sans aucun trade : le collecteur ne tournait pas (pump.fun n'est jamais silencieux)."""
    if trades.empty:
        return set()
    m = (trades["recv_ms"] // 60_000).astype(np.int64)
    present = set(m.unique().tolist())
    return {x for x in range(min(present), max(present) + 1) if x not in present}


def build_day(cfg: Config, day: str, force: bool = False, cache: bool = True) -> pd.DataFrame | None:
    """Features + labels des tokens crees ce jour-la. ``cache=False`` pour un jour pas termine
    (recalcule a chaque fois : seuls les tokens dont le resultat est deja connu sont inclus)."""
    data = cfg.data
    out = feature_path(cfg, day)
    if cache and out.exists() and not force:
        return pd.read_parquet(out)
    s = cfg["strategy"]
    nxt = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    tok = dedupe_tokens(read(data, "tokens", [day]))
    if tok.empty:
        return None
    trades = read(data, "trades", [day, nxt])
    if trades.empty:
        return None
    mig = migrations_chain(read(data, "migrations", [day, nxt]))
    meta = read(data, "metadata", [day, nxt])
    soc = read(data, "social", [day, nxt])
    meta = meta[meta.ok].drop_duplicates("mint").set_index("mint") if not meta.empty else pd.DataFrame()
    soc = soc.drop_duplicates("mint").set_index("mint") if not soc.empty else pd.DataFrame()
    gaps = gap_minutes(trades)
    prev = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    out_df = read(data, "outages", [prev, day, nxt])
    outages = list(zip(out_df.start_ms, out_df.end_ms)) if not out_df.empty else []
    # minutes sans aucun trade (collecteur arrete) = coupures d'une minute, fusionnees si consecutives
    for m in sorted(gaps):
        if outages and outages[-1][1] == m * 60_000 - 6_000:
            outages[-1] = (outages[-1][0], (m + 1) * 60_000 - 6_000)
        else:
            outages.append((m * 60_000 - 6_000, (m + 1) * 60_000 - 6_000))  # -6 s : heure murale -> on-chain
    # dernier instant couvert, en heure on-chain (le flux a quelques secondes de retard)
    tsv = trades["ts"][trades["ts"] > 1_000_000_000] if "ts" in trades else pd.Series(dtype=float)
    last_ms = int(tsv.max()) * 1000 if len(tsv) else int(trades["recv_ms"].max()) - 6_000

    _, horizon = horizon_ms(s)
    trades = trades.sort_values(["mint", "recv_ms", "slot"])
    groups = {m: g for m, g in trades.groupby("mint", sort=False)}
    rows = []
    strict = cfg["collector"].get("trades_source") == "per_token"
    for t in tok.itertuples(index=False):
        if strict and not (pd.notna(t.sampled) and bool(t.sampled)):
            continue  # coin non suivi integralement (ou donnees anciennes, partielles)
        g = groups.get(t.mint)
        tr = F.Trades.from_rows(g.to_dict("records")) if g is not None else F.Trades.from_rows([])
        token = _nn(t._asdict())
        created = F.created_chain_ms(token, tr)
        token["recv_ms"] = created  # tout en heure on-chain a partir d'ici
        if created + horizon > last_ms:
            continue  # resultat du dernier point d'analyse pas encore connu
        mig_ms = int(mig[t.mint]) if t.mint in mig.index else None
        rows += token_rows(token, tr, _nn(meta.loc[t.mint].to_dict()) if t.mint in meta.index else None,
                           _nn(soc.loc[t.mint].to_dict()) if t.mint in soc.index else None, s, mig_ms, outages)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if cache:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(out)
    log.info("features %s%s : %d coins retenus sur %d (%d lignes, une par point d'analyse)", day,
             "" if cache else " (jour en cours)", df.mint.nunique(), len(tok), len(df))
    return df


def feature_path(cfg: Config, day: str) -> Path:
    """Cache des features d'un jour. Le nom depend de la version des features ET de la strategie
    (points d'analyse, TP, SL, frais...) : changer la strategie change les resultats simules."""
    import hashlib
    import json as _json
    sig = hashlib.md5(_json.dumps(cfg["strategy"], sort_keys=True).encode()).hexdigest()[:8]
    return cfg.data / "features" / f"date={day}" / f"v{F.FEATURE_VERSION}-{sig}.parquet"


def checkpoints_ms(s: dict) -> list[int]:
    """Ages (ms) auxquels chaque coin est analyse : achat possible au premier point ou le score passe."""
    return [int(c * 1000) for c in s["checkpoints_s"]]


def hold_ms(s: dict) -> int:
    """Duree entre un point d'analyse et le moment ou son resultat est connu."""
    return int((s["latency_s"] * 2 + s["max_hold_s"]) * 1000)


def horizon_ms(s: dict) -> tuple[int, int]:
    """(dernier point d'analyse, instant ou tous les resultats d'un coin sont connus), depuis la creation."""
    last = checkpoints_ms(s)[-1]
    return last, last + hold_ms(s)


def token_rows(token: dict, tr: F.Trades, mrow: dict | None, srow: dict | None, s: dict,
               mig_ms: int | None, outages: list | None = None) -> list[dict]:
    """Une ligne par point d'analyse encore jouable (meme calcul pour le live et les coins importes).
    ``first_cp`` marque la premiere : son resultat sert de "resultat du coin" (memoire, createur, tendance)."""
    created = int(token["recv_ms"])
    rows = []
    for dec in checkpoints_ms(s):
        if outages and collection_broken(outages, created, dec + int(s["latency_s"] * 1000), dec + hold_ms(s)):
            continue  # trades manquants a ce point : features ou label faux
        r = token_row(token, tr, mrow, srow, s, mig_ms, dec)
        if r is None:
            break  # courbe devenue non standard ou deja migre : plus jouable ensuite non plus
        if r == "wait":
            continue
        rows.append(r)
    if rows:
        rows[0]["first_cp"] = 1.0
        for r in rows[1:]:
            r["first_cp"] = 0.0
    return rows


def token_row(token: dict, tr: F.Trades, mrow: dict | None, srow: dict | None, s: dict,
              mig_ms: int | None, dec: int) -> dict | None:
    """Features (connues a ``dec`` ms d'age) + resultat d'un achat a cet instant."""
    created = int(token["recv_ms"])
    horizon = dec + hold_ms(s)
    if not F.is_standard_curve(tr.upto(created + dec)):
        return None
    if mig_ms is not None and mig_ms <= created + dec:
        return None  # deja migre au moment de decider : hors strategie
    # Les metadonnees / infos X recues APRES la decision ne sont pas utilisables.
    # (heure de reception : la decision live a lieu a creation + dec + latence, en heure murale)
    cutoff = created + dec + int(s["latency_s"] * 1000)
    if mrow and mrow["recv_ms"] > cutoff:
        mrow = None
    if srow and srow["recv_ms"] > cutoff:
        srow = None
    if s.get("require_link") and mrow is None:
        return "wait"  # metadonnees pas encore la : on ne sait pas encore s'il y a un lien
    if F.excluded(token, mrow, tr.upto(created + dec), s, srow):
        return None  # jamais achete (lancement groupe, vieux tweet, reponse parasite) : hors strategie
    if F.tweet_pending(token, mrow, srow, created + dec, s):
        return "wait"  # tweet frais pas encore lu : pas de decision a ce point
    f = {"mint": token["mint"], "created_ms": created, "creator": token.get("creator"), "name": token.get("name"),
         "symbol": token.get("symbol"), "text": F.text_of(token, mrow, srow),
         "image_path": (mrow or {}).get("image_path"), "twitter": F.x_link(mrow),
         "coin_text": F.coin_text(token, mrow), "tweet_text": ((srow or {}).get("tweet_text") or "")[:600],
         "meta_ms": (mrow or {}).get("recv_ms")}
    f["age_s"] = dec / 1000
    f["decision_ms"] = created + dec
    f.update(F.flow_features(created, token.get("creator"), tr, created + dec))
    f.update(F.meta_features(token, mrow, srow))
    f.update(F.time_features(created))
    o = F.simulate(tr, created + dec, s, mig_ms)
    f.update(pnl=o.pnl_sol, exit_reason=o.exit_reason, hold_s=o.hold_s, max_mult_after=o.max_mult_after,
             entry_mult=o.entry_mult, migrated=float(mig_ms is not None and mig_ms <= created + horizon),
             resolve_ms=created + horizon)
    return f


def complete_days(cfg: Config) -> list[str]:
    """Jours dont tous les labels sont connus (le lendemain a commence depuis > horizon)."""
    s = cfg["strategy"]
    horizon_h = horizon_ms(s)[1] / 3_600_000 + 0.25
    now = datetime.now(timezone.utc)
    return [d for d in raw_days(cfg.data)
            if now >= datetime.fromisoformat(d).replace(tzinfo=timezone.utc) + timedelta(days=1, hours=horizon_h)]


def creator_activity(creators: np.ndarray, created: np.ndarray, pop_creator: np.ndarray,
                     pop_created: np.ndarray) -> dict[str, np.ndarray]:
    """Coins lances par le meme createur dans l'heure et les 6 h precedentes, parmi TOUS les coins vus
    (y compris ceux exclus de la strategie) : les bots en serie lancent beaucoup de coins exclus."""
    by: dict[str, np.ndarray] = {}
    order = np.argsort(pop_created)
    pc, pt = pop_creator[order], pop_created[order]
    for cr in set(creators):
        if isinstance(cr, str):
            by[cr] = pt[pc == cr]
    n1 = np.zeros(len(creators), np.float32)
    n6 = np.zeros(len(creators), np.float32)
    for i, (cr, t) in enumerate(zip(creators, created)):
        ts = by.get(cr)
        if ts is None or not len(ts):
            continue
        hi = np.searchsorted(ts, t, side="left")
        n1[i] = hi - np.searchsorted(ts, t - 3_600_000)
        n6[i] = hi - np.searchsorted(ts, t - 6 * 3_600_000)
    return {"creator_n_1h": n1, "creator_n_6h": n6}


def creator_features(df: pd.DataFrame) -> pd.DataFrame:
    """Historique du createur, sans fuite : seuls les coins precedents (et resolus pour le PnL).
    Calcule sur une ligne par coin (premier point d'analyse), puis recopie sur toutes ses lignes."""
    full = df
    df = df[df["first_cp"] > 0] if "first_cp" in df else df
    df = df.sort_values("created_ms")
    n_prev = np.zeros(len(df))
    prev_pnl = np.full(len(df), np.nan)
    prev_mig = np.full(len(df), np.nan)
    pos = {i: k for k, i in enumerate(df.index)}
    for _, g in df.groupby("creator", sort=False):
        if len(g) == 1:
            continue
        c, r, p, mg = (g[k].to_numpy() for k in ("created_ms", "resolve_ms", "pnl", "migrated"))
        for j, ix in enumerate(g.index):
            k = pos[ix]
            n_prev[k] = j
            known = r[:j] <= c[j]
            if known.any():
                prev_pnl[k] = p[:j][known].mean()
                prev_mig[k] = mg[:j][known].mean()
    per = pd.DataFrame({"mint": df["mint"].to_numpy(), "creator_prev_tokens": n_prev, "creator_prev_pnl": prev_pnl,
                        "creator_prev_migrated": prev_mig})
    full = full.drop(columns=[c for c in per.columns if c != "mint" and c in full], errors="ignore")
    return full.merge(per, on="mint", how="left")


def load_window(cfg: Config, days: list[str]) -> pd.DataFrame:
    """Jours termines (cache) + jours en cours (recalcules) : on apprend sans attendre la fin du jour."""
    complete = set(complete_days(cfg))
    parts = [df for d in days if (df := build_day(cfg, d, cache=d in complete)) is not None]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def population(cfg: Config, days: list[str]) -> pd.DataFrame:
    """TOUS les coins vus (memes exclus de la strategie) + leurs metadonnees : la memoire des copies."""
    tok = dedupe_tokens(read(cfg.data, "tokens", days))
    if tok.empty:
        return tok
    meta = read(cfg.data, "metadata", days)
    if not meta.empty:
        meta = meta[meta.ok].sort_values("recv_ms").drop_duplicates("mint")
        meta["twitter"] = [F.x_link({"twitter": a, "website": b}) for a, b in zip(meta["twitter"], meta["website"])]
        meta = meta[["mint", "recv_ms", "twitter", "image_path"]]
        tok = tok.merge(meta.rename(columns={"recv_ms": "meta_ms"}), on="mint", how="left")
    for c in ("meta_ms", "twitter", "image_path"):
        if c not in tok:
            tok[c] = None
    # creation en heure on-chain (memes references que les features)
    tok["recv_ms"] = tok["chain_ms"].fillna(tok["recv_ms"] - 6_000).astype(np.int64)
    return tok.sort_values("recv_ms").reset_index(drop=True)
