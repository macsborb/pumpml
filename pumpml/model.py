"""Apprentissage continu champion / challenger, registre de modeles, prediction.

* Toutes les ``retrain_every_hours`` : on reentraine sur toutes les donnees dont le resultat est connu
  (jours termines + jour en cours). Les decisions passees de l'IA et leurs resultats en font partie :
  elle apprend de ses erreurs.
* Decoupage TEMPOREL : validation = les heures les plus recentes, jamais vues a l'entrainement.
* Plusieurs reglages (hyperparametres) sont essayes a chaque fois ; on garde le meilleur sur la fin du
  train. Le seuil d'achat est choisi la aussi, jamais sur la validation.
* Le nouveau modele ne remplace le champion que s'il fait mieux en PnL simule sur la MEME validation.
* Tes avis : ponderation des exemples + modele "gout perso". Les coins importes par CA servent
  d'exemples supplementaires (entrainement seulement).
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from . import dataset as D
from . import fame as FM
from . import narrative as N
from . import protos as P
from . import search as S
from . import vamp as V
from .config import Config
from .feedback import Journal

log = logging.getLogger("train")

NON_FEATURES = {"mint", "created_ms", "creator", "name", "symbol", "text", "image_path", "pnl", "exit_reason",
                "coin_text", "tweet_text", "decision_ms", "first_cp",
                "hold_s", "max_mult_after", "entry_mult", "migrated", "resolve_ms", "day", "y", "w", "verdict",
                "twitter", "meta_ms", "is_backfill", "truncated",
                "fame_known"}  # nombre de gagnants connus : simple mesure du temps qui passe, pas un signal
# Features qui dependent des AUTRES coins du moment : inconnues pour un coin importe (NaN).
CONTEXT = ["creator_prev_tokens", "creator_prev_pnl", "creator_prev_migrated", "creator_n_1h", "creator_n_6h", "sim_1h", "nn_pnl", "nn_win",
           "nn_sim", "dup_symbol", "dup_name", "dup_twitter", "dup_image", "dup_any", "is_first",
           "dup_first_age_min", "dup_best_mult", "dup_any_migrated", "dup_best_pnl", "dup_strong"]
TASTE_WEIGHT = 0.3
MIN_FEEDBACK_FOR_TASTE = 40
KEEP_MODELS = 200


def build_memory(cfg: Config, pop: pd.DataFrame, outcomes: pd.DataFrame, hc: V.HashCache) -> V.Memory:
    """Memoire des coins vus. Les filtres temporels de Memory.features evitent toute fuite du futur."""
    mem = V.Memory()
    for r in pop.itertuples(index=False):
        mem.add(r.mint, int(r.recv_ms), r.name, r.symbol)
        if pd.notna(r.meta_ms):
            img = r.image_path if isinstance(r.image_path, str) else None
            mem.set_meta(r.mint, int(r.meta_ms), r.twitter if isinstance(r.twitter, str) else None,
                         hc.get(cfg.data, img))
    for r in outcomes.itertuples(index=False):
        mem.set_outcome(r.mint, int(r.resolve_ms), float(r.entry_mult * r.max_mult_after), float(r.pnl),
                        bool(r.migrated))
    return mem


def vamp_frame(cfg: Config, mem: V.Memory, df: pd.DataFrame, hc: V.HashCache) -> pd.DataFrame:
    lat = int(cfg["strategy"]["latency_s"] * 1000)  # meme instant que le live (infos recues)
    rows = []
    for r in df.itertuples(index=False):
        img = r.image_path if isinstance(r.image_path, str) else None
        tw = r.twitter if isinstance(r.twitter, str) else None
        f, _ = mem.features(r.mint, int(r.created_ms), r.name, r.symbol, tw, hc.get(cfg.data, img),
                            int(r.decision_ms) + lat)
        rows.append(f)
    return pd.DataFrame(rows, index=df.index)


def embeddings(cfg: Config, df: pd.DataFrame, emb_store: N.EmbStore, embedder: N.Embedder) -> dict:
    """Embeddings du contenu connu a chaque point d'analyse (l'image ou le tweet peuvent arriver plus tard)."""
    items = df[["mint", "created_ms", "text", "image_path"]].rename(columns={"created_ms": "recv_ms"})
    items = items.astype(object).where(items.notna(), None).to_dict("records")
    for it in items:
        it["key"] = N.emb_key(it["mint"], it["text"], it["image_path"])
    embs = N.ensure_embeddings(emb_store, embedder, items, cfg.data)
    keys = [it["key"] for it in items]
    return {"T": np.stack([embs[k][0] for k in keys]), "I": np.stack([embs[k][1] for k in keys]),
            "has_img": np.array([embs[k][2] for k in keys])}


def add_context(df: pd.DataFrame, cfg: Config, emb_store: N.EmbStore, embedder: N.Embedder,
                pop: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Historique createur, tendance de narrative, copies (vamps) et embeddings (texte, image)."""
    df = D.creator_features(df).sort_values("created_ms").reset_index(drop=True)
    emb = embeddings(cfg, df, emb_store, embedder)
    T, I, has_img = emb["T"], emb["I"], emb["has_img"]
    df["img_text_sim"] = np.where(has_img, (T * I).sum(1), np.nan)
    # Tendance : comparaison avec les coins sortis avant le point d'analyse, dont le resultat etait connu.
    # Le "resultat d'un coin" = celui de son premier point d'analyse (meme definition qu'en live).
    first = (df["first_cp"] > 0).to_numpy() if "first_cp" in df else np.ones(len(df), bool)
    lat = int(cfg["strategy"]["latency_s"] * 1000)
    tr = N.trend_features_at(df.created_ms.to_numpy()[first], df.mint.to_numpy()[first], T[first],
                             df.pnl.to_numpy()[first], df.resolve_ms.to_numpy()[first],
                             (df.decision_ms.to_numpy() + lat) if "decision_ms" in df else df.created_ms.to_numpy(),
                             df.mint.to_numpy(), T)
    for k, v in tr.items():
        df[k] = v
    add_topics(df, embedder)
    add_protos(df, emb, cfg, emb_store, embedder)
    add_fame(df, emb, cfg, emb_store, embedder)
    add_kol(df, cfg)
    add_kolnarr(df, emb, cfg, emb_store, embedder)
    add_percees(df, emb, cfg, emb_store, embedder)
    if pop is not None and not pop.empty:
        ca = D.creator_activity(df.creator.to_numpy(), df.created_ms.to_numpy(), pop.creator.to_numpy(),
                                pop.recv_ms.to_numpy())
        for k, v in ca.items():
            df[k] = v
        hc = V.HashCache(cfg.data / "emb.db")
        mem = build_memory(cfg, pop, df[first], hc)
        vf = vamp_frame(cfg, mem, df, hc)
        for c in vf.columns:
            df[c] = vf[c]
    return df, emb


def add_protos(df: pd.DataFrame, emb: dict, cfg: Config, emb_store: N.EmbStore, embedder: N.Embedder):
    """Ressemblance a tes bons / mauvais exemples et a tes categories (bibliotheque de narratives)."""
    lib = P.build(cfg, emb_store, embedder, D.horizon_ms(cfg["strategy"])[1])
    lat = int(cfg["strategy"]["latency_s"] * 1000)
    q_time = (df.decision_ms.to_numpy() + lat) if "decision_ms" in df else df.created_ms.to_numpy()
    for k, v in lib.features(df.mint.to_numpy(), q_time, emb["T"], emb["I"], emb["has_img"]).items():
        df[k] = v


_FAME: dict = {}


def add_fame(df: pd.DataFrame, emb: dict, cfg: Config, emb_store: N.EmbStore, embedder: N.Embedder):
    """Memoire longue des gagnants : createur deja gagnant, ressemblance / clin d'oeil a un ancien succes."""
    if "f" not in _FAME:
        _FAME["f"] = FM.build(cfg, emb_store, embedder)
    lat = int(cfg["strategy"]["latency_s"] * 1000)
    q_time = (df.decision_ms.to_numpy() + lat) if "decision_ms" in df else df.created_ms.to_numpy()
    cr = df["creator"].to_numpy(object) if "creator" in df else np.full(len(df), None, object)
    for k, v in _FAME["f"].features(df.mint.to_numpy(object), cr, df["name"].to_numpy(object),
                                    df["symbol"].to_numpy(object), emb["T"], emb["I"], emb["has_img"],
                                    q_time).items():
        df[k] = v


def add_kol(df: pd.DataFrame, cfg: Config):
    """KOL acheteurs avant l'instant de decision, et leur flair passe (sans fuite du futur)."""
    if not cfg.raw.get("kol", {}).get("enabled"):
        return
    from . import kol as K
    if "k" not in _KOL:
        _KOL["k"] = K.build(cfg)
    K.add(df, _KOL["k"], int(cfg["strategy"]["latency_s"] * 1000))


_KOL: dict = {}


def add_kolnarr(df: pd.DataFrame, emb: dict, cfg: Config, emb_store: N.EmbStore, embedder: N.Embedder):
    """Ressemblance (nom, ticker, description, image, createur) aux coins achetes par les gros KOL, connue a la
    decision : predire ce qu'ils vont acheter avant eux, sans regarder le prix."""
    if not cfg.raw.get("kol", {}).get("enabled"):
        return
    from . import kolnarr as KN
    if "kn" not in _KOL:
        _KOL["kn"] = KN.build(cfg, emb_store, embedder)
    lat = int(cfg["strategy"]["latency_s"] * 1000)
    q_time = (df.decision_ms.to_numpy() + lat) if "decision_ms" in df else df.created_ms.to_numpy()
    cr = df["creator"].to_numpy(object) if "creator" in df else np.full(len(df), None, object)
    for k, v in _KOL["kn"].features(df.mint.to_numpy(object), cr, df["name"].to_numpy(object),
                                    df["symbol"].to_numpy(object), emb["T"], emb["I"], emb["has_img"], q_time).items():
        df[k] = v


_PERC: dict = {}


def add_percees(df: pd.DataFrame, emb: dict, cfg: Config, emb_store: N.EmbStore, embedder: N.Embedder):
    """Ressemblance aux coins qui ont perce pendant nos collectes (~30 000 $ ou migres), connue a la decision."""
    from . import percees as PC
    if "percees" not in cfg["training"]["feature_groups"] and not cfg["training"].get("force_percees"):
        return
    if "p" not in _PERC:
        _PERC["p"] = PC.build(cfg, emb_store, embedder)
    lat = int(cfg["strategy"]["latency_s"] * 1000)
    q_time = (df.decision_ms.to_numpy() + lat) if "decision_ms" in df else df.created_ms.to_numpy()
    cr = df["creator"].to_numpy(object) if "creator" in df else np.full(len(df), None, object)
    for k, v in _PERC["p"].features(df.mint.to_numpy(object), cr, emb["T"], emb["I"], emb["has_img"],
                                    q_time).items():
        df[k] = v


def add_topics(df: pd.DataFrame, embedder: N.Embedder):
    """Themes (monde, politique, animal...) et gravite du coin et du tweet."""
    ct = df["coin_text"].fillna("").tolist() if "coin_text" in df else df["text"].fillna("").tolist()
    tt = df["tweet_text"].fillna("").tolist() if "tweet_text" in df else [""] * len(df)
    for k, v in N.topic_features(embedder, ct, tt).items():
        df[k] = v


class Predictor:
    def __init__(self, model_dir: Path):
        self.dir = model_dir
        self.meta = json.loads((model_dir / "meta.json").read_text())
        if self.meta.get("members"):  # ensemble (LightGBM et/ou XGBoost)
            self.booster = S.Ensemble.load(model_dir, self.meta["members"])
        else:  # anciens modeles : un seul LightGBM
            self.booster = S.Ensemble([S.Member("lgb", lgb.Booster(model_file=str(model_dir / "model.txt")))])
        self.pca_t, self.pca_i = joblib.load(model_dir / "pca.joblib")
        tp = model_dir / "taste.txt"
        self.taste = lgb.Booster(model_file=str(tp)) if tp.exists() else None
        self.id = model_dir.name
        self.threshold = self.meta["threshold"]
        self.features = self.meta["features"]

    @staticmethod
    def embed_cols(pca_t: PCA, pca_i: PCA, T: np.ndarray, I: np.ndarray, has_img: np.ndarray) -> pd.DataFrame:
        a = pca_t.transform(T)
        b = pca_i.transform(I)
        b[~has_img] = np.nan
        cols = {f"txt_{k}": a[:, k] for k in range(a.shape[1])}
        cols.update({f"img_{k}": b[:, k] for k in range(b.shape[1])})
        return pd.DataFrame(cols)

    def matrix(self, df: pd.DataFrame, emb: dict) -> pd.DataFrame:
        E = self.embed_cols(self.pca_t, self.pca_i, emb["T"], emb["I"], emb["has_img"])
        X = pd.concat([df.reset_index(drop=True), E], axis=1)
        for c in self.features:
            if c not in X:
                X[c] = np.nan
        return X[self.features].astype(np.float32)

    def score(self, df: pd.DataFrame, emb: dict) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        X = self.matrix(df, emb)
        p = self.booster.predict(X)
        t = self.taste.predict(X) if self.taste is not None else None
        final = p if t is None else (1 - TASTE_WEIGHT) * p + TASTE_WEIGHT * t
        return p, t, final

    def reasons(self, df: pd.DataFrame, emb: dict, top: int = 5) -> list[list[tuple[str, float]]]:
        """Contributions SHAP : pourquoi le modele aime / n'aime pas ce coin."""
        X = self.matrix(df, emb)
        c = self.booster.contrib(X)
        out = []
        for row in c:
            idx = np.argsort(-np.abs(row))[:top]
            out.append([(self.features[i], float(row[i])) for i in idx])
        return out


def prev_cummax(score: np.ndarray, mint: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Pour chaque ligne : meilleur score obtenu par ce coin AUX POINTS PRECEDENTS (-inf au premier)."""
    order = np.lexsort((age, mint))
    out = np.full(len(score), -np.inf)
    prev_m, best = None, -np.inf
    for i in order:
        if mint[i] != prev_m:
            prev_m, best = mint[i], -np.inf
        out[i] = best
        best = max(best, score[i])
    return out


def bought(score: np.ndarray, pcm: np.ndarray, thr: float) -> np.ndarray:
    """Regle d'achat : la ligne ou le score depasse le seuil POUR LA PREMIERE FOIS (un achat par coin)."""
    return (score >= thr) & (pcm < thr)


def pnl_at(score: np.ndarray, pnl: np.ndarray, thr: float, pcm: np.ndarray | None = None) -> tuple[float, int]:
    sel = bought(score, pcm, thr) if pcm is not None else score >= thr
    return float(pnl[sel].sum()), int(sel.sum())


def best_threshold(score: np.ndarray, pnl: np.ndarray, min_trades: int,
                   pcm: np.ndarray | None = None) -> tuple[float, float, int]:
    """Seuil qui maximise le PnL simule de la regle d'achat (avec un minimum d'achats).
    Retourne (seuil, pnl, nombre d'achats)."""
    best, best_thr, best_n = -np.inf, 1.01, 0
    for thr in np.unique(np.quantile(score, np.linspace(0.3, 0.999, 150))):
        v, n = pnl_at(score, pnl, thr, pcm)
        if n >= min_trades and v > best:
            best, best_thr, best_n = v, float(thr), n
    return best_thr, (best if np.isfinite(best) else 0.0), best_n


class Registry:
    def __init__(self, root: Path):
        self.root = root
        self.file = root / "registry.json"
        self.state = json.loads(self.file.read_text()) if self.file.exists() else {"champion": None, "history": []}

    def save(self):
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str))
        tmp.replace(self.file)

    def latest(self) -> Predictor | None:
        """Dernier modele entraine (meme non promu)."""
        for h in reversed(self.state.get("history", [])):
            if (self.root / h["id"] / "meta.json").exists():  # ancien (model.txt) ou ensemble (member*)
                return Predictor(self.root / h["id"])
        return None

    def champion(self) -> Predictor | None:
        c = self.state.get("champion")
        return Predictor(self.root / c) if c and (self.root / c).exists() else None


def _bins(score: np.ndarray, pnl: np.ndarray, y: np.ndarray, n: int = 10) -> list[dict]:
    """Par tranche de score : le modele classe-t-il bien ? (les hauts scores doivent mieux rapporter)"""
    if len(score) < n:
        return []
    qs = np.quantile(score, np.linspace(0, 1, n + 1))
    out = []
    for i in range(n):
        m = (score >= qs[i]) & ((score < qs[i + 1]) if i < n - 1 else (score <= qs[i + 1]))
        if m.any():
            out.append({"lo": float(qs[i]), "hi": float(qs[i + 1]), "n": int(m.sum()), "pred": float(score[m].mean()),
                        "win": float(y[m].mean()), "pnl_mean": float(pnl[m].mean()), "pnl_sum": float(pnl[m].sum())})
    return out


def train(cfg: Config, deep: bool = False) -> dict:
    """``deep`` : recherche de reglages en profondeur (des centaines d'essais, carte graphique) ;
    sinon quelques essais repartant des meilleurs reglages de la derniere recherche profonde."""
    t0 = time.time()
    _KOL.clear()  # historique KOL a jour a chaque entrainement
    _PERC.clear()
    tc = cfg["training"]
    days = D.raw_days(cfg.data)[-(tc["window_days"] + 1):]
    if not days:
        return {"status": "skip", "reason": "aucune donnee"}
    live = D.load_window(cfg, days)
    n_coins = live.mint.nunique() if not live.empty else 0
    if n_coins < tc["min_train_rows"]:
        return {"status": "skip", "reason": f"pas encore assez de coins avec un resultat connu ({n_coins} / "
                                            f"{tc['min_train_rows']})"}
    emb_store = N.EmbStore(cfg.data / "emb.db")
    embedder = N.Embedder(tc["text_model"], tc["image_model"], tc["device"])
    live, emb = add_context(live, cfg, emb_store, embedder, D.population(cfg, days))
    live["is_backfill"] = 0.0

    # Coins importes par CA (exemples) : entrainement seulement, sans contexte du moment.
    from . import backfill as B
    bf = B.load_rows(cfg)
    if not bf.empty:
        # anciens imports (avant les points d'analyse) : ignores, ils sont reimportes automatiquement
        bf = bf[~bf.mint.isin(live.mint)] if "age_s" in bf else bf.iloc[0:0]
        bf = bf[bf["age_s"].notna()].reset_index(drop=True) if not bf.empty else bf
    if not bf.empty:
        bf_emb = embeddings(cfg, bf, emb_store, embedder)
        bf["img_text_sim"] = np.where(bf_emb["has_img"], (bf_emb["T"] * bf_emb["I"]).sum(1), np.nan)
        add_topics(bf, embedder)
        add_protos(bf, bf_emb, cfg, emb_store, embedder)
        add_fame(bf, bf_emb, cfg, emb_store, embedder)
        add_kol(bf, cfg)
        add_kolnarr(bf, bf_emb, cfg, emb_store, embedder)
        add_percees(bf, bf_emb, cfg, emb_store, embedder)
        # Contexte du moment inconnu pour un coin importe. Le laisser vide permettrait au modele de
        # reconnaitre les imports a ce seul trou : on recopie le contexte d'un coin live tire au hasard.
        donors = np.random.default_rng(0).integers(0, len(live), len(bf))
        for c in CONTEXT:
            bf[c] = live[c].to_numpy()[donors] if c in live else np.nan
        bf["is_backfill"] = 1.0
        df = pd.concat([live, bf], ignore_index=True)
        emb = {k: np.concatenate([emb[k], bf_emb[k]]) for k in emb}
    else:
        df = live
    df = df.copy()  # defragmente (beaucoup de colonnes ajoutees une par une)
    df["y"] = (df.pnl > 0).astype(int)

    # Avis utilisateur : poids x3 (coin live) ou x5 (exemple importe)
    fb = Journal(cfg.data / "journal.db").feedback_df()
    verdict = df.mint.map(dict(zip(fb.mint, fb.verdict))) if not fb.empty else pd.Series(np.nan, index=df.index)
    blind = df.mint.map(dict(zip(fb.mint, fb.resolved_at_rating == 0))) if not fb.empty else pd.Series(False, index=df.index)
    df["verdict"] = verdict
    rated = verdict.notna().to_numpy()
    isbf = df.is_backfill.to_numpy() > 0
    df["w"] = np.where(rated, np.where(isbf, 5.0, 3.0), 1.0)

    # Decoupage temporel PAR COIN (toutes les lignes d'un coin du meme cote) :
    # validation = coins des heures les plus recentes, es = fin du train (reglages + seuil).
    lv = np.where(~isbf)[0]
    c = df.created_ms.to_numpy()
    tmax, tmin = c[lv].max(), c[lv].min()
    # Debut de la validation : au plus valid_hours avant la fin, et au plus valid_frac_max des COINS
    # (quantile sur les coins, robuste a un coin isole tres ancien qui etirerait la plage de dates).
    coin_t = np.unique(c[lv])
    vstart = max(tmax - tc["valid_hours"] * 3600_000, np.quantile(coin_t, 1 - tc["valid_frac_max"]))
    vspan = tmax - vstart
    va_idx = lv[c[lv] >= vstart]
    tr_live = lv[c[lv] < vstart]
    cut_t = np.quantile(np.unique(c[tr_live]), 0.8) if len(tr_live) else 0
    es_idx = tr_live[c[tr_live] >= cut_t]
    fit_idx = np.concatenate([tr_live[c[tr_live] < cut_t], np.where(isbf)[0]])
    mints, ages = df.mint.to_numpy(), df.age_s.to_numpy()
    n_va, n_es, n_fit = (len(set(mints[i])) for i in (va_idx, es_idx, fit_idx))
    if min(n_va, n_es, n_fit) < tc.get("min_split_rows", 30):
        return {"status": "skip", "reason": f"decoupage trop petit (coins : fit {n_fit}, es {n_es}, valid {n_va})"}

    dims = min(tc["embedding_dims"], len(fit_idx) // 4)
    pca_t = PCA(dims, random_state=0).fit(emb["T"][fit_idx])
    img_rows = fit_idx[emb["has_img"][fit_idx]]
    pca_i = PCA(max(2, dims // 2), random_state=0).fit(emb["I"][img_rows] if len(img_rows) > dims else emb["I"][fit_idx])
    E = Predictor.embed_cols(pca_t, pca_i, emb["T"], emb["I"], emb["has_img"])
    X = pd.concat([df.reset_index(drop=True), E], axis=1)
    feats = [k for k in X.columns if k not in NON_FEATURES and pd.api.types.is_numeric_dtype(X[k])]
    if tc.get("feature_groups"):  # familles de signaux autorisees (ex. narrative seule)
        feats = [k for k in feats if S.group_of(k) in tc["feature_groups"]]
    X = X[feats].astype(np.float32)
    y, w, pnl = df.y.to_numpy(), df.w.to_numpy(), df.pnl.to_numpy()
    min_tr = tc["min_valid_trades"]
    min_es = max(3, int(min_tr * n_es / max(1, n_va)))

    # --- recherche de reglages (pumpml/search.py) : chaque essai est entraine sur fit et juge sur es ---
    def thr_fn(score, idx, mn=None):
        return best_threshold(score, pnl[idx], mn or min_es, prev_cummax(score, mints[idx], ages[idx]))

    # Periodes de jugement : chaque essai est aussi entraine sur le debut du train et juge sur les tranches
    # suivantes (avant es). Un reglage doit tenir sur plusieurs periodes, pas sur une seule (gain par chance).
    folds = []
    ut = np.unique(c[tr_live])
    for qa, qb in ((0.5, 0.65), (0.65, 0.8))[: int(tc.get("folds", 3)) - 1]:
        ta, tb = np.quantile(ut, qa), np.quantile(ut, qb)
        ev = tr_live[(c[tr_live] >= ta) & (c[tr_live] < tb)]
        fi = np.concatenate([tr_live[c[tr_live] < ta], np.where(isbf)[0]])
        if len(set(mints[ev])) >= tc.get("min_split_rows", 30):
            folds.append((fi, ev, max(2, int(min_es * len(set(mints[ev])) / max(1, n_es)))))
    res = S.run(tc, cfg.models, X, y, w, pnl, df.created_ms.to_numpy(), isbf, fit_idx, es_idx, mints, ages,
                min_es, deep, None, thr_fn, folds)
    booster, thr, objective = res["ensemble"], res["threshold"], res["objective"]
    trials, best_i = res["trials"], res["best"]

    taste = None
    rated_blind = np.where(rated & blind.fillna(False).to_numpy(bool) & ~isbf)[0]
    rated_blind = rated_blind[np.isin(rated_blind, fit_idx)]
    if len(rated_blind) >= MIN_FEEDBACK_FOR_TASTE and objective != "regression":  # echelles incompatibles
        tp = dict(objective="binary", learning_rate=0.05, num_leaves=7, min_data_in_leaf=5, verbose=-1, seed=0)
        taste = lgb.train(tp, lgb.Dataset(X.iloc[rated_blind], (df.verdict.to_numpy()[rated_blind] > 0).astype(int)), 200)

    def final_of(idx):
        p = booster.predict(X.iloc[idx])
        return p if taste is None else (1 - TASTE_WEIGHT) * p + TASTE_WEIGHT * taste.predict(X.iloc[idx])

    if taste is not None:
        pe = final_of(es_idx)
        thr, _, _ = best_threshold(pe, pnl[es_idx], min_es, prev_cummax(pe, mints[es_idx], ages[es_idx]))

    # --- evaluation sur la validation (jamais vue), avec la vraie regle d'achat ---
    va_final = final_of(va_idx)
    pv, yv = pnl[va_idx], y[va_idx]
    va_pcm = prev_cummax(va_final, mints[va_idx], ages[va_idx])
    va_pnl, va_n = pnl_at(va_final, pv, thr, va_pcm)
    auc = float(roc_auc_score(yv, va_final)) if len(set(yv)) > 1 else float("nan")
    sel = bought(va_final, va_pcm, thr)
    sub = df.iloc[va_idx].reset_index(drop=True)
    dms = sub.decision_ms.to_numpy()
    order = np.argsort(dms)
    curve = np.cumsum(np.where(sel, pv, 0.0)[order])
    step = max(1, len(curve) // 200)
    first_va = (sub.first_cp > 0).to_numpy()
    coin_win = sub[first_va].pnl > 0  # coin "gagnant" = gagnant s'il est achete au premier point

    def coin(k):
        r = sub.iloc[k]
        return {"mint": r.mint, "name": r["name"], "symbol": r.symbol, "score": float(va_final[k]),
                "pnl": float(pv[k]), "max_mult_after": float(r.max_mult_after), "exit": r.exit_reason,
                "age_s": float(r.age_s)}

    bad_buys = [coin(k) for k in np.argsort(-va_final) if sel[k] and pv[k] < 0][:12]
    bought_mints = set(sub.mint[sel])
    best_rows = sub.assign(_k=np.arange(len(sub))).sort_values("pnl", ascending=False).drop_duplicates("mint")
    missed = [coin(k) for k in best_rows._k if sub.mint[k] not in bought_mints and pv[k] > 0][:12]
    metrics = {"valid_pnl": va_pnl, "valid_trades": va_n, "valid_auc": auc, "threshold": thr,
               "valid_buy_all_pnl": float(sub[first_va].pnl.sum()), "valid_tokens": int(n_va),
               "valid_rows": int(len(va_idx)),
               "valid_win_rate_selected": float((pv[sel] > 0).mean()) if va_n else None,
               "valid_buy_age_s": float(sub.age_s[sel].mean()) if va_n else None,
               "base_win_rate": float(coin_win.mean()) if len(coin_win) else 0.0,
               "train_tokens": int(n_fit + n_es),
               "backfill_used": int(df.mint[isbf].nunique()), "feedback_used": int(df.mint[rated].nunique()),
               "taste_model": taste is not None, "best_iter": int(booster.members[0].it or 0),
               "valid_hours": round(vspan / 3600_000, 2), "data_hours": round((tmax - tmin) / 3600_000, 1),
               "trials": res["n_trials"], "best_trial": best_i, "objective": objective, "deep": deep,
               "family": res["params"]["family"], "ensemble_size": res["ensemble_size"], "train_seconds": round(time.time() - t0, 1),
               "copies_in_valid": float((sub[first_va].dup_strong > 0).mean()) if "dup_strong" in sub else None}

    reg = Registry(cfg.models)
    champ = reg.champion()
    champ_pnl = None
    if champ is not None and champ.meta.get("embedder") != embedder.tag:
        log.warning("champion %s entraine avec d'autres embeddings : comparaison impossible", champ.id)
        champ = None
    if champ is not None:
        _, _, cf = champ.score(sub, {k: emb[k][va_idx] for k in emb})
        champ_pnl, champ_n = pnl_at(cf, pv, champ.threshold, prev_cummax(cf, mints[va_idx], ages[va_idx]))
        metrics.update(champion=champ.id, champion_valid_pnl=champ_pnl, champion_valid_trades=champ_n,
                       champion_valid_auc=float(roc_auc_score(yv, cf)) if len(set(yv)) > 1 else None)

    ok = va_n >= min_tr and va_pnl > 0
    promote = ok and (champ_pnl is None or va_pnl > champ_pnl + tc["min_improvement_sol"])
    metrics["promoted"] = bool(promote)

    mid = time.strftime("%Y%m%d-%H%M%S")
    mdir = cfg.models / mid
    mdir.mkdir(parents=True)
    members = booster.save(mdir)
    if taste is not None:
        taste.save_model(str(mdir / "taste.txt"))
    joblib.dump((pca_t, pca_i), mdir / "pca.joblib")
    imp = sorted(zip(feats, booster.importance(feats)), key=lambda x: -x[1])
    meta = {"features": feats, "threshold": thr, "metrics": metrics, "embedder": embedder.tag, "members": members,
            "importance": [(f, float(g)) for f, g in imp[:40]], "trials": trials[:60],
            "bins": _bins(va_final, pv, yv), "bad_buys": bad_buys, "missed": missed,
            "valid_curve": [[int(dms[order][k]), float(curve[k])] for k in range(0, len(curve), step)],
            "params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in res["params"].items()}}
    (mdir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    reg.state["history"].append({"id": mid, **metrics})
    if promote:
        reg.state["champion"] = mid
        log.info("NOUVEAU CHAMPION %s", mid)
    reg.save()
    keep = {h["id"] for h in reg.state["history"][-KEEP_MODELS:]} | {reg.state.get("champion")}
    for p in cfg.models.iterdir():
        if p.is_dir() and p.name not in keep:
            shutil.rmtree(p, ignore_errors=True)
    log.info("entrainement %s : %s", mid, json.dumps(metrics, default=str))
    return {"status": "ok", "id": mid, **metrics}
