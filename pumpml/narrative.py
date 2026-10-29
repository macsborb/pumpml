"""Narrative : embeddings texte + image (CLIP, sur GPU) et features de tendance.

Le texte (multilingue) et l'image sont encodes dans le MEME espace CLIP, ce qui permet
de mesurer si l'image colle au nom, et de retrouver les coins passes qui ressemblent.
Les embeddings sont mis en cache dans data/emb.db (un par mint, calcule une seule fois).
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
DIM = 512


class Embedder:
    def __init__(self, text_model: str, image_model: str, device: str = "cuda"):
        self.text_name, self.image_name, self.device = text_model, image_model, device
        self._t = self._i = None
        self.backend = None

    def _load(self):
        if self.backend:
            return
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            dev = self.device if (self.device != "cuda" or torch.cuda.is_available()) else "cpu"
            self._t = SentenceTransformer(self.text_name, device=dev)
            self._i = SentenceTransformer(self.image_name, device=dev)
            self.backend = f"clip:{dev}"
        except Exception as e:  # pas de torch : repli sans GPU, texte seulement
            log.warning("CLIP indisponible (%s) : repli sur un hachage de n-grammes", e)
            from sklearn.feature_extraction.text import HashingVectorizer
            self._t = HashingVectorizer(n_features=DIM, analyzer="char_wb", ngram_range=(2, 4), norm="l2",
                                        alternate_sign=False)
            self.backend = "hash"
        log.info("embeddings : %s", self.backend)

    @property
    def tag(self) -> str:
        self._load()
        return self.backend.split(":")[0]

    def text(self, texts: list[str]) -> np.ndarray:
        self._load()
        if self.backend == "hash":
            return self._t.transform(texts).toarray().astype(np.float32)
        return self._t.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False).astype(np.float32)

    def images(self, paths: list[Path]) -> np.ndarray:
        """Une ligne de zeros pour chaque image manquante ou illisible."""
        self._load()
        out = np.zeros((len(paths), DIM), np.float32)
        if self.backend == "hash":
            return out
        from PIL import Image
        imgs, idx = [], []
        for k, p in enumerate(paths):
            try:
                imgs.append(Image.open(p).convert("RGB"))
                idx.append(k)
            except Exception:
                pass
        if imgs:
            e = self._i.encode(imgs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=False)
            out[idx] = e
        return out


# Themes : chaque texte (coin, tweet) est compare a ces descriptions dans l'espace du modele texte
# multilingue. Score = ressemblance au theme moins la ressemblance moyenne a tous les themes.
TOPICS = {
    "monde": ["actualite mondiale majeure, geopolitique, guerre, pays", "breaking world news, war, global event"],
    "politique": ["politique, president, gouvernement, election, Trump", "politics, president, government, election"],
    "drame": ["catastrophe, accident, mort, tragedie, attentat, crash", "disaster, death, tragedy, attack, accident"],
    "celebrite": ["celebrite, star, influenceur, rappeur, streamer", "celebrity, famous influencer, rapper, streamer"],
    "animal": ["animal mignon, chien, chat, singe, hamster, pingouin", "cute animal, dog, cat, monkey, frog, penguin"],
    "crypto": ["crypto, bitcoin, solana, memecoin, pump", "crypto, bitcoin, solana, memecoin, token launch"],
    "tech_ia": ["intelligence artificielle, technologie, robot, Elon Musk", "artificial intelligence, AI agent, tech, robot"],
    "sport": ["sport, football, match, joueur, coupe", "sports, football, match, player, championship"],
    "humour": ["meme, blague, humour, absurde, troll", "funny meme, joke, absurd humor, troll"],
    "finance": ["bourse, economie, banque, dollar, inflation", "stock market, economy, bank, dollar"],
    "viral": ["video virale, tendance internet, TikTok, buzz", "viral video, internet trend, TikTok"],
}
GRAVITY = (["actualite majeure tres grave et urgente dans le monde entier", "huge breaking news, major global event"],
           ["petite anecdote sans importance", "minor trivial unimportant story"])


def topic_features(emb: "Embedder", coin_texts: list[str], tweet_texts: list[str]) -> dict[str, np.ndarray]:
    """Scores de themes + gravite pour le texte du coin et pour le tweet (NaN si pas de tweet)."""
    if not hasattr(emb, "_topic_cache"):
        def mean_emb(prompts):
            e = emb.text(prompts).mean(0)
            return e / (np.linalg.norm(e) + 1e-9)
        names = list(TOPICS)
        emb._topic_cache = (names, np.stack([mean_emb(TOPICS[k]) for k in names]),
                            mean_emb(GRAVITY[0]) - mean_emb(GRAVITY[1]))
    names, P, G = emb._topic_cache
    out: dict[str, np.ndarray] = {}
    embs = {}
    for prefix, texts in (("coin", coin_texts), ("tw", tweet_texts)):
        texts = [t or "" for t in texts]
        has = np.array([bool(t.strip()) for t in texts])
        uniq = sorted({t for t in texts if t.strip()} | {"-"})
        idx = {t: k for k, t in enumerate(uniq)}
        U = emb.text(uniq)  # chaque texte distinct n'est encode qu'une fois
        E = U[[idx[t] if t.strip() else idx["-"] for t in texts]]
        sims = E @ P.T
        rel = sims - sims.mean(1, keepdims=True)
        for k, n in enumerate(names):
            out[f"{prefix}_th_{n}"] = np.where(has, rel[:, k], np.nan).astype(np.float32)
        out[f"{prefix}_gravite"] = np.where(has, E @ G, np.nan).astype(np.float32)
        embs[prefix] = (E, has)
    # le coin parle-t-il de la meme chose que le tweet ? (nom / description vs texte du tweet)
    (Ec, hc), (Et, ht) = embs["coin"], embs["tw"]
    out["coin_tweet_sim"] = np.where(hc & ht, (Ec * Et).sum(1), np.nan).astype(np.float32)
    return out


class EmbStore:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.lock = threading.Lock()
        with self.db:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("CREATE TABLE IF NOT EXISTS emb(mint TEXT PRIMARY KEY, recv_ms INTEGER, backend TEXT,"
                            " text BLOB, img BLOB, has_img INTEGER)")

    def get(self, mints: list[str], backend: str) -> dict[str, tuple[np.ndarray, np.ndarray, bool]]:
        out = {}
        with self.lock:
            for i in range(0, len(mints), 900):
                chunk = mints[i:i + 900]
                q = f"SELECT mint, text, img, has_img FROM emb WHERE backend=? AND mint IN ({','.join('?' * len(chunk))})"
                for m, t, im, h in self.db.execute(q, [backend, *chunk]):
                    out[m] = (np.frombuffer(t, np.float16).astype(np.float32),
                              np.frombuffer(im, np.float16).astype(np.float32), bool(h))
        return out

    def put(self, rows: list[tuple[str, int, str, np.ndarray, np.ndarray, bool]]):
        with self.lock, self.db:
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES(?,?,?,?,?,?)",
                                [(m, t, b, te.astype(np.float16).tobytes(), ie.astype(np.float16).tobytes(), int(h))
                                 for m, t, b, te, ie, h in rows])


def ensure_embeddings(store: EmbStore, emb: Embedder, items: list[dict], data_root: Path,
                      batch: int = 256) -> dict[str, tuple[np.ndarray, np.ndarray, bool]]:
    """items : dicts avec mint, recv_ms, text, image_path (relatif a data/). Calcule ce qui manque."""
    backend = emb.tag
    for it in items:
        it.setdefault("key", it["mint"])
    have = store.get(list({it["key"] for it in items}), backend)
    todo = list({it["key"]: it for it in items if it["key"] not in have}.values())
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        te = emb.text([it["text"] or "" for it in part])
        paths = [data_root / it["image_path"] if it.get("image_path") else Path("__none__") for it in part]
        ie = emb.images(paths)
        rows = [(it["key"], it["recv_ms"], backend, te[k], ie[k], bool(ie[k].any())) for k, it in enumerate(part)]
        store.put(rows)
        for m, _, _, t, im, h in rows:
            have[m] = (t, im, h)
    if todo:
        log.info("embeddings calcules : %d", len(todo))
    return have


def emb_key(mint: str, text: str | None, image_path: str | None) -> str:
    """Cle de cache : le contenu connu peut changer entre deux points d'analyse (image, tweet arrives)."""
    import hashlib
    h = hashlib.md5(f"{text or ''}|{image_path or ''}".encode()).hexdigest()[:12]
    return f"{mint}|{h}"


def trend_features_at(c_created: np.ndarray, c_mint: np.ndarray, c_T: np.ndarray, c_pnl: np.ndarray,
                      c_resolve: np.ndarray, q_time: np.ndarray, q_mint: np.ndarray, q_T: np.ndarray
                      ) -> dict[str, np.ndarray]:
    """Tendance de narrative pour chaque requete (coin a un point d'analyse, a l'instant ``q_time``) :
    * sim_1h : coins (autres que lui) sortis dans l'heure precedente avec cos > 0.85 ;
    * nn_pnl / nn_win / nn_sim : 20 plus proches voisins des 6 h precedentes dont le resultat etait connu."""
    order = np.argsort(c_created)
    c_created, c_mint, c_T, c_pnl, c_resolve = (a[order] for a in (c_created, c_mint, c_T, c_pnl, c_resolve))
    n = len(q_time)
    out = {k: np.full(n, np.nan, np.float32) for k in ("sim_1h", "nn_pnl", "nn_win", "nn_sim")}
    for i in range(n):
        t = q_time[i]
        a = int(np.searchsorted(c_created, t - 6 * 3600_000))
        b = int(np.searchsorted(c_created, t, side="left"))
        if b <= a:
            out["sim_1h"][i] = 0.0
            continue
        other = c_mint[a:b] != q_mint[i]
        sims = c_T[a:b] @ q_T[i]
        recent = c_created[a:b] >= t - 3600_000
        out["sim_1h"][i] = float(((sims > 0.85) & recent & other).sum())
        known = (c_resolve[a:b] <= t) & other & ~np.isnan(c_pnl[a:b])
        if known.sum() >= 5:
            s, p = sims[known], c_pnl[a:b][known]
            top = np.argsort(s)[-20:]
            out["nn_pnl"][i] = float(p[top].mean())
            out["nn_win"][i] = float((p[top] > 0).mean())
            out["nn_sim"][i] = float(s[top].mean())
    return out


def trend_features(created_ms: np.ndarray, text: np.ndarray, pnl: np.ndarray, resolve_ms: np.ndarray,
                   targets: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Features de tendance de narrative, sans fuite du futur.

    Pour chaque token i (decision a created_ms[i]) :
      * sim_1h        : nb de tokens crees dans l'heure precedente avec cos > 0.85 (copies, vague)
      * nn_pnl / nn_win / nn_sim : PnL moyen, taux de gain et similarite des 20 voisins les plus
        proches parmi les tokens des 6 h precedentes dont le resultat etait CONNU (resolve_ms <= decision)
    Les tableaux doivent etre tries par created_ms. ``targets`` : indices a calculer (defaut : tous).
    """
    n = len(created_ms)
    idx = np.arange(n) if targets is None else targets
    out = {k: np.full(len(idx), np.nan, np.float32) for k in ("sim_1h", "nn_pnl", "nn_win", "nn_sim")}
    for o, i in enumerate(idx):
        t = created_ms[i]
        a = int(np.searchsorted(created_ms, t - 6 * 3600_000))
        b = int(np.searchsorted(created_ms, t, side="left"))
        if b <= a:
            continue
        sims = text[a:b] @ text[i]
        h = int(np.searchsorted(created_ms[a:b], t - 3600_000))
        out["sim_1h"][o] = float((sims[h:] > 0.85).sum())
        known = resolve_ms[a:b] <= t
        if known.sum() >= 5:
            s, p = sims[known], pnl[a:b][known]
            top = np.argsort(s)[-20:]
            out["nn_pnl"][o] = float(p[top].mean())
            out["nn_win"][o] = float((p[top] > 0).mean())
            out["nn_sim"][o] = float(s[top].mean())
    return out
