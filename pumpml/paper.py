"""Paper trader en direct : analyse chaque nouveau coin des sa sortie, a plusieurs points d'analyse
(5 s, 10 s, ... 7 min), et achete au PREMIER point ou le score depasse le seuil. Apres 7 min, plus
aucun achat sur ce coin.

Lit le hot store du collecteur, calcule les features avec EXACTEMENT les memes fonctions que
l'entrainement, note avec le champion (ou le dernier modele), et ecrit tout dans data/journal.db
(lu par le dashboard). Les prix d'achat et de vente viennent de vraies cotations Jupiter.
Chaque coin est "resolu" une fois son resultat connu (achat simule a son premier point d'analyse),
y compris s'il n'a pas ete achete (opportunites manquees).
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from . import dataset as D
from . import features as F
from . import narrative as N
from . import protos as P
from . import pump
from . import vamp as V
from .config import Config
from .feedback import Journal
from .jupiter import Jupiter
from .model import Registry
from .storage import LONG_OUTAGE_MS, HotStore, now_ms, overlaps

log = logging.getLogger("paper")

TRADE_COLS = "recv_ms, slot, is_buy, sol_amount, token_amount, user, vsol, vtok, real_tok, real_sol, ts"
# Creation en heure on-chain (sinon reception moins le retard typique du flux).
CREATED = "COALESCE(chain_ms, recv_ms - 2000)"
BUY_RESERVE = 12  # cotations Jupiter par minute gardees pour les achats


class Paper:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.s = cfg["strategy"]
        self.cps = D.checkpoints_ms(self.s)          # points d'analyse (ms depuis la creation)
        self.lat = int(self.s["latency_s"] * 1000)
        self.hold = D.hold_ms(self.s)                 # du point d'analyse au resultat connu
        self.hot = HotStore(cfg.data / "hot.db")
        self.j = Journal(cfg.data / "journal.db")
        self.reg_mtime = 0.0
        self.model = None
        self.done: dict[str, set[int]] = {}           # mint -> points deja analyses
        self.closed: set[str] = set()                 # achete, exclu, ou 7 min passees
        self.counts: dict[str, int] = {}
        t = cfg["training"]
        self.emb_store = N.EmbStore(cfg.data / "emb.db")
        self.embedder = N.Embedder(t["text_model"], t["image_model"], t["device"])
        self.creator_hist = self._load_creator_history()
        self.hc = V.HashCache(cfg.data / "emb.db")
        self.jup = Jupiter(cfg.secrets.jupiter_key, cfg["paper"]["slippage_bps"])
        self.lib, self.lib_at = None, 0.0  # bibliotheque de narratives (tes exemples), rechargee toutes les 5 min
        self.kol, self.kol_at = None, 0.0
        self.pc, self.pc_at = None, 0.0      # memoire des coins perces (~30 000 $), rechargee toutes les 15 min
        self.kn, self.kn_at = None, 0.0      # memoire « ce que les gros KOL achetent », rechargee toutes les 15 min    # KOL : historique et flair, recharge toutes les 15 min
        self.fame, self.fame_at = None, 0.0  # memoire des gagnants (tout l'historique), rechargee toutes les 15 min
        self.mem, self.ing_tok, self.ing_meta = V.Memory(), 0, 0
        self._load_memory()
        for r in self.j.query("SELECT mint FROM decisions WHERE action='buy'"):
            self.closed.add(r["mint"])
        # apres migration : observation seulement (verdict anti-rug, cotation Jupiter, resultat simule)
        self.amm = None
        if cfg["amm"].get("enabled"):
            from .amm import Observer
            self.amm = Observer(cfg, self.hot, self.j, self.jup, BUY_RESERVE)

    def coin_resolve_ms(self, created: int, first_age_s: float | None) -> int:
        """Instant ou le resultat "du coin" (achat a son premier point d'analyse) est connu."""
        first = int((first_age_s or self.cps[0] / 1000) * 1000)
        return created + first + self.hold

    # ---------- memoire, historiques ----------
    def _load_memory(self):
        """Memoire des copies : tous les coins des 7 derniers jours (archive) + resultats connus."""
        days = D.raw_days(self.cfg.data)[-8:]
        pop = D.population(self.cfg, days)
        for r in pop.itertuples(index=False):
            self.mem.add(r.mint, int(r.recv_ms), r.name, r.symbol)
            if pd.notna(r.meta_ms):
                img = r.image_path if isinstance(r.image_path, str) else None
                self.mem.set_meta(r.mint, int(r.meta_ms), r.twitter if isinstance(r.twitter, str) else None,
                                  self.hc.get(self.cfg.data, img))
        for d in D.complete_days(self.cfg)[-8:]:
            p = D.feature_path(self.cfg, d)
            if p.exists():
                f = pd.read_parquet(p, columns=["mint", "resolve_ms", "entry_mult", "max_mult_after", "pnl",
                                                "migrated", "first_cp"])
                for r in f[f.first_cp > 0].itertuples(index=False):
                    self.mem.set_outcome(r.mint, int(r.resolve_ms), r.entry_mult * r.max_mult_after, r.pnl, r.migrated)
        for r in self.j.query("SELECT mint, created_ms, first_age_s, pnl_sim, max_mult_after, entry_mult_sim,"
                              " exit_reason_sim FROM decisions WHERE resolved=1 AND pnl_sim IS NOT NULL"):
            self.mem.set_outcome(r["mint"], self.coin_resolve_ms(r["created_ms"], r["first_age_s"]),
                                 (r["entry_mult_sim"] or 1.0) * (r["max_mult_after"] or 1.0), r["pnl_sim"],
                                 r["exit_reason_sim"] == "migrated")
        log.info("memoire des copies : %d coins", len(self.mem.entries))

    def _ingest(self):
        """Ajoute a la memoire les coins et metadonnees arrives depuis le dernier passage."""
        for r in self.hot.query(f"SELECT mint, recv_ms, {CREATED} AS c, name, symbol FROM tokens WHERE recv_ms > ?"
                                " ORDER BY recv_ms", (self.ing_tok,)):
            self.mem.add(r["mint"], r["c"], r["name"], r["symbol"])
            self.ing_tok = max(self.ing_tok, r["recv_ms"])
        for r in self.hot.query("SELECT mint, recv_ms, twitter, website, image_path FROM metadata WHERE recv_ms > ?"
                                " ORDER BY recv_ms", (self.ing_meta,)):
            self.mem.set_meta(r["mint"], r["recv_ms"], F.x_link(dict(r)), self.hc.get(self.cfg.data, r["image_path"]))
            self.ing_meta = max(self.ing_meta, r["recv_ms"])

    def _load_creator_history(self) -> pd.DataFrame:
        days = D.complete_days(self.cfg)[-self.cfg["training"]["window_days"]:]
        parts = []
        for d in days:
            p = D.feature_path(self.cfg, d)
            if p.exists():
                f = pd.read_parquet(p, columns=["creator", "created_ms", "resolve_ms", "pnl", "migrated", "first_cp"])
                parts.append(f[f.first_cp > 0].drop(columns="first_cp"))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["creator", "created_ms", "resolve_ms", "pnl", "migrated"])

    def _reload_model(self):
        f = self.cfg.models / "registry.json"
        if f.exists() and f.stat().st_mtime != self.reg_mtime:
            self.reg_mtime = f.stat().st_mtime
            reg = Registry(self.cfg.models)
            # Sans champion, on note quand meme avec le dernier modele (marque "non promu" au dashboard).
            self.model = reg.champion() or reg.latest()
            if self.model and "age_s" not in self.model.features and not self.model.meta.get("members"):
                log.info("modele %s anterieur aux points d'analyse : observation seulement", self.model.id)
                self.model = None
            if self.model:
                log.info("modele charge : %s (%s, seuil %.3f)", self.model.id,
                         "champion" if self.model.id == reg.state.get("champion") else "non promu", self.model.threshold)

    def trades(self, mint: str) -> F.Trades:
        rows = self.hot.query(f"SELECT {TRADE_COLS} FROM trades WHERE mint=?", (mint,))
        return F.Trades.from_rows([dict(r) for r in rows])

    def watermark(self) -> int:
        """Jusqu'ou (heure on-chain) les trades sont arrives."""
        if self.cfg["collector"].get("trades_source") == "per_token":
            # flux coin par coin : complet jusqu'a maintenant - latence tant qu'il est connecte et a jour
            r = self.hot.query("SELECT ms, info FROM heartbeat WHERE name='tokstream'")
            if r:
                st = json.loads(r[0]["info"] or "{}")
                if st.get("connected") and not st.get("pending_gapfill") and now_ms() - r[0]["ms"] < 20_000:
                    return now_ms() - self.lat
                return min(st.get("down_since") or r[0]["ms"], r[0]["ms"]) - self.lat
        r = self.hot.query("SELECT MAX(ts) t FROM trades")
        return int(r[0]["t"]) * 1000 if r and r[0]["t"] else now_ms() - 6_000

    def outages(self) -> list[tuple[int, int]]:
        return [(r["start_ms"], r["end_ms"]) for r in self.hot.query("SELECT start_ms, end_ms FROM outages")]

    def _creator_feats(self, creator: str, created: int) -> dict:
        live = pd.DataFrame(self.j.query(
            "SELECT creator, created_ms, created_ms + COALESCE(first_age_s, ?) * 1000 + ? AS resolve_ms,"
            " pnl_sim AS pnl, CASE WHEN exit_reason_sim='migrated' THEN 1.0 ELSE 0.0 END AS migrated FROM decisions"
            " WHERE creator=? AND created_ms < ?", (self.cps[0] / 1000, self.hold, creator, created)))
        h = self.creator_hist
        hist = pd.concat([h[(h.creator == creator) & (h.created_ms < created)], live], ignore_index=True)
        hist = hist.drop_duplicates(["creator", "created_ms"])
        known = hist[(hist.resolve_ms <= created) & hist.pnl.notna()]
        return {"creator_prev_tokens": float(len(hist)),
                "creator_prev_pnl": float(known.pnl.mean()) if len(known) else np.nan,
                "creator_prev_migrated": float(known.migrated.mean()) if len(known) else np.nan}

    def _kol(self, mint: str, created: int, at_ms: int) -> dict:
        """KOL acheteurs du coin avant la decision (flux PumpPortal en direct) et leur flair passe."""
        if not self.cfg.raw.get("kol", {}).get("enabled"):
            return {}
        from . import kol as K
        if self.kol is None or time.monotonic() - self.kol_at > 900:
            self.kol, self.kol_at = K.build(self.cfg), time.monotonic()
        wallets, _ = K.kols(self.cfg)
        try:
            rows = [dict(r) for r in self.hot.query("SELECT signature, mint, recv_ms, wallet, is_buy, sol FROM kol_trades"
                                                    " WHERE mint=? AND recv_ms<=?", (mint, at_ms))]
        except Exception:  # table pas encore creee par le collecteur
            rows = []
        tr = pd.DataFrame(rows, columns=["signature", "mint", "recv_ms", "wallet", "is_buy", "sol"])
        tr["is_buy"] = tr.is_buy.astype(bool)
        tr["person"] = tr.wallet.map(wallets)
        return {k: float(v) for k, v in self.kol.features(mint, created, at_ms, tr).items()}

    def _trend(self, mint: str, at_ms: int, t_emb: np.ndarray) -> dict:
        """Tendance de narrative au moment de l'analyse (meme calcul qu'a l'entrainement)."""
        rows = self.j.query("SELECT mint, created_ms, first_age_s, pnl_sim, text, image_path FROM decisions"
                            " WHERE created_ms >= ? AND created_ms < ? AND mint != ?",
                            (at_ms - 6 * 3600_000, at_ms, mint))
        empty = {"sim_1h": 0.0, "nn_pnl": np.nan, "nn_win": np.nan, "nn_sim": np.nan}
        if not rows:
            return empty
        keys = [N.emb_key(r["mint"], r["text"], r["image_path"]) for r in rows]
        embs = self.emb_store.get(keys, self.embedder.tag)
        rows = [(r, k) for r, k in zip(rows, keys) if k in embs]
        if not rows:
            return empty
        out = N.trend_features_at(
            np.array([r["created_ms"] for r, _ in rows], np.int64), np.array([r["mint"] for r, _ in rows], object),
            np.stack([embs[k][0] for _, k in rows]),
            np.array([r["pnl_sim"] if r["pnl_sim"] is not None else np.nan for r, _ in rows]),
            np.array([self.coin_resolve_ms(r["created_ms"], r["first_age_s"]) for r, _ in rows], np.int64),
            np.array([at_ms], np.int64), np.array([mint], object), t_emb[None])
        return {k: float(v[0]) for k, v in out.items()}

    def _copy_veto(self, mrow: dict | None, srow: dict | None, orig: dict) -> str | None:
        """Regle block_copies (F.copy_veto) : compare le tweet de la copie a celui de l'original."""
        o = self.hot.query("SELECT json FROM social WHERE mint=?", (orig["mint"],))
        otext = (json.loads(o[0]["json"]) or {}).get("tweet_text") if o else None
        text = (srow or {}).get("tweet_text")
        sim = None
        if otext and text:
            a, b = self.embedder.text([text[:600], otext[:600]])
            sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        return F.copy_veto(mrow, srow, orig, otext, sim, self.s.get("copy_max_tweet_sim", 0.9))

    # ---------- analyse d'un coin a un point d'analyse ----------
    def decide(self, tok: dict, cp: int, last: bool):
        mint = tok["mint"]
        tr = self.trades(mint)
        created = F.created_chain_ms(tok, tr)
        tok["recv_ms"] = created  # heure on-chain, comme a l'entrainement
        decision_ms = created + cp
        cutoff = decision_ms + self.lat  # infos recues (heure murale) utilisables, comme a l'entrainement
        mig = self.hot.query("SELECT recv_ms FROM migrations WHERE mint=?", (mint,))
        reason = None
        if mig and mig[0]["recv_ms"] <= decision_ms:
            reason = "migre"
        elif not F.is_standard_curve(tr.upto(decision_ms)):
            reason = "courbe"
        elif overlaps(self.outages(), created, cutoff):
            reason = "coupure"
        else:
            m0 = self.hot.query("SELECT * FROM metadata WHERE mint=? AND recv_ms <= ?", (mint, cutoff))
            s0 = self.hot.query("SELECT json FROM social WHERE mint=? AND recv_ms <= ?", (mint, cutoff))
            m0, s0 = (dict(m0[0]) if m0 else None), (json.loads(s0[0]["json"]) if s0 else None)
            if self.s.get("require_link") and m0 is None:
                return  # metadonnees pas encore la : on revient au point d'analyse suivant
            reason = F.excluded(tok, m0, tr.upto(decision_ms), self.s, s0)
            if not reason and F.tweet_pending(tok, m0, s0, decision_ms, self.s):
                self.counts["attente_tweet"] = self.counts.get("attente_tweet", 0) + 1
                return  # tweet frais pas encore lu : on revient au point d'analyse suivant
        if reason:
            self.closed.add(mint)  # memes exclusions qu'a l'entrainement (et plus jouable ensuite)
            self.counts[f"exclu_{reason}"] = self.counts.get(f"exclu_{reason}", 0) + 1
            return
        self.counts["analyses"] = self.counts.get("analyses", 0) + 1
        m = self.hot.query("SELECT * FROM metadata WHERE mint=?", (mint,))
        mrow = dict(m[0]) if m and m[0]["recv_ms"] <= cutoff else None
        sr = self.hot.query("SELECT recv_ms, json FROM social WHERE mint=?", (mint,))
        srow = json.loads(sr[0]["json"]) if sr and sr[0]["recv_ms"] <= cutoff else None
        f = {"age_s": cp / 1000}
        f.update(F.flow_features(created, tok["creator"], tr, decision_ms))
        f.update(F.meta_features(tok, mrow, srow))
        f.update(F.time_features(created))
        f.update(self._creator_feats(tok["creator"], created))
        # activite du createur sur TOUS les coins vus (meme calcul qu'a l'entrainement)
        cr = self.hot.query(f"SELECT {CREATED} AS c FROM tokens WHERE creator=? AND {CREATED} < ? AND {CREATED} >= ?",
                            (tok["creator"], created, created - 6 * 3_600_000)) if tok.get("creator") else []
        f["creator_n_6h"] = float(len(cr))
        f["creator_n_1h"] = float(sum(1 for r in cr if r["c"] >= created - 3_600_000))
        text = F.text_of(tok, mrow, srow)
        img = (mrow or {}).get("image_path")
        key = N.emb_key(mint, text, img)
        e = N.ensure_embeddings(self.emb_store, self.embedder,
                                [{"mint": mint, "key": key, "recv_ms": created, "text": text, "image_path": img}],
                                self.cfg.data)[key]
        f["img_text_sim"] = float(e[0] @ e[1]) if e[2] else np.nan
        tops = N.topic_features(self.embedder, [F.coin_text(tok, mrow)], [((srow or {}).get("tweet_text") or "")[:600]])
        f.update({k: float(v[0]) for k, v in tops.items()})
        f.update(self._trend(mint, cutoff, e[0]))
        if self.lib is None or time.monotonic() - self.lib_at > 300:
            self.lib, self.lib_at = P.build(self.cfg, self.emb_store, self.embedder, D.horizon_ms(self.s)[1]), time.monotonic()
        pf = self.lib.features(np.array([mint], object), np.array([cutoff], np.int64), e[0][None], e[1][None],
                               np.array([e[2]]))
        f.update({k: float(v[0]) for k, v in pf.items()})
        if self.fame is None or time.monotonic() - self.fame_at > 900:
            from . import fame as FM
            self.fame, self.fame_at = FM.build(self.cfg, self.emb_store, self.embedder), time.monotonic()
        ff = self.fame.features(np.array([mint], object), np.array([tok.get("creator")], object),
                                np.array([tok.get("name")], object), np.array([tok.get("symbol")], object),
                                e[0][None], e[1][None], np.array([e[2]]), np.array([cutoff], np.int64))
        f.update({k: float(v[0]) for k, v in ff.items()})
        f.update(self._kol(mint, created, cutoff))
        if self.cfg.raw.get("kol", {}).get("enabled"):
            if self.kn is None or time.monotonic() - self.kn_at > 900:
                from . import kolnarr as KN
                self.kn, self.kn_at = KN.build(self.cfg, self.emb_store, self.embedder), time.monotonic()
            kf = self.kn.features(np.array([mint], object), np.array([tok.get("creator")], object),
                                  np.array([tok.get("name")], object), np.array([tok.get("symbol")], object),
                                  e[0][None], e[1][None], np.array([e[2]]), np.array([cutoff], np.int64))
            f.update({k: float(v[0]) for k, v in kf.items()})
        if "percees" in self.cfg["training"]["feature_groups"]:
            if self.pc is None or time.monotonic() - self.pc_at > 900:
                from . import percees as PC
                self.pc, self.pc_at = PC.build(self.cfg, self.emb_store, self.embedder), time.monotonic()
            pcf = self.pc.features(np.array([mint], object), np.array([tok.get("creator")], object), e[0][None],
                                   e[1][None], np.array([e[2]]), np.array([cutoff], np.int64))
            f.update({k: float(v[0]) for k, v in pcf.items()})
        vf, vinfo = self.mem.features(mint, created, tok.get("name"), tok.get("symbol"), F.x_link(mrow),
                                      self.hc.get(self.cfg.data, img), cutoff)
        f.update(vf)
        if vinfo:
            orig = self.j.query("SELECT name, symbol FROM decisions WHERE mint=?", (vinfo["mint"],))
            orig = orig or self.hot.query("SELECT name, symbol FROM tokens WHERE mint=?", (vinfo["mint"],))
            if orig:
                vinfo.update(name=orig[0]["name"], symbol=orig[0]["symbol"])
        score = taste = final = None
        action, reasons, thr = "observe", [], None
        if self.model is not None:
            row = pd.DataFrame([f])
            emb = {"T": e[0][None], "I": e[1][None], "has_img": np.array([e[2]])}
            p, t, fin = self.model.score(row, emb)
            score, final, thr = float(p[0]), float(fin[0]), self.model.threshold
            taste = float(t[0]) if t is not None else None
            action = "buy" if final >= thr else ("skip" if last else "watch")
            reasons = self.model.reasons(row, emb)[0]
        elif not last:
            action = "watch"
        if action == "buy" and self.s.get("block_copies") and vinfo and vinfo.get("orig"):
            veto = self._copy_veto(mrow, srow, vinfo["orig"])
            if veto:
                vinfo["veto"] = veto
                action = "skip"  # la copie ne change plus : coin ferme
                self.closed.add(mint)
                self.counts["veto_copie"] = self.counts.get("veto_copie", 0) + 1
                log.info("COPIE NON ACHETEE %s (%s) score=%.3f : %s", tok.get("symbol"), mint[:8], final, veto)
        # historique des scores du coin (un point par analyse)
        prev = self.j.query("SELECT scores, first_age_s FROM decisions WHERE mint=?", (mint,))
        scores = json.loads(prev[0]["scores"] or "[]") if prev else []
        scores.append([cp / 1000, final])
        first_age = prev[0]["first_age_s"] if prev and prev[0]["first_age_s"] is not None else cp / 1000
        self.j.exec("INSERT OR REPLACE INTO decisions(mint, decided_ms, created_ms, model_id, score, taste, final,"
                    " threshold, action, name, symbol, image_path, twitter, description, mcap_sol, features,"
                    " top_reasons, text, creator, vamp, age_s, first_age_s, scores, resolved)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
                    (mint, now_ms(), created, self.model.id if self.model else None, score, taste, final, thr,
                     action, tok.get("name"), tok.get("symbol"), img, F.x_link(mrow),
                     ((mrow or {}).get("description") or "")[:500], f.get("mcap_sol"),
                     json.dumps({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in f.items()}),
                     json.dumps(reasons), text, tok.get("creator"), json.dumps(vinfo) if vinfo else None,
                     cp / 1000, first_age, json.dumps(scores)))
        if action == "buy":
            self.closed.add(mint)
            self.counts["achats"] = self.counts.get("achats", 0) + 1
            log.info("BUY %s (%s) a %ds score=%.3f", tok.get("symbol"), mint[:8], cp // 1000, final)
            self.open_position(mint, decision_ms)
        elif last:
            self.closed.add(mint)

    # ---------- positions simulees (prix Jupiter) ----------
    def open_position(self, mint: str, decision_ms: int):
        """Achat simule au prix d'une vraie cotation Jupiter (SOL -> token), prise maintenant.
        Pas de cotation = pas d'achat (jamais de repli sur un calcul de courbe)."""
        cost = self.s["buy_sol"] + 2 * self.s["tx_cost_sol"]
        n_open = self.j.query("SELECT COUNT(*) n FROM positions WHERE status='open'")[0]["n"]
        if n_open >= self.cfg["paper"]["max_open_positions"]:
            self.j.exec("INSERT OR IGNORE INTO positions(mint, entry_ms, cost, status, exit_reason, entry_wall_ms)"
                        " VALUES(?,?,?,?,?,?)", (mint, decision_ms + self.lat, cost, "cap",
                                                 "plafond de positions ouvertes atteint", now_ms()))
            return
        q = None
        for _ in range(4):  # jusqu'a ~6 s pour obtenir une place dans le budget de cotations
            if self.jup.budget() > 0:
                q = self.jup.buy(mint, self.s["buy_sol"])
                if q is not None:
                    break
            time.sleep(1.5)
        if q is None:
            self.j.exec("INSERT OR IGNORE INTO positions(mint, entry_ms, cost, status, exit_reason, entry_wall_ms)"
                        " VALUES(?,?,?,?,?,?)", (mint, decision_ms + self.lat, cost, "no_quote",
                                                 "pas de cotation Jupiter", now_ms()))
            log.info("ACHAT IMPOSSIBLE %s : pas de cotation Jupiter", mint[:8])
            return
        self.j.exec("INSERT OR IGNORE INTO positions(mint, entry_ms, tokens, cost, status, value, max_value,"
                    " entry_wall_ms, src, last_quote_ms, quotes_n, impact, route) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (mint, decision_ms + self.lat, q["out"], cost, "open", None, None, now_ms(), "jupiter", 0, 1,
                     q["impact"], q["route"]))
        log.info("ACHAT %s : %.3f SOL -> %d tokens (Jupiter %s)", mint[:8], self.s["buy_sol"], q["out"], q["route"])

    def _curve_value(self, mint: str, tokens: float) -> float | None:
        """Estimation par la courbe : sert UNIQUEMENT a declencher une cotation prioritaire."""
        r = self.hot.query("SELECT vsol, vtok FROM trades WHERE mint=? AND vsol>0 ORDER BY slot DESC LIMIT 1", (mint,))
        if not r:
            return None
        return pump.sim_sell(int(r[0]["vsol"]), int(r[0]["vtok"]), tokens, self.s["fee_bps"])

    def update_positions(self):
        """Ventes au prix Jupiter (token -> SOL). Priorite : sorties declenchees et timeouts, puis suivi."""
        s = self.s
        tp, sl = s["buy_sol"] * (1 + s["take_profit"]), s["buy_sol"] * (1 - s["stop_loss"])
        now, hold = now_ms(), int(s["max_hold_s"] * 1000)
        requote = int(self.cfg["paper"]["requote_seconds"] * 1000)
        self.j.exec("UPDATE positions SET status='annule', exit_reason='ancienne version (prix courbe)'"
                    " WHERE status IN ('pending', 'open', 'closed') AND src IS NULL")
        urgent, routine = [], []
        stale_ms = int(s["stale_exit_s"] * 1000) if s.get("stale_exit_s") else None
        stale_thr = s["buy_sol"] * (1 + s.get("stale_min_gain", 0.1))
        for p in self.j.query("SELECT * FROM positions WHERE status='open'"):
            cv = self._curve_value(p["mint"], p["tokens"])
            # "stale" : apres N s, si la meilleure cotation n'a jamais atteint +X %, on sort (meme regle qu'a l'entrainement)
            stale = (stale_ms is not None and now - p["entry_wall_ms"] >= stale_ms
                     and max(p["max_value"] or 0.0, cv or 0.0) < stale_thr)
            timeout = now - p["entry_wall_ms"] >= hold or stale
            if timeout or (cv is not None and (cv >= tp or cv <= sl)):
                urgent.append((p, cv, timeout))
            elif now - (p["last_quote_ms"] or 0) >= requote:
                routine.append((p, cv, False))
        routine.sort(key=lambda x: x[0]["last_quote_ms"] or 0)
        for k, (p, cv, timeout) in enumerate(urgent + routine):
            # suivi de routine : on laisse toujours des cotations libres pour les achats (prioritaires)
            q = self.jup.sell(p["mint"], int(p["tokens"]), reserve=2 if k < len(urgent) else BUY_RESERVE)
            if q is None:
                if timeout and now - p["entry_wall_ms"] >= hold + 600_000:
                    # 10 min apres le timeout sans aucune cotation : on considere la position invendable
                    self.j.exec("UPDATE positions SET status='closed', exit_ms=?, exit_reason=?, pnl=?, value=0"
                                " WHERE mint=?", (now, "invendable (pas de route Jupiter)", -p["cost"], p["mint"]))
                continue
            value = q["sol"]
            mx = max(value, p["max_value"] or 0.0)
            is_stale = stale_ms is not None and now - p["entry_wall_ms"] >= stale_ms and max(mx, value) < stale_thr
            reason = ("tp" if value >= tp else "sl" if value <= sl else "stale" if is_stale
                      else "timeout" if now - p["entry_wall_ms"] >= hold else None)
            if reason:
                pnl = value - p["cost"]
                self.j.exec("UPDATE positions SET status='closed', exit_ms=?, exit_reason=?, pnl=?, value=?, max_value=?,"
                            " last_quote_ms=?, quotes_n=quotes_n+1, curve_value=?, exit_route=? WHERE mint=?",
                            (now, reason, pnl, value, mx, now, cv, q["route"], p["mint"]))
                log.info("VENTE %s %s : %.4f SOL (Jupiter %s) pnl=%+.4f", p["mint"][:8], reason, value, q["route"], pnl)
            else:
                self.j.exec("UPDATE positions SET value=?, max_value=?, last_quote_ms=?, quotes_n=quotes_n+1,"
                            " curve_value=? WHERE mint=?", (value, mx, now, cv, p["mint"]))

    def resolve(self):
        """Resultat du coin (achat simule a son premier point d'analyse), une fois connu."""
        wm = self.watermark()
        outs = self.outages()
        for d in self.j.query("SELECT mint, created_ms, first_age_s FROM decisions WHERE resolved=0"
                              " AND created_ms < ? LIMIT 500", (wm - self.cps[0] - self.hold,)):
            first = int((d["first_age_s"] or self.cps[0] / 1000) * 1000)
            res_ms = d["created_ms"] + first + self.hold
            if res_ms + 5_000 > wm:
                continue
            if overlaps(outs, d["created_ms"] + first, res_ms, LONG_OUTAGE_MS):
                # trades manquants : pas de resultat fiable (donc ignore par les features de tendance)
                self.j.exec("UPDATE decisions SET resolved=1, exit_reason_sim='coupure' WHERE mint=?", (d["mint"],))
                continue
            tr = self.trades(d["mint"])
            mig = self.hot.query("SELECT recv_ms FROM migrations WHERE mint=?", (d["mint"],))
            mig_ms = mig[0]["recv_ms"] if mig else None
            o = F.simulate(tr, d["created_ms"] + first, self.s, mig_ms)
            self.j.exec("UPDATE decisions SET pnl_sim=?, max_mult_after=?, exit_reason_sim=?, entry_mult_sim=?,"
                        " resolved=1 WHERE mint=?", (o.pnl_sol, o.max_mult_after, o.exit_reason, o.entry_mult, d["mint"]))
            self.mem.set_outcome(d["mint"], res_ms, o.entry_mult * o.max_mult_after, o.pnl_sol,
                                 mig_ms is not None and mig_ms <= res_ms)

    # ---------- boucle ----------
    def step(self):
        wm = self.watermark()
        last_cp = self.cps[-1]
        strict = "AND sampled=1" if self.cfg["collector"].get("trades_source") == "per_token" else ""
        toks = self.hot.query(f"SELECT *, {CREATED} AS c FROM tokens WHERE {CREATED} <= ? AND {CREATED} >= ?"
                              f" {strict} ORDER BY c", (wm - self.cps[0], wm - last_cp - 60_000))
        for tok in toks:
            mint = tok["mint"]
            if mint in self.closed:
                continue
            done = self.done.setdefault(mint, set())
            due = [cp for cp in self.cps if tok["c"] + cp <= wm and cp not in done]
            if not due:
                continue
            cp = max(due)  # si on est en retard, on n'analyse que le point le plus recent
            done.update(due)
            try:
                self.decide(dict(tok), cp, last=cp == last_cp)
            except Exception:
                log.exception("analyse %s a %ds", mint, cp // 1000)
        cut = wm - last_cp - 120_000
        if len(self.done) > 5000:
            keep = {r["mint"] for r in toks}
            self.done = {m: v for m, v in self.done.items() if m in keep}
        if len(self.closed) > 100_000:
            self.closed = {m for m in self.closed if m in self.done}
        return cut

    def run(self):
        log.info("paper trader demarre (analyses a %s s, achat jusqu'a %d s)",
                 ", ".join(str(c // 1000) for c in self.cps), self.cps[-1] // 1000)
        last_resolve = 0.0
        self.last_count_log = time.monotonic()
        while not (self.cfg.data / "PAUSE").exists():
            t0 = time.monotonic()
            try:
                self._reload_model()
                self._ingest()
                self.step()
                self.update_positions()
                if self.amm is not None:
                    self.amm.step()
                if time.monotonic() - last_resolve > 30:
                    self.resolve()
                    if self.amm is not None:
                        self.amm.resolve()
                    self.hot.heartbeat("paper", json.dumps({"model": self.model.id if self.model else None,
                                                            **self.counts}))
                    last_resolve = time.monotonic()
                if time.monotonic() - self.last_count_log > 3600:
                    log.info("derniere heure : %s", json.dumps(self.counts))
                    self.counts, self.last_count_log = {}, time.monotonic()
                    self.mem.prune(now_ms())
            except Exception:
                log.exception("boucle paper")
            time.sleep(max(0.0, min(1.0, self.cfg["paper"]["poll_seconds"]) - (time.monotonic() - t0)))


def main(cfg: Config):
    Paper(cfg).run()
