"""Dashboard local : http://127.0.0.1:8787 (n'ecoute que sur la machine)."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from pathlib import Path

import pandas as pd

from aiohttp import web

from . import backfill as B
from . import perf as P
from . import pump
from .config import ROOT, Config
from .feedback import Journal
from .model import Registry
from .storage import HotStore, now_ms

log = logging.getLogger("dashboard")
PORT = 8787
VERDICTS = {-1, 0, 1}


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _rows(rows):
    return [{k: _clean(v) for k, v in r.items()} for r in rows]


def _deep(v):
    """NaN/inf -> None partout (le JSON du navigateur ne les accepte pas)."""
    if isinstance(v, dict):
        return {k: _deep(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_deep(x) for x in v]
    return _clean(v)


def _same_origin(req):
    origin = req.headers.get("Origin")
    if origin and origin not in (f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"):
        raise web.HTTPForbidden()


def candles(series: list) -> tuple[list, int]:
    """Bougies OHLC (market cap) + volumes achat / vente en SOL. Intervalle choisi selon la duree."""
    if len(series) < 2:
        return [], 0
    span = (series[-1][0] - series[0][0]) / 1000
    step = next((s for s, lim in ((1, 120), (5, 600), (15, 1800), (60, 7200), (300, 43200), (900, 172800))
                 if span <= lim), 3600)
    out: dict[int, list] = {}
    for t, v, buy, sol, _p in series:
        b = int(t // 1000 // step * step)
        c = out.get(b)
        if c is None:
            out[b] = [b, v, v, v, v, 0.0, 0.0]
        else:
            c[2], c[3], c[4] = max(c[2], v), min(c[3], v), v
        out[b][5 if buy else 6] += sol or 0.0
    rows = sorted(out.values())
    for a, b in zip(rows, rows[1:]):  # chaque bougie ouvre au prix de cloture de la precedente
        b[1] = a[4]
        b[2], b[3] = max(b[2], b[1]), min(b[3], b[1])
    return rows, step


class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.hot = HotStore(cfg.data / "hot.db")
        self.j = Journal(cfg.data / "journal.db")
        self.importer = B.Importer(cfg, self.j)
        self._sol = (0.0, None)  # (instant, prix du SOL en dollars)
        self._kol = (0.0, None)  # (instant, carnet des KOL)

    def sol_usd(self) -> float | None:
        """Prix du SOL en dollars (Jupiter), garde 5 min."""
        import time as _t
        import urllib.request
        if self._sol[1] is None or _t.monotonic() - self._sol[0] > 300:
            try:
                u = "https://lite-api.jup.ag/price/v3?ids=So11111111111111111111111111111111111111112"
                with urllib.request.urlopen(u, timeout=8) as r:
                    self._sol = (_t.monotonic(), float(next(iter(json.load(r).values()))["usdPrice"]))
            except Exception as e:
                log.debug("prix du SOL indisponible : %s", e)
        return self._sol[1]

    def kol_trades(self, mint: str, tr, k: float) -> list[dict]:
        """Trades des KOL sur ce coin (flux PumpPortal + trades suivis), avec leur flair actuel."""
        from . import kol as K
        if not self.cfg.raw.get("kol", {}).get("enabled"):
            return []
        wallets, emoji = K.kols(self.cfg)
        import time as _t
        if self._kol[1] is None or _t.monotonic() - self._kol[0] > 900:
            self._kol = (_t.monotonic(), K.build(self.cfg))
        book = self._kol[1]
        rows = {}
        try:
            for r in self.hot.query("SELECT signature, recv_ms, wallet, is_buy, sol, mcap_sol FROM kol_trades WHERE mint=?",
                                    (mint,)):
                rows[r["signature"]] = dict(r)
        except Exception:
            pass
        for r in (dict(x) for x in tr):
            if r.get("user") in wallets:
                ms = r["ts"] * 1000 if r.get("ts") else r["recv_ms"]
                rows.setdefault(f"{r['user']}{ms}{r['sol_amount']}", {
                    "recv_ms": ms, "wallet": r["user"], "is_buy": r["is_buy"], "sol": r["sol_amount"] / 1e9,
                    "mcap_sol": pump.mcap_sol(r["vsol"], r["vtok"]) if r.get("vsol") else None})
        out = []
        for r in sorted(rows.values(), key=lambda r: r["recv_ms"]):
            p = wallets.get(r["wallet"], r["wallet"][:6])
            s, n = book.score(p, now_ms(), exclude=mint)
            out.append({"ms": r["recv_ms"], "person": p, "emoji": emoji.get(p, ""), "wallet": r["wallet"],
                        "is_buy": bool(r["is_buy"]), "sol": r["sol"], "mcap": (r["mcap_sol"] or 0) * k or None,
                        "score": s, "known": n, "base": book.base_rate(now_ms())})
        return out

    def live_price(self, mints: list[str]) -> dict[str, dict]:
        out = {}
        for i in range(0, len(mints), 500):
            ch = mints[i:i + 500]
            q = (f"SELECT t.mint, t.vsol, t.vtok, t.real_tok, t.recv_ms FROM trades t JOIN ("
                 f"SELECT mint, MAX(recv_ms) m FROM trades WHERE mint IN ({','.join('?' * len(ch))}) GROUP BY mint)"
                 f" x ON t.mint=x.mint AND t.recv_ms=x.m")
            for r in self.hot.query(q, tuple(ch)):
                out[r["mint"]] = {"mcap_now": pump.mcap_sol(r["vsol"], r["vtok"]),
                                  "progress_now": pump.progress(r["real_tok"]), "last_trade_ms": r["recv_ms"]}
        return out

    async def state(self, _):
        hb = {r["name"]: {"ms": r["ms"], "info": json.loads(r["info"] or "{}")}
              for r in self.hot.query("SELECT * FROM heartbeat")}
        reg = Registry(self.cfg.models).state
        champ = None
        if reg.get("champion"):
            mp = self.cfg.models / reg["champion"] / "meta.json"
            if mp.exists():
                m = json.loads(mp.read_text())
                champ = {"id": reg["champion"], "metrics": m["metrics"], "importance": m["importance"][:15]}
        day = now_ms() - 86_400_000
        pos = self.j.query("SELECT COUNT(*) n, SUM(status='open') open, SUM(CASE WHEN status='closed' THEN pnl END) pnl,"
                           " SUM(status='closed' AND pnl>0) wins, SUM(status='closed') closed FROM positions")[0]
        pos24 = self.j.query("SELECT SUM(pnl) pnl, COUNT(*) n FROM positions WHERE status='closed' AND exit_ms>=?", (day,))[0]
        dec = self.j.query("SELECT COUNT(*) n, SUM(action='buy') buys FROM decisions WHERE decided_ms>=?", (day,))[0]
        fb = self.j.query("SELECT COUNT(*) n, SUM(verdict>0) up, SUM(verdict<0) down FROM feedback")[0]
        missed = self.j.query("SELECT COUNT(*) n, SUM(pnl_sim>0) good FROM decisions WHERE resolved=1 AND action!='buy'"
                              " AND decided_ms>=?", (day,))[0]
        return web.json_response({"now": now_ms(), "sol_usd": await asyncio.to_thread(self.sol_usd),
                                  "heartbeat": hb, "champion": champ,
                                  "history": reg.get("history", [])[-30:], "positions": pos, "positions_24h": pos24,
                                  "decisions_24h": dec, "feedback": fb, "missed_24h": missed,
                                  "strategy": self.cfg["strategy"]})

    async def decisions(self, req):
        lim = min(int(req.query.get("limit", 80)), 300)
        flt = req.query.get("filter", "all")
        where = {"buy": "d.action='buy'", "skip": "d.action='skip'", "observe": "d.action='observe'",
                 "watch": "d.action='watch'",
                 "open": "p.status='open'", "rated": "f.verdict IS NOT NULL", "unrated": "f.verdict IS NULL",
                 "missed": "d.action!='buy' AND d.pnl_sim>0"}.get(flt, "1=1")
        q = (f"SELECT d.mint, d.decided_ms, d.created_ms, d.model_id, d.score, d.taste, d.final, d.threshold, d.action,"
             f" d.name, d.symbol, d.image_path, d.twitter, d.description, d.mcap_sol, d.top_reasons, d.pnl_sim,"
             f" d.max_mult_after, d.exit_reason_sim, d.resolved, d.age_s, p.status pos_status, p.pnl pos_pnl, p.value pos_value,"
             f" p.cost pos_cost, p.exit_reason pos_exit, f.verdict, f.tags, f.comment, d.vamp"
             f" FROM decisions d LEFT JOIN positions p ON p.mint=d.mint LEFT JOIN feedback f ON f.mint=d.mint"
             f" WHERE {where} ORDER BY d.decided_ms DESC LIMIT ?")
        rows = _rows(self.j.query(q, (lim,)))
        lp = self.live_price([r["mint"] for r in rows])
        for r in rows:
            r.update(lp.get(r["mint"], {}))
            r["top_reasons"] = json.loads(r["top_reasons"] or "[]")
            r["vamp"] = json.loads(r["vamp"]) if r.get("vamp") else None
        return web.json_response(rows)

    async def token(self, req):
        mint = req.match_info["mint"]
        d = _rows(self.j.query("SELECT * FROM decisions WHERE mint=?", (mint,)))
        tr = self.hot.query("SELECT recv_ms, ts, is_buy, sol_amount, user, vsol, vtok, real_tok FROM trades WHERE mint=?"
                            " ORDER BY COALESCE(ts * 1000, recv_ms), recv_ms", (mint,))
        soc = self.hot.query("SELECT json FROM social WHERE mint=?", (mint,))
        meta = self.hot.query("SELECT * FROM metadata WHERE mint=?", (mint,))
        out = d[0] if d else {"mint": mint}
        if not d:  # coin importe (pas vu en direct)
            ex = _rows(self.j.query("SELECT * FROM examples WHERE mint=?", (mint,)))
            if ex:
                out.update(ex[0], resolved=1, action="exemple")
                bp = self.cfg.data / "backfill" / "trades" / f"{mint}.parquet"
                rows = pd.read_parquet(bp) if bp.exists() else pd.DataFrame()
                br = B.load_rows(self.cfg)
                br = br[br.mint == mint] if not br.empty else br
                if not br.empty:
                    out["features"] = {k: v for k, v in br.iloc[0].to_dict().items()
                                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
                    out["exit_reason_sim"] = br.iloc[0]["exit_reason"]
                tr = rows.to_dict("records") if not rows.empty else []
        if out.get("vamp") and isinstance(out["vamp"], str):
            out["vamp"] = json.loads(out["vamp"])
        if out.get("features") and isinstance(out["features"], str):
            out["features"] = json.loads(out["features"])
        if not isinstance(out.get("top_reasons"), list):
            out["top_reasons"] = json.loads(out.get("top_reasons") or "[]")
        # Market cap en DOLLARS (prix du SOL Jupiter), heure on-chain ; apres migration : echanges PumpSwap.
        usd = await asyncio.to_thread(self.sol_usd)
        k = usd or 1.0
        series = [[(r["ts"] * 1000 if r.get("ts") else r["recv_ms"]), pump.mcap_sol(r["vsol"], r["vtok"]) * k,
                   int(bool(r["is_buy"])), r["sol_amount"] / 1e9, pump.progress(r["real_tok"]) if r.get("real_tok") is not None else None]
                  for r in (dict(x) for x in tr) if r["vsol"] > 0]
        sw = self.hot.query("SELECT ts, is_buy, sol_amount, sol_swap, token_amount FROM swaps WHERE mint=? ORDER BY ts",
                            (mint,))
        series += [[r["ts"] * 1000, r["sol_swap"] / r["token_amount"] * 1e15 / 1e9 * k, int(bool(r["is_buy"])),
                    r["sol_amount"] / 1e9, 1.0] for r in sw if r["token_amount"]]
        series.sort(key=lambda s: s[0])
        out["candles"], out["candle_s"] = candles(series)
        out["kol"] = await asyncio.to_thread(self.kol_trades, mint, tr, k)
        if len(series) > 800:
            step = len(series) / 800
            series = [series[int(i * step)] for i in range(800)] + [series[-1]]
        out["series"] = series
        out["usd"] = usd is not None
        # market cap a laquelle la courbe est complete (migration)
        kk = pump.INITIAL_VSOL * pump.INITIAL_VTOK
        vt_end = pump.INITIAL_VTOK - 793_100_000_000_000
        out["migration_mcap"] = pump.mcap_sol(kk / vt_end, vt_end) * k
        mg = self.hot.query("SELECT recv_ms FROM migrations WHERE mint=?", (mint,))
        out["migrated_ms"] = mg[0]["recv_ms"] if mg else None
        pos = self.j.query("SELECT COALESCE(entry_wall_ms, entry_ms) AS entry_ms, exit_ms, exit_reason, pnl, status"
                           " FROM positions WHERE mint=?", (mint,))
        out["position"] = pos[0] if pos else None
        out["social"] = json.loads(soc[0]["json"]) if soc else None
        out["meta"] = dict(meta[0]) if meta else None
        out["n_trades"] = len(tr)
        fb = self.j.query("SELECT verdict, tags, comment FROM feedback WHERE mint=?", (mint,))
        out.update(fb[0] if fb else {"verdict": None, "tags": "", "comment": ""})
        return web.json_response(_deep(out))

    # ---------- recherche / import par CA ----------
    async def search(self, req):
        mint = B.extract_mint(req.query.get("q", ""))
        if not mint:
            return web.json_response({"mint": None})
        seen = bool(self.j.query("SELECT 1 FROM decisions WHERE mint=?", (mint,)))
        ex = _rows(self.j.query("SELECT * FROM examples WHERE mint=?", (mint,)))
        hot = self.hot.query("SELECT name, symbol FROM tokens WHERE mint=?", (mint,))
        return web.json_response({"mint": mint, "seen": seen, "example": ex[0] if ex else None,
                                  "collecting": dict(hot[0]) if hot else None})

    async def import_coin(self, req):
        _same_origin(req)
        mint = B.extract_mint((await req.json()).get("q", ""))
        if not mint:
            raise web.HTTPBadRequest(text="CA invalide")
        return web.json_response({"ok": self.importer.submit(mint), "mint": mint})

    async def examples(self, _):
        rows = _rows(self.j.query("SELECT e.*, f.verdict, f.tags, f.comment FROM examples e"
                                  " LEFT JOIN feedback f ON f.mint=e.mint ORDER BY e.added_ms DESC LIMIT 300"))
        return web.json_response(rows)

    async def amm(self, _):
        """Apres migration : verdicts anti-rug et resultats simules (phase d'observation)."""
        rows = _rows(self.j.query("SELECT * FROM amm_decisions ORDER BY migrated_ms DESC LIMIT 300"))
        summ = _rows(self.j.query(
            "SELECT CASE WHEN reasons='' THEN 'eligible' ELSE 'exclu' END AS grp, COUNT(*) n,"
            " SUM(resolved) resolus, AVG(pnl_sim) pnl_moy, SUM(pnl_sim) pnl_total,"
            " AVG(CASE WHEN pnl_sim > 0 THEN 1.0 ELSE 0.0 END) gagnants, AVG(max_mult) max_mult_moy,"
            " AVG(CASE WHEN jup_tokens IS NOT NULL THEN 1.0 END) routables,"
            " AVG(CASE WHEN jup_tokens > 0 AND sim_tokens > 0 THEN jup_tokens / sim_tokens END) jup_vs_sim"
            " FROM amm_decisions WHERE migrated_ms > ? GROUP BY grp", (now_ms() - 7 * 86_400_000,)))
        return web.json_response({"rows": rows, "summary": summ})

    # ---------- IA : modeles et apprentissage ----------
    async def models(self, _):
        reg = Registry(self.cfg.models).state
        live = {r["model_id"]: r for r in _rows(self.j.query(
            "SELECT model_id, COUNT(*) n, SUM(action='buy') buys, SUM(resolved=1 AND pnl_sim IS NOT NULL) resolved,"
            " SUM(CASE WHEN action='buy' THEN pnl_sim END) buy_pnl,"
            " AVG(CASE WHEN action='buy' AND pnl_sim IS NOT NULL THEN pnl_sim > 0 END) buy_win,"
            " SUM(CASE WHEN action='buy' AND pnl_sim IS NOT NULL THEN 1 ELSE 0 END) buy_resolved,"
            " AVG(pnl_sim) all_pnl_mean, MIN(decided_ms) first_ms, MAX(decided_ms) last_ms"
            " FROM decisions WHERE model_id IS NOT NULL GROUP BY model_id"))}
        return web.json_response(_deep({"champion": reg.get("champion"), "history": reg.get("history", []),
                                        "live": live}))

    async def perf(self, _):
        r = await asyncio.to_thread(P.build, self.j, Registry(self.cfg.models).state)
        r["stake"] = self.cfg["strategy"].get("buy_sol")
        return web.json_response(_deep(r))

    async def model(self, req):
        mid = req.match_info["id"]
        if not re.fullmatch(r"[0-9-]{8,20}", mid):
            raise web.HTTPNotFound()
        p = self.cfg.models / mid / "meta.json"
        if not p.exists():
            raise web.HTTPNotFound()
        m = json.loads(p.read_text())
        m.pop("features", None)
        return web.json_response(_deep(m))

    async def datastats(self, _):
        since = now_ms() - 48 * 3600_000
        hours = _rows(self.j.query(
            "SELECT (decided_ms/3600000)*3600000 h, COUNT(*) n, SUM(action='buy') buys,"
            " SUM(resolved=1 AND pnl_sim IS NOT NULL) resolved, AVG(pnl_sim > 0) win, AVG(pnl_sim) pnl_mean,"
            " SUM(vamp IS NOT NULL) copies FROM decisions WHERE decided_ms >= ? GROUP BY h ORDER BY h", (since,)))
        tot = self.j.query("SELECT COUNT(*) n, SUM(resolved=1 AND pnl_sim IS NOT NULL) resolved FROM decisions")[0]
        # Fidelite de la simulation par la courbe (labels d'entrainement) vs les vrais prix Jupiter
        calib = self.j.query("SELECT COUNT(*) n, AVG(p.pnl - d.pnl_sim) gap, AVG(ABS(p.pnl - d.pnl_sim)) abs_gap,"
                             " AVG(p.impact) impact FROM positions p JOIN decisions d ON d.mint=p.mint"
                             " WHERE p.status='closed' AND p.src='jupiter' AND d.pnl_sim IS NOT NULL")[0]
        hb = {r["name"]: json.loads(r["info"] or "{}") for r in self.hot.query("SELECT * FROM heartbeat")}
        lp = self.cfg.logs / "learn.log"
        lines = lp.read_text(encoding="utf-8", errors="replace").splitlines() if lp.exists() else []
        last_learn = [l for l in lines if "entrainement" in l or "skip" in l or "reglages" in l][-4:]
        return web.json_response(_deep({"hours": hours, "total": tot, "paper": hb.get("paper", {}), "calib": calib,
                                        "collector": hb.get("collector", {}), "last_learn": last_learn,
                                        "min_train_rows": self.cfg["training"]["min_train_rows"],
                                        "retrain_every_hours": self.cfg["training"]["retrain_every_hours"]}))

    async def feedback(self, req):
        _same_origin(req)
        b = await req.json()
        mint, verdict = str(b.get("mint", ""))[:64], b.get("verdict")
        if not mint or verdict not in VERDICTS:
            raise web.HTTPBadRequest(text="mint / verdict invalides")
        if verdict == 0 and not b.get("comment") and not b.get("tags"):
            self.j.exec("DELETE FROM feedback WHERE mint=?", (mint,))
        else:
            self.j.set_feedback(mint, now_ms(), verdict, str(b.get("tags", ""))[:200], str(b.get("comment", ""))[:2000],
                                after_outcome=bool(b.get("after_outcome")))
        return web.json_response({"ok": True})

    async def image(self, req):
        p = (self.cfg.data / req.match_info["path"]).resolve()
        if not p.is_relative_to((self.cfg.data / "images").resolve()) or not p.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(p, headers={"Cache-Control": "max-age=86400"})

    async def index(self, _):
        return web.FileResponse(ROOT / "web" / "index.html")


def main(cfg: Config):
    a = App(cfg)
    app = web.Application()
    app.router.add_get("/", a.index)
    app.router.add_get("/api/state", a.state)
    app.router.add_get("/api/decisions", a.decisions)
    app.router.add_get("/api/token/{mint}", a.token)
    app.router.add_post("/api/feedback", a.feedback)
    app.router.add_get("/api/search", a.search)
    app.router.add_post("/api/import", a.import_coin)
    app.router.add_get("/api/examples", a.examples)
    app.router.add_get("/api/amm", a.amm)
    app.router.add_get("/api/models", a.models)
    app.router.add_get("/api/model/{id}", a.model)
    app.router.add_get("/api/perf", a.perf)
    app.router.add_get("/api/datastats", a.datastats)
    app.router.add_get("/img/{path:.+}", a.image)
    app.router.add_static("/static/", ROOT / "web")
    log.info("dashboard : http://127.0.0.1:%d", PORT)
    async def pause_watch(_app):
        async def loop():
            while not (cfg.data / "PAUSE").exists():
                await asyncio.sleep(2)
            log.info("pause demandee")
            os._exit(0)  # rien a sauvegarder ici
        _app["pause"] = asyncio.create_task(loop())

    app.on_startup.append(pause_watch)

    async def resume_imports(_app):
        a.importer.resume_pending()

    app.on_startup.append(resume_imports)
    web.run_app(app, host="127.0.0.1", port=PORT, print=None)
