"""Collecteur 24h/24.

Flux :
* PumpPortal (une seule connexion, gratuit) : creations de tokens + migrations.
* Websocket Solana (Helius) : ``logsSubscribe`` sur le programme pump.fun, decodage des
  TradeEvent / CreateEvent / CompleteEvent. Les creations on-chain servent aussi de secours
  si PumpPortal tombe.
* Workers : metadonnees IPFS -> image -> lien X -> fournisseur X.

Chaque flux se reconnecte seul (backoff exponentiel + detection de silence). Les files des
workers sont bornees : si elles debordent, on perd de l'enrichissement, jamais des trades.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import signal
import time

import aiohttp
import websockets

from . import metadata as md
from . import features as F
from . import pump, social
from .config import Config
from .storage import HotStore, ParquetSink, day_of, now_ms

SLOT_GAP = 25  # ~10 s sans aucune notification pump.fun

log = logging.getLogger("collector")


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class Collector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        c = cfg["collector"]
        self.c = c
        data = cfg.data
        self.sinks = {t: ParquetSink(data, t, c["flush_rows"], c["flush_seconds"])
                      for t in ("tokens", "trades", "migrations", "metadata", "social", "outages", "swaps", "pools", "kol")}
        self.hot = HotStore(data / "hot.db")
        self.tracked: dict[str, int] = {}  # mint -> recv_ms de creation
        self.seen_meta: set[str] = set()
        self.hot_buf: dict[str, list[tuple]] = {k: [] for k in ("tokens", "trades", "migrations", "metadata", "social",
                                                                "outages", "swaps", "pools", "sampled", "spam", "kol")}
        self.meta_q: asyncio.Queue = asyncio.Queue(maxsize=5000)
        self.social_q: asyncio.Queue = asyncio.LifoQueue(maxsize=5000)  # les plus recents d'abord
        self.stats: dict[str, int] = {}
        self.last_msg = {"pumpportal": time.monotonic(), "chain": time.monotonic()}
        self.connected = {"pumpportal": False, "chain": False}
        self.chain_failures = 0
        self.fallback_until = 0.0
        # Dernier slot recu. Au redemarrage : dernier trade connu -> l'arret du collecteur est
        # enregistre comme une coupure (detectee au premier message, par le trou de slots).
        last = self.hot.query("SELECT slot, ts, recv_ms FROM trades ORDER BY slot DESC LIMIT 1")
        if last:
            r = last[0]
            self.last_slot = r["slot"]
            self.anchor = (r["slot"], r["ts"] * 1000 + 500 if r["ts"] else r["recv_ms"] - 6_000)
        else:
            self.last_slot, self.anchor = None, (0, 0)
        self.stop = asyncio.Event()
        # Mode "per_token" : trades recus coin par coin (Helius), complets, sur un echantillon de coins.
        self.per_token = c.get("trades_source") == "per_token" and bool(cfg.secrets.helius_key)
        self.sampled: dict[str, bool | None] = {}  # None : en attente des metadonnees
        self.pending_link: dict[str, int] = {}     # mint -> creation, suivi decide a l'arrivee des metadonnees
        from . import kol as K
        self.kol_wallets = set(K.kols(cfg)[0]) if cfg.raw.get("kol", {}).get("enabled") else set()
        self.kol_seen: set[str] = set()
        self.tok = None
        self.amm = None
        if self.per_token:
            from .tokstream import TokenStream
            self.tok = TokenStream(self)
            if cfg["amm"].get("enabled"):
                from .amm import Migrations
                self.amm = Migrations(self, self.tok)
            self.last_msg["tokstream"] = time.monotonic()
            self.connected["tokstream"] = False
        elif c.get("trades_source") == "per_token":
            log.warning("trades_source=per_token demande HELIUS_API_KEY : retour au flux public complet")
        self._restore_tracked()
        if self.amm is not None:
            self.amm.restore()

    # ---------- utilitaires ----------
    def bump(self, k: str, n: int = 1):
        self.stats[k] = self.stats.get(k, 0) + n

    def _restore_tracked(self):
        cut = now_ms() - int(self.c["track_hours"] * 3600_000)
        for r in self.hot.query("SELECT mint, recv_ms FROM tokens WHERE recv_ms >= ?", (cut,)):
            self.tracked[r["mint"]] = r["recv_ms"]
        for r in self.hot.query("SELECT mint FROM metadata"):
            self.seen_meta.add(r["mint"])
        if self.tracked:
            log.info("reprise : %d tokens suivis restaures", len(self.tracked))
        if self.per_token:
            # coins encore dans leur fenetre : reabonnement + rattrapage de ce qui a ete manque pendant l'arret
            cut = now_ms() - self.tok.track_ms
            for r in self.hot.query("SELECT mint, sampled, COALESCE(chain_ms, recv_ms) c FROM tokens WHERE recv_ms >= ?",
                                    (now_ms() - 6 * 3600_000,)):
                self.sampled[r["mint"]] = bool(r["sampled"])
                if r["sampled"] and r["c"] >= cut:
                    from . import backfill
                    curve = backfill.bonding_curve_pda(r["mint"])
                    self.tok.active[curve] = {"mint": r["mint"], "until": r["c"] + self.tok.track_ms, "sub": None}
            last = self.hot.query("SELECT MAX(ts) t FROM trades")
            if self.tok.active:
                self.tok.down_since = (last[0]["t"] * 1000) if last and last[0]["t"] else cut
                log.info("reprise : %d coins suivis a reabonner", len(self.tok.active))

    def register_token(self, row: dict):
        mint = row["mint"]
        if mint not in self.sampled and self.per_token and self.cfg["strategy"].get("require_link"):
            # suivi decide a l'arrivee des metadonnees (~2 s) : seulement les coins avec Twitter ou site web.
            # Les trades d'avant l'abonnement sont rattrapes par RPC, comme d'habitude.
            self.sampled[mint] = None
            self.pending_link[mint] = row.get("chain_ms") or row["recv_ms"]
        if mint not in self.sampled:
            if self.per_token:
                ok = self.tok.want(mint)
                if ok:
                    self.tok.add(mint, row.get("chain_ms") or row["recv_ms"])
                    self.bump("tokens_suivis")
                elif self.tok.over_budget():
                    self.bump("tokens_hors_budget")
            else:
                ok = True  # flux complet : tous les coins (donnees partielles avec le websocket public)
            self.sampled[mint] = ok
        if len(self.pending_link) > 20_000:
            self.pending_link = dict(list(self.pending_link.items())[-10_000:])
        if len(self.sampled) > 200_000:
            self.sampled = dict(list(self.sampled.items())[-100_000:])
        if len(self.kol_seen) > 200_000:
            self.kol_seen = set()
        row["sampled"] = self.sampled.get(mint) if self.per_token else None
        self.sinks["tokens"].add(row)
        if mint not in self.tracked:
            self.tracked[mint] = row["recv_ms"]
            self.bump("tokens_new")
        self.hot_buf["tokens"].append((mint, row["recv_ms"], row.get("name"), row.get("symbol"), row.get("uri"),
                                       row.get("creator"), row.get("initial_buy"), row.get("sol_amount"),
                                       row.get("mcap_sol"), row.get("chain_ms"),
                                       None if row["sampled"] is None else int(row["sampled"])))
        if row.get("uri") and mint not in self.seen_meta:
            self.seen_meta.add(mint)
            try:
                self.meta_q.put_nowait((mint, row["uri"]))
            except asyncio.QueueFull:
                self.bump("meta_dropped")

    def mark_spam(self, mint: str):
        """Coin noye sous les robots de volume : plus suivi, exclu de l'entrainement et des achats."""
        self.sampled[mint] = False
        self.sinks["tokens"].add({"recv_ms": now_ms(), "source": "spam", "mint": mint, "sampled": False})
        self.hot_buf["spam"].append((mint,))
        self.bump("tokens_spam")

    def decide_tracking(self, mint: str, meta: dict):
        """Metadonnees arrivees : suivi des trades seulement si Twitter ou site web (et tirage + budget)."""
        created = self.pending_link.pop(mint, None)
        if created is None:
            return
        if not F.has_link(meta):
            self.sampled[mint] = False
            self.bump("tokens_sans_lien")
            return
        ok = self.tok.want(mint)
        self.sampled[mint] = ok
        if ok:
            self.tok.add(mint, created)
            self.bump("tokens_suivis")
            self.sinks["tokens"].add({"recv_ms": now_ms(), "source": "sample", "mint": mint, "sampled": True})
            self.hot_buf["sampled"].append((mint,))
        elif self.tok.over_budget():
            self.bump("tokens_hors_budget")

    # ---------- PumpPortal ----------
    def on_pumpportal(self, msg: dict):
        t = now_ms()
        tx = msg.get("txType")
        if tx == "create" and msg.get("mint"):
            self.register_token({
                "recv_ms": t, "source": "pumpportal", "mint": msg["mint"], "name": msg.get("name"),
                "symbol": msg.get("symbol"), "uri": msg.get("uri"), "creator": msg.get("traderPublicKey"),
                "bonding_curve": msg.get("bondingCurveKey"), "initial_buy": _f(msg.get("initialBuy")),
                "sol_amount": _f(msg.get("solAmount")), "vsol": _f(msg.get("vSolInBondingCurve")),
                "vtok": _f(msg.get("vTokensInBondingCurve")), "mcap_sol": _f(msg.get("marketCapSol")),
                "signature": msg.get("signature"), "raw": json.dumps(msg)[:4000],
            })
        elif tx in ("buy", "sell") and msg.get("traderPublicKey") in self.kol_wallets and msg.get("mint"):
            self.on_kol(msg, t)
        elif (tx == "migrate" or "migrat" in str(msg.get("txType", ""))) and msg.get("mint"):
            self.on_migration(msg["mint"], "pumpportal", msg.get("signature"), json.dumps(msg)[:4000])
        elif "message" in msg:
            log.info("pumpportal: %s", msg.get("message"))
        else:
            self.bump("pp_other")

    def on_kol(self, msg: dict, t: int, source: str = "pumpportal"):
        """Trade d'un KOL (PumpPortal subscribeAccountTrade) : n'importe quel coin pump.fun, meme non suivi."""
        sig = msg.get("signature")
        if not sig or sig in self.kol_seen:
            return
        self.kol_seen.add(sig)
        row = {"recv_ms": t, "signature": sig, "mint": msg["mint"], "wallet": msg["traderPublicKey"],
               "is_buy": msg.get("txType") == "buy", "sol": _f(msg.get("solAmount")),
               "tokens": _f(msg.get("tokenAmount")), "mcap_sol": _f(msg.get("marketCapSol")), "source": source}
        self.sinks["kol"].add(row)
        self.hot_buf["kol"].append((sig, msg["mint"], t, row["wallet"], int(row["is_buy"]), row["sol"], row["tokens"],
                                    row["mcap_sol"]))
        self.bump("kol_trades")

    def on_migration(self, mint: str, source: str, sig: str | None, raw: str = "", chain_ms: int | None = None):
        t = now_ms()
        self.sinks["migrations"].add({"recv_ms": t, "source": source, "mint": mint, "signature": sig, "raw": raw,
                                      "chain_ms": chain_ms})
        self.hot_buf["migrations"].append((mint, chain_ms or t - 6_000))
        self.bump(f"migrations_{source}")
        if self.amm is not None and source == "pumpportal" and sig and mint not in self.amm.done:
            # secours : la signature PumpPortal est celle de la transaction de migration
            self.amm.q.put_nowait((sig, 0))

    async def run_pumpportal(self):
        url = self.cfg.pumpportal_url()
        backoff = 5.0  # PumpPortal bannit les reconnexions trop rapides : on reste prudent
        while not self.stop.is_set():
            self.connected["pumpportal"] = False
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_size=2 ** 22,
                                              open_timeout=20) as ws:
                    await ws.send(json.dumps({"method": "subscribeNewToken"}))
                    await ws.send(json.dumps({"method": "subscribeMigration"}))
                    if self.kol_wallets and self.cfg.secrets.pumpportal_key:  # cle PumpPortal approvisionnee requise
                        keys = sorted(self.kol_wallets)
                        for i in range(0, len(keys), 200):
                            await ws.send(json.dumps({"method": "subscribeAccountTrade", "keys": keys[i:i + 200]}))
                        log.info("pumpportal : suivi de %d wallets KOL", len(keys))
                    log.info("pumpportal connecte")
                    self.connected["pumpportal"] = True
                    self.last_msg["pumpportal"] = time.monotonic()
                    backoff = 5.0
                    async for raw in ws:
                        self.last_msg["pumpportal"] = time.monotonic()
                        try:
                            self.on_pumpportal(json.loads(raw))
                        except Exception:
                            log.exception("message pumpportal illisible")
                            self.bump("pp_errors")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("pumpportal deconnecte: %s", e)
            self.connected["pumpportal"] = False
            self.bump("pp_reconnects")
            await asyncio.sleep(backoff + random.random() * 2)
            backoff = min(backoff * 2, 120)

    # ---------- chaine (Helius) ----------
    def on_logs(self, slot: int, value: dict, source: str = "firehose"):
        """``source`` : "firehose" (tout pump.fun) ou "token" (abonnement coin par coin)."""
        if value.get("err") is not None:
            return
        if self.per_token and source == "firehose":
            # le flux global ne sert plus qu'a voir les creations / migrations (si PumpPortal tombe)
            for ev in pump.iter_events(value.get("logs") or []):
                if isinstance(ev, pump.Create):
                    self.register_token({
                        "recv_ms": now_ms(), "source": "chain", "mint": ev.mint, "name": ev.name, "symbol": ev.symbol,
                        "uri": ev.uri, "creator": ev.creator or ev.user, "bonding_curve": ev.bonding_curve,
                        "signature": value.get("signature"), "vsol": float(pump.INITIAL_VSOL),
                        "vtok": float(pump.INITIAL_VTOK), "chain_ms": ev.ts * 1000 if ev.ts else None})
                    self.bump("creates_chain")
                elif isinstance(ev, pump.Complete):
                    self.on_migration(ev.mint, "chain", value.get("signature"),
                                      chain_ms=ev.ts * 1000 if ev.ts else None)
            return
        sig = value.get("signature")
        logs = value.get("logs") or []
        if any("Log truncated" in l for l in logs):
            self.bump("logs_truncated")
        if self.amm is not None and source == "token":
            self.amm.on_logs(slot, sig, logs)
        t = now_ms()
        # Pertes silencieuses : pump.fun a de l'activite a presque chaque slot (0,4 s). Un trou de plus de
        # SLOT_GAP slots entre deux notifications = donnees perdues (meme si la connexion est restee ouverte).
        if source == "firehose" and self.last_slot is not None and slot > self.last_slot + SLOT_GAP:
            self.record_outage(self.slot_ms(self.last_slot) - 2_000, self.slot_ms(slot) + 2_000)
        if self.last_slot is None or slot > self.last_slot:
            self.last_slot = slot
        for ev in pump.iter_events(logs):
            if isinstance(ev, pump.Trade):
                if ev.ts:
                    self.anchor = (slot, ev.ts * 1000 + 500)
                if ev.vsol <= 0 or ev.vtok <= 0:
                    # Reserves a 0 : famille de coins pump.fun sans prix en SOL (jamais une courbe standard).
                    # On les garde quand meme, pour que l'entrainement et le live les excluent explicitement.
                    self.bump("trades_unpriced")
                if ev.mint not in self.tracked or (self.per_token and not self.sampled.get(ev.mint)):
                    self.bump("trades_untracked")
                    continue
                self.sinks["trades"].add({
                    "recv_ms": t, "slot": slot, "signature": sig, "mint": ev.mint, "is_buy": ev.is_buy,
                    "sol_amount": ev.sol_amount, "token_amount": ev.token_amount, "user": ev.user, "ts": ev.ts,
                    "vsol": ev.vsol, "vtok": ev.vtok, "real_sol": ev.real_sol, "real_tok": ev.real_tok})
                self.hot_buf["trades"].append((ev.mint, t, slot, int(ev.is_buy), ev.sol_amount, ev.token_amount,
                                               ev.user, ev.vsol, ev.vtok, ev.real_tok, sig, ev.real_sol, ev.ts))
                self.bump("trades")
                if ev.user in self.kol_wallets:  # achat / vente d'un KOL sur un coin suivi
                    self.on_kol({"signature": sig, "mint": ev.mint, "traderPublicKey": ev.user,
                                 "txType": "buy" if ev.is_buy else "sell", "solAmount": ev.sol_amount / 1e9,
                                 "tokenAmount": ev.token_amount / 1e6,
                                 "marketCapSol": pump.mcap_sol(ev.vsol, ev.vtok) if ev.vsol > 0 else None},
                                t, source="chain")
            elif isinstance(ev, pump.Create):
                self.register_token({
                    "recv_ms": t, "source": "chain", "mint": ev.mint, "name": ev.name, "symbol": ev.symbol,
                    "uri": ev.uri, "creator": ev.creator or ev.user, "bonding_curve": ev.bonding_curve,
                    "signature": sig, "vsol": float(pump.INITIAL_VSOL), "vtok": float(pump.INITIAL_VTOK),
                    "chain_ms": ev.ts * 1000 if ev.ts else None})
                self.bump("creates_chain")
            elif isinstance(ev, pump.Complete):
                self.on_migration(ev.mint, "chain", sig, chain_ms=ev.ts * 1000 if ev.ts else None)

    def slot_ms(self, slot: int) -> int:
        """Heure on-chain estimee d'un slot (ancre sur le dernier trade horodate, 400 ms par slot)."""
        a_slot, a_ms = self.anchor
        return int(a_ms + (slot - a_slot) * 400)

    def record_outage(self, start: int, end: int):
        """Trou dans le flux de trades (reconnexion, arret, pertes) en heure on-chain : les tokens
        concernes ont des trades manquants, l'entrainement et le paper trader les excluent."""
        self.sinks["outages"].add({"recv_ms": end, "start_ms": start, "end_ms": end, "stream": "chain"})
        self.hot_buf["outages"].append((start, end, end))
        self.bump("outages")
        self.bump("outage_s", (end - start) / 1000)
        log.warning("coupure du flux de trades : %.1f s", (end - start) / 1000)

    def _chain_url(self) -> str:
        """Source principale (config ``chain_primary``) ; secours 10 min apres 3 echecs de suite."""
        helius = self.cfg.chain_ws_url() if self.cfg.secrets.helius_key else None
        primary, backup = self.c["public_ws"], helius
        if self.c.get("chain_primary") == "helius" and helius:
            primary, backup = helius, self.c["public_ws"]
        if backup is None:
            return primary
        if time.monotonic() < self.fallback_until:
            return backup
        if self.chain_failures >= 3:
            log.warning("source principale en echec %d fois : secours pendant 10 min", self.chain_failures)
            self.fallback_until = time.monotonic() + 600
            self.chain_failures = 0
            self.bump("chain_fallbacks")
            return backup
        return primary

    def pp_healthy(self) -> bool:
        return self.connected["pumpportal"] and time.monotonic() - self.last_msg["pumpportal"] < 45

    async def run_chain(self):
        log.info("trades : source principale %s, secours %s", self.c.get("chain_primary", "public"),
                 "helius" if self.cfg.secrets.helius_key else "aucun (pas de HELIUS_API_KEY)")
        sub = {"jsonrpc": "2.0", "id": 1, "method": "logsSubscribe",
               "params": [{"mentions": [pump.PUMP_PROGRAM]}, {"commitment": self.c["commitment"]}]}
        backoff = 1.0
        while not self.stop.is_set():
            self.connected["chain"] = False
            if self.per_token and self.pp_healthy():
                await asyncio.sleep(5)  # PumpPortal fournit les creations : flux global inutile
                continue
            url = self._chain_url()
            got_data = False
            try:
                async with websockets.connect(url, ping_interval=30, ping_timeout=30, max_size=2 ** 24,
                                              open_timeout=20) as ws:
                    await ws.send(json.dumps(sub))
                    log.info("websocket chaine connecte")
                    self.connected["chain"] = True
                    self.last_msg["chain"] = time.monotonic()
                    backoff = 1.0
                    async for raw in ws:
                        if self.per_token and self.pp_healthy():
                            log.info("PumpPortal de retour : flux global coupe")
                            break
                        self.last_msg["chain"] = time.monotonic()
                        self.bump("chain_msgs")
                        self.bump("chain_kb", len(raw) / 1024)
                        m = json.loads(raw)
                        if m.get("method") == "logsNotification":
                            if not got_data:
                                got_data, self.chain_failures = True, 0
                            res = m["params"]["result"]
                            try:
                                self.on_logs(res["context"]["slot"], res["value"])
                            except Exception:
                                log.exception("logs illisibles")
                                self.bump("chain_errors")
                        elif "error" in m:
                            log.error("erreur RPC: %s", m["error"])
                            break  # abonnement refuse (quota...) : on compte un echec et on se reconnecte
                        elif "result" in m:
                            log.info("abonnement logs id=%s", m["result"])
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("websocket chaine deconnecte: %s", e)
            self.connected["chain"] = False
            if not got_data:
                self.chain_failures += 1
            self.bump("chain_reconnects")
            await asyncio.sleep(backoff + random.random())
            backoff = min(backoff * 2, 60)

    # ---------- enrichissement ----------
    async def meta_worker(self, s: aiohttp.ClientSession):
        m = self.cfg["metadata"]
        imgdir = self.cfg.data / "images"
        while True:
            mint, uri = await self.meta_q.get()
            try:
                j = await md.fetch_json(s, uri, m["ipfs_gateways"], m["timeout_seconds"])
                row = {"mint": mint, "uri": uri, "ok": j is not None}
                if j:
                    row.update(md.parse_metadata(j))
                    row["raw"] = json.dumps(j)[:8000]
                    if m["download_images"] and row.get("image_uri"):
                        p = imgdir / day_of(now_ms()) / f"{mint}.jpg"
                        if await md.fetch_image(s, row["image_uri"], m["ipfs_gateways"], m["timeout_seconds"],
                                                m["max_image_bytes"], p):
                            row["image_path"] = str(p.relative_to(self.cfg.data))
                    links = [row.get("twitter"), row.get("website")]
                    tw = next((l for l in links if social.parse_link(l)["kind"] in ("tweet", "profile", "community")), None)
                    if tw:
                        try:
                            self.social_q.put_nowait((mint, tw))
                        except asyncio.QueueFull:
                            self.bump("social_dropped")
                self.decide_tracking(mint, row)
                # Horodatage APRES le telechargement : c'est le moment ou l'info est reellement disponible
                # (l'entrainement s'en sert pour savoir si elle etait connue au moment de decider).
                row["recv_ms"] = now_ms()
                self.sinks["metadata"].add(row)
                self.hot_buf["metadata"].append((mint, row["recv_ms"], row.get("description"), row.get("twitter"),
                                                 row.get("telegram"), row.get("website"), row.get("image_path")))
                self.bump("meta_ok" if j else "meta_fail")
            except Exception:
                log.exception("metadata %s", mint)
            finally:
                self.meta_q.task_done()

    async def social_worker(self, s: aiohttp.ClientSession, provider: social.Provider, rl: social.RateLimiter):
        while True:
            mint, url = await self.social_q.get()
            try:
                link = social.parse_link(url)
                row = {"mint": mint, "url": url, "kind": link["kind"], "handle": link["handle"],
                       "tweet_id": link["tweet_id"] or link["community_id"], "provider": provider.name, "ok": True}
                sc = self.cfg["social"]
                created = self.tracked.get(mint, now_ms())
                too_late = lambda: now_ms() - created > sc["max_coin_age_s"] * 1000
                stale = link["tweet_ms"] is not None and (created - link["tweet_ms"]) > sc["fresh_minutes"] * 60_000
                if provider.name != "none" and link["kind"] in ("tweet", "profile"):
                    # On ne paie pas : un vieux tweet (deja utilise ailleurs ; son age, gratuit, suffit) ou une
                    # lecture dont la reponse arriverait apres la decision a 60 s.
                    if not stale and not too_late():
                        await rl.wait()
                    if stale or too_late():
                        row["provider"] = "ignore:" + ("vieux_tweet" if stale else "trop_tard")
                        self.bump("social_skipped")
                    else:
                        try:
                            info = await provider.fetch(link)
                            row.update({k: v for k, v in info.items() if v is not None})
                            self.bump("social_ok")
                            if row.get("rel") == "reply" and not row.get("parent_handle") and not too_late():
                                # reponse a un autre tweet : on lit aussi le tweet d'origine pour comparer
                                await rl.wait()
                                try:
                                    row.update({k: v for k, v in (await social.fetch_parent(
                                        provider, row["parent_id"])).items() if v is not None})
                                    self.bump("social_parent")
                                except Exception as e:
                                    log.debug("tweet d'origine %s: %s", row["parent_id"], e)
                        except Exception as e:
                            row["ok"] = False
                            self.bump("social_fail")
                            log.debug("social %s: %s", url, e)
                row["recv_ms"] = now_ms()  # apres l'appel au fournisseur (voir metadata)
                self.sinks["social"].add(row)
                self.hot_buf["social"].append((mint, row["recv_ms"], json.dumps(
                    {k: v for k, v in row.items() if k != "raw"})))
            except Exception:
                log.exception("social %s", mint)
            finally:
                self.social_q.task_done()

    # ---------- boucles de maintenance ----------
    async def flush_loop(self):
        sql = {
            # premiere source gagnante, mais l'heure on-chain (source "chain") est toujours completee
            "tokens": "INSERT INTO tokens VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(mint) DO UPDATE SET"
                      " chain_ms=COALESCE(tokens.chain_ms, excluded.chain_ms),"
                      " sampled=COALESCE(tokens.sampled, excluded.sampled)",
            "trades": "INSERT INTO trades VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            "migrations": "INSERT OR IGNORE INTO migrations VALUES(?,?)",
            "metadata": "INSERT OR REPLACE INTO metadata VALUES(?,?,?,?,?,?,?)",
            "social": "INSERT OR REPLACE INTO social VALUES(?,?,?)",
            "outages": "INSERT INTO outages VALUES(?,?,?)",
            "swaps": "INSERT INTO swaps VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            "pools": "INSERT OR REPLACE INTO pools VALUES(" + ",".join("?" * 17) + ")",
            "kol": "INSERT OR IGNORE INTO kol_trades VALUES(?,?,?,?,?,?,?,?)",
            "sampled": "UPDATE tokens SET sampled=1 WHERE mint=?",
            "spam": "UPDATE tokens SET sampled=0 WHERE mint=?",
        }
        while True:
            await asyncio.sleep(1.0)
            bufs = {k: v for k, v in self.hot_buf.items() if v}
            for k in bufs:
                self.hot_buf[k] = []
            try:
                for k, rows in bufs.items():
                    await asyncio.to_thread(self.hot.write, sql[k], rows)
                for sink in self.sinks.values():
                    if sink.due():
                        await asyncio.to_thread(sink.flush)
            except Exception:
                log.exception("ecriture disque")

    async def watchdog(self):
        stall = self.c["stall_seconds"]
        while True:
            await asyncio.sleep(15)
            for name, t in self.last_msg.items():
                if self.connected[name] and time.monotonic() - t > stall:
                    log.error("aucun message %s depuis %ds", name, int(time.monotonic() - t))
                    self.bump(f"{name}_stalls")
                    self.last_msg[name] = time.monotonic()
                    for task in self.tasks:
                        if task.get_name() == name:
                            self.restart.add(name)
                            task.cancel()  # relance par _supervise

    async def housekeeping(self):
        n = 0
        while True:
            await asyncio.sleep(60)
            n += 1
            cut = now_ms() - int(self.c["track_hours"] * 3600_000)
            for mint in [m for m, t in self.tracked.items() if t < cut]:
                del self.tracked[mint]
            self.stats["chain_kb"] = round(self.stats.get("chain_kb", 0))
            if "outage_s" in self.stats:
                self.stats["outage_s"] = round(self.stats["outage_s"], 1)
            if self.per_token:
                self.stats.update(suivis_actifs=len(self.tok.active),
                                  credits_jour=round(self.tok.credits_today), budget_jour=self.tok.budget)
                self.stats["helius_credits"] = round(self.stats.get("helius_credits", 0))
                self.stats["credits_pools"] = round(self.stats.get("credits_pools", 0))
                self.stats["credits_ws"] = round(self.stats.get("credits_ws", 0))
            info = json.dumps({**self.stats, "tracked": len(self.tracked), "meta_q": self.meta_q.qsize(),
                               "social_q": self.social_q.qsize()})
            log.info("stats %s", info)
            await asyncio.to_thread(self.hot.heartbeat, "collector", info)
            self.stats = {}
            if n % 30 == 0:
                await asyncio.to_thread(self.hot.prune, self.c["hot_retention_hours"])
                if len(self.seen_meta) > 500_000:
                    self.seen_meta = set(self.tracked)

    async def pause_watch(self):
        """Fichier data/PAUSE (scripts/pause.ps1) : arret propre, les tampons sont vides sur disque."""
        while not (self.cfg.data / "PAUSE").exists():
            await asyncio.sleep(2)
        log.info("pause demandee")
        self.stop.set()

    async def _supervise(self, name: str, factory):
        """Relance une tache si elle meurt ou est annulee par le watchdog."""
        while not self.stop.is_set():
            task = asyncio.create_task(factory(), name=name)
            self.tasks.append(task)
            try:
                await task
            except asyncio.CancelledError:
                if name not in self.restart:
                    raise
                self.restart.discard(name)
                log.warning("tache %s relancee", name)
            except Exception:
                log.exception("tache %s plantee, relance", name)
                await asyncio.sleep(5)
            finally:
                self.tasks.remove(task)

    async def run(self):
        self.tasks: list[asyncio.Task] = []
        self.restart: set[str] = set()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop.set)
            except (NotImplementedError, RuntimeError):
                pass  # Windows : Ctrl+C leve KeyboardInterrupt, gere dans main()
        conn = aiohttp.TCPConnector(limit=64, ttl_dns_cache=300)
        async with aiohttp.ClientSession(connector=conn, headers={"User-Agent": "pumpml/1.0"}) as s:
            sc = self.cfg["social"]
            provider = social.make_provider(self.cfg.secrets.twitter_provider, self.cfg.secrets.twitter_key, s)
            log.info("fournisseur X : %s", provider.name)
            rl = social.RateLimiter(sc["rate_per_minute"])
            runners = [
                self._supervise("pumpportal", self.run_pumpportal),
                self._supervise("chain", self.run_chain),
                self._supervise("flush", self.flush_loop),
                self._supervise("watchdog", self.watchdog),
                self._supervise("housekeeping", self.housekeeping),
                self._supervise("pause", self.pause_watch),
            ]
            if self.per_token:
                runners += [self._supervise("tokstream", self.tok.run),
                            self._supervise("tokstream_hk", self.tok.housekeeping),
                            self._supervise("gapfill", self.tok.gapfill_worker)]
                if self.amm is not None:
                    runners.append(self._supervise("migrations", self.amm.worker))
                log.info("trades coin par coin (Helius) : %.0f %% des coins, budget %d credits/jour",
                         100 * self.tok.fraction, self.tok.budget)
            runners += [self._supervise(f"meta{i}", lambda: self.meta_worker(s))
                        for i in range(self.cfg["metadata"]["concurrency"])]
            runners += [self._supervise(f"social{i}", lambda: self.social_worker(s, provider, rl))
                        for i in range(sc["concurrency"])]
            main = asyncio.gather(*runners)
            stopper = asyncio.create_task(self.stop.wait())
            try:
                await asyncio.wait([main, stopper], return_when=asyncio.FIRST_COMPLETED)
            finally:
                main.cancel()
                try:
                    await main
                except (asyncio.CancelledError, Exception):
                    pass  # arret normal des taches
                self.shutdown()

    def shutdown(self):
        log.info("arret : vidage des tampons")
        for sink in self.sinks.values():
            try:
                sink.flush()
            except Exception:
                log.exception("flush final")


def main(cfg: Config):
    c = Collector(cfg)
    try:
        asyncio.run(c.run())
    except KeyboardInterrupt:
        c.shutdown()
