"""Suivi coin par coin via Helius : un abonnement ``logsSubscribe`` par courbe de bonding.

Pourquoi : le flux complet du programme pump.fun (~47 Mo/min) est trop gros, et le websocket public
gratuit en perd environ deux tiers. Ici on ne recoit que les trades des coins suivis (~7 Mo/min pour
100 % des coins), et ils sont COMPLETS :
* les trades passes avant l'abonnement (achat du dev, snipers) sont rattrapes par RPC ;
* apres une reconnexion, les trades manques sont rattrapes de la meme facon ;
* si un rattrapage echoue, la periode est enregistree comme coupure (le coin sera exclu).

Echantillonnage : ``sample_fraction`` des coins (tirage deterministe sur l'adresse), pour tenir dans
le budget de credits Helius. Un compteur estime la consommation et arrete d'ajouter des coins
si le budget du jour est atteint.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from collections import OrderedDict

import aiohttp
import websockets

from . import backfill
from .amm import MIGRATION_ACCOUNT
from .storage import now_ms

log = logging.getLogger("tokstream")
# Autorite de mint de pump.fun : presente dans TOUTES les transactions de creation (et seulement celles-la).
# S'y abonner donne 100 % des creations pour quelques credits par minute (PumpPortal en rate ~la moitie).
MINT_AUTHORITY = "TSLvdd1pWpHVjahSpsvCXUbgwsL3JAcvokwaKt1eokM"
BYTES_PER_CREDIT = 51_200  # 2 credits par 0,1 Mo de flux websocket


def in_sample(mint: str, fraction: float) -> bool:
    """Tirage deterministe : le meme coin est toujours dans (ou hors de) l'echantillon."""
    return int(hashlib.sha256(mint.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


class TokenStream:
    def __init__(self, col):
        self.col = col
        cfg = col.cfg
        c = cfg["collector"]
        self.key = cfg.secrets.helius_key
        self.ws_url = cfg.chain_ws_url()
        self.rpc_url = f"{c['helius_rpc']}?api-key={self.key}"
        self.track_ms = int(c["track_minutes"] * 60_000)
        self.fraction = c["sample_fraction"]
        self.base_budget = c["helius_daily_credit_budget"]
        # budget exceptionnel pour UN jour (date "AAAA-MM-JJ") : le budget normal revient tout seul ensuite
        self.override = c.get("helius_budget_override") or {}
        self.active: dict[str, dict] = {}          # courbe -> {mint, until, sub}
        self.by_sub: dict[int, str] = {}           # id d'abonnement -> courbe
        self.req: dict[int, str] = {}              # id de requete -> courbe
        self.next_id = 1
        self.out: asyncio.Queue = asyncio.Queue()  # messages a envoyer
        self.gap_q: asyncio.Queue = asyncio.Queue()
        self.connected = False
        self.down_since: int | None = None
        self.day = time.strftime("%Y-%m-%d")
        self.credits_today = 0.0
        # le compteur du jour survit aux redemarrages (sinon le garde-fou du budget repartirait de zero)
        try:
            r = col.hot.query("SELECT info FROM heartbeat WHERE name='tokstream'")
            st = json.loads(r[0]["info"] or "{}") if r else {}
            if st.get("day") == self.day:
                self.credits_today = float(st.get("credits_today") or 0)
        except Exception:
            pass
        self.seen: OrderedDict[str, None] = OrderedDict()  # signatures deja traitees
        self.reconnect_pending = 0  # rattrapages apres reconnexion pas encore faits
        self.addr_n: dict[str, int] = {}  # notifications par adresse suivie (diagnostic du cout)

    # ---------- budget ----------
    def spend(self, credits: float):
        d = time.strftime("%Y-%m-%d")
        if d != self.day:
            self.day, self.credits_today = d, 0.0
        self.credits_today += credits
        self.col.bump("helius_credits", credits)

    @property
    def budget(self) -> int:
        if self.override.get("date") == time.strftime("%Y-%m-%d"):
            return int(self.override["credits"])
        return self.base_budget

    def over_budget(self) -> bool:
        return self.credits_today >= self.budget

    def mark_seen(self, sig: str) -> bool:
        """True si la signature est nouvelle (un meme tx peut toucher plusieurs courbes suivies)."""
        if sig in self.seen:
            return False
        self.seen[sig] = None
        if len(self.seen) > 300_000:
            self.seen.popitem(last=False)
        return True

    # ---------- suivi ----------
    def want(self, mint: str) -> bool:
        return in_sample(mint, self.fraction) and not self.over_budget()

    def add(self, mint: str, created_ms: int):
        self.watch(backfill.bonding_curve_pda(mint), mint, created_ms + self.track_ms, "curve")

    def watch(self, address: str, mint: str, until_ms: int, kind: str, since_ms: int = 0):
        """Suit une adresse (courbe ou pool PumpSwap) jusqu'a ``until_ms``."""
        if address in self.active:
            return
        self.active[address] = {"mint": mint, "until": until_ms, "sub": None, "kind": kind}
        if self.connected:
            self._subscribe(address)
        # rattrapage des tout premiers echanges (avant que l'abonnement soit actif) : vite, pour l'analyse a 5 s,
        # puis une seconde passe pour ce qui n'etait pas encore indexe par le RPC
        self.gap_q.put_nowait((address, since_ms, time.monotonic() + 2))
        self.gap_q.put_nowait((address, since_ms, time.monotonic() + 12))

    def _subscribe(self, curve: str):
        rid = self.next_id
        self.next_id += 1
        self.req[rid] = curve
        self.out.put_nowait({"jsonrpc": "2.0", "id": rid, "method": "logsSubscribe",
                             "params": [{"mentions": [curve]}, {"commitment": "confirmed"}]})

    def _drop(self, address: str):
        a = self.active.pop(address, None)
        if a and a["sub"] is not None and self.connected:
            self.out.put_nowait({"jsonrpc": "2.0", "id": self.next_id, "method": "logsUnsubscribe", "params": [a["sub"]]})
            self.next_id += 1
            self.by_sub.pop(a["sub"], None)

    def _expire(self):
        now = now_ms()
        for curve, a in list(self.active.items()):
            if a["until"] < now:
                if a["sub"] is not None and self.connected:
                    self.out.put_nowait({"jsonrpc": "2.0", "id": self.next_id, "method": "logsUnsubscribe",
                                         "params": [a["sub"]]})
                    self.next_id += 1
                    self.by_sub.pop(a["sub"], None)
                del self.active[curve]

    async def run(self):
        backoff = 1.0
        while True:
            self.connected = False
            self.by_sub.clear()
            self.req.clear()
            while not self.out.empty():
                self.out.get_nowait()
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20, max_size=2 ** 24,
                                              open_timeout=20) as ws:
                    self.connected = True
                    self.col.connected["tokstream"] = True
                    self.out.put_nowait({"jsonrpc": "2.0", "id": 0, "method": "logsSubscribe",
                                         "params": [{"mentions": [MINT_AUTHORITY]}, {"commitment": "confirmed"}]})
                    if self.col.amm is not None:
                        # toutes les migrations vers PumpSwap
                        self.out.put_nowait({"jsonrpc": "2.0", "id": -1, "method": "logsSubscribe",
                                             "params": [{"mentions": [MIGRATION_ACCOUNT]}, {"commitment": "confirmed"}]})
                    for curve, a in self.active.items():
                        a["sub"] = None
                        self._subscribe(curve)
                    if self.down_since is not None:
                        # rattrapage de tout ce qui a ete manque pendant la deconnexion
                        for curve in list(self.active):
                            self.gap_q.put_nowait((curve, self.down_since, time.monotonic() + 3, "reconnect"))
                            self.reconnect_pending += 1
                        self.down_since = None
                    log.info("suivi coin par coin connecte (%d coins)", len(self.active))
                    backoff = 1.0
                    writer = asyncio.create_task(self._writer(ws))
                    try:
                        async for raw in ws:
                            self.spend(len(raw) / BYTES_PER_CREDIT)
                            self.col.bump("credits_ws", len(raw) / BYTES_PER_CREDIT)
                            self.col.last_msg["tokstream"] = time.monotonic()
                            m = json.loads(raw)
                            if m.get("method") == "logsNotification":
                                p = m["params"]
                                a = self.active.get(self.by_sub.get(p.get("subscription"), ""))
                                if a is not None and a.get("kind") == "pool":
                                    self.col.bump("credits_pools", len(raw) / BYTES_PER_CREDIT)
                                # mesure du cout par type d'abonnement (courbe, pool, creations / migrations)
                                kind = a.get("kind", "curve") if a is not None else "global"
                                self.col.bump(f"kb_{kind}", len(raw) / 1024)
                                self.col.bump(f"n_{kind}")
                                if a is not None:
                                    ad = self.by_sub.get(p.get("subscription"))
                                    self.addr_n[ad] = self.addr_n.get(ad, 0) + 1
                                v = p["result"]["value"]
                                if self.mark_seen(v.get("signature") or ""):
                                    self.col.on_logs(p["result"]["context"]["slot"], v, "token")
                            elif "id" in m and m["id"] in self.req:
                                curve = self.req.pop(m["id"])
                                if "result" in m and curve in self.active:
                                    self.active[curve]["sub"] = m["result"]
                                    self.by_sub[m["result"]] = curve
                                elif "error" in m:
                                    log.warning("abonnement refuse : %s", m["error"])
                                    self.col.bump("sub_errors")
                    finally:
                        writer.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("suivi coin par coin deconnecte : %s", e)
            self.col.connected["tokstream"] = False
            if self.down_since is None:
                self.down_since = now_ms()
            self.col.bump("token_ws_reconnects")
            await asyncio.sleep(backoff + random.random())
            backoff = min(backoff * 2, 30)

    async def _writer(self, ws):
        while True:
            msg = await self.out.get()
            await ws.send(json.dumps(msg))

    async def housekeeping(self):
        n = 0
        while True:
            await asyncio.sleep(5)
            self._expire()
            n += 1
            if n % 12 == 0 and self.addr_n:
                # Robots de volume : des milliers de transactions (souvent echouees) par minute sur un coin, toutes
                # facturees. Faux volume : suivi coupe, coin exclu de l'entrainement et des achats.
                cap = self.col.c.get("max_notifications_per_min", 1000)
                for a, c in self.addr_n.items():
                    st = self.active.get(a)
                    if c > cap and st is not None and st.get("kind", "curve") == "curve":
                        log.info("coin %s : %d transactions/min (robots de volume) -> suivi coupe, coin exclu",
                                 st["mint"][:8], c)
                        self._drop(a)
                        self.col.mark_spam(st["mint"])
                self.addr_n = {}
            # etat lu par le paper trader : jusqu'a quand les donnees sont completes
            info = json.dumps({"connected": self.connected, "down_since": self.down_since,
                               "pending_gapfill": self.reconnect_pending, "active": len(self.active),
                               "day": self.day, "credits_today": round(self.credits_today)})
            await asyncio.to_thread(self.col.hot.heartbeat, "tokstream", info)

    # ---------- rattrapage par RPC ----------
    async def _rpc(self, s: aiohttp.ClientSession, method: str, params: list):
        for attempt in range(6):
            self.spend(1)
            self.col.bump(f"rpc_{method}")
            try:
                async with s.post(self.rpc_url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                                  timeout=aiohttp.ClientTimeout(total=30)) as r:
                    if r.status == 429 or r.status >= 500:
                        raise aiohttp.ClientError(f"HTTP {r.status}")
                    j = await r.json()
                if "error" in j:
                    raise aiohttp.ClientError(str(j["error"]))
                return j["result"]
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(1 + attempt * 2 + random.random())
        raise RuntimeError(f"{method} : echec")

    async def gapfill_worker(self):
        """Recupere par RPC les trades d'un coin suivi absents du flux (debut, reconnexion).
        Les rattrapages s'executent en parallele (chacun attend son heure sans bloquer les autres)."""
        sem = asyncio.Semaphore(8)
        running: set[asyncio.Task] = set()

        async def one(s, curve, since_ms, not_before, reconnect):
            try:
                await self._gapfill(s, sem, curve, since_ms, not_before)
            finally:
                if reconnect:
                    self.reconnect_pending = max(0, self.reconnect_pending - 1)

        async with aiohttp.ClientSession() as s:
            while True:
                item = await self.gap_q.get()
                curve, since_ms, not_before = item[:3]
                t = asyncio.create_task(one(s, curve, since_ms, not_before, len(item) > 3))
                running.add(t)
                t.add_done_callback(running.discard)

    async def _gapfill(self, s, sem, curve: str, since_ms: int, not_before: float):
        delay = not_before - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        a = self.active.get(curve)
        if a is None:
            return
        if a.get("kind") == "pool":
            self.col.bump("credits_pools", 1)
        try:
            sigs = await self._rpc(s, "getSignaturesForAddress", [curve, {"limit": 1000, "commitment": "confirmed"}])
            todo = [x for x in sigs if x.get("err") is None and x["signature"] not in self.seen
                    and (x.get("blockTime") or 0) * 1000 >= since_ms - 5_000]

            async def fetch(x):
                async with sem:
                    tx = await self._rpc(s, "getTransaction", [x["signature"], {
                        "encoding": "json", "maxSupportedTransactionVersion": 1, "commitment": "confirmed"}])
                if tx and (tx.get("meta") or {}).get("err") is None and self.mark_seen(x["signature"]):
                    self.col.on_logs(x["slot"], {"signature": x["signature"], "err": None,
                                                 "logs": (tx.get("meta") or {}).get("logMessages") or []},
                                     "token")

            await asyncio.gather(*(fetch(x) for x in reversed(todo)))
            self.col.bump("gapfill_tx", len(todo))
        except Exception as e:
            # rattrapage impossible : periode enregistree comme coupure (coin(s) exclus)
            log.warning("rattrapage impossible pour %s : %s", a["mint"][:8], e)
            start = since_ms or (a["until"] - self.track_ms)
            self.col.record_outage(start - 2_000, now_ms())
