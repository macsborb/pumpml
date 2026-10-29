"""Apres la migration : coins echanges sur PumpSwap (AMM de pump.fun).

* Les migrations sont vues via un compte present dans TOUTES les transactions ``MigrateV2`` (et
  seulement celles-la) : 100 % des migrations pour quelques credits par jour.
* Chaque pool cree est suivi (abonnement aux logs + rattrapage RPC, comme les courbes) pendant
  ``track_minutes`` : evenements BuyEvent / SellEvent du programme PumpSwap.
* Detection des lancements "groupes" (rug) : le createur achete toute ou une grosse partie de la courbe
  au bloc 0 avec ses propres wallets, pour simuler une tendance puis revendre sur les acheteurs.
  Mesures : part de la courbe achetee dans le slot de creation (+2 slots), nombre de wallets, delai
  creation -> migration, concentration des plus gros detenteurs juste apres la migration.

Format des evenements verifie sur de vraies transactions (soldes avant/apres) :
reserves du pool AVANT le trade ; frais totaux ~1 % a ces niveaux (lp + protocole + createur).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import struct
from dataclasses import dataclass

import aiohttp
import numpy as np

from . import pump

log = logging.getLogger("amm")

AMM_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
# Compte present dans chaque transaction de migration pump.fun -> PumpSwap (et seulement celles-la).
MIGRATION_ACCOUNT = "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"
WSOL = "So11111111111111111111111111111111111111112"
CURVE_SUPPLY = 793_100_000_000_000  # tokens vendus par la courbe (unites brutes)
TOTAL_RAW = pump.TOTAL_SUPPLY * 10 ** pump.TOKEN_DECIMALS


def _disc(name: str) -> bytes:
    return hashlib.sha256(f"event:{name}".encode()).digest()[:8]


DISC_BUY = _disc("BuyEvent")
DISC_SELL = _disc("SellEvent")


@dataclass
class Swap:
    ts: int
    is_buy: bool
    token_amount: int   # tokens recus (achat) ou vendus (vente)
    sol_amount: int     # cote utilisateur, frais inclus : paye (achat) ou recu (vente)
    sol_swap: int       # avant frais : sert au prix d'execution
    pool_tok: int       # reserves du pool AVANT le trade
    pool_sol: int
    pool: str
    user: str

    @property
    def price(self) -> float:
        """Prix d'execution avant frais, en lamports par unite brute de token."""
        return self.sol_swap / self.token_amount if self.token_amount else 0.0

    @property
    def fee(self) -> float:
        """Frais totaux en fraction (lp + protocole + createur)."""
        if self.is_buy:
            return self.sol_amount / self.sol_swap - 1 if self.sol_swap else 0.0
        return 1 - self.sol_amount / self.sol_swap if self.sol_swap else 0.0


def decode_swap(raw: bytes) -> Swap | None:
    d = raw[:8]
    if d not in (DISC_BUY, DISC_SELL) or len(raw) < 8 + 14 * 8 + 64:
        return None
    v = struct.unpack_from("<q13Q", raw, 8)
    o = 8 + 14 * 8
    pool = pump.base58.b58encode(raw[o:o + 32]).decode()
    user = pump.base58.b58encode(raw[o + 32:o + 64]).decode()
    if d == DISC_BUY:
        # ts, base_out, max_quote_in, user_base, user_quote, pool_base, pool_quote, quote_in (paye),
        # lp_bps, lp_fee, proto_bps, proto_fee, quote_in_with_lp_fee, user_quote_in (net, avant frais)
        return Swap(v[0], True, v[1], v[7], v[13], v[5], v[6], pool, user)
    # ts, base_in, min_quote_out, user_base, user_quote, pool_base, pool_quote, quote_out (brut),
    # lp_bps, lp_fee, proto_bps, proto_fee, quote_out_without_lp_fee, user_quote_out (net recu)
    return Swap(v[0], False, v[1], v[13], v[7], v[5], v[6], pool, user)


def iter_swaps(logs: list[str]):
    for line in pump.program_data(logs, AMM_PROGRAM):
        try:
            sw = decode_swap(base64.b64decode(line[14:], validate=False))
        except (ValueError, struct.error):
            continue
        if sw is not None and sw.token_amount > 0 and sw.sol_swap > 0:
            yield sw


def is_migration(logs: list[str]) -> bool:
    return any(l.startswith("Program log: Instruction: Migrate") for l in logs)


def parse_migration(tx: dict) -> tuple[str, str, str] | None:
    """(mint, pool, coffre de tokens du pool) d'une transaction de migration."""
    meta = tx["meta"]
    msg = tx["transaction"]["message"]
    keys = msg["accountKeys"] + (meta.get("loadedAddresses") or {}).get("writable", []) + \
        (meta.get("loadedAddresses") or {}).get("readonly", [])
    # pool = proprietaire qui detient a la fois le coin et des WSOL apres la migration
    by_owner: dict[str, dict] = {}
    for b in meta.get("postTokenBalances") or []:
        by_owner.setdefault(b.get("owner"), {})[b["mint"]] = (int(b["uiTokenAmount"]["amount"]), keys[b["accountIndex"]])
    for owner, mints in by_owner.items():
        coins = [m for m in mints if m != WSOL]
        if WSOL in mints and coins and mints[coins[0]][0] > 0:
            return coins[0], owner, mints[coins[0]][1]
    return None


# ---------- lancement groupe / concentration ----------

def rug_reasons(row: dict, a: dict) -> list[str]:
    """Raisons d'exclure un coin migre (seuils dans [amm] de config.toml). Liste vide = eligible."""
    out = []
    if row.get("block0_share") is None:
        # historique de courbe illisible (trop long, coin tres ancien) : prudence, on ne sait pas qui a achete
        out.append("historique_inconnu")
    if row.get("curve_s") is not None and row["curve_s"] < a["min_curve_seconds"]:
        out.append("migre_trop_vite")
    if (row.get("block0_share") or 0) >= a["max_block0_share"]:
        out.append("achat_groupe_bloc0")
    if (row.get("top10_share") or 0) >= a["max_top10_share"]:
        out.append("detention_concentree")
    return out


class Migrations:
    """Branche "apres migration" du collecteur. Reutilise le websocket et le rattrapage du suivi coin
    par coin (TokenStream) : un pool est une adresse suivie de plus."""

    def __init__(self, col, tok):
        self.col, self.tok = col, tok
        self.a = col.cfg["amm"]
        self.track_ms = int(self.a["track_minutes"] * 60_000)
        self.pools: dict[str, str] = {}   # pool -> mint
        self.q: asyncio.Queue = asyncio.Queue()
        self.done: set[str] = set()
        self.sigs_seen: set[str] = set()

    def restore(self):
        """Pools encore dans leur fenetre de suivi apres un redemarrage."""
        for r in self.col.hot.query("SELECT mint, pool, migrated_ms FROM pools WHERE migrated_ms >= ?",
                                    (pump_now() - self.track_ms,)):
            self.pools[r["pool"]] = r["mint"]
            self.done.add(r["mint"])
            # pas de rattrapage initial (deja fait) : la reconnexion rattrape la periode d'arret
            self.tok.active[r["pool"]] = {"mint": r["mint"], "until": r["migrated_ms"] + self.track_ms, "sub": None,
                                          "kind": "pool"}

    def on_logs(self, slot: int, sig: str, logs: list[str]):
        """Appele pour chaque transaction recue par le suivi coin par coin."""
        if is_migration(logs):
            self.q.put_nowait((sig, slot))
        t = pump_now()
        for sw in iter_swaps(logs):
            mint = self.pools.get(sw.pool)
            if mint is None:
                self.col.bump("swaps_untracked")
                continue
            row = {"recv_ms": t, "slot": slot, "signature": sig, "mint": mint, "pool": sw.pool, "is_buy": sw.is_buy,
                   "sol_amount": sw.sol_amount, "sol_swap": sw.sol_swap, "token_amount": sw.token_amount,
                   "user": sw.user, "ts": sw.ts, "pool_sol": sw.pool_sol, "pool_tok": sw.pool_tok}
            self.col.sinks["swaps"].add(row)
            self.col.hot_buf["swaps"].append((mint, t, slot, int(sw.is_buy), sw.sol_amount, sw.sol_swap,
                                              sw.token_amount, sw.user, sw.ts, sw.pool_sol, sw.pool_tok, sig))
            self.col.bump("swaps")

    async def worker(self):
        """Migrations traitees en parallele (chacune fait des dizaines d'appels RPC : en serie, la file
        s'allongeait et ~80 % des migrations n'etaient jamais traitees)."""
        sem = asyncio.Semaphore(int(self.a.get("concurrency", 6)))
        running: set[asyncio.Task] = set()

        async def one(s, sig):
            async with sem:
                try:
                    await self._handle(s, sig)
                except Exception as e:
                    log.warning("migration %s illisible : %s", sig[:10], e)
                    self.col.bump("amm_errors")

        async with aiohttp.ClientSession() as s:
            while True:
                sig, slot = await self.q.get()
                if sig in self.sigs_seen:
                    continue
                self.sigs_seen.add(sig)
                t = asyncio.create_task(one(s, sig))
                running.add(t)
                t.add_done_callback(running.discard)
                self.col.stats["amm_file"] = self.q.qsize() + len(running)

    async def _handle(self, s, sig: str):
        tx = await self.tok._rpc(s, "getTransaction", [sig, {"encoding": "json", "maxSupportedTransactionVersion": 1,
                                                             "commitment": "confirmed"}])
        if not tx or (tx.get("meta") or {}).get("err") is not None:
            return
        got = parse_migration(tx)
        if got is None or got[0] in self.done:
            return
        mint, pool, vault = got
        self.done.add(mint)
        mig_ms = (tx.get("blockTime") or 0) * 1000 or pump_now()
        self.pools[pool] = mint
        self.tok.watch(pool, mint, mig_ms + self.track_ms, "pool", since_ms=mig_ms)
        self.col.on_migration(mint, "amm", sig, chain_ms=mig_ms)
        self.col.bump("pools")
        row = {"recv_ms": pump_now(), "mint": mint, "pool": pool, "signature": sig, "migrated_ms": mig_ms,
               "migrate_slot": tx["slot"], "vault": vault}
        try:
            row.update(await self._curve_history(s, mint, tx["slot"], mig_ms))
        except Exception as e:
            log.info("historique de courbe %s indisponible : %s", mint[:8], e)
        await asyncio.sleep(3)  # laisse les premiers echanges se faire avant la photo des detenteurs
        try:
            row.update(await self._holders(s, mint, vault))
        except Exception as e:
            log.info("detenteurs %s indisponibles : %s", mint[:8], e)
        row["reasons"] = ",".join(rug_reasons(row, self.a))
        if row["reasons"] and pool in self.tok.active:
            # coin exclu : suivi court, juste de quoi comparer aux eligibles (le suivi complet coute cher)
            self.tok.active[pool]["until"] = mig_ms + int(self.a["excluded_track_minutes"] * 60_000)
        self.col.sinks["pools"].add(row)
        self.col.hot_buf["pools"].append(tuple(row.get(c) for c in POOL_COLS))
        log.info("migration %s : courbe %.0f s, bloc 0 %.0f %% (%s wallets), top10 %.0f %% -> %s", mint[:8],
                 -1 if row.get("curve_s") is None else row["curve_s"], 100 * (row.get("block0_share") or 0), row.get("block0_wallets"),
                 100 * (row.get("top10_share") or 0), row["reasons"] or "eligible")

    async def _curve_history(self, s, mint: str, mig_slot: int, mig_ms: int) -> dict:
        """Toutes les signatures de la courbe (1 credit / 1000), puis decodage des seuls trades du bloc 0."""
        from .backfill import bonding_curve_pda
        curve = bonding_curve_pda(mint)
        sigs, before = [], None
        for _ in range(int(self.a["max_sig_pages"])):
            p = {"limit": 1000, "commitment": "confirmed"}
            if before:
                p["before"] = before
            page = await self.tok._rpc(s, "getSignaturesForAddress", [curve, p])
            sigs += page
            if len(page) < 1000:
                break
            before = page[-1]["signature"]
        else:
            return {"curve_tx": len(sigs)}  # trop long : historique tronque, pas de mesure du bloc 0
        ok = [x for x in sigs if x.get("err") is None]
        if not ok:
            return {}
        s0 = min(x["slot"] for x in ok)
        t0 = min((x.get("blockTime") or 0) for x in ok if x["slot"] == s0) * 1000
        first = sorted((x for x in ok if x["slot"] <= s0 + 2), key=lambda x: x["slot"])[:40]
        tok_bought, wallets, sol = 0, set(), 0
        for x in first:
            t = await self.tok._rpc(s, "getTransaction", [x["signature"], {
                "encoding": "json", "maxSupportedTransactionVersion": 1, "commitment": "confirmed"}])
            for ev in pump.iter_events(((t or {}).get("meta") or {}).get("logMessages") or []):
                if isinstance(ev, pump.Trade) and ev.mint == mint and ev.is_buy:
                    tok_bought += ev.token_amount
                    sol += ev.sol_amount
                    wallets.add(ev.user)
        return {"curve_tx": len(ok), "create_slot": s0, "curve_s": (mig_ms - t0) / 1000 if t0 else None,
                "block0_share": tok_bought / CURVE_SUPPLY, "block0_wallets": len(wallets), "block0_sol": sol / 1e9,
                "block0_tx": len(first)}

    async def _holders(self, s, mint: str, vault: str | None) -> dict:
        res = await self.tok._rpc(s, "getTokenLargestAccounts", [mint, {"commitment": "confirmed"}])
        sup = await self.tok._rpc(s, "getTokenSupply", [mint, {"commitment": "confirmed"}])
        total = int(sup["value"]["amount"]) or TOTAL_RAW  # certains coins ont 2 milliards de tokens
        amts = [int(a["amount"]) for a in res.get("value", []) if a.get("address") != vault]
        amts.sort(reverse=True)
        return {"top1_share": amts[0] / total if amts else 0.0, "top10_share": sum(amts[:10]) / total,
                "supply": total}


POOL_COLS = ("mint", "recv_ms", "pool", "signature", "migrated_ms", "migrate_slot", "create_slot", "curve_s",
             "curve_tx", "block0_share", "block0_wallets", "block0_sol", "block0_tx", "top1_share", "top10_share",
             "reasons", "supply")


def pump_now() -> int:
    from .storage import now_ms
    return now_ms()


# ---------- simulation (prix d'execution des echanges du pool) ----------

class Swaps:
    """Echanges d'un pool, tries en heure on-chain."""

    def __init__(self, rows: list[dict]):
        rows = sorted(rows, key=lambda r: (r["ts"] or 0, r["slot"]))
        self.ms = np.array([(r["ts"] or 0) * 1000 for r in rows], dtype=np.int64)
        self.price = np.array([r["sol_swap"] / r["token_amount"] for r in rows], dtype=np.float64)
        self.fee = np.array([(r["sol_amount"] / r["sol_swap"] - 1) if r["is_buy"] else (1 - r["sol_amount"] / r["sol_swap"])
                             for r in rows], dtype=np.float64)
        self.pool_sol = np.array([r["pool_sol"] for r in rows], dtype=np.float64)

    def __len__(self):
        return len(self.ms)


def simulate(sw: Swaps, decision_ms: int, s: dict) -> dict | None:
    """Achat de ``buy_sol`` a decision + latence au dernier prix connu, sortie TP / SL / stale / duree max.
    Valeur de vente = tokens x prix x (1 - frais), avec l'impact de notre propre ordre sur la reserve SOL."""
    entry = decision_ms + int(s["latency_s"] * 1000)
    i = int(np.searchsorted(sw.ms, entry, side="right"))
    if i == 0:
        return None
    p0, f0, r0 = sw.price[i - 1], max(sw.fee[i - 1], 0.0), sw.pool_sol[i - 1]
    net = s["buy_sol"] * 1e9 / (1 + f0)
    tokens = net / p0 * r0 / (r0 + net)
    cost = s["buy_sol"] + 2 * s["tx_cost_sol"]

    def value(j):
        g = tokens * sw.price[j]
        return g * sw.pool_sol[j] / (sw.pool_sol[j] + g) * (1 - max(sw.fee[j], 0.0)) / 1e9

    tp = s["buy_sol"] * (1 + s["take_profit"]) if s.get("take_profit", 0) > 0 else float("inf")
    sl = s["buy_sol"] * (1 - s["stop_loss"])
    end = entry + int(s["max_hold_s"] * 1000)
    stale = entry + int(s["stale_exit_s"] * 1000) if s.get("stale_exit_s") else None
    stale_thr = s["buy_sol"] * (1 + s.get("stale_min_gain", 0.1))
    maxv, best, k, reason = 0.0, p0, i - 1, "timeout"
    for j in range(i, len(sw)):
        if sw.ms[j] > end:
            break
        if stale is not None and sw.ms[j] >= stale and maxv < stale_thr:
            reason = "stale"
            break
        k = j
        v = value(j)
        maxv, best = max(maxv, v), max(best, sw.price[j])
        if v >= tp:
            reason = "tp"
            break
        if v <= sl:
            reason = "sl"
            break
    if reason in ("tp", "sl"):
        k = int(np.searchsorted(sw.ms, sw.ms[k] + int(s["latency_s"] * 1000), side="right")) - 1
    out = value(k) if k >= i else value(i - 1)
    return {"pnl": out - cost, "reason": reason, "max_mult": best / p0, "entry_price": p0,
            "hold_s": max(0, int(sw.ms[k]) - entry) / 1000 if k >= i else 0.0}


# ---------- observation dans le paper trader (phase 1 : aucun achat) ----------

class Observer:
    """Pour chaque coin migre : verdict (eligible ou raison d'exclusion) au premier point d'analyse,
    cotation Jupiter reelle pour les eligibles (le pool est-il routable, a quel prix), puis resultat simule
    une fois la duree de detention passee. Sert a valider les seuils anti-rug avant tout achat."""

    def __init__(self, cfg, hot, journal, jup, reserve: int):
        self.s, self.a = cfg["strategy"], cfg["amm"]
        self.hot, self.j, self.jup, self.reserve = hot, journal, jup, reserve
        self.cp0 = int(self.a["checkpoints_s"][0] * 1000)
        self.hold = int((self.s["latency_s"] + self.s["max_hold_s"]) * 1000)
        self.done = {r["mint"] for r in self.j.query("SELECT mint FROM amm_decisions")}

    def step(self):
        now = pump_now()
        for p in self.hot.query("SELECT * FROM pools WHERE migrated_ms >= ? AND migrated_ms <= ?",
                                (now - 600_000, now - self.cp0)):
            if p["mint"] in self.done:
                continue
            self.done.add(p["mint"])
            tok = self.hot.query("SELECT name, symbol, COALESCE(chain_ms, recv_ms) c FROM tokens WHERE mint=?",
                                 (p["mint"],))
            t = dict(tok[0]) if tok else {}
            decided = max(p["migrated_ms"] + self.cp0, p["recv_ms"])
            q = None
            if not p["reasons"]:
                q = self.jup.buy(p["mint"], self.s["buy_sol"]) if self.jup.budget() > self.reserve else None
            self.j.exec("INSERT OR IGNORE INTO amm_decisions(mint, pool, migrated_ms, created_ms, name, symbol, reasons,"
                        " curve_s, block0_share, block0_wallets, top10_share, decided_ms, jup_tokens, jup_route,"
                        " jup_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (p["mint"], p["pool"], p["migrated_ms"], t.get("c"), t.get("name"), t.get("symbol"),
                         p["reasons"], p["curve_s"], p["block0_share"], p["block0_wallets"], p["top10_share"], decided,
                         q["out"] if q else None, q["route"] if q else None, now if q else None))

    def resolve(self):
        now = pump_now()
        for d in self.j.query("SELECT mint, decided_ms, jup_tokens, jup_ms FROM amm_decisions WHERE resolved=0"
                              " AND decided_ms < ?", (now - self.hold - 10_000,)):
            rows = self.hot.query("SELECT is_buy, sol_amount, sol_swap, token_amount, ts, slot, pool_sol FROM swaps"
                                  " WHERE mint=?", (d["mint"],))
            sw = Swaps([dict(r) for r in rows if r["token_amount"] and r["sol_swap"]])
            o = simulate(sw, d["decided_ms"], self.s) if len(sw) else None
            sim_tokens = None
            if o and d["jup_tokens"]:
                # tokens que la simulation aurait obtenus au moment de la cotation Jupiter (ecart sim / reel)
                i = int(np.searchsorted(sw.ms, d["jup_ms"], side="right"))
                if i:
                    net = self.s["buy_sol"] * 1e9 / (1 + max(sw.fee[i - 1], 0.0))
                    sim_tokens = net / sw.price[i - 1] * sw.pool_sol[i - 1] / (sw.pool_sol[i - 1] + net)
            self.j.exec("UPDATE amm_decisions SET resolved=1, n_swaps=?, pnl_sim=?, max_mult=?, exit_reason_sim=?,"
                        " hold_s=?, sim_tokens=? WHERE mint=?",
                        (len(sw), o["pnl"] if o else None, o["max_mult"] if o else None,
                         o["reason"] if o else "aucun_echange", o["hold_s"] if o else None, sim_tokens, d["mint"]))
