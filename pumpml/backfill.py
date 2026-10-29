"""Import d'un coin passe par son adresse (CA) : reconstitue ses premieres minutes depuis la blockchain.

Etapes : compte de la courbe de bonding (PDA) -> toutes ses signatures -> transactions des premieres
minutes (creation + trades jusqu'a l'horizon de la strategie) -> metadonnees, image, lien X
-> MEME calcul de features et de resultat simule que pour les coins collectes en direct.
Le coin devient un exemple d'entrainement (pondere davantage si tu l'as note).

Limites : l'horodatage est a la seconde (blockTime) ; les infos du compte X sont celles
d'aujourd'hui, pas celles du jour du lancement.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import re
from pathlib import Path

import aiohttp
import pandas as pd

from . import dataset as D
from . import features as F
from . import metadata as md
from . import pump, social
from .config import Config
from .feedback import Journal
from .storage import now_ms

log = logging.getLogger("backfill")
B58 = re.compile(r"[1-9A-HJ-NP-Za-km-z]{32,44}")
MAX_SIG_PAGES = 150
MAX_TX = 6000
CHUNK = 150


def extract_mint(text: str) -> str | None:
    """Accepte un CA brut ou un lien (pump.fun, dexscreener, solscan...)."""
    cands = B58.findall(text or "")
    cands.sort(key=lambda s: (not s.endswith("pump"), -len(s)))
    return cands[0] if cands else None


def bonding_curve_pda(mint: str) -> str:
    from solders.pubkey import Pubkey
    pda, _ = Pubkey.find_program_address([b"bonding-curve", bytes(Pubkey.from_string(mint))],
                                         Pubkey.from_string(pump.PUMP_PROGRAM))
    return str(pda)


class Rpc:
    def __init__(self, cfg: Config, s: aiohttp.ClientSession):
        key = cfg.secrets.helius_key
        self.url = f"{cfg['collector']['helius_rpc']}?api-key={key}" if key else "https://api.mainnet-beta.solana.com"
        self.s = s
        self.sem = asyncio.Semaphore(12 if key else 3)
        self.gap = 1 / (30 if key else 4)  # requetes par seconde (Helius Developer 50/s, RPC public plus strict)
        self.next = 0.0
        self.lock = asyncio.Lock()

    async def _pace(self):
        async with self.lock:
            now = asyncio.get_running_loop().time()
            if self.next > now:
                await asyncio.sleep(self.next - now)
            self.next = max(now, self.next) + self.gap

    async def call(self, method: str, params: list | dict):
        delay = 2.0
        for _ in range(15):
            await self._pace()
            async with self.sem:
                try:
                    async with self.s.post(self.url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                                           timeout=aiohttp.ClientTimeout(total=40)) as r:
                        if r.status == 429 or r.status >= 500:
                            raise aiohttp.ClientError(f"HTTP {r.status}")
                        j = await r.json()
                    if "error" in j:
                        if j["error"].get("code") in (-32005, 429, -32429):
                            raise aiohttp.ClientError(str(j["error"]))
                        raise RuntimeError(j["error"].get("message", str(j["error"])))
                    return j["result"]
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
            await asyncio.sleep(delay + random.random())
            delay = min(delay * 1.7, 30)
        raise RuntimeError(f"RPC {method} : trop d'echecs (limite de debit ?)")


async def run(cfg: Config, mint: str, j: Journal, s: aiohttp.ClientSession) -> dict:
    def status(st: str, msg: str = "", **kw):
        cols = ", ".join(f"{k}=?" for k in kw)
        j.exec(f"UPDATE examples SET status=?, msg=?{', ' + cols if cols else ''} WHERE mint=?",
               (st, msg, *kw.values(), mint))

    if not cfg.secrets.helius_key:
        # Le RPC public renvoie 429 sur getTransaction des qu'on en enchaine quelques-unes.
        raise RuntimeError("ajoute HELIUS_API_KEY dans .env puis relance PumpML : le RPC public bloque l'import")
    strat = cfg["strategy"]
    dec, horizon = D.horizon_ms(strat)
    rpc = Rpc(cfg, s)
    bc = bonding_curve_pda(mint)

    # 1. signatures (de la plus recente a la plus ancienne)
    sigs, before = [], None
    for page in range(MAX_SIG_PAGES):
        opts = {"limit": 1000, **({"before": before} if before else {})}
        res = await rpc.call("getSignaturesForAddress", [bc, opts])
        sigs += res
        status("signatures", f"{len(sigs)} transactions trouvees")
        if len(res) < 1000:
            break
        before = res[-1]["signature"]
    else:
        raise NotTradable(f"plus de {MAX_SIG_PAGES * 1000} transactions avant migration : trop actif pour rejouer les trades")
    if not sigs:
        raise RuntimeError("aucune transaction : ce n'est pas un coin pump.fun (ou CA invalide)")
    sigs = [x for x in reversed(sigs) if x.get("blockTime")]  # plus ancienne d'abord
    t0 = sigs[0]["blockTime"]
    window = [x for x in sigs if x["blockTime"] * 1000 <= t0 * 1000 + horizon + 10_000 and x.get("err") is None]
    # 2. transactions -> evenements pump, par paquets dans l'ordre chronologique. On s'arrete des que la
    #    sortie simulee (TP / SL / migration) est connue : le resultat est identique (la simulation
    #    s'arrete au meme endroit) et un gagnant tres actif s'importe en quelques dizaines de secondes.
    lat = int(strat["latency_s"] * 1000)
    create, trades, mig_ms = None, [], None
    per_second: dict[int, int] = {}
    failed = done = 0
    stopped_early = False

    async def fetch(x):
        try:
            return x, await rpc.call("getTransaction", [x["signature"], {
                "encoding": "json", "maxSupportedTransactionVersion": 1, "commitment": "confirmed"}])
        except RuntimeError:
            return x, None

    for i in range(0, min(len(window), MAX_TX), CHUNK):
        part = await asyncio.gather(*(fetch(x) for x in window[i:i + CHUNK]))
        for x, tx in part:
            done += 1
            if tx is None:
                failed += 1
                continue
            if (tx.get("meta") or {}).get("err") is not None:
                continue
            bt = x["blockTime"]
            for ev in pump.iter_events((tx.get("meta") or {}).get("logMessages") or []):
                k = per_second.get(bt, 0)
                per_second[bt] = k + 1
                ms = bt * 1000 + min(k, 999)  # ordre conserve a l'interieur d'une seconde
                if isinstance(ev, pump.Create) and ev.mint == mint:
                    create = (ev, ms)
                elif isinstance(ev, pump.Trade) and ev.mint == mint:
                    trades.append({"recv_ms": ms, "ts": bt, "slot": x["slot"], "is_buy": ev.is_buy, "sol_amount": ev.sol_amount,
                                   "token_amount": ev.token_amount, "user": ev.user, "vsol": ev.vsol,
                                   "vtok": ev.vtok, "real_tok": ev.real_tok, "real_sol": ev.real_sol})
                elif isinstance(ev, pump.Complete) and ev.mint == mint and mig_ms is None:
                    mig_ms = ms
        status("transactions", f"{done} transactions lues")
        if create and trades and trades[-1]["recv_ms"] > create[1] + dec + lat:
            # arret anticipe si la sortie simulee de CHAQUE point d'analyse est deja connue
            trd = F.Trades.from_rows(trades)
            known = True
            for cp in D.checkpoints_ms(strat):
                o = F.simulate(trd, create[1] + cp, strat, mig_ms)
                exit_ms = create[1] + cp + lat + int(o.hold_s * 1000)
                if o.exit_reason not in ("tp", "sl", "migrated") or trades[-1]["recv_ms"] <= exit_ms + lat + 1000:
                    known = False
                    break
            if known:
                stopped_early = True
                break
    if failed > max(2, 0.02 * done):
        raise RuntimeError(f"{failed} transactions illisibles sur {done} (RPC sature) : reessaie plus tard")
    truncated = not stopped_early and len(window) > MAX_TX
    if create is None:
        raise RuntimeError("creation introuvable (coin cree avant la fenetre lue, ou pas un coin pump.fun)")
    cev, created = create
    token = {"mint": mint, "recv_ms": created, "chain_ms": created, "name": cev.name, "symbol": cev.symbol, "uri": cev.uri,
             "creator": cev.creator or cev.user, "sol_amount": None}

    # 3. metadonnees, image, lien X
    status("metadonnees", "metadonnees et image")
    m = cfg["metadata"]
    mrow = None
    jm = await md.fetch_json(s, cev.uri, m["ipfs_gateways"], m["timeout_seconds"])
    if jm:
        mrow = {"recv_ms": created, **md.parse_metadata(jm), "image_path": None}
        if mrow.get("image_uri"):
            p = cfg.data / "images" / "backfill" / f"{mint}.jpg"
            if await md.fetch_image(s, mrow["image_uri"], m["ipfs_gateways"], m["timeout_seconds"],
                                    m["max_image_bytes"], p):
                mrow["image_path"] = str(p.relative_to(cfg.data))
    srow = None
    if mrow and F.x_link(mrow):
        link = social.parse_link(F.x_link(mrow))
        srow = {"recv_ms": created, "kind": link["kind"], "handle": link["handle"]}
        prov = social.make_provider(cfg.secrets.twitter_provider, cfg.secrets.twitter_key, s)
        if prov.name != "none" and link["kind"] in ("tweet", "profile"):
            try:
                srow.update({k: v for k, v in (await prov.fetch(link)).items() if v is not None and k != "raw"})
            except Exception as e:
                log.info("X %s : %s", mint, e)

    # 4. features + resultat simule (meme code que le live)
    tr = F.Trades.from_rows(trades)
    rows = D.token_rows(token, tr, mrow, srow, strat, mig_ms)
    if not rows:
        raise NotTradable("pas achetable sur la courbe (migre des la creation, ex. createur qui achete tout, ou courbe speciale)")
    for r in rows:
        r["truncated"] = float(truncated)
    row = rows[0]
    out = cfg.data / "backfill"
    (out / "trades").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_parquet(out / "trades" / f"{mint}.parquet", index=False)
    save_rows(cfg, rows)
    status("ok", "importe" + (" (fenetre tronquee)" if truncated else ""), name=cev.name, symbol=cev.symbol,
           image_path=(mrow or {}).get("image_path"), created_ms=created, pnl_sim=row["pnl"],
           max_mult_after=row["max_mult_after"], entry_mult=row["entry_mult"], n_trades=len(trades),
           migrated=row["migrated"], twitter=(mrow or {}).get("twitter"),
           description=((mrow or {}).get("description") or "")[:500])
    log.info("import %s (%s) : %d trades, pnl simule %.4f", mint, cev.symbol, len(trades), row["pnl"])
    return row


class NotTradable(RuntimeError):
    """Coin impossible a rejouer comme trade (migre des la creation, courbe speciale, trop actif)."""


async def narrative_only(cfg: Config, mint: str, j: Journal, s: aiohttp.ClientSession, reason: str) -> None:
    """Import "narratif seul" : nom, description, image, lien X et date, sans les trades.
    Le coin ne sert pas d'exemple de trading, mais entre dans la bibliotheque de narratives."""
    rpc = Rpc(cfg, s)
    asset = await rpc.call("getAsset", {"id": mint})  # Helius DAS
    content = asset.get("content") or {}
    md_ = content.get("metadata") or {}
    uri = content.get("json_uri")
    name, symbol = md_.get("name"), md_.get("symbol")
    m = cfg["metadata"]
    meta = (await md.fetch_json(s, uri, m["ipfs_gateways"], m["timeout_seconds"])) if uri else None
    meta = md.parse_metadata(meta) if meta else {}
    img_path = None
    img_uri = meta.get("image_uri") or (content.get("links") or {}).get("image")
    if img_uri:
        p = cfg.data / "images" / "backfill" / f"{mint}.jpg"
        if await md.fetch_image(s, img_uri, m["ipfs_gateways"], m["timeout_seconds"], m["max_image_bytes"], p):
            img_path = str(p.relative_to(cfg.data))
    created = None
    try:  # date : premiere paire DexScreener (au plus tard la migration, donc proche de la creation)
        async with s.get(f"https://api.dexscreener.com/latest/dex/tokens/{mint}", timeout=aiohttp.ClientTimeout(total=20)) as r:
            pairs = (await r.json()).get("pairs") or []
        ts = [p["pairCreatedAt"] for p in pairs if p.get("pairCreatedAt")]
        created = min(ts) if ts else None
    except Exception:
        pass
    j.exec("UPDATE examples SET status='narratif', msg=?, name=?, symbol=?, image_path=?, created_ms=?, twitter=?,"
           " description=? WHERE mint=?",
           (f"narratif seul : {reason}"[:300], name, symbol, img_path, created or 0, meta.get("twitter"),
            (meta.get("description") or "")[:500], mint))
    log.info("import narratif seul %s (%s) : %s", mint[:8], symbol, reason)


def save_rows(cfg: Config, rows: list[dict]):
    p = cfg.data / "backfill" / "rows.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    if not df.empty:
        df = df[df.mint != rows[0]["mint"]]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    tmp = p.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)


def load_rows(cfg: Config) -> pd.DataFrame:
    p = cfg.data / "backfill" / "rows.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


class Importer:
    """File d'imports (une a la fois, pour menager le RPC), executee dans le processus du dashboard."""

    def __init__(self, cfg: Config, j: Journal):
        self.cfg, self.j = cfg, j
        self.q: asyncio.Queue = asyncio.Queue()
        self.task = None

    def submit(self, mint: str) -> bool:
        cur = self.j.query("SELECT status FROM examples WHERE mint=?", (mint,))
        if cur and cur[0]["status"] not in ("erreur", "ok", "narratif") and self.task is not None:
            return False  # deja en cours
        self.j.exec("INSERT OR REPLACE INTO examples(mint, added_ms, status, msg) VALUES(?,?,?,?)",
                    (mint, now_ms(), "en attente", ""))
        self.q.put_nowait(mint)
        if self.task is None:
            self.task = asyncio.create_task(self._loop())
        return True

    def resume_pending(self):
        """Au demarrage : relance les imports interrompus (pause, redemarrage), et refait ceux importes avec
        un ancien format (avant les points d'analyse)."""
        rows = load_rows(self.cfg)
        fresh = set(rows.mint[rows["age_s"].notna()]) if not rows.empty and "age_s" in rows else set()
        for r in self.j.query("SELECT mint FROM examples WHERE status='ok'"):
            if r["mint"] not in fresh:
                self.j.exec("UPDATE examples SET status='en attente', msg='mise a jour du format' WHERE mint=?",
                            (r["mint"],))
        # anciens refus "hors strategie" / "trop actif" : on les refait en narratif seul
        self.j.exec("UPDATE examples SET status='en attente', msg='nouvel essai (narratif seul possible)'"
                    " WHERE status='erreur' AND (msg LIKE 'coin hors strategie%' OR msg LIKE 'plus de %')")
        if self.task is None and self.j.query("SELECT 1 FROM examples WHERE status NOT IN ('ok','erreur','narratif') LIMIT 1"):
            self.task = asyncio.create_task(self._loop())

    async def _loop(self):
        # imports interrompus par un redemarrage : on les relance
        for r in self.j.query("SELECT mint FROM examples WHERE status NOT IN ('ok','erreur','narratif')"):
            if r["mint"] not in self.q._queue:  # noqa: SLF001
                self.q.put_nowait(r["mint"])
        async with aiohttp.ClientSession(headers={"User-Agent": "pumpml/1.0"}) as s:
            while True:
                mint = await self.q.get()
                try:
                    try:
                        await run(self.cfg, mint, self.j, s)
                    except NotTradable as e:
                        await narrative_only(self.cfg, mint, self.j, s, str(e))
                except Exception as e:
                    log.warning("import %s : %s", mint, e)
                    self.j.exec("UPDATE examples SET status='erreur', msg=? WHERE mint=?", (str(e)[:300], mint))
