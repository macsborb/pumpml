"""Verification rapide : cles, flux PumpPortal et Helius, decodage des evenements pump.fun, GPU."""
from __future__ import annotations

import asyncio
import json
import time

import aiohttp
import websockets

from . import pump
from .config import Config


async def _pumpportal(cfg: Config, secs: float) -> dict:
    n, sample = 0, None
    async with websockets.connect(cfg.pumpportal_url(), open_timeout=20) as ws:
        await ws.send(json.dumps({"method": "subscribeNewToken"}))
        end = time.monotonic() + secs
        while time.monotonic() < end:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), end - time.monotonic()))
            except asyncio.TimeoutError:
                break
            if m.get("txType") == "create":
                n += 1
                sample = sample or {k: m.get(k) for k in ("mint", "name", "symbol", "pool", "marketCapSol")}
    return {"creations": n, "exemple": sample}


async def _chain(cfg: Config, secs: float) -> dict:
    c = {"notifications": 0, "trades": 0, "creates": 0, "completes": 0}
    sub = {"jsonrpc": "2.0", "id": 1, "method": "logsSubscribe",
           "params": [{"mentions": [pump.PUMP_PROGRAM]}, {"commitment": "confirmed"}]}
    sample = None
    async with websockets.connect(cfg.chain_ws_url(), open_timeout=20, max_size=2 ** 24) as ws:
        await ws.send(json.dumps(sub))
        end = time.monotonic() + secs
        while time.monotonic() < end:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), end - time.monotonic()))
            except asyncio.TimeoutError:
                break
            if "error" in m:
                return {"erreur": m["error"]}
            if m.get("method") != "logsNotification":
                continue
            c["notifications"] += 1
            for ev in pump.iter_events(m["params"]["result"]["value"].get("logs") or []):
                if isinstance(ev, pump.Trade) and ev.vsol <= 0:
                    c["unpriced"] = c.get("unpriced", 0) + 1
                elif isinstance(ev, pump.Trade):
                    c["trades"] += 1
                    if sample is None:
                        sample = {"mint": ev.mint, "buy": ev.is_buy, "sol": ev.sol_amount / 1e9,
                                  "mcap_sol": round(pump.mcap_sol(ev.vsol, ev.vtok), 2)}
                elif isinstance(ev, pump.Create):
                    c["creates"] += 1
                elif isinstance(ev, pump.Complete):
                    c["completes"] += 1
    c["exemple_trade"] = sample
    return c


async def _helius_rpc(cfg: Config) -> str:
    if not cfg.secrets.helius_key:
        return "pas de cle"
    url = f"{cfg['collector']['helius_rpc']}?api-key={cfg.secrets.helius_key}"
    async with aiohttp.ClientSession() as s:
        async with s.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"}) as r:
            return (await r.json()).get("result", "?")


def main(cfg: Config) -> int:
    ok = True
    print("PumpPortal cle :", "oui" if cfg.secrets.pumpportal_key else "non (les flux gratuits marchent quand meme)")
    print("Helius cle     :", "oui" if cfg.secrets.helius_key else "NON -> websocket public")
    print("Fournisseur X  :", cfg.secrets.twitter_provider, "(cle presente)" if cfg.secrets.twitter_key else "")
    try:
        print("Helius RPC     :", asyncio.run(_helius_rpc(cfg)))
    except Exception as e:
        ok = False
        print("Helius RPC     : ECHEC", e)
    for name, fn in (("PumpPortal 20s", _pumpportal), ("Chaine 20s", _chain)):
        try:
            r = asyncio.run(fn(cfg, 20))
            print(f"{name:<15}:", json.dumps(r, ensure_ascii=False))
            if not (r.get("creations") or r.get("trades")):
                ok = False
        except Exception as e:
            ok = False
            print(f"{name:<15}: ECHEC {e!r}")
    try:
        import torch
        print("GPU            :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA indisponible")
    except Exception as e:
        print("GPU            : torch absent", e)
    print("RESULTAT       :", "OK" if ok else "PROBLEME")
    return 0 if ok else 1
