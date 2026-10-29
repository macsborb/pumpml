"""Cotations Jupiter : prix reellement executables (route, frais du pool, impact) pour le paper trading.

Sans cle : 30 requetes/min (fenetre glissante de 60 s). Avec JUPITER_API_KEY (gratuite) : 60/min.
Le limiteur garde une marge et sert les demandes par priorite : achats > ventes declenchees > suivi.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque

log = logging.getLogger("jupiter")
SOL = "So11111111111111111111111111111111111111112"


class Jupiter:
    def __init__(self, api_key: str = "", slippage_bps: int = 1000):
        self.key = api_key
        self.base = "https://api.jup.ag/swap/v1/quote" if api_key else "https://lite-api.jup.ag/swap/v1/quote"
        self.limit = 55 if api_key else 27  # par minute, avec marge
        self.sent: deque[float] = deque()
        self.slippage = slippage_bps
        self.errors = 0

    def budget(self) -> int:
        """Requetes encore disponibles dans la fenetre de 60 s."""
        now = time.monotonic()
        while self.sent and now - self.sent[0] > 60:
            self.sent.popleft()
        return self.limit - len(self.sent)

    def quote(self, input_mint: str, output_mint: str, amount: int, reserve: int = 0) -> dict | None:
        """Cotation ExactIn. ``reserve`` : nombre de requetes a laisser aux demandes plus prioritaires.
        Retourne {"out": int, "impact": float, "route": str} ou None (pas de budget, pas de route, erreur)."""
        if amount <= 0 or self.budget() <= reserve:
            return None
        self.sent.append(time.monotonic())
        q = urllib.parse.urlencode({"inputMint": input_mint, "outputMint": output_mint, "amount": int(amount),
                                    "slippageBps": self.slippage, "swapMode": "ExactIn"})
        req = urllib.request.Request(f"{self.base}?{q}", headers={"Accept": "application/json",
                                                                 **({"x-api-key": self.key} if self.key else {})})
        try:
            with urllib.request.urlopen(req, timeout=8) as r:
                j = json.load(r)
        except urllib.error.HTTPError as e:
            self.errors += 1
            if e.code == 429:
                # le serveur nous juge trop rapides : on remplit la fenetre pour ralentir
                self.sent.extend([time.monotonic()] * 10)
            log.debug("jupiter %s : HTTP %s", output_mint[:8], e.code)
            return None
        except Exception as e:
            self.errors += 1
            log.debug("jupiter %s : %s", output_mint[:8], e)
            return None
        if "outAmount" not in j:
            return None
        route = "+".join(p.get("swapInfo", {}).get("label", "?") for p in j.get("routePlan", []))
        return {"out": int(j["outAmount"]), "impact": float(j.get("priceImpactPct") or 0), "route": route}

    def buy(self, mint: str, sol: float) -> dict | None:
        return self.quote(SOL, mint, int(sol * 1e9))

    def sell(self, mint: str, tokens: int, reserve: int = 0) -> dict | None:
        q = self.quote(mint, SOL, int(tokens), reserve)
        if q:
            q["sol"] = q["out"] / 1e9
        return q
