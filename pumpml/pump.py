"""Programme pump.fun : decodage des evenements on-chain et maths de la courbe de bonding.

Les evenements Anchor apparaissent dans les logs sous la forme
``Program data: <base64>`` ; les 8 premiers octets sont sha256("event:<Nom>")[:8].
Le decodage lit un prefixe stable et ignore les champs ajoutes plus tard par pump.fun,
pour survivre aux mises a jour du programme.
"""
from __future__ import annotations

import base64
import hashlib
import struct
from dataclasses import dataclass

import base58

PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
TOKEN_DECIMALS = 6
TOTAL_SUPPLY = 1_000_000_000  # tokens (unites entieres)
INITIAL_VSOL = 30_000_000_000  # lamports
INITIAL_VTOK = 1_073_000_000_000_000  # unites brutes (6 decimales)


def _disc(name: str) -> bytes:
    return hashlib.sha256(f"event:{name}".encode()).digest()[:8]


DISC_TRADE = _disc("TradeEvent")
DISC_CREATE = _disc("CreateEvent")
DISC_COMPLETE = _disc("CompleteEvent")


class _Reader:
    __slots__ = ("b", "o")

    def __init__(self, b: bytes, o: int = 8):
        self.b, self.o = b, o

    def left(self) -> int:
        return len(self.b) - self.o

    def take(self, n: int) -> bytes:
        if self.o + n > len(self.b):
            raise ValueError("truncated")
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def pk(self) -> str:
        return base58.b58encode(self.take(32)).decode()

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.take(8))[0]

    def bool(self) -> bool:
        return self.take(1) != b"\x00"

    def string(self) -> str:
        n = struct.unpack("<I", self.take(4))[0]
        if n > 10_000:
            raise ValueError("string too long")
        return self.take(n).decode("utf-8", "replace")


@dataclass(slots=True)
class Trade:
    mint: str
    sol_amount: int
    token_amount: int
    is_buy: bool
    user: str
    ts: int
    vsol: int
    vtok: int
    real_sol: int
    real_tok: int
    creator: str | None = None


@dataclass(slots=True)
class Create:
    name: str
    symbol: str
    uri: str
    mint: str
    bonding_curve: str
    user: str
    creator: str | None = None
    ts: int | None = None  # horodatage on-chain (s)


@dataclass(slots=True)
class Complete:
    user: str
    mint: str
    bonding_curve: str
    ts: int


def decode_trade(raw: bytes) -> Trade:
    r = _Reader(raw)
    t = Trade(r.pk(), r.u64(), r.u64(), r.bool(), r.pk(), r.i64(), r.u64(), r.u64(), r.u64(), r.u64())
    # Champs optionnels (versions recentes) : fee_recipient, fee_bps, fee, creator.
    if r.left() >= 32 + 8 + 8 + 32:
        r.take(32 + 8 + 8)
        t.creator = r.pk()
    return t


def decode_create(raw: bytes) -> Create:
    r = _Reader(raw)
    c = Create(r.string(), r.string(), r.string(), r.pk(), r.pk(), r.pk())
    if r.left() >= 32:
        c.creator = r.pk()
    if r.left() >= 8:
        ts = r.i64()
        c.ts = ts if 1_600_000_000 < ts < 4_000_000_000 else None
    return c


def decode_complete(raw: bytes) -> Complete:
    r = _Reader(raw)
    return Complete(r.pk(), r.pk(), r.pk(), r.i64())


def decode_log_line(line: str):
    """Retourne un Trade/Create/Complete, ou None si la ligne n'est pas un evenement pump connu."""
    if not line.startswith("Program data: "):
        return None
    try:
        raw = base64.b64decode(line[14:], validate=False)
    except Exception:
        return None
    d = raw[:8]
    try:
        if d == DISC_TRADE:
            return decode_trade(raw)
        if d == DISC_CREATE:
            return decode_create(raw)
        if d == DISC_COMPLETE:
            return decode_complete(raw)
    except (ValueError, struct.error):
        return None
    return None


def program_data(logs: list[str], program: str):
    """Lignes ``Program data:`` emises PAR ``program`` uniquement.

    Suit la pile d'appels (``Program X invoke [n]`` / ``Program X success|failed``) : un autre
    programme de la transaction peut definir un evenement Anchor du meme nom, donc avec le
    meme discriminant. On s'arrete a ``Log truncated`` (la suite est incomplete).
    """
    stack: list[str] = []
    for line in logs:
        if line.startswith("Program data: "):
            if stack and stack[-1] == program:
                yield line
            continue
        if line.startswith("Log truncated"):
            return
        if not line.startswith("Program ") or line.startswith("Program log:") or line.startswith("Program return:"):
            continue
        parts = line.split(" ")
        if len(parts) >= 3 and parts[2] == "invoke":
            stack.append(parts[1])
        elif len(parts) >= 3 and (parts[2] == "success" or parts[2].startswith("failed")) and stack:
            stack.pop()


def iter_events(logs: list[str]):
    """Evenements emis PAR le programme pump.fun uniquement (Trade/Create/Complete)."""
    for line in program_data(logs, PUMP_PROGRAM):
        ev = decode_log_line(line)
        if ev is not None:
            yield ev


# ---------- courbe de bonding (produit constant sur reserves virtuelles) ----------

def price_sol(vsol: float, vtok: float) -> float:
    """Prix d'un token entier en SOL."""
    if vtok <= 0:
        return 0.0
    return (vsol / 1e9) / (vtok / 10 ** TOKEN_DECIMALS)


def mcap_sol(vsol: float, vtok: float) -> float:
    return price_sol(vsol, vtok) * TOTAL_SUPPLY


def progress(real_tok: float) -> float:
    """Avancement de la courbe : 0 au debut, 1 a la migration."""
    initial_real = 793_100_000_000_000
    return max(0.0, min(1.0, 1 - real_tok / initial_real))


def sim_buy(vsol: int, vtok: int, sol_in: float, fee_bps: int) -> tuple[float, int, int]:
    """Achat de ``sol_in`` SOL (frais inclus). Retourne (tokens bruts recus, vsol, vtok apres)."""
    net = sol_in * 1e9 / (1 + fee_bps / 1e4)
    k = vsol * vtok
    new_vsol = vsol + net
    new_vtok = k / new_vsol
    return vtok - new_vtok, int(new_vsol), int(new_vtok)


def sim_sell(vsol: int, vtok: int, tokens: float, fee_bps: int) -> float:
    """Vente de ``tokens`` (unites brutes). Retourne les SOL nets recus."""
    if tokens <= 0:
        return 0.0
    k = vsol * vtok
    new_vtok = vtok + tokens
    gross = vsol - k / new_vtok
    return max(0.0, gross / 1e9 * (1 - fee_bps / 1e4))
