"""Features et labels. UNE seule implementation, utilisee a l'entrainement ET en live.

Regle d'or : les features d'un token n'utilisent que des donnees recues avant
``decision_ms = created_ms + age`` (un point d'analyse). Le label n'utilise que ce qui suit.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from . import pump, social

FEATURE_VERSION = 7
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


@dataclass
class Trades:
    """Trades d'un token, dans l'ordre on-chain.

    ``recv_ms`` contient l'heure ON-CHAIN du trade (horodatage du bloc, en ms), pas l'heure de reception :
    le websocket public a plusieurs secondes de retard variable. Dans une meme seconde, l'ordre
    (slot, reception) est conserve en ajoutant 1 ms par trade. Sans horodatage (anciennes lignes),
    on retombe sur l'heure de reception.
    """
    recv_ms: np.ndarray
    slot: np.ndarray
    is_buy: np.ndarray
    sol: np.ndarray       # lamports
    tok: np.ndarray       # unites brutes
    user: np.ndarray      # object (str)
    vsol: np.ndarray
    vtok: np.ndarray
    real_tok: np.ndarray
    real_sol: np.ndarray

    @classmethod
    def from_rows(cls, rows) -> "Trades":
        """rows : iterable de dict/Row avec les colonnes des trades."""
        rows = sorted(rows, key=lambda r: (_chain_base(r), r["slot"], r["recv_ms"]))
        times, prev, k = [], None, 0
        for r in rows:
            base = _chain_base(r)
            if _has_ts(r):
                k = k + 1 if base == prev else 0
                times.append(base + min(k, 999))
            else:
                times.append(base)
            prev = base
        g = lambda k, dt: np.array([r[k] for r in rows], dtype=dt)
        return cls(np.array(times, dtype=np.int64), g("slot", np.int64), g("is_buy", bool), g("sol_amount", np.float64),
                   g("token_amount", np.float64), g("user", object), g("vsol", np.float64), g("vtok", np.float64),
                   g("real_tok", np.float64), g("real_sol", np.float64))

    def __len__(self):
        return len(self.recv_ms)

    def upto(self, ms: int) -> "Trades":
        i = int(np.searchsorted(self.recv_ms, ms, side="right"))
        return self.slice(0, i)

    def slice(self, a: int, b: int) -> "Trades":
        return Trades(*(getattr(self, f)[a:b] for f in self.__dataclass_fields__))


def _has_ts(r) -> bool:
    ts = r.get("ts") if hasattr(r, "get") else (r["ts"] if "ts" in r.keys() else None)
    try:
        return ts is not None and not math.isnan(float(ts)) and float(ts) > 1_000_000_000
    except (TypeError, ValueError):
        return False


def _chain_base(r) -> int:
    return int(float(r["ts"])) * 1000 if _has_ts(r) else int(r["recv_ms"])


def created_chain_ms(token: dict, tr: Trades) -> int:
    """Heure on-chain de creation : horodatage de l'evenement de creation, sinon premier trade
    (l'achat du dev est dans la meme transaction), sinon reception moins le retard typique."""
    c = token.get("chain_ms")
    try:
        if c is not None and not math.isnan(float(c)) and float(c) > 0:
            return int(c)
    except (TypeError, ValueError):
        pass
    if len(tr):
        return int(tr.recv_ms[0]) // 1000 * 1000
    return int(token["recv_ms"]) - 6_000


def curve_ref(tr: Trades) -> tuple[float, float]:
    """Prix initial et reserve reelle initiale du token, reconstitues depuis ses propres reserves.

    Produit constant : k = vsol*vtok ne bouge pas ; au depart real_sol = 0 donc
    vsol0 = vsol - real_sol. Marche aussi pour les courbes aux parametres non standard.
    """
    if len(tr):
        vsol0 = tr.vsol[0] - tr.real_sol[0]
        if vsol0 > 0:
            vtok0 = tr.vsol[0] * tr.vtok[0] / vsol0
            return vsol0 / vtok0, vtok0 - (tr.vtok[0] - tr.real_tok[0])
    return pump.INITIAL_VSOL / pump.INITIAL_VTOK, 793_100_000_000_000.0


def is_standard_curve(tr: Trades) -> bool:
    """Courbe pump.fun standard cotee en SOL (30 SOL virtuels au depart).

    Certaines courbes pump.fun ont d'autres reserves (autre actif de cotation, reserves modifiees) :
    acheter "0.1 SOL" dessus n'a pas de sens, on les exclut de l'entrainement et du live.
    A appeler sur les trades connus AU MOMENT DE DECIDER, a l'entrainement comme en live.
    """
    if not len(tr):
        return True
    if (tr.vsol <= 0).any() or (tr.vtok <= 0).any():
        return False  # trades sans prix en SOL : autre famille de coins
    # Sur une courbe standard, vsol - real_sol reste egal a 30 SOL a chaque trade. Certaines courbes
    # voient leurs reserves virtuelles modifiees en cours de route : on les exclut aussi.
    offset = tr.vsol - tr.real_sol
    return bool((np.abs(offset - pump.INITIAL_VSOL) <= 0.02 * pump.INITIAL_VSOL).all())


def _log1p(x) -> float:
    try:
        return math.log1p(max(0.0, float(x)))
    except (TypeError, ValueError):
        return float("nan")


CURVE_SUPPLY = 793_100_000_000_000  # tokens vendus par la courbe (unites brutes)


def block0(tr: Trades) -> tuple[float, int]:
    """Lancement groupe : part de la courbe achetee dans le slot de creation et les 2 suivants (~1 s), dev
    compris, et nombre de wallets. Un createur qui achete une grosse part avec ses propres wallets simule
    une tendance pour revendre sur les acheteurs suivants (rug)."""
    if not len(tr):
        return 0.0, 0
    m = tr.is_buy & (tr.slot <= tr.slot[0] + 2)
    return float(tr.tok[m].sum() / CURVE_SUPPLY), len(set(tr.user[m]))


def x_link(meta: dict | None) -> str | None:
    """Lien X du coin : champ twitter, sinon le site web quand c'est un lien X (cas frequent)."""
    meta = meta or {}
    for u in (meta.get("twitter"), meta.get("website")):
        if social.parse_link(u)["kind"] in ("tweet", "profile", "community"):
            return u
    return meta.get("twitter")


def has_link(meta: dict | None) -> bool:
    """Twitter ou site web dans les metadonnees : sans eux, pas de narrative verifiable."""
    meta = meta or {}
    return any(isinstance(meta.get(k), str) and meta[k].strip() for k in ("twitter", "website"))


def excluded(token: dict, meta: dict | None, tr: Trades, s: dict, soc: dict | None = None) -> str | None:
    """Coins jamais achetes (memes regles a l'entrainement et en direct) :
    * ni Twitter ni site web dans les metadonnees (``require_link``) ;
    * lancement groupe (``max_block0_share``) ;
    * tweet de plus de ``max_tweet_age_min`` au lancement : deja exploite par d'autres coins ;
    * reponse "parasite" a un tweet viral d'un autre compte (``block_parasite``)."""
    if s.get("require_link") and meta is not None and not has_link(meta):
        return "sans_lien"
    if s.get("block_parasite") and is_parasite(soc):
        return "parasite"
    if s.get("max_block0_share") and block0(tr)[0] >= s["max_block0_share"]:
        return "groupe"
    if s.get("max_tweet_age_min") and meta:
        tms = social.parse_link(x_link(meta))["tweet_ms"]
        if tms and (int(token["recv_ms"]) - tms) > s["max_tweet_age_min"] * 60_000:
            return "vieux_tweet"
    return None


def flow_features(created_ms: int, creator: str | None, tr: Trades, decision_ms: int) -> dict:
    """Activite on-chain du token entre sa creation et le moment de decision."""
    t = tr.upto(decision_ms)
    f: dict[str, float] = {}
    n = len(t)
    f["n_trades"] = n
    if n == 0:
        f.update(price_mult=1.0, max_mult=1.0, progress=0.0, mcap_sol=pump.mcap_sol(pump.INITIAL_VSOL, pump.INITIAL_VTOK))
        return f
    buys, sells = t.is_buy, ~t.is_buy
    sol = t.sol / 1e9
    f["n_buys"] = int(buys.sum())
    f["n_sells"] = int(sells.sum())
    f["buy_sol"] = float(sol[buys].sum())
    f["sell_sol"] = float(sol[sells].sum())
    f["net_sol"] = f["buy_sol"] - f["sell_sol"]
    f["sell_buy_ratio"] = f["sell_sol"] / (f["buy_sol"] + 1e-9)
    ub = set(t.user[buys])
    f["uniq_buyers"] = len(ub)
    f["uniq_sellers"] = len(set(t.user[sells]))
    f["uniq_traders"] = len(set(t.user))
    f["avg_buy_sol"] = f["buy_sol"] / max(1, f["n_buys"])
    f["max_buy_sol"] = float(sol[buys].max()) if f["n_buys"] else 0.0
    f["small_buys_frac"] = float((sol[buys] < 0.05).mean()) if f["n_buys"] else 0.0
    # Dev
    dev = t.user == creator if creator else np.zeros(n, bool)
    f["dev_buy_sol"] = float(sol[dev & buys].sum())
    dev_bought = float(t.tok[dev & buys].sum())
    dev_sold = float(t.tok[dev & sells].sum())
    f["dev_sold_frac"] = dev_sold / dev_bought if dev_bought > 0 else 0.0
    f["dev_n_trades"] = int(dev.sum())
    # Concentration des acheteurs (hors dev)
    if f["n_buys"]:
        holders: dict[str, float] = {}
        for u, b, k in zip(t.user, t.is_buy, t.tok):
            holders[u] = holders.get(u, 0.0) + (k if b else -k)
        bal = np.array([v for u, v in holders.items() if u != creator and v > 0])
        tot = bal.sum()
        s = np.sort(bal)[::-1]
        f["top1_share"] = float(s[:1].sum() / tot) if tot > 0 else 0.0
        f["top5_share"] = float(s[:5].sum() / tot) if tot > 0 else 0.0
        f["n_holders"] = int((bal > 0).sum())
        # Bundle / snipers : achats dans le slot de creation et le suivant
        s0 = t.slot[0]
        early = buys & (t.slot <= s0 + 1) & ~dev
        f["early_buyers"] = len(set(t.user[early]))
        f["early_sol"] = float(sol[early].sum())
        f["block0_share"], f["block0_wallets"] = block0(t)
    # Repetitions (bots de volume)
    _, counts = np.unique(t.user, return_counts=True)
    f["repeat_traders_frac"] = float((counts > 2).mean())
    # Prix
    price = t.vsol / t.vtok
    p0, real0 = curve_ref(t)
    f["price_mult"] = float(price[-1] / p0)
    f["max_mult"] = float(price.max() / p0)
    f["drawdown"] = float(price[-1] / price.max() - 1)
    f["progress"] = float(max(0.0, min(1.0, 1 - t.real_tok[-1] / real0))) if real0 > 0 else 0.0
    f["mcap_sol"] = pump.mcap_sol(t.vsol[-1], t.vtok[-1])
    # Dynamique : 15 dernieres secondes vs le reste
    recent = t.recv_ms >= decision_ms - 15_000
    f["recent_trades"] = int(recent.sum())
    f["recent_net_sol"] = float(sol[recent & buys].sum() - sol[recent & sells].sum())
    j = int(np.searchsorted(t.recv_ms, decision_ms - 15_000))
    f["recent_price_chg"] = float(price[-1] / price[j - 1] - 1) if j > 0 else f["price_mult"] - 1
    f["since_last_trade_s"] = (decision_ms - int(t.recv_ms[-1])) / 1000
    f["first_trade_delay_s"] = (int(t.recv_ms[0]) - created_ms) / 1000
    return f


def meta_features(token: dict, meta: dict | None, soc: dict | None) -> dict:
    """Nom, ticker, description, liens, compte X et tweet."""
    name, sym = token.get("name") or "", token.get("symbol") or ""
    f = {
        "name_len": len(name), "symbol_len": len(sym),
        "symbol_upper": float(sym.isupper()), "name_has_digit": float(any(c.isdigit() for c in name)),
        "name_emoji": len(_EMOJI.findall(name + sym)), "name_eq_symbol": float(name.lower() == sym.lower()),
        "initial_buy_sol": float(token.get("sol_amount") or 0.0),
    }
    meta = meta or {}
    f["meta_ok"] = float(bool(meta))
    desc = meta.get("description") or ""
    f["desc_len"] = len(desc)
    f["desc_words"] = len(desc.split())
    xl = x_link(meta)
    f["has_twitter"] = float(bool(xl))
    f["has_telegram"] = float(bool(meta.get("telegram")))
    f["has_website"] = float(bool(meta.get("website")))
    f["has_image"] = float(bool(meta.get("image_path")))
    link = social.parse_link(xl)
    for k in ("tweet", "profile", "community", "other"):
        f[f"x_{k}"] = float(link["kind"] == k)
    soc = soc or {}
    f["x_followers"] = _log1p(soc.get("followers")) if soc.get("followers") is not None else float("nan")
    f["x_following"] = _log1p(soc.get("following")) if soc.get("following") is not None else float("nan")
    f["x_verified"] = float(bool(soc.get("verified"))) if soc else float("nan")
    vt = (soc.get("verified_type") or "").lower()
    f["x_verified_business"] = float(vt in ("business", "government")) if soc else float("nan")
    created = token.get("recv_ms") or 0
    acm = soc.get("account_created_ms")
    f["x_account_age_d"] = (created - acm) / 86_400_000 if acm else float("nan")
    f["x_statuses"] = _log1p(soc.get("statuses")) if soc.get("statuses") is not None else float("nan")
    for k in ("likes", "retweets", "replies", "quotes", "views"):
        f[f"x_{k}"] = _log1p(soc.get(k)) if soc.get(k) is not None else float("nan")
    # Age du tweet au lancement du coin (lu dans l'identifiant du tweet : toujours disponible).
    # Un vieux tweet a deja servi a d'autres coins ; un tweet de moins de 2 min est "frais".
    tcm = link.get("tweet_ms") or soc.get("tweet_created_ms")
    age = (created - tcm) / 60_000 if tcm else float("nan")
    f["x_tweet_age_min"] = age
    f["x_tweet_fresh"] = float(-1 <= age <= 2) if tcm else float("nan")
    f["x_is_reply"] = float(bool(soc.get("is_reply"))) if soc.get("tweet_text") else float("nan")
    f.update(parent_features(token, soc))
    f["x_mentions"] = len([m for m in (soc.get("mentions") or "").split(",") if m]) if soc else float("nan")
    f.update(link_features(token, meta, soc))
    # Le handle du compte X est-il dans le nom/ticker ? (coin "officiel" vs reprise)
    h = (soc.get("handle") or link.get("handle") or "").lower()
    f["x_handle_in_name"] = float(bool(h) and (h in name.lower().replace(" ", "") or h == sym.lower()))
    return f


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def link_features(token: dict, meta: dict, soc: dict) -> dict:
    """Le nom du coin a-t-il un lien direct avec ce qui est tweete, et avec son site web ?"""
    nan = float("nan")
    words = [w for w in {_norm(token.get("name") or ""), _norm(token.get("symbol") or "")} if len(w) >= 3]
    tw = soc.get("tweet_text") if soc else None
    f = {"x_name_in_tweet": nan, "x_ticker_cashtag": nan, "web_real": 0.0, "web_name_match": nan}
    if tw:
        t = _norm(tw)
        f["x_name_in_tweet"] = float(any(w in t for w in words))
        sym = (token.get("symbol") or "").lower()
        f["x_ticker_cashtag"] = float(bool(sym) and f"${sym}" in tw.lower())
    web = meta.get("website") if isinstance(meta.get("website"), str) else ""
    if web.strip() and social.parse_link(web)["kind"] not in ("tweet", "profile", "community"):
        # vrai site (pas un lien X) : son adresse reprend-elle le nom ou le ticker du coin ?
        f["web_real"] = 1.0
        host = re.sub(r"^https?://(www\.)?", "", web.strip().lower()).split("/")[0]
        f["web_name_match"] = float(any(w in _norm(host) for w in words))
    return f


def parent_features(token: dict, soc: dict) -> dict:
    """Tweet du coin qui REPOND a (ou cite) un autre tweet, souvent viral.
    Deux cas tres differents :
    * suite / precision apportee par l'auteur lui-meme ou par un compte credible -> vraie info ;
    * petit compte qui s'accroche au tweet viral d'un autre pour faire du buzz -> "parasite".
    Mesures : qui parle (meme auteur ? abonnes compares), d'ou vient le nom du coin (reponse ou origine)."""
    nan = float("nan")
    rel = soc.get("rel")
    f = {"x_rel_reply": float(rel == "reply"), "x_rel_quote": float(rel == "quote")} if soc.get("tweet_text") else {
        "x_rel_reply": nan, "x_rel_quote": nan}
    keys = ("x_parent_followers", "x_parent_verified", "x_parent_likes", "x_parent_views", "x_parent_age_min",
            "x_same_author", "x_follower_gap", "x_name_in_reply", "x_name_in_parent", "x_parasite")
    if not rel or not soc.get("parent_handle"):
        f.update({k: nan for k in keys})
        return f
    pf, rf = soc.get("parent_followers"), soc.get("followers")
    f["x_parent_followers"] = _log1p(pf) if pf is not None else nan
    f["x_parent_verified"] = float(bool(soc.get("parent_verified")))
    f["x_parent_likes"] = _log1p(soc.get("parent_likes")) if soc.get("parent_likes") is not None else nan
    f["x_parent_views"] = _log1p(soc.get("parent_views")) if soc.get("parent_views") is not None else nan
    pc = soc.get("parent_created_ms")
    f["x_parent_age_min"] = ((token.get("recv_ms") or 0) - pc) / 60_000 if pc else nan
    same = (soc.get("handle") or "") == (soc.get("parent_handle") or "")
    f["x_same_author"] = float(same)
    # ecart d'audience (log) : tres negatif = petit compte qui repond a un gros compte
    f["x_follower_gap"] = (_log1p(rf) - _log1p(pf)) if rf is not None and pf is not None else nan
    words = [w for w in {_norm(token.get("name") or ""), _norm(token.get("symbol") or "")} if len(w) >= 3]
    reply, parent = _norm(soc.get("tweet_text") or ""), _norm(soc.get("parent_text") or "")
    f["x_name_in_reply"] = float(any(w in reply for w in words))
    f["x_name_in_parent"] = float(any(w in parent for w in words))
    f["x_parasite"] = float(is_parasite(soc))
    return f


def is_parasite(soc: dict | None) -> bool:
    """Reponse / citation d'un AUTRE compte, au moins 20x plus petit que l'auteur du tweet d'origine, et non
    certifie : il surfe sur le buzz de quelqu'un d'autre. On n'achete pas (regle ``block_parasite``)."""
    soc = soc or {}
    if not soc.get("rel") or not soc.get("parent_handle") or soc.get("handle") == soc.get("parent_handle"):
        return False
    rf, pf = soc.get("followers"), soc.get("parent_followers")
    if rf is None or pf is None or soc.get("verified_type") in ("business", "government"):
        return False
    return (pf + 1) / (rf + 1) >= 20


def copy_veto(meta: dict | None, soc: dict | None, orig: dict | None, orig_text: str | None,
              text_sim: float | None, max_sim: float = 0.9) -> str | None:
    """Regle ``block_copies`` : le modele veut acheter une copie (meme image, meme lien X, ou meme nom + ticker
    qu'un coin deja sorti). On n'achete pas, sauf si son tweet apporte une information nouvelle sur l'original :
    un AUTRE tweet que celui de l'original, publie APRES le lancement de l'original, lu, et pas une redite de son
    texte (similarite < ``max_sim``). Retourne la raison du refus, ou None si l'achat reste permis."""
    if not orig:
        return None
    link = social.parse_link(x_link(meta or {}))
    if link["kind"] != "tweet" or not link["tweet_id"]:
        return "copie sans tweet"
    if orig.get("tw") == f"t:{link['tweet_id']}":
        return "copie : meme tweet que l'original"
    if not link["tweet_ms"] or link["tweet_ms"] <= orig["created_ms"]:
        return "copie : tweet anterieur a l'original"
    if not (soc or {}).get("tweet_text"):
        return "copie : tweet non lu"
    if orig_text and text_sim is not None and text_sim >= max_sim:
        return "copie : meme texte que le tweet de l'original"
    return None


def tweet_pending(token: dict, meta: dict | None, soc: dict | None, decision_ms: int, s: dict) -> bool:
    """Le coin renvoie a un tweet frais pas encore lu : on attend sa lecture (jusqu'a ``wait_tweet_s``)
    au lieu d'acheter sans connaitre son contenu."""
    if soc is not None or not meta or not s.get("wait_tweet_s"):
        return False
    link = social.parse_link(x_link(meta))
    if link["kind"] != "tweet" or not link["tweet_ms"]:
        return False
    created = int(token["recv_ms"])
    return (created - link["tweet_ms"]) <= s.get("max_tweet_age_min", 2) * 60_000 and \
        decision_ms - created < s["wait_tweet_s"] * 1000


def time_features(created_ms: int) -> dict:
    hour = (created_ms // 3_600_000) % 24
    return {"hour_sin": math.sin(2 * math.pi * hour / 24), "hour_cos": math.cos(2 * math.pi * hour / 24),
            "dow": float(((created_ms // 86_400_000) + 3) % 7)}


def coin_text(token: dict, meta: dict | None) -> str:
    """Le coin lui-meme : nom, ticker, description (sans le tweet)."""
    parts = [token.get("name") or "", token.get("symbol") or "", (meta or {}).get("description") or ""]
    return " | ".join(p.strip() for p in parts if p and p.strip())[:600]


def text_of(token: dict, meta: dict | None, soc: dict | None) -> str:
    """Texte de la narrative (pour l'embedding)."""
    parts = [token.get("name") or "", token.get("symbol") or "", (meta or {}).get("description") or "",
             (soc or {}).get("tweet_text") or ""]
    return " | ".join(p.strip() for p in parts if p and p.strip())[:1000]


# ---------- label : simulation de la strategie ----------

@dataclass
class Outcome:
    pnl_sol: float
    exit_reason: str
    hold_s: float
    max_mult_after: float
    entry_mult: float


def simulate(tr: Trades, decision_ms: int, s: dict, migrated_ms: int | None = None) -> Outcome:
    """Achete ``buy_sol`` a decision+latence, sort sur TP/SL/timeout/migration, avec frais et slippage.

    Options de sortie (absentes = comportement d'origine) :
    * ``take_profit`` <= 0 : pas de plafond de gain ;
    * ``stale_exit_s`` + ``stale_min_gain`` : apres N s, si la valeur n'a jamais atteint +X %, on vend ("stale") ;
    * ``trail_activate`` + ``trail_pct`` : une fois +A % atteint, on vend si la valeur retombe de T % sous
      son plus haut ("trail")."""
    lat = int(s["latency_s"] * 1000)
    entry_ms = decision_ms + lat
    i = int(np.searchsorted(tr.recv_ms, entry_ms, side="right"))
    vsol, vtok = (tr.vsol[i - 1], tr.vtok[i - 1]) if i > 0 else (float(pump.INITIAL_VSOL), float(pump.INITIAL_VTOK))
    p0, _ = curve_ref(tr)
    entry_mult = (vsol / vtok) / p0
    tokens, _, _ = pump.sim_buy(int(vsol), int(vtok), s["buy_sol"], s["fee_bps"])
    cost = s["buy_sol"] + 2 * s["tx_cost_sol"]
    tp = s["buy_sol"] * (1 + s["take_profit"]) if s.get("take_profit", 0) > 0 else float("inf")
    sl = s["buy_sol"] * (1 - s["stop_loss"])
    end_ms = entry_ms + int(s["max_hold_s"] * 1000)
    stale_ms = entry_ms + int(s["stale_exit_s"] * 1000) if s.get("stale_exit_s") else None
    stale_thr = s["buy_sol"] * (1 + s.get("stale_min_gain", 0.1))
    trail_on = s["buy_sol"] * (1 + s["trail_activate"]) if s.get("trail_activate") else None
    trail_pct = s.get("trail_pct", 0.25)
    best = entry_mult
    maxv = 0.0
    trigger, reason = None, "timeout"
    for j in range(i, len(tr)):
        t = tr.recv_ms[j]
        if t > end_ms:
            break
        if stale_ms is not None and t >= stale_ms and maxv < stale_thr:
            # le coin n'a rien fait : on sort a l'echeance (au dernier etat connu avant elle)
            reason = "stale"
            break
        if migrated_ms is not None and t >= migrated_ms:
            trigger, reason = j, "migrated"
            break
        best = max(best, (tr.vsol[j] / tr.vtok[j]) / p0)
        v = pump.sim_sell(int(tr.vsol[j]), int(tr.vtok[j]), tokens, s["fee_bps"])
        maxv = max(maxv, v)
        if v >= tp:
            trigger, reason = j, "tp"
            break
        if v <= sl:
            trigger, reason = j, "sl"
            break
        if trail_on is not None and maxv >= trail_on and v <= maxv * (1 - trail_pct):
            trigger, reason = j, "trail"
            break
    if trigger is None and reason != "stale" and stale_ms is not None and maxv < stale_thr and stale_ms <= end_ms:
        # aucun trade apres l'echeance : la sortie "stale" a lieu quand meme a l'echeance
        reason = "stale"
    if trigger is None:
        stop = stale_ms if reason == "stale" else end_ms
        k = int(np.searchsorted(tr.recv_ms, stop, side="right")) - 1
        exit_ms = stop
    else:
        if reason == "migrated":
            k = trigger - 1 if trigger > 0 else 0  # derniere valeur connue sur la courbe
            exit_ms = int(tr.recv_ms[trigger])
        else:
            exit_ms = int(tr.recv_ms[trigger]) + lat
            k = int(np.searchsorted(tr.recv_ms, exit_ms, side="right")) - 1
    if k >= i and k >= 0:
        vs, vt = tr.vsol[k], tr.vtok[k]
    else:
        vs, vt = vsol, vtok  # personne n'a trade apres nous
    out = pump.sim_sell(int(vs), int(vt), tokens, s["fee_bps"])
    return Outcome(out - cost, reason, (exit_ms - entry_ms) / 1000, best / entry_mult if entry_mult else 1.0, entry_mult)
