"""Performance reelle (paper trading au prix Jupiter) des modeles, pour l'onglet Modeles du dashboard.

Chaque position est rattachee au modele qui a decide l'achat (``decisions.model_id``). On mesure, par
fenetre de temps (1 h, 4 h, 12 h, 24 h, 48 h, depuis la mise en service) et par modele : achats executes,
gain realise, taux de gain, gain moyen / perte moyenne, facteur de profit, pire baisse (drawdown), raisons de
sortie. Reference honnete sur les memes heures : le gain simule moyen par coin si on achetait tout, compare au
gain simule moyen des achats du modele (meme simulation, donc sans l'ecart Jupiter / courbe)."""
from __future__ import annotations

import json
import time

EXEC = ("closed", "open")
WINDOWS = [("1 h", 1), ("4 h", 4), ("12 h", 12), ("24 h", 24), ("48 h", 48)]
EXITS = {"tp": "Objectif atteint (+50 %)", "sl": "Stop (-40 %)", "stale": "Vente a 2 min (rien ne se passe)",
         "timeout": "Duree max (30 min)"}


def stats(pos: list[dict]) -> dict:
    """Statistiques d'une liste de positions (dicts de la table positions)."""
    ex = [p for p in pos if p["status"] in EXEC]
    closed = sorted((p for p in ex if p["status"] == "closed" and p["pnl"] is not None), key=lambda p: p["exit_ms"] or 0)
    pnl = [p["pnl"] for p in closed]
    wins, losses = [x for x in pnl if x > 0], [x for x in pnl if x <= 0]
    cum = peak = dd = 0.0
    for x in pnl:
        cum += x
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    reasons: dict[str, dict] = {}
    for p in closed:
        r = p["exit_reason"] or "?"
        k = r if r in EXITS else ("invendable" if "invendable" in r else r)
        o = reasons.setdefault(k, {"n": 0, "pnl": 0.0})
        o["n"] += 1
        o["pnl"] += p["pnl"]
    opn = [p for p in ex if p["status"] == "open"]
    cost = sum((p["cost"] or 0) for p in closed)
    return {
        "trades": len(ex), "closed": len(closed), "open": len(opn),
        "not_executed": len(pos) - len(ex),
        "pnl": sum(pnl), "unrealized": sum(((p["value"] or 0) - (p["cost"] or 0)) for p in opn),
        "win_rate": len(wins) / len(pnl) if pnl else None,
        "avg_win": sum(wins) / len(wins) if wins else None,
        "avg_loss": sum(losses) / len(losses) if losses else None,
        "per_trade": sum(pnl) / len(pnl) if pnl else None,
        "roi": sum(pnl) / cost if cost else None,
        "profit_factor": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else None,
        "best": max(pnl) if pnl else None, "worst": min(pnl) if pnl else None,
        "max_drawdown": dd, "reasons": reasons,
    }


def sim_ref(decs: list[dict]) -> dict:
    """Meme simulation pour les achats du modele et pour tous les coins notes (reference « acheter tout »)."""
    res = [d for d in decs if d["resolved"] and d["pnl_sim"] is not None]
    buys = [d["pnl_sim"] for d in res if d["action"] == "buy"]
    allc = [d["pnl_sim"] for d in res]
    return {"coins": len(decs), "resolved": len(res), "buys": sum(d["action"] == "buy" for d in decs),
            "sim_buy_mean": sum(buys) / len(buys) if buys else None,
            "sim_all_mean": sum(allc) / len(allc) if allc else None,
            "sim_all_win": sum(x > 0 for x in allc) / len(allc) if allc else None}


def build(j, registry: dict, now: int | None = None) -> dict:
    now = now or int(time.time() * 1000)
    pos = [dict(r) for r in j.query(
        "SELECT p.mint, p.status, p.pnl, p.cost, p.value, p.exit_ms, p.exit_reason, p.entry_wall_ms, p.impact,"
        " d.model_id, d.name, d.symbol, d.final, d.threshold, d.age_s FROM positions p LEFT JOIN decisions d ON d.mint=p.mint")]
    decs = [dict(r) for r in j.query("SELECT model_id, decided_ms, action, resolved, pnl_sim FROM decisions"
                                     " WHERE model_id IS NOT NULL")]
    champ = registry.get("champion")
    hist = {h["id"]: h for h in registry.get("history", [])}

    # periodes de service : premiere et derniere decision de chaque modele
    service: dict[str, list[int]] = {}
    for d in decs:
        s = service.setdefault(d["model_id"], [d["decided_ms"], d["decided_ms"]])
        s[0], s[1] = min(s[0], d["decided_ms"]), max(s[1], d["decided_ms"])
    live_id = max(service, key=lambda m: service[m][1]) if service else None

    def window(pp, dd, since):
        return {**stats([p for p in pp if (p["entry_wall_ms"] or 0) >= since]),
                **sim_ref([d for d in dd if d["decided_ms"] >= since])}

    out_models = []
    for mid, (a, b) in sorted(service.items(), key=lambda x: -x[1][0]):
        pp = [p for p in pos if p["model_id"] == mid]
        dd = [d for d in decs if d["model_id"] == mid]
        h = hist.get(mid, {})
        out_models.append({
            "id": mid, "champion": mid == champ, "live": mid == live_id, "start_ms": a, "end_ms": b,
            "hours": (b - a) / 3_600_000, **stats(pp), **sim_ref(dd),
            "valid": {k: h.get(k) for k in ("valid_pnl", "valid_trades", "valid_tokens", "valid_auc",
                                            "valid_buy_all_pnl", "valid_win_rate_selected", "promoted", "family",
                                            "objective", "threshold", "ensemble_size", "deep", "train_tokens",
                                            "champion_valid_pnl", "valid_hours")},
        })
    # modeles entraines mais jamais mis en service
    for mid, h in hist.items():
        if mid not in service:
            out_models.append({"id": mid, "champion": mid == champ, "live": False, "start_ms": None, "end_ms": None,
                               "hours": 0, "trades": 0, "valid": {k: h.get(k) for k in (
                                   "valid_pnl", "valid_trades", "valid_tokens", "valid_auc", "valid_buy_all_pnl",
                                   "valid_win_rate_selected", "promoted", "family", "objective", "threshold",
                                   "ensemble_size", "deep", "train_tokens", "champion_valid_pnl", "valid_hours")}})

    focus = live_id or champ
    fp = [p for p in pos if p["model_id"] == focus]
    fd = [d for d in decs if d["model_id"] == focus]
    start = service.get(focus, [now])[0]
    windows = [{"label": l, "hours": hh, **window(fp, fd, now - hh * 3_600_000)} for l, hh in WINDOWS]
    windows.append({"label": "depuis sa mise en service", "hours": (now - start) / 3_600_000, **window(fp, fd, start)})
    all_windows = [{"label": l, "hours": hh, **window(pos, decs, now - hh * 3_600_000)} for l, hh in WINDOWS]

    # courbe des gains realises (toutes les positions), avec le modele qui a achete
    eq, cum = [], 0.0
    for p in sorted((p for p in pos if p["status"] == "closed" and p["pnl"] is not None), key=lambda p: p["exit_ms"] or 0):
        cum += p["pnl"]
        eq.append([p["exit_ms"], round(cum, 5), p["model_id"], p["name"], round(p["pnl"], 5)])
    # gain par heure du modele suivi (48 h)
    hours: dict[int, dict] = {}
    for p in fp:
        if p["status"] == "closed" and p["pnl"] is not None and (p["exit_ms"] or 0) >= now - 48 * 3_600_000:
            k = p["exit_ms"] // 3_600_000 * 3_600_000
            o = hours.setdefault(k, {"h": k, "pnl": 0.0, "n": 0, "wins": 0})
            o["pnl"] += p["pnl"]
            o["n"] += 1
            o["wins"] += p["pnl"] > 0
    recent = sorted((p for p in fp if p["status"] in EXEC), key=lambda p: -(p["entry_wall_ms"] or 0))[:25]
    return {"now": now, "champion": champ, "live_model": live_id, "focus": focus,
            "focus_meta": {**hist.get(focus, {}), "start_ms": start},
            "windows": windows, "all_windows": all_windows, "models": out_models, "equity": eq,
            "hourly": sorted(hours.values(), key=lambda o: o["h"]), "recent": recent, "exits": EXITS}


if __name__ == "__main__":  # apercu en ligne de commande
    from .config import Config
    from .feedback import Journal
    from .model import Registry
    cfg = Config()
    r = build(Journal(cfg.data / "journal.db"), Registry(cfg.models).state)
    print(json.dumps({k: r[k] for k in ("focus", "windows")}, indent=1, default=str)[:4000])
