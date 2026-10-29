import base64
import struct

import base58
import numpy as np
import pytest

from pumpml import features as F
from pumpml import pump, social

MINT = base58.b58encode(bytes(range(32))).decode()
USER = base58.b58encode(bytes([7] * 32)).decode()
CREATOR = base58.b58encode(bytes([9] * 32)).decode()


def _pk(s):
    return base58.b58decode(s)


def trade_bytes(is_buy=True, extended=True):
    b = pump.DISC_TRADE + _pk(MINT) + struct.pack("<QQ?", 1_000_000_000, 35_000_000_000_000, is_buy) + _pk(USER)
    b += struct.pack("<qQQQQ", 1_700_000_000, 31_000_000_000, 1_038_000_000_000_000, 1_000_000_000, 758_100_000_000_000)
    if extended:
        b += bytes(32) + struct.pack("<QQ", 95, 9_500_000) + _pk(CREATOR) + struct.pack("<QQ", 30, 3_000_000)
        b += b"\x01" + bytes(40)  # champs futurs : doivent etre ignores
    return b


def test_decode_trade():
    for ext in (True, False):
        ev = pump.decode_log_line("Program data: " + base64.b64encode(trade_bytes(extended=ext)).decode())
        assert isinstance(ev, pump.Trade)
        assert ev.mint == MINT and ev.user == USER and ev.is_buy
        assert ev.sol_amount == 1_000_000_000 and ev.vsol == 31_000_000_000
        assert ev.creator == (CREATOR if ext else None)


def test_decode_create_and_garbage():
    s = lambda x: struct.pack("<I", len(x.encode())) + x.encode()
    b = pump.DISC_CREATE + s("Doge Moon") + s("DGM") + s("https://ipfs.io/ipfs/Qm") + _pk(MINT) + bytes(32) + _pk(USER) + _pk(CREATOR)
    ev = pump.decode_log_line("Program data: " + base64.b64encode(b).decode())
    assert isinstance(ev, pump.Create) and ev.name == "Doge Moon" and ev.creator == CREATOR
    assert pump.decode_log_line("Program data: " + base64.b64encode(pump.DISC_TRADE + b"\x00" * 10).decode()) is None
    assert pump.decode_log_line("Program log: Instruction: Buy") is None
    assert pump.decode_log_line("Program data: !!!notbase64") is None


def test_curve_roundtrip_loses_fees():
    tok, vs, vt = pump.sim_buy(pump.INITIAL_VSOL, pump.INITIAL_VTOK, 1.0, 125)
    back = pump.sim_sell(vs, vt, tok, 125)
    assert 0.97 < back < 1.0
    assert pump.mcap_sol(pump.INITIAL_VSOL, pump.INITIAL_VTOK) == pytest.approx(27.96, rel=1e-3)


def make_trades(points, start=0):
    """points : liste de (t_ms, is_buy, sol) ; reserves recalculees comme sur la courbe."""
    vs, vt = pump.INITIAL_VSOL, pump.INITIAL_VTOK
    rows = []
    for i, (t, buy, sol) in enumerate(points):
        if buy:
            tok, vs, vt = pump.sim_buy(vs, vt, sol, 0)
        else:
            k = vs * vt
            new_vs = vs - sol * 1e9
            tok = k / new_vs - vt
            vs, vt = int(new_vs), int(k / new_vs)
        rows.append({"recv_ms": start + t, "slot": i, "is_buy": buy, "sol_amount": int(sol * 1e9),
                     "token_amount": int(tok), "user": f"u{i % 5}", "vsol": vs, "vtok": vt,
                     "real_tok": 793_100_000_000_000 - (pump.INITIAL_VTOK - vt), "real_sol": vs - pump.INITIAL_VSOL})
    return F.Trades.from_rows(rows)


STRAT = {"latency_s": 1.0, "buy_sol": 0.1, "take_profit": 1.0, "stop_loss": 0.4, "max_hold_s": 600,
         "fee_bps": 125, "tx_cost_sol": 0.0005}


def test_no_lookahead():
    base = [(1000, True, 1.0), (5000, True, 0.5), (20_000, False, 0.2)]
    a = make_trades(base)
    b = make_trades(base + [(61_000, True, 50.0), (90_000, False, 1.0)])
    fa = F.flow_features(0, "u0", a, 60_000)
    fb = F.flow_features(0, "u0", b, 60_000)
    assert fa == fb


def test_simulate_tp_and_sl():
    pump_up = make_trades([(1000, True, 0.5)] + [(62_000 + i * 1000, True, 3.0) for i in range(15)])
    o = F.simulate(pump_up, 60_000, STRAT)
    assert o.exit_reason == "tp" and o.pnl_sol > 0.05
    dump = make_trades([(1000, True, 20.0), (62_000, False, 19.0)])  # prix x0.38
    o = F.simulate(dump, 60_000, STRAT)
    assert o.exit_reason == "sl" and o.pnl_sol < -0.03
    dead = make_trades([(1000, True, 0.5)])
    o = F.simulate(dead, 60_000, STRAT)
    assert o.exit_reason == "timeout" and -0.01 < o.pnl_sol < 0


def test_parse_link():
    assert social.parse_link("https://x.com/elonmusk/status/123")["kind"] == "tweet"
    assert social.parse_link("twitter.com/SomeOne")["handle"] == "someone"
    assert social.parse_link("https://x.com/i/communities/1789")["kind"] == "community"
    assert social.parse_link("https://t.me/abc")["kind"] == "other"
    assert social.parse_link(None)["kind"] == "none"
    assert social.parse_link("https://x.com/search?q=doge")["kind"] == "other"


def test_trend_no_future():
    from pumpml import narrative as N
    rng = np.random.default_rng(0)
    T = rng.normal(size=(50, 8)).astype(np.float32)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    c = np.arange(50) * 60_000
    pnl = rng.normal(size=50)
    res = c + 30 * 60_000
    a = N.trend_features(c, T, pnl, res)
    pnl2 = pnl.copy()
    pnl2[45:] = 100  # changer le futur ne doit rien changer avant
    b = N.trend_features(c, T, pnl2, res)
    for k in a:
        np.testing.assert_array_equal(a[k][:45], b[k][:45])


def test_curve_filters():
    std = make_trades([(1000, True, 1.0), (2000, True, 0.5)])
    assert F.is_standard_curve(std)
    unpriced = make_trades([(1000, True, 1.0)])
    unpriced.vsol[:] = 0
    assert not F.is_standard_curve(unpriced)
    other = make_trades([(1000, True, 1.0)])
    other.vsol[:] = other.vsol / 2  # autres reserves initiales
    other.real_sol[:] = 0
    assert not F.is_standard_curve(other)
    # pile d'appels : un evenement emis par un autre programme est ignore
    data = "Program data: " + base64.b64encode(trade_bytes()).decode()
    logs = ["Program Other111 invoke [1]", data, "Program Other111 success",
            f"Program {pump.PUMP_PROGRAM} invoke [1]", data, f"Program {pump.PUMP_PROGRAM} success"]
    assert len(list(pump.iter_events(logs))) == 1
    assert list(pump.iter_events(["Log truncated", data])) == []


def test_curve_modified_midway():
    tr = make_trades([(1000, True, 1.0), (2000, True, 0.5), (90_000, True, 0.5)])
    tr.vsol[1] -= 5e9  # reserves virtuelles modifiees avant la decision
    assert not F.is_standard_curve(tr.upto(60_000))
    tr2 = make_trades([(1000, True, 1.0), (2000, True, 0.5), (90_000, True, 0.5)])
    tr2.vsol[2] -= 5e9  # modifiees APRES la decision : invisible au moment de decider
    assert F.is_standard_curve(tr2.upto(60_000))


def test_vamp_memory():
    from pumpml import vamp as V
    m = V.Memory()
    t0 = 1_000_000_000
    m.add("ORIG", t0, "Doge Moon", "$DOGE")
    m.set_meta("ORIG", t0 + 5_000, "https://x.com/doge/status/42", 0x1234)
    m.set_outcome("ORIG", t0 + 31 * 60_000, 8.0, 0.1, True)
    # copie 2 h apres : meme ticker (casse/$ differents), meme tweet, image quasi identique
    m.add("COPY", t0 + 7_200_000, "doge moon", "doge")
    f, info = m.features("COPY", t0 + 7_200_000, "doge moon", "doge", "https://twitter.com/doge/status/42",
                         0x1234 ^ 0b11, t0 + 7_260_000)
    assert f["is_first"] == 0 and f["dup_symbol"] == 1 and f["dup_twitter"] == 1 and f["dup_image"] == 1
    assert f["dup_best_mult"] == 8.0 and f["dup_any_migrated"] == 1.0 and f["dup_strong"] == 1
    assert info["mint"] == "ORIG" and "ticker" in info["how"]
    # copie 10 min apres : le resultat de l'original n'est pas encore connu -> pas de fuite
    f2, _ = m.features("EARLY", t0 + 600_000, "x", "DOGE", None, None, t0 + 660_000)
    assert f2["dup_symbol"] == 1 and np.isnan(f2["dup_best_mult"])
    # l'original lui-meme : rien avant lui
    f3, info3 = m.features("ORIG", t0, "Doge Moon", "$DOGE", None, 0x1234, t0 + 60_000)
    assert f3["is_first"] == 1 and info3 is None


def test_buy_rule_first_checkpoint_only():
    from pumpml.model import bought, prev_cummax
    mint = np.array(["A", "A", "A", "B", "B"], object)
    age = np.array([5, 10, 15, 5, 10])
    score = np.array([0.2, 0.6, 0.9, 0.7, 0.8])
    pcm = prev_cummax(score, mint, age)
    sel = bought(score, pcm, 0.5)
    # A : achete a 10 s (pas une 2e fois a 15 s) ; B : achete des 5 s
    assert sel.tolist() == [False, True, False, True, False]
    # lignes melangees : l'ordre d'entree ne doit pas compter
    perm = np.array([4, 2, 0, 3, 1])
    sel2 = bought(score[perm], prev_cummax(score[perm], mint[perm], age[perm]), 0.5)
    assert sel2.tolist() == sel[perm].tolist()


def test_narrative_library_no_leak():
    from pumpml.protos import Library
    rng = np.random.default_rng(0)
    T = rng.normal(size=(3, 16)).astype(np.float32)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    lib = Library(mints=np.array(["G1", "G2", "B1"], object), known_ms=np.array([100, 5_000, 100], np.int64),
                  verdict=np.array([1, 1, -1]), T=T, I=np.zeros_like(T), has_img=np.zeros(3, bool),
                  tags=[{"animal"}, {"news du jour"}, set()], categories=["animal", "news du jour"])
    q = T[[1]]  # requete identique au bon exemple G2
    early = lib.features(np.array(["X"], object), np.array([1_000]), q, np.zeros_like(q), np.array([False]))
    late = lib.features(np.array(["X"], object), np.array([10_000]), q, np.zeros_like(q), np.array([False]))
    self_q = lib.features(np.array(["G2"], object), np.array([10_000]), q, np.zeros_like(q), np.array([False]))
    assert early["proto_bon_texte"][0] < 0.99          # G2 pas encore utilisable (resultat pas connu)
    assert late["proto_bon_texte"][0] > 0.99           # G2 utilisable ensuite
    assert self_q["proto_bon_texte"][0] < 0.99         # jamais pour le coin lui-meme
    assert late["cat_news_du_jour"][0] > 0.99 and np.isnan(early["cat_news_du_jour"][0])


def test_simulate_stale_and_trailing_exits():
    flat = make_trades([(1000, True, 0.5)] + [(62_000 + i * 20_000, True, 0.01) for i in range(20)])
    s = {**STRAT, "stale_exit_s": 120, "stale_min_gain": 0.10}
    o = F.simulate(flat, 60_000, s)
    assert o.exit_reason == "stale" and 115 <= o.hold_s <= 125
    # sans trade apres l'echeance : sortie "stale" quand meme
    dead = make_trades([(1000, True, 0.5)])
    o = F.simulate(dead, 60_000, s)
    assert o.exit_reason == "stale"
    # stop suiveur : monte fort puis retombe -> "trail", pas de plafond
    up_down = make_trades([(1000, True, 0.5)] + [(62_000 + i * 1000, True, 3.0) for i in range(8)]
                          + [(80_000, False, 12.0)])
    t = {**STRAT, "take_profit": 0, "trail_activate": 0.3, "trail_pct": 0.25}
    o = F.simulate(up_down, 60_000, t)
    assert o.exit_reason == "trail" and o.pnl_sol > 0


def test_amm_swap_decode_and_rug_rules():
    """Evenement PumpSwap (valeurs d'une vraie transaction verifiees sur les soldes) et regles anti-rug."""
    import base64
    import struct
    from pumpml import amm
    pool, user = bytes(range(32)), bytes(range(32, 64))
    buy = amm.DISC_BUY + struct.pack("<q13Q", 1793006949, 74772700152, 313877663, 0, 313877663, 65149284258427,
                                     252762712382, 313877663, 20, 621233, 5, 155309, 311237424, 310616191) + pool + user
    logs = ["Program pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA invoke [1]",
            "Program data: " + base64.b64encode(buy).decode(),
            "Program pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA success"]
    sw = list(amm.iter_swaps(logs))
    assert len(sw) == 1 and sw[0].is_buy and sw[0].token_amount == 74772700152
    assert sw[0].sol_amount == 313877663 and sw[0].pool_sol == 252762712382
    assert abs(sw[0].fee - 0.0105) < 0.0005  # ~1,05 % de frais totaux a l'achat
    # un evenement du meme nom emis par un AUTRE programme est ignore
    assert not list(amm.iter_swaps(["Program Other1111 invoke [1]"] + logs[1:2] + ["Program Other1111 success"]))
    a = {"min_curve_seconds": 60, "max_block0_share": 0.25, "max_top10_share": 0.4}
    assert amm.rug_reasons({"curve_s": 0.0, "block0_share": 1.0, "top10_share": 0.1}, a) == [
        "migre_trop_vite", "achat_groupe_bloc0"]
    assert amm.rug_reasons({"curve_s": 900.0, "block0_share": 0.003, "top10_share": 0.2}, a) == []
    assert amm.rug_reasons({"curve_s": 150.0, "block0_share": 0.0, "top10_share": 0.88}, a) == ["detention_concentree"]
    assert amm.rug_reasons({"curve_tx": 20000, "top10_share": 0.1}, a) == ["historique_inconnu"]


def test_amm_simulate_tp_and_stale():
    from pumpml import amm
    s = {"latency_s": 3.0, "buy_sol": 0.1, "take_profit": 0.5, "stop_loss": 0.4, "max_hold_s": 1800,
         "stale_exit_s": 120, "stale_min_gain": 0.10, "tx_cost_sol": 0.0005}

    def rows(prices):
        return [{"ts": 1000 + 10 * k, "slot": k, "is_buy": True, "sol_amount": int(1.01e9), "sol_swap": int(1e9),
                 "token_amount": int(1e9 / p), "pool_sol": int(85e9)} for k, p in enumerate(prices)]
    up = amm.simulate(amm.Swaps(rows([1.0, 1.0, 1.3, 1.7, 2.0, 2.0])), 1_000_000, s)
    assert up["reason"] == "tp" and up["pnl"] > 0.04
    flat = amm.simulate(amm.Swaps(rows([1.0] * 30)), 1_000_000, s)
    assert flat["reason"] == "stale" and -0.004 < flat["pnl"] < 0  # ~2 % de frais aller-retour + tx
    assert amm.simulate(amm.Swaps(rows([1.0])), 900_000, s) is None  # aucun prix avant l'achat


def test_reply_parasite_and_tweet_wait():
    """Reponse a un tweet viral : meme auteur ou compte credible -> ok ; petit compte qui s'accroche -> parasite."""
    from pumpml import features as F
    tok = {"recv_ms": 1_800_000_000_000, "name": "Moo Deng", "symbol": "MOODENG"}
    base = {"tweet_text": "moodeng just did it again", "rel": "reply", "parent_handle": "zoo", "parent_followers": 500_000,
            "parent_text": "our hippo moodeng", "parent_created_ms": tok["recv_ms"] - 600_000}
    parasite = {**base, "handle": "randomguy", "followers": 300}
    assert F.is_parasite(parasite) and F.excluded(tok, None, F.Trades.from_rows([]), {"block_parasite": True}, parasite)
    assert not F.is_parasite({**base, "handle": "zoo", "followers": 500_000})          # suite par l'auteur
    assert not F.is_parasite({**base, "handle": "reporter", "followers": 80_000})      # compte credible
    assert not F.is_parasite({**base, "rel": None, "handle": "randomguy", "followers": 300})  # pas une reponse
    f = F.parent_features(tok, parasite)
    assert f["x_parasite"] == 1 and f["x_same_author"] == 0 and f["x_name_in_parent"] == 1 and f["x_follower_gap"] < -5
    assert abs(f["x_parent_age_min"] - 10) < 1e-6
    # tweet frais (30 s) pas encore lu : attente jusqu'a 60 s d'age du coin, puis decision sans lui
    tid = (tok["recv_ms"] - 30_000 - 1288834974657) << 22
    meta = {"twitter": None, "website": f"https://x.com/a/status/{tid}"}  # lien X range dans "site web"
    s = {"wait_tweet_s": 60, "max_tweet_age_min": 2}
    assert F.tweet_pending(tok, meta, None, tok["recv_ms"] + 10_000, s)
    assert not F.tweet_pending(tok, meta, {"tweet_text": "x"}, tok["recv_ms"] + 10_000, s)
    assert not F.tweet_pending(tok, meta, None, tok["recv_ms"] + 61_000, s)
    old = (tok["recv_ms"] - 5 * 60_000 - 1288834974657) << 22
    assert F.excluded(tok, {"twitter": f"https://x.com/a/status/{old}"}, F.Trades.from_rows([]),
                      {"max_tweet_age_min": 2}) == "vieux_tweet"


def test_require_link():
    """Ni Twitter ni site web : pas de narrative, pas d'achat ; metadonnees absentes : on attend."""
    from pumpml import features as F
    tok = {"recv_ms": 1_800_000_000_000}
    s, empty = {"require_link": True}, F.Trades.from_rows([])
    assert F.excluded(tok, {"twitter": None, "website": " ", "telegram": "t.me/x"}, empty, s) == "sans_lien"
    assert F.excluded(tok, {"twitter": None, "website": "https://coin.fun"}, empty, s) is None
    assert F.excluded(tok, {"twitter": "https://x.com/coin"}, empty, s) is None
    assert F.excluded(tok, None, empty, s) is None  # pas encore de metadonnees : decide plus tard


def test_search_ensemble_roundtrip(tmp_path):
    """Recherche courte (LightGBM + XGBoost), ensemble sauvegarde puis recharge : memes predictions."""
    import pandas as pd
    from pumpml import search as S
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"net_sol": rng.normal(size=n), "x_followers": rng.normal(size=n), "fame_text_sim": rng.random(n)})
    y = (X.net_sol + rng.normal(scale=0.5, size=n) > 0.8).astype(int).to_numpy()
    pnl = np.where(y == 1, 0.05, -0.01)
    created = np.arange(n) * 1000
    idx = np.arange(n)
    tc = {"search_trials": 6, "families": ["lgb", "xgb"], "device": "cpu", "threads": 2, "ensemble": 3}

    def thr_fn(score, ix, mn=None):
        best = max(((t, float(pnl[ix][score >= t].sum()), int((score >= t).sum()))
                    for t in np.quantile(score, np.linspace(0.5, 0.99, 20))), key=lambda r: r[1])
        return best
    res = S.run(tc, tmp_path, X, y, np.ones(n), pnl, created, np.zeros(n, bool), idx[:2000], idx[2000:], idx, idx,
                5, False, None, thr_fn)
    ens = res["ensemble"]
    members = ens.save(tmp_path)
    back = S.Ensemble.load(tmp_path, members)
    assert np.allclose(back.predict(X.iloc[:50]), ens.predict(X.iloc[:50]), atol=1e-5)
    assert back.contrib(X.iloc[:5]).shape == (5, 3)
    assert S.group_of("fame_twist") == "gagnants" and S.group_of("x_parasite") == "tweet"


def test_fame_hold_rule():
    """Regle de tenue : meche / vidage / courbe sans foule = flop ; tenue pas mesuree = hors memoire."""
    import importlib.util
    import pandas as pd
    from pathlib import Path
    from pumpml import fame as FM
    spec = importlib.util.spec_from_file_location("imp", Path(__file__).parent.parent / "scripts" / "import_fame.py")
    imp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imp)
    sup, mig = 1e9, 1_000_000
    # plus haut 150 k$ a la 2e bougie, puis 60 k$ (40 %) 1 h apres ; bougies sans trade = derniere cloture
    bars = [[mig, 5e-5, 9e-5, 5e-5, 8e-5, 1], [mig + 900, 8e-5, 15e-5, 7e-5, 9e-5, 1], [mig + 2700, 9e-5, 9e-5, 6e-5, 6e-5, 1],
            [mig + 9000, 6e-5, 6e-5, 1e-6, 2e-6, 1]]
    peak, pk_ms, hold = imp.hold_of(bars, sup, mig, 3600, mig + 99_999)
    assert peak == pytest.approx(150_000) and pk_ms == (mig + 1800) * 1000 and hold == pytest.approx(0.4)
    assert imp.hold_of(bars, sup, mig, 3600, mig + 3000) is None  # pas encore mesurable
    cfg = {"fame": {"min_peak_mcap_usd": 100_000, "win_rule": "tenue", "hold_minutes": 60, "min_hold_ratio": 0.3,
                    "min_curve_tx": 100}}
    f = pd.DataFrame({"migrated_ms": [0] * 6, "win_ms": [5] * 6,
                      "peak_mcap_usd": [150e3, 150e3, 150e3, 150e3, 150e3, 50e3],
                      "peak15_usd": [150e3, 53e3, 150e3, 150e3, np.nan, np.nan],
                      "peak15_ms": [10, 10, 10, 10, np.nan, np.nan],
                      "hold_ratio": [0.4, 0.9, 0.02, 0.4, np.nan, np.nan],
                      "curve_tx": [np.nan, 500, 500, 7, 500, 500]})
    win, known = FM.label(f, cfg)
    assert list(win.fillna(-1)) == [1, 0, 0, 0, -1, 0]
    assert known[0] == 10 + 3_600_000 and known[5] == 86_400_000
    old, _ = FM.label(f, {"fame": {**cfg["fame"], "win_rule": "plus_haut"}})
    assert list(old) == [1, 1, 1, 1, 1, 0]


def test_perf_stats():
    """Onglet Modeles : gain, taux de gain, facteur de profit, drawdown ; les achats non executes a part."""
    from pumpml import perf
    P = lambda st, pnl, t, r="sl": {"status": st, "pnl": pnl, "cost": 0.1, "value": 0.12, "exit_ms": t, "exit_reason": r}
    s = perf.stats([P("closed", 0.05, 1, "tp"), P("closed", -0.02, 2), P("closed", -0.04, 3), P("open", None, None),
                    P("no_quote", None, None)])
    assert s["trades"] == 4 and s["closed"] == 3 and s["open"] == 1 and s["not_executed"] == 1
    assert s["pnl"] == pytest.approx(-0.01) and s["win_rate"] == pytest.approx(1 / 3)
    assert s["profit_factor"] == pytest.approx(0.05 / 0.06) and s["max_drawdown"] == pytest.approx(-0.06)
    assert s["unrealized"] == pytest.approx(0.02) and s["reasons"]["tp"]["n"] == 1


def test_kol_no_lookahead():
    """KOL : seuls les achats avant la decision comptent ; le flair n'utilise que des resultats deja connus."""
    import pandas as pd
    from pumpml import kol as K
    H = 3_600_000
    h = pd.DataFrame({"signature": list("abcde"), "mint": ["m1", "m2", "m3", "m3", "m3"],
                      "recv_ms": [0, 10, 5 * H, 5 * H + 1000, 5 * H + 90_000],
                      "wallet": ["w1", "w1", "w1", "w2", "w1"], "person": ["A", "A", "A", "B", "A"],
                      "is_buy": [True, True, True, True, False], "sol": [1.0, 1.0, 2.0, 0.5, 2.0]})
    b = K.Book(h, {"m1": H})                       # m1 migre 1 h apres l'achat de A
    p0 = b.base_rate(30 * 60_000)
    assert b.score("A", 30 * 60_000)[1] == 0          # migration de m1 pas encore vue
    s, n = b.score("A", 2 * H)
    assert n == 1 and s > p0                          # m1 connu (gagnant), m2 pas encore (24 h)
    f = b.features("m3", 5 * H - 5000, 5 * H + 500)  # seul l'achat de A est passe
    assert f["kol_n"] == 1 and f["kol_buy_sol"] == 2.0 and f["kol_sells"] == 0 and f["kol_first_age_s"] == 5
    f = b.features("m3", 5 * H - 5000, 5 * H + 100_000)
    assert f["kol_n"] == 2 and f["kol_sells"] == 1 and f["kol_holding_n"] == 1
    assert b.score("A", 5 * H, exclude="m1")[1] == 0  # le coin lui-meme n'entre jamais dans le flair


def test_kolnarr_no_lookahead():
    """Memoire des gros KOL : un coin n'est visible qu'apres le 1er achat d'un gros KOL ; createur compte ; fenetre 48 h."""
    import pandas as pd
    from pumpml import fame as FMm, kolnarr as KN
    H = 3_600_000
    rows = pd.DataFrame({"mint": ["a", "b", "c"], "creator": ["C1", "C2", "C1"], "known_ms": [0, 100 * H, 10 * H],
                         "created_ms": [0, 100 * H, 10 * H], "name": ["Dog", "Cat", "Frog"], "symbol": ["DOG", "CAT", "FROG"]})
    e = {k: (np.eye(4, dtype=np.float32)[i], np.eye(4, dtype=np.float32)[i], True) for i, k in enumerate("abc")}
    bank = FMm.Bank.make(rows, e, ["a", "b", "c"], 4)
    kn = KN.KolNarr(bank, None, FMm.Bank.make(rows.iloc[:0], {}, [], 4), FMm.Bank.make(rows.iloc[:0], {}, [], 4),
                    np.array([1, 5, 3]))
    q = np.eye(4, dtype=np.float32)[[1, 1, 0]]          # requetes : ressemble a b, a b, a a
    f = kn.features(np.array(["x", "y", "z"], object), np.array(["C1", "C9", "C1"], object),
                    np.array(["Cat", "q", "Dog"], object), np.array(["", "", ""], object), q, q, np.array([True] * 3),
                    np.array([50 * H, 101 * H, 60 * H]))
    assert f["kolnarr_text_sim"][0] < 0.5                 # b pas encore achete a 50 h
    assert f["kolnarr_text_sim"][1] > 0.99 and f["kolnarr_same_name"][1] == 0
    assert f["kolnarr_creator_n"][0] == 2 and f["kolnarr_creator_n"][1] == 0   # C1 : a (0 h) et c (10 h)
    assert np.isnan(f["kolnarr_recent_sim"][2])            # a et c ont plus de 48 h a 60 h


def test_copy_veto():
    from pumpml import vamp as V
    def tid(ms):
        return str((ms - 1288834974657) << 22)
    t0 = 1_800_000_000_000
    mem = V.Memory()
    mem.add("ORIG", t0, "Dog Wif Hat", "WIF")
    mem.set_meta("ORIG", t0 + 1000, f"https://x.com/a/status/{tid(t0 - 30_000)}", None)
    mem.add("TICK", t0 + 5000, "autre chose", "WIF")  # ticker seul : pas une copie forte
    f, info = mem.features("COPY", t0 + 600_000, "dog wif hat", "WIF", None, None, t0 + 600_000)
    assert f["dup_strong"] == 1 and info["orig"]["mint"] == "ORIG"
    _, info2 = mem.features("NEW", t0 + 600_000, "chat", "WIF", None, None, t0 + 600_000)
    assert "orig" not in info2
    orig = info["orig"]
    meta = lambda ms: {"twitter": f"https://x.com/b/status/{tid(ms)}"}  # noqa: E731
    soc = {"tweet_text": "breaking: the dog got a new hat"}
    assert F.copy_veto(meta(t0 + 500_000), soc, None, None, None) is None          # pas une copie
    assert F.copy_veto({"twitter": "https://x.com/b"}, soc, orig, None, None)       # pas de tweet
    assert "meme tweet" in F.copy_veto({"twitter": f"https://x.com/a/status/{tid(t0 - 30_000)}"}, soc, orig, None, None)
    assert "anterieur" in F.copy_veto(meta(t0 - 10_000), soc, orig, None, None)
    assert "non lu" in F.copy_veto(meta(t0 + 500_000), None, orig, None, None)
    assert "meme texte" in F.copy_veto(meta(t0 + 500_000), soc, orig, "the dog", 0.95)
    assert F.copy_veto(meta(t0 + 500_000), soc, orig, "the dog", 0.5) is None       # info nouvelle : permis


def test_percees_no_lookahead():
    import pandas as pd
    from pumpml import percees as PC
    rng = np.random.default_rng(0)
    n, dim = 60, 8
    T = rng.normal(size=(n, dim)).astype(np.float32)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    m = pd.DataFrame({"mint": [f"m{i}" for i in range(n)], "creator": ["c0"] + [f"c{i}" for i in range(1, n)],
                      "name": "x", "symbol": "x", "created_ms": np.arange(n) * 1000,
                      "known_ms": 10_000 + np.arange(n) * 1000, "perc": (np.arange(n) % 6 == 0).astype(int),
                      "kol": np.arange(n) == 0})
    keys = list(m.mint)
    embs = {k: (T[i], T[i], True) for i, k in enumerate(keys)}
    p = PC.make(m, embs, keys, dim)
    q = T[[0]]
    # avant que la percee m0 soit connue : rien sur elle ; apres : elle-meme exclue, mais un clone la voit
    f = p.features(np.array(["new"], object), np.array(["c0"], object), q, q, np.array([True]), np.array([9_000]))
    assert np.isnan(f["perc_text_sim"][0]) and f["perc_creator_n"][0] == 0 and np.isnan(f["perc_rate"][0])
    f = p.features(np.array(["new"], object), np.array(["c0"], object), q, q, np.array([True]), np.array([200_000]))
    assert f["perc_text_sim"][0] > 0.99 and f["perc_kol_sim"][0] > 0.99 and f["perc_creator_n"][0] == 1
    assert 0 < f["perc_rate"][0] < 1
    f = p.features(np.array(["m0"], object), np.array(["c0"], object), q, q, np.array([True]), np.array([200_000]))
    assert f["perc_creator_n"][0] == 0 and (np.isnan(f["perc_kol_sim"][0]) or f["perc_kol_sim"][0] < 0.99)
