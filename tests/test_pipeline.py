"""Pipeline complet sur donnees synthetiques : parquet -> features -> entrainement -> registre -> prediction."""
import json
import tomllib
from datetime import date, datetime, timedelta, timezone

import numpy as np

from pumpml import pump
from pumpml.config import ROOT, Config
from pumpml.storage import ParquetSink


def _cfg(tmp_path):
    raw = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    lines = [f'[paths]\ndata = "{(tmp_path / "data").as_posix()}"\nmodels = "{(tmp_path / "models").as_posix()}"\n'
             f'logs = "{(tmp_path / "logs").as_posix()}"\n']
    for sec, vals in raw.items():
        if sec == "paths":
            continue
        if sec == "training":
            vals = {**vals, "min_train_rows": 200, "min_valid_trades": 5, "text_model": "__none__", "search_trials": 3, "deep_trials": 4, "deep_hours": 0.2, "feature_groups": [],
                    "image_model": "__none__", "device": "cpu", "embedding_dims": 8}
        lines.append(f"[{sec}]")
        def toml(v):  # JSON convient sauf pour les tableaux en ligne
            if isinstance(v, dict):
                return "{ " + ", ".join(f"{k} = {toml(x)}" for k, x in v.items()) + " }"
            return json.dumps(v)
        lines += [f"{k} = {toml(v)}" for k, v in vals.items()]
    p = tmp_path / "config.toml"
    p.write_text("\n".join(lines), encoding="utf-8")
    return Config(p)


def _fill(cfg, days, per_day=400, seed=0):
    rng = np.random.default_rng(seed)
    sinks = {t: ParquetSink(cfg.data, t, 10**9, 10**9) for t in ("tokens", "trades", "metadata")}
    for d in days:
        t0 = int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() * 1000)
        for i in range(per_day):
            created = t0 + int(i * 86_400_000 / per_day)
            mint = f"M{d}{i:05d}"
            good = rng.random() < 0.3
            name = ("cat " if good else "rug ") + str(i)
            sinks["tokens"].add({"recv_ms": created, "source": "pumpportal", "mint": mint, "name": name,
                                 "symbol": name[:3].upper(), "uri": "u", "creator": f"c{i % 50}", "sol_amount": 0.5,
                                 "raw": '{"pool":"pump"}', "sampled": True})
            sinks["metadata"].add({"recv_ms": created + 3000, "mint": mint, "uri": "u", "ok": True,
                                   "description": name, "twitter": f"https://x.com/a/status/{(created - 30_000 - 1288834974657) << 22}" if good else "https://x.com/rugger"})
            vs, vt = pump.INITIAL_VSOL, pump.INITIAL_VTOK
            n_early = rng.integers(8, 20) if good else rng.integers(1, 6)
            ts = sorted(rng.integers(1000, 60_000, n_early).tolist())
            ts += sorted(rng.integers(63_000, 1_500_000, 30).tolist())
            for k, t in enumerate(ts):
                late = t > 60_000
                buy = (rng.random() < (0.8 if good else 0.25)) if late else True
                sol = float(rng.uniform(0.2, 1.5))
                if buy:
                    tok, vs, vt = pump.sim_buy(vs, vt, sol, 0)
                else:
                    sol = min(sol, (vs - pump.INITIAL_VSOL) / 1e9 * 0.5)
                    if sol <= 0:
                        continue
                    kk = vs * vt
                    nv = vs - sol * 1e9
                    tok = kk / nv - vt
                    vs, vt = int(nv), int(kk / nv)
                sinks["trades"].add({"recv_ms": created + t, "slot": k, "signature": "s", "mint": mint, "is_buy": buy,
                                     "sol_amount": int(sol * 1e9), "token_amount": int(tok), "user": f"u{rng.integers(1e6)}",
                                     "ts": 0, "vsol": int(vs), "vtok": int(vt), "real_sol": int(vs - pump.INITIAL_VSOL),
                                     "real_tok": int(793_100_000_000_000 - (pump.INITIAL_VTOK - vt))})
        # trades "bruit" chaque minute, pour qu'il n'y ait pas de trou de collecte
        for m in range(0, 86_400_000 + 3_600_000, 30_000):
            sinks["trades"].add({"recv_ms": t0 + m, "slot": 0, "signature": "n", "mint": "NOISE", "is_buy": True,
                                 "sol_amount": 1, "token_amount": 1, "user": "n", "ts": 0, "vsol": pump.INITIAL_VSOL,
                                 "vtok": pump.INITIAL_VTOK, "real_sol": 0, "real_tok": 0})
    for s in sinks.values():
        s.flush()


def test_pipeline(tmp_path):
    cfg = _cfg(tmp_path)
    days = [(date.today() - timedelta(days=k)).isoformat() for k in (6, 5, 4)]
    _fill(cfg, days)
    from pumpml import dataset as D
    from pumpml.model import Registry, train
    assert D.complete_days(cfg) == days
    df = D.load_window(cfg, days)
    assert len(df) > 1000 and df.pnl.notna().all()
    assert df.x_tweet.sum() > 0
    r = train(cfg)
    assert r["status"] == "ok", r
    assert r["valid_auc"] > 0.7, r
    reg = Registry(cfg.models)
    assert reg.state["history"]
    if r["promoted"]:
        assert reg.champion() is not None
    # 2e entrainement : compare au champion sans planter
    r2 = train(cfg)
    assert r2["status"] == "ok"
