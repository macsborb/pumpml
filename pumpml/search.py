"""Recherche de reglages (hyperparametres) du modele, en profondeur.

Chaque essai choisit :
* la famille de modele : LightGBM (processeur) ou XGBoost (carte graphique, CUDA) ;
* ses reglages (taille des arbres, vitesse d'apprentissage, regularisation, echantillonnage...) ;
* l'objectif d'apprentissage : gagne / ne gagne pas, pondere par l'ampleur du gain ou de la perte
  (avec un exposant), ou PnL directement ;
* le poids des donnees recentes (demi-vie en jours) ;
* le poids de chaque famille de signaux (narrative, tweets, createur, gros gagnants, copies, flux de
  trades...) : un multiplicateur sur l'interet d'une coupure (LightGBM ``feature_contri``) ou sur la
  probabilite d'etre tire (XGBoost ``feature_weights``).

Optuna (TPE) propose les essais en apprenant des precedents. Chaque essai est entraine sur "fit" et juge
sur "es" (fin du train) avec la VRAIE regle d'achat et le PnL simule. La validation (heures les plus
recentes) n'est jamais regardee ici : elle sert seulement au verdict final, honnete.
Le modele final est la moyenne des meilleurs essais (ensemble), plus stable qu'un seul.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger("search")

GROUPS = {
    "narrative": r"^(txt_|img_|proto_|cat_|coin_th_|tw_th_|coin_gravite|tw_gravite|img_text_sim|coin_tweet_sim|sim_1h|nn_)",
    "tweet": r"^x_",
    "meta": r"^(name_|symbol_|desc_|has_|meta_ok|web_)",
    "createur": r"^creator_",
    "gagnants": r"^(fame_|flop_)",
    "copies": r"^(dup_|is_first)",
    "percees": r"^perc_",
    "kol": r"^kol_",
    "kolnarr": r"^kolnarr_",
    "flux": r"^(n_|buy_sol|sell_sol|net_sol|sell_buy|uniq_|avg_buy|max_buy|small_buys|dev_|top|early|block0|initial_buy|"
            r"repeat|price_mult|max_mult|drawdown|progress|mcap|recent_|since_last|first_trade)",
    "temps": r"^(hour_|dow|age_s)",
}
OBJECTIVES = ["binary", "pnl_weighted", "regression"]


def group_of(feat: str) -> str:
    for g, rx in GROUPS.items():
        if re.match(rx, feat):
            return g
    return "autre"


# ---------- modeles ----------

class Member:
    """Un modele de l'ensemble (LightGBM ou XGBoost)."""

    def __init__(self, kind: str, model, it: int | None = None):
        self.kind, self.model, self.it = kind, model, it

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "lgb":
            return self.model.predict(X, num_iteration=self.it or None)
        import xgboost as xgb
        r = (0, self.it + 1) if self.it is not None else (0, 0)
        return self.model.predict(xgb.DMatrix(X.to_numpy(np.float32), missing=np.nan), iteration_range=r)

    def contrib(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "lgb":
            return self.model.predict(X, pred_contrib=True, num_iteration=self.it or None)[:, :-1]
        import xgboost as xgb
        r = (0, self.it + 1) if self.it is not None else (0, 0)
        return self.model.predict(xgb.DMatrix(X.to_numpy(np.float32), missing=np.nan), pred_contribs=True,
                                  iteration_range=r)[:, :-1]

    def importance(self, feats: list[str]) -> np.ndarray:
        if self.kind == "lgb":
            return np.asarray(self.model.feature_importance("gain"), float)
        sc = self.model.get_score(importance_type="total_gain")
        return np.array([sc.get(f"f{i}", 0.0) for i in range(len(feats))])

    def save(self, path: Path) -> dict:
        if self.kind == "lgb":
            self.model.save_model(str(path.with_suffix(".txt")), num_iteration=self.it or None)
            return {"kind": "lgb", "file": path.with_suffix(".txt").name}
        self.model.save_model(str(path.with_suffix(".json")))
        return {"kind": "xgb", "file": path.with_suffix(".json").name, "it": self.it}

    @staticmethod
    def load(d: Path, m: dict) -> "Member":
        if m["kind"] == "lgb":
            return Member("lgb", lgb.Booster(model_file=str(d / m["file"])))
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(str(d / m["file"]))
        b.set_param({"device": "cpu"})  # prediction en direct : un coin a la fois, le CPU suffit
        return Member("xgb", b, m.get("it"))


class Ensemble:
    """Moyenne des membres (meme objectif, donc meme echelle de score)."""

    def __init__(self, members: list[Member]):
        self.members = members

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.members], axis=0)

    def contrib(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.contrib(X) for m in self.members], axis=0)

    def importance(self, feats: list[str]) -> np.ndarray:
        imps = [m.importance(feats) for m in self.members]
        return np.mean([i / max(i.sum(), 1e-9) for i in imps], axis=0)

    def save(self, d: Path) -> list[dict]:
        return [m.save(d / f"member{k}") for k, m in enumerate(self.members)]

    @staticmethod
    def load(d: Path, members: list[dict]) -> "Ensemble":
        return Ensemble([Member.load(d, m) for m in members])


# ---------- essais ----------

def _target(obj: str, y, pnl, w, power: float):
    if obj == "binary":
        return y.astype(float), w
    if obj == "pnl_weighted":
        a = np.abs(pnl) / max(np.abs(pnl).mean(), 1e-9)
        return y.astype(float), w * (a ** power + 0.1)
    return pnl * 10, w  # regression


def _suggest(trial, n_fit: int, families: list[str], tune_weights: bool, objectives: list[str],
             ranges: dict | None = None) -> dict:
    p = {"family": trial.suggest_categorical("family", families),
         "objective": trial.suggest_categorical("objective", objectives),
         "power": trial.suggest_float("power", 0.3, 2.0),
         "half_life_d": trial.suggest_categorical("half_life_d", [0.0, 1.0, 2.0, 4.0, 8.0])}
    if tune_weights:
        for g in GROUPS:  # groupes prioritaires (config training.group_ranges) : poids plus eleve
            lo, hi = (ranges or {}).get(g, (0.25, 3.0))
            p[f"w_{g}"] = trial.suggest_float(f"w_{g}", lo, hi, log=True)
    if p["family"] == "lgb":
        p.update(learning_rate=trial.suggest_float("lgb_lr", 0.005, 0.2, log=True),
                 num_leaves=trial.suggest_int("lgb_leaves", 4, 256, log=True),
                 min_data_in_leaf=min(trial.suggest_int("lgb_min_leaf", 5, 400, log=True), max(5, n_fit // 20)),
                 feature_fraction=trial.suggest_float("lgb_ff", 0.2, 1.0),
                 bagging_fraction=trial.suggest_float("lgb_bf", 0.4, 1.0),
                 bagging_freq=trial.suggest_categorical("lgb_bfreq", [0, 1, 5]),
                 lambda_l1=trial.suggest_float("lgb_l1", 1e-3, 50, log=True),
                 lambda_l2=trial.suggest_float("lgb_l2", 1e-3, 100, log=True),
                 min_gain_to_split=trial.suggest_float("lgb_gain", 0.0, 1.0),
                 max_depth=trial.suggest_categorical("lgb_depth", [-1, 4, 6, 8, 12]),
                 extra_trees=trial.suggest_categorical("lgb_extra", [False, True]),
                 is_unbalance=trial.suggest_categorical("lgb_unbal", [False, True]))
    else:
        p.update(eta=trial.suggest_float("xgb_eta", 0.005, 0.2, log=True),
                 max_depth=trial.suggest_int("xgb_depth", 2, 10),
                 min_child_weight=trial.suggest_float("xgb_mcw", 1.0, 100.0, log=True),
                 subsample=trial.suggest_float("xgb_sub", 0.4, 1.0),
                 colsample_bynode=trial.suggest_float("xgb_col", 0.2, 1.0),
                 reg_lambda=trial.suggest_float("xgb_l2", 1e-3, 100, log=True),
                 reg_alpha=trial.suggest_float("xgb_l1", 1e-3, 50, log=True),
                 gamma=trial.suggest_float("xgb_gamma", 0.0, 5.0),
                 scale_pos_weight=trial.suggest_float("xgb_spw", 1.0, 20.0, log=True))
    return p


def _fit(p: dict, X: pd.DataFrame, target, wt, fit_idx, es_idx, feats: list[str], device: str,
         threads: int) -> Member:
    gw = np.array([p.get(f"w_{group_of(f)}", 1.0) for f in feats])
    reg = p["objective"] == "regression"
    if p["family"] == "lgb":
        params = dict(objective="huber" if reg else "binary", verbose=-1, seed=0, num_threads=threads,
                      feature_contri=list(gw), **{k: p[k] for k in (
                          "learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction", "bagging_fraction",
                          "bagging_freq", "lambda_l1", "lambda_l2", "min_gain_to_split", "max_depth", "extra_trees")})
        if reg:
            params["alpha"] = 0.5
        else:
            params["is_unbalance"] = p["is_unbalance"]
        dfit = lgb.Dataset(X.iloc[fit_idx], target[fit_idx], weight=wt[fit_idx], free_raw_data=False)
        des = lgb.Dataset(X.iloc[es_idx], target[es_idx], weight=wt[es_idx], reference=dfit)
        b = lgb.train(params, dfit, 3000, valid_sets=[des], callbacks=[lgb.early_stopping(100, verbose=False)])
        return Member("lgb", b, b.best_iteration)
    import xgboost as xgb
    params = dict(device=device, tree_method="hist", seed=0, nthread=threads,
                  objective="reg:pseudohubererror" if reg else "binary:logistic",
                  eval_metric="mphe" if reg else "logloss",
                  **{k: p[k] for k in ("eta", "max_depth", "min_child_weight", "subsample", "colsample_bynode",
                                       "reg_lambda", "reg_alpha", "gamma")})
    if not reg:
        params["scale_pos_weight"] = p["scale_pos_weight"]
    Xa = X.to_numpy(np.float32)
    dfit = xgb.DMatrix(Xa[fit_idx], target[fit_idx], weight=wt[fit_idx], missing=np.nan)
    dfit.set_info(feature_weights=gw)
    des = xgb.DMatrix(Xa[es_idx], target[es_idx], weight=wt[es_idx], missing=np.nan)
    b = xgb.train(params, dfit, 3000, evals=[(des, "es")], early_stopping_rounds=100, verbose_eval=False)
    return Member("xgb", b, b.best_iteration)


def run(tc: dict, models_dir: Path, X: pd.DataFrame, y, w, pnl, created, isbf, fit_idx, es_idx,
        mints, ages, min_es: int, deep: bool, score_fn, threshold_fn, folds=()) -> dict:
    """Recherche ; retourne l'ensemble, son seuil (choisi sur es), l'objectif et le detail des essais.
    ``score_fn(score, idx) -> (pnl, n)`` et ``threshold_fn(score, idx) -> (seuil, pnl, n)`` appliquent la
    regle d'achat (premier point d'analyse au-dessus du seuil) sur les lignes ``idx``.
    ``folds`` : periodes supplementaires (fit, eval, achats minimum) ; le score d'un essai est la somme de
    ses PnL sur toutes les periodes, et une periode perdante est penalisee (le reglage doit tenir partout)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    feats = list(X.columns)
    n_trials = int(tc["deep_trials"] if deep else tc["search_trials"])
    families = list(tc.get("families", ["lgb", "xgb"]))
    device = "cuda" if tc.get("device", "cuda") == "cuda" else "cpu"
    threads = int(tc.get("threads", 16))
    objectives = OBJECTIVES if tc.get("search_objective", True) else [tc.get("objective", "pnl_weighted")]
    t_end = created[fit_idx][~isbf[fit_idx]].max() if (~isbf[fit_idx]).any() else created.max()
    keep = int(tc.get("ensemble", 5))
    pool: list[tuple[float, int, dict, Member, np.ndarray]] = []  # meilleurs essais (score, n, params, modele, pe)
    trials_log = []

    sampler = optuna.samplers.TPESampler(seed=int(time.time()) % 10_000, n_startup_trials=min(20, max(4, n_trials // 5)))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    best_file = models_dir / "search_best.json"
    if best_file.exists():  # on repart des meilleurs reglages trouves par la derniere recherche profonde
        for prm in json.loads(best_file.read_text())[: max(1, n_trials // 3)]:
            study.enqueue_trial(prm, skip_if_exists=True)

    def objective(trial):
        p = _suggest(trial, len(fit_idx), families, tc.get("search_weights", True), objectives,
                     {g: tuple(v) for g, v in tc.get("group_ranges", {}).items()})
        target, wt = _target(p["objective"], y, pnl, w, p["power"])
        if p["half_life_d"] > 0:  # donnees recentes plus importantes (les coins importes gardent leur poids)
            age_d = np.clip((t_end - created) / 86_400_000, 0, None)
            wt = np.where(isbf, wt, wt * 0.5 ** (age_d / p["half_life_d"]))
        try:
            m = _fit(p, X, target, wt, fit_idx, es_idx, feats, device, threads)
        except Exception as e:  # reglage invalide : essai ignore
            log.debug("essai %d en echec : %s", trial.number, e)
            return -1e9
        pe = m.predict(X.iloc[es_idx])
        thr, es_pnl, es_n = threshold_fn(pe, es_idx)
        score = es_pnl if es_n >= min_es else -1e3 + es_n
        fold_pnl = []
        if score > -1e3:
            for fi, ev, mn in folds:
                try:
                    mf = _fit(p, X, target, wt, fi, ev, feats, device, threads)
                except Exception:
                    return -1e9
                pf = mf.predict(X.iloc[ev])
                _, fp, fn = threshold_fn(pf, ev, mn)
                if fn < mn:
                    return -500 + fn  # pas assez d'achats sur cette periode
                fold_pnl.append(fp)
                trial.report(score + sum(fold_pnl), len(fold_pnl))
            # somme des periodes ; une periode perdante compte double
            score += sum(fp if fp >= 0 else 2 * fp for fp in fold_pnl)
        trials_log.append({"i": trial.number, "family": p["family"], "objective": p["objective"], "es_pnl": es_pnl,
                           "folds_pnl": fold_pnl, "score": score,
                           "es_trades": es_n, "threshold": thr, "best_iter": int(m.it or 0),
                           "params": {k: v for k, v in trial.params.items()}})
        pool.append((score, trial.number, p, m, pe))
        pool.sort(key=lambda r: -r[0])
        del pool[3 * keep:]
        return score

    timeout = float(tc.get("deep_hours", 6)) * 3600 if deep else None
    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)
    best_score, best_i, best_p, best_m, best_pe = pool[0]
    # ensemble : les meilleurs essais du MEME objectif que le meilleur (scores sur la meme echelle)
    same = [r for r in pool if r[2]["objective"] == best_p["objective"] and r[0] > -1e3][:keep]
    single_thr, single_pnl, single_n = threshold_fn(best_pe, es_idx)
    ens_pe = np.mean([r[4] for r in same], axis=0)
    ens_thr, ens_pnl, ens_n = threshold_fn(ens_pe, es_idx)
    use_ens = len(same) > 1 and ens_n >= min_es and ens_pnl >= single_pnl
    members = [r[3] for r in same] if use_ens else [best_m]
    thr = ens_thr if use_ens else single_thr
    if deep:
        top = sorted(study.trials, key=lambda t: -(t.value if t.value is not None else -1e18))[:10]
        best_file.write_text(json.dumps([t.params for t in top if t.value is not None and t.value > -1e3], indent=1))
    log.info("%d essais ; meilleur #%d (%s, %s) es %.2f SOL / %d achats ; ensemble de %d : %s", len(study.trials),
             best_i, best_p["family"], best_p["objective"], single_pnl, single_n, len(same),
             "retenu" if use_ens else "non retenu")
    return {"ensemble": Ensemble(members), "threshold": thr, "objective": best_p["objective"],
            "trials": sorted(trials_log, key=lambda r: -r["score"]), "best": best_i, "params": best_p,
            "n_trials": len(study.trials), "ensemble_size": len(members)}
