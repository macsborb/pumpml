"""python -m pumpml <commande>"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta

from . import logutil
from .config import Config

CMDS = {
    "supervise": "tout lancer et surveiller 24h/24 (collecte, paper, dashboard, entrainement quotidien)",
    "collect": "collecteur seul",
    "paper": "paper trader seul",
    "dashboard": "dashboard seul (http://127.0.0.1:8787)",
    "learn": "apprentissage : features a jour + entrainement (lance toutes les 4 h par supervise)",
    "daily": "comme learn + compaction de l'archive (lance chaque nuit)",
    "train": "entrainement seul",
    "deepsearch": "recherche de reglages en profondeur (des centaines d'essais, carte graphique)",
    "features": "recalculer les features (--force pour tout refaire)",
    "status": "etat des donnees et du modele",
    "check": "verifier les cles et la connexion aux flux",
}


_LOCK = None


def single_instance(cfg: Config, name: str) -> bool:
    """Verrou de fichier (libere automatiquement si le processus meurt)."""
    global _LOCK
    f = open(cfg.data / f"{name}.lock", "a+")
    try:
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return False
    _LOCK = f
    return True


def main():
    ap = argparse.ArgumentParser(prog="pumpml", description="IA pump.fun",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(f"  {k:<10} {v}" for k, v in CMDS.items()))
    ap.add_argument("cmd", choices=list(CMDS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = Config()
    if a.cmd in ("collect", "paper", "dashboard", "daily", "learn", "deepsearch") and (cfg.data / "PAUSE").exists():
        sys.exit(0)  # en pause : scripts/resume.ps1 pour reprendre
    # la recherche profonde prend le verrou de l'apprentissage horaire (un seul entrainement a la fois)
    lock = "learn" if a.cmd in ("deepsearch", "daily") else a.cmd
    if a.cmd in ("supervise", "collect", "paper", "dashboard", "daily", "learn", "deepsearch") and             not single_instance(cfg, lock):
        print(f"{a.cmd} tourne deja (verrou data/{a.cmd}.lock) : rien a faire.")
        sys.exit(3)
    logutil.setup(a.cmd, cfg.logs)

    if a.cmd == "supervise":
        from .supervisor import main as m
        m(cfg)
    elif a.cmd == "collect":
        from .collector import main as m
        m(cfg)
    elif a.cmd == "paper":
        from .paper import main as m
        m(cfg)
    elif a.cmd == "dashboard":
        from .dashboard import main as m
        m(cfg)
    elif a.cmd in ("daily", "features", "learn", "deepsearch"):
        from . import dataset as D
        from .storage import compact_day
        days = D.complete_days(cfg)
        for d in days:
            D.build_day(cfg, d, force=a.force)
        if a.cmd == "daily":
            old = [d for d in D.raw_days(cfg.data) if d < (date.today() - timedelta(days=1)).isoformat()]
            for d in old:
                for t in ("tokens", "trades", "migrations", "metadata", "social", "outages", "swaps", "pools", "kol"):
                    compact_day(cfg.data, t, d)
        if a.cmd in ("daily", "learn", "deepsearch"):
            import logging
            from .model import train
            r = train(cfg, deep=a.cmd in ("daily", "deepsearch"))  # chaque nuit : recherche profonde
            logging.getLogger("train").info("resultat : %s", json.dumps(r, default=str)[:600])
            print(json.dumps(r, indent=2, default=str))
    elif a.cmd == "train":
        from .model import train
        print(json.dumps(train(cfg), indent=2, default=str))
    elif a.cmd == "status":
        from . import dataset as D
        from .model import Registry
        print("jours bruts     :", D.raw_days(cfg.data))
        print("jours complets  :", D.complete_days(cfg))
        print("champion        :", Registry(cfg.models).state.get("champion"))
    elif a.cmd == "check":
        from .check import main as m
        sys.exit(m(cfg))


if __name__ == "__main__":
    main()
