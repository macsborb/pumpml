"""Superviseur : lance et relance collecteur, paper trader et dashboard ; apprentissage regulier
(toutes les ``retrain_every_hours``) + passe quotidienne (compaction). Pause : fichier data/PAUSE.

Chaque service est un processus separe : un plantage (ou l'entrainement GPU) ne bloque
jamais la collecte. Relance avec backoff exponentiel, remis a zero apres 5 min de bon
fonctionnement.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

from .config import ROOT, Config

log = logging.getLogger("supervisor")
SERVICES = ["collect", "paper", "dashboard"]
DAILY_AT = (4, 30)  # heure locale


class Proc:
    def __init__(self, name: str):
        self.name, self.p, self.started, self.backoff, self.next_start = name, None, 0.0, 2.0, 0.0

    def start(self):
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        self.p = subprocess.Popen([sys.executable, "-m", "pumpml", self.name], cwd=ROOT,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        self.started = time.monotonic()
        log.info("demarre %s (pid %d)", self.name, self.p.pid)

    def check(self):
        now = time.monotonic()
        if self.p is None:
            if now >= self.next_start:
                self.start()
            return
        code = self.p.poll()
        if code is None:
            if now - self.started > 300:
                self.backoff = 2.0
            return
        log.error("%s s'est arrete (code %s), relance dans %.0fs", self.name, code, self.backoff)
        self.p = None
        self.next_start = now + self.backoff
        self.backoff = min(self.backoff * 2, 300)

    def stop(self):
        if self.p and self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(15)
            except subprocess.TimeoutExpired:
                self.p.kill()


def main(cfg: Config):
    pause = cfg.data / "PAUSE"
    if pause.exists():
        log.info("en pause (fichier data/PAUSE present) : rien ne demarre")
        return
    # Fichiers temporaires des services sur le disque des donnees : un C: plein (vu le 26/10) ne doit
    # pas faire planter la collecte ni l'entrainement.
    tmp = cfg.data / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(tmp), TMP=str(tmp), TMPDIR=str(tmp))
    procs = [Proc(n) for n in SERVICES]
    learn: subprocess.Popen | None = None
    every = cfg["training"]["retrain_every_hours"] * 3600
    last_learn = time.monotonic() - every + 600  # premier apprentissage 10 min apres le demarrage
    # date de la derniere passe quotidienne (fichier) : un redemarrage ne la relance pas
    done_file = cfg.data / "daily.done"
    last_daily = done_file.read_text().strip() if done_file.exists() else None
    last_disk = 0.0
    try:
        while True:
            if pause.exists():
                log.info("pause demandee : arret propre des services")
                break
            for p in procs:
                p.check()
            now = datetime.now()
            daily_due = (now.hour, now.minute) >= DAILY_AT and last_daily != now.date().isoformat()
            if learn is None and (daily_due or time.monotonic() - last_learn >= every):
                cmd = "daily" if daily_due else "learn"
                log.info("lancement de l'apprentissage (%s)", cmd)
                learn = subprocess.Popen([sys.executable, "-m", "pumpml", cmd], cwd=ROOT,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                last_learn = time.monotonic()
                if daily_due:
                    last_daily = now.date().isoformat()
                    done_file.write_text(last_daily)
            if learn is not None and learn.poll() is not None:
                log.info("apprentissage termine (code %s)", learn.returncode)
                learn = None
            if time.monotonic() - last_disk > 600:
                free = shutil.disk_usage(cfg.data).free / 1e9
                (log.warning if free < 20 else log.info)("espace disque libre : %.0f Go", free)
                last_disk = time.monotonic()
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        # Les services surveillent aussi data/PAUSE et s'arretent seuls (le collecteur vide ses
        # tampons) ; on leur laisse 20 s avant de forcer.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and any(p.p and p.p.poll() is None for p in procs):
            time.sleep(0.5)
        for p in procs:
            p.stop()
        if learn and learn.poll() is None:
            learn.terminate()
        log.info("superviseur arrete")
