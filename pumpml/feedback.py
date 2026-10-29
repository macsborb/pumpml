"""Avis de l'utilisateur sur les choix de l'IA, et journal des decisions du paper trader."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pandas as pd

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS feedback(
  mint TEXT PRIMARY KEY, ts INTEGER, verdict INTEGER, tags TEXT, comment TEXT,
  resolved_at_rating INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS decisions(
  mint TEXT PRIMARY KEY, decided_ms INTEGER, created_ms INTEGER, model_id TEXT, score REAL, taste REAL,
  final REAL, threshold REAL, action TEXT, name TEXT, symbol TEXT, image_path TEXT, twitter TEXT,
  description TEXT, mcap_sol REAL, features TEXT, top_reasons TEXT, text TEXT, creator TEXT,
  pnl_sim REAL, max_mult_after REAL, exit_reason_sim TEXT, resolved INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS positions(
  mint TEXT PRIMARY KEY, entry_ms INTEGER, entry_vsol REAL, entry_vtok REAL, tokens REAL, cost REAL,
  status TEXT, exit_ms INTEGER, exit_reason TEXT, pnl REAL, value REAL, max_value REAL, last_price REAL);
CREATE INDEX IF NOT EXISTS dec_time ON decisions(decided_ms);
CREATE TABLE IF NOT EXISTS examples(
  mint TEXT PRIMARY KEY, added_ms INTEGER, status TEXT, msg TEXT, name TEXT, symbol TEXT, image_path TEXT,
  created_ms INTEGER, pnl_sim REAL, max_mult_after REAL, entry_mult REAL, n_trades INTEGER, migrated REAL,
  twitter TEXT, description TEXT);
CREATE TABLE IF NOT EXISTS amm_decisions(
  mint TEXT PRIMARY KEY, pool TEXT, migrated_ms INTEGER, created_ms INTEGER, name TEXT, symbol TEXT,
  reasons TEXT, curve_s REAL, block0_share REAL, block0_wallets INTEGER, top10_share REAL, decided_ms INTEGER,
  jup_tokens REAL, jup_route TEXT, jup_ms INTEGER, sim_tokens REAL, n_swaps INTEGER, pnl_sim REAL, max_mult REAL,
  exit_reason_sim TEXT, hold_s REAL, resolved INTEGER DEFAULT 0);
"""
# Colonnes ajoutees apres coup (bases deja existantes).
MIGRATIONS = ["ALTER TABLE decisions ADD COLUMN vamp TEXT", "ALTER TABLE decisions ADD COLUMN entry_mult_sim REAL"] + [
    f"ALTER TABLE positions ADD COLUMN {c}" for c in (
        "entry_wall_ms INTEGER", "src TEXT", "last_quote_ms INTEGER", "quotes_n INTEGER", "impact REAL",
        "route TEXT", "exit_route TEXT", "curve_value REAL")] + [
    f"ALTER TABLE decisions ADD COLUMN {c}" for c in ("age_s REAL", "first_age_s REAL", "scores TEXT")]


class Journal:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.db:
            self.db.executescript(SCHEMA)
        for m in MIGRATIONS:
            try:
                with self.db:
                    self.db.execute(m)
            except sqlite3.OperationalError:
                pass  # deja faite

    def exec(self, sql: str, args=()):
        with self.lock, self.db:
            return self.db.execute(sql, args)

    def query(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def set_feedback(self, mint: str, ts: int, verdict: int, tags: str = "", comment: str = "",
                     after_outcome: bool = False):
        """``resolved_at_rating`` : l'issue du coin etait-elle deja connue quand tu l'as note ?
        Seuls les avis donnes AVANT servent au modele "gout perso" (sinon il apprendrait juste le prix).
        ``after_outcome`` : annotation ecrite en connaissant le resultat (analyse des gros gagnants)."""
        r = self.query("SELECT resolved FROM decisions WHERE mint=?", (mint,))
        resolved = int(bool(r and r[0]["resolved"]))
        if self.query("SELECT 1 FROM examples WHERE mint=?", (mint,)):
            resolved = 1  # coin importe : son resultat est deja connu
        old = self.query("SELECT verdict, resolved_at_rating FROM feedback WHERE mint=?", (mint,))
        if old and old[0]["resolved_at_rating"] == 0 and old[0]["verdict"] == verdict:
            resolved = 0  # meme verdict, juste un commentaire ajoute apres coup : il reste "a l'aveugle"
        if after_outcome:
            resolved = 1
        self.exec("INSERT OR REPLACE INTO feedback VALUES(?,?,?,?,?,?)", (mint, ts, verdict, tags, comment, resolved))

    def feedback_df(self) -> pd.DataFrame:
        with self.lock:
            return pd.read_sql_query("SELECT * FROM feedback", self.db)
