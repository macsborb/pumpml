"""Stockage.

* Archive brute en Parquet, append-only, partitionnee par jour :
  ``data/raw/<table>/date=YYYY-MM-DD/*.parquet``. Les fichiers sont ecrits en .tmp puis
  renommes : un crash ne laisse jamais de fichier a moitie ecrit.
  Ces donnees ne sont jamais modifiees, toutes les features peuvent etre recalculees.
* Hot store SQLite (WAL) : les dernieres heures, lu en direct par le paper trader.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

SCHEMAS: dict[str, pa.Schema] = {
    "tokens": pa.schema([
        ("recv_ms", pa.int64()), ("source", pa.string()), ("mint", pa.string()), ("name", pa.string()),
        ("symbol", pa.string()), ("uri", pa.string()), ("creator", pa.string()), ("bonding_curve", pa.string()),
        ("initial_buy", pa.float64()), ("sol_amount", pa.float64()), ("vsol", pa.float64()), ("vtok", pa.float64()),
        ("mcap_sol", pa.float64()), ("signature", pa.string()), ("raw", pa.string()),
        ("chain_ms", pa.int64()),  # horodatage on-chain de la creation
        ("sampled", pa.bool_()),   # trades suivis integralement (mode coin par coin)
    ]),
    "trades": pa.schema([
        ("recv_ms", pa.int64()), ("slot", pa.int64()), ("signature", pa.string()), ("mint", pa.string()),
        ("is_buy", pa.bool_()), ("sol_amount", pa.int64()), ("token_amount", pa.int64()), ("user", pa.string()),
        ("ts", pa.int64()), ("vsol", pa.int64()), ("vtok", pa.int64()), ("real_sol", pa.int64()),
        ("real_tok", pa.int64()),
    ]),
    "migrations": pa.schema([
        ("recv_ms", pa.int64()), ("source", pa.string()), ("mint", pa.string()), ("signature", pa.string()),
        ("raw", pa.string()), ("chain_ms", pa.int64()),
    ]),
    "metadata": pa.schema([
        ("recv_ms", pa.int64()), ("mint", pa.string()), ("uri", pa.string()), ("ok", pa.bool_()),
        ("description", pa.string()), ("twitter", pa.string()), ("telegram", pa.string()),
        ("website", pa.string()), ("image_uri", pa.string()), ("image_path", pa.string()), ("raw", pa.string()),
    ]),
    # apres migration : echanges sur PumpSwap (reserves du pool AVANT le trade)
    "swaps": pa.schema([
        ("recv_ms", pa.int64()), ("slot", pa.int64()), ("signature", pa.string()), ("mint", pa.string()),
        ("pool", pa.string()), ("is_buy", pa.bool_()), ("sol_amount", pa.int64()), ("sol_swap", pa.int64()),
        ("token_amount", pa.int64()), ("user", pa.string()), ("ts", pa.int64()), ("pool_sol", pa.int64()),
        ("pool_tok", pa.int64()),
    ]),
    # trades des KOL (PumpPortal subscribeAccountTrade), tous les coins pump.fun ; montants en SOL / tokens
    "kol": pa.schema([
        ("recv_ms", pa.int64()), ("signature", pa.string()), ("mint", pa.string()), ("wallet", pa.string()),
        ("is_buy", pa.bool_()), ("sol", pa.float64()), ("tokens", pa.float64()), ("mcap_sol", pa.float64()),
        ("source", pa.string()),
    ]),
    # une ligne par migration : pool, historique de la courbe, achat groupe au bloc 0, concentration
    "pools": pa.schema([
        ("recv_ms", pa.int64()), ("mint", pa.string()), ("pool", pa.string()), ("signature", pa.string()),
        ("migrated_ms", pa.int64()), ("migrate_slot", pa.int64()), ("vault", pa.string()),
        ("create_slot", pa.int64()), ("curve_s", pa.float64()), ("curve_tx", pa.int64()),
        ("block0_share", pa.float64()), ("block0_wallets", pa.int64()), ("block0_sol", pa.float64()),
        ("block0_tx", pa.int64()), ("top1_share", pa.float64()), ("top10_share", pa.float64()),
        ("reasons", pa.string()), ("supply", pa.int64()),
    ]),
    "outages": pa.schema([
        ("recv_ms", pa.int64()), ("start_ms", pa.int64()), ("end_ms", pa.int64()), ("stream", pa.string()),
    ]),
    "social": pa.schema([
        ("recv_ms", pa.int64()), ("mint", pa.string()), ("url", pa.string()), ("kind", pa.string()),
        ("handle", pa.string()), ("tweet_id", pa.string()), ("provider", pa.string()), ("ok", pa.bool_()),
        ("followers", pa.int64()), ("following", pa.int64()), ("verified", pa.bool_()),
        ("verified_type", pa.string()), ("account_created_ms", pa.int64()), ("statuses", pa.int64()),
        ("bio", pa.string()), ("tweet_text", pa.string()), ("tweet_created_ms", pa.int64()),
        ("likes", pa.int64()), ("retweets", pa.int64()), ("replies", pa.int64()), ("quotes", pa.int64()),
        ("views", pa.int64()), ("is_reply", pa.bool_()), ("mentions", pa.string()), ("raw", pa.string()),
        # reponse / citation : tweet d'origine
        ("rel", pa.string()), ("parent_id", pa.string()), ("parent_handle", pa.string()),
        ("parent_followers", pa.int64()), ("parent_verified", pa.bool_()), ("parent_text", pa.string()),
        ("parent_created_ms", pa.int64()), ("parent_likes", pa.int64()), ("parent_views", pa.int64()),
    ]),
}


def now_ms() -> int:
    return int(time.time() * 1000)


def day_of(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


class ParquetSink:
    """Tampon thread-safe vide sur disque par taille ou par temps."""

    def __init__(self, root: Path, table: str, flush_rows: int, flush_seconds: float):
        self.dir = root / "raw" / table
        self.table, self.schema = table, SCHEMAS[table]
        self.flush_rows, self.flush_seconds = flush_rows, flush_seconds
        self.buf: list[dict] = []
        self.lock = threading.Lock()
        self.last_flush = time.monotonic()
        self.written = 0

    def add(self, row: dict) -> None:
        with self.lock:
            self.buf.append(row)

    def due(self) -> bool:
        return len(self.buf) >= self.flush_rows or (self.buf and time.monotonic() - self.last_flush >= self.flush_seconds)

    def flush(self) -> int:
        with self.lock:
            rows, self.buf = self.buf, []
            self.last_flush = time.monotonic()
        if not rows:
            return 0
        by_day: dict[str, list[dict]] = {}
        for r in rows:
            by_day.setdefault(day_of(r["recv_ms"]), []).append(r)
        for day, part in by_day.items():
            d = self.dir / f"date={day}"
            d.mkdir(parents=True, exist_ok=True)
            cols = {f.name: [r.get(f.name) for r in part] for f in self.schema}
            tbl = pa.table(cols, schema=self.schema)
            name = f"{now_ms()}-{uuid.uuid4().hex[:8]}.parquet"
            tmp = d / (name + ".tmp")
            pq.write_table(tbl, tmp, compression="zstd")
            os.replace(tmp, d / name)
        self.written += len(rows)
        return len(rows)


LONG_OUTAGE_MS = 180_000  # au-dela, on risque d'avoir rate un vrai pic pendant la detention


def overlaps(outages: list[tuple[int, int]], a: int, b: int, min_len: int = 0) -> bool:
    """Une coupure de collecte (d'au moins ``min_len`` ms) chevauche-t-elle [a, b] ?"""
    return any(e - s >= min_len and s <= b and e >= a for s, e in outages)


def collection_broken(outages: list[tuple[int, int]], created: int, dec: int, horizon: int) -> bool:
    """Donnees d'un coin inutilisables ?
    * toute coupure pendant les ``dec`` premieres ms : les indicateurs de decision seraient faux ;
    * apres, seulement une coupure longue : chaque trade porte l'etat complet des reserves, donc le prix
      reste exact apres une courte coupure (on peut juste rater un pic tres bref)."""
    return overlaps(outages, created, created + dec) or overlaps(outages, created, created + horizon, LONG_OUTAGE_MS)


def compact_day(root: Path, table: str, day: str) -> None:
    """Fusionne les petits fichiers d'un jour termine en un seul (operation idempotente)."""
    d = root / "raw" / table / f"date={day}"
    files = sorted(p for p in d.glob("*.parquet") if not p.name.startswith("compact-"))
    if len(files) <= 1:
        return
    olds = sorted(d.glob("compact-*.parquet"))
    tbl = pa.concat_tables([pq.read_table(p, schema=SCHEMAS[table]) for p in olds + files])
    tmp = d / "compact.tmp"
    pq.write_table(tbl, tmp, compression="zstd")
    final = d / f"compact-{now_ms()}.parquet"
    os.replace(tmp, final)
    for p in olds + files:
        p.unlink()
    log.info("compact %s %s: %d fichiers -> 1 (%d lignes)", table, day, len(olds) + len(files), tbl.num_rows)


HOT_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS tokens(
  mint TEXT PRIMARY KEY, recv_ms INTEGER, name TEXT, symbol TEXT, uri TEXT, creator TEXT,
  initial_buy REAL, sol_amount REAL, mcap_sol REAL, chain_ms INTEGER, sampled INTEGER);
CREATE TABLE IF NOT EXISTS trades(
  mint TEXT, recv_ms INTEGER, slot INTEGER, is_buy INTEGER, sol_amount INTEGER, token_amount INTEGER,
  user TEXT, vsol INTEGER, vtok INTEGER, real_tok INTEGER, signature TEXT, real_sol INTEGER, ts INTEGER);
CREATE INDEX IF NOT EXISTS trades_mint ON trades(mint, recv_ms);
CREATE INDEX IF NOT EXISTS trades_time ON trades(recv_ms);
CREATE TABLE IF NOT EXISTS migrations(mint TEXT PRIMARY KEY, recv_ms INTEGER);
CREATE TABLE IF NOT EXISTS metadata(mint TEXT PRIMARY KEY, recv_ms INTEGER, description TEXT,
  twitter TEXT, telegram TEXT, website TEXT, image_path TEXT);
CREATE TABLE IF NOT EXISTS social(mint TEXT PRIMARY KEY, recv_ms INTEGER, json TEXT);
CREATE TABLE IF NOT EXISTS heartbeat(name TEXT PRIMARY KEY, ms INTEGER, info TEXT);
CREATE TABLE IF NOT EXISTS outages(start_ms INTEGER, end_ms INTEGER, recv_ms INTEGER);
CREATE TABLE IF NOT EXISTS swaps(
  mint TEXT, recv_ms INTEGER, slot INTEGER, is_buy INTEGER, sol_amount INTEGER, sol_swap INTEGER,
  token_amount INTEGER, user TEXT, ts INTEGER, pool_sol INTEGER, pool_tok INTEGER, signature TEXT);
CREATE INDEX IF NOT EXISTS swaps_mint ON swaps(mint, ts);
CREATE TABLE IF NOT EXISTS kol_trades(signature TEXT PRIMARY KEY, mint TEXT, recv_ms INTEGER, wallet TEXT,
  is_buy INTEGER, sol REAL, tokens REAL, mcap_sol REAL);
CREATE INDEX IF NOT EXISTS kol_mint ON kol_trades(mint, recv_ms);
CREATE TABLE IF NOT EXISTS pools(
  mint TEXT PRIMARY KEY, recv_ms INTEGER, pool TEXT, signature TEXT, migrated_ms INTEGER, migrate_slot INTEGER,
  create_slot INTEGER, curve_s REAL, curve_tx INTEGER, block0_share REAL, block0_wallets INTEGER,
  block0_sol REAL, block0_tx INTEGER, top1_share REAL, top10_share REAL, reasons TEXT, supply INTEGER);
"""


class HotStore:
    """SQLite WAL : un ecrivain (le collecteur), plusieurs lecteurs."""

    def __init__(self, path: Path, readonly: bool = False):
        self.path = path
        uri = f"file:{path.as_posix()}?mode=ro" if readonly else f"file:{path.as_posix()}"
        self.db = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        if not readonly:
            self.db.executescript(HOT_SCHEMA)
            for m in ("ALTER TABLE tokens ADD COLUMN chain_ms INTEGER", "ALTER TABLE trades ADD COLUMN ts INTEGER",
                      "ALTER TABLE tokens ADD COLUMN sampled INTEGER"):
                try:
                    with self.db:
                        self.db.execute(m)
                except sqlite3.OperationalError:
                    pass  # colonne deja la

    def write(self, sql: str, rows: list[tuple]) -> None:
        if not rows:
            return
        with self.lock, self.db:
            self.db.executemany(sql, rows)

    def query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def heartbeat(self, name: str, info: str = "") -> None:
        self.write("INSERT OR REPLACE INTO heartbeat VALUES(?,?,?)", [(name, now_ms(), info)])

    def prune(self, keep_hours: float) -> None:
        cut = now_ms() - int(keep_hours * 3600_000)
        with self.lock, self.db:
            for t in ("trades", "tokens", "migrations", "metadata", "social", "outages", "swaps", "pools", "kol_trades"):
                self.db.execute(f"DELETE FROM {t} WHERE recv_ms < ?", (cut,))
        with self.lock:
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
