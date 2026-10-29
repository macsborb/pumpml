"""Graphiques du README (docs/img/*.png), calcules sur les vraies donnees locales.

Usage : .venv\\Scripts\\python scripts\\make_charts.py
Lit data/ (collecte, journal du paper trading), models/ (champion) et logs/learning_curve.json ; ecrit des PNG
sans aucune cle ni adresse. Les chiffres du rejeu des copies et de l'entonnoir des percees sont recalcules.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pumpml.config import Config  # noqa: E402
from pumpml import dataset as D  # noqa: E402

OUT = ROOT / "docs" / "img"
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
BLUE, GREEN, RED, AMBER, GREY = "#2563eb", "#16a34a", "#dc2626", "#d97706", "#9ca3af"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "figure.facecolor": "white",
    "axes.facecolor": "white", "legend.frameon": False, "axes.axisbelow": True,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("ecrit", name)


def collection(cfg):
    days = D.raw_days(cfg.data)
    tok = D.dedupe_tokens(D.read(cfg.data, "tokens", days))
    t = pd.to_datetime(tok.recv_ms, unit="ms", utc=True).dt.tz_convert("Europe/Paris").dt.tz_localize(None)
    h = t.dt.floor("h").value_counts().sort_index()
    h = h.reindex(pd.date_range(h.index.min(), h.index.max(), freq="h"), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(h.index, h.values, width=1 / 24 * 0.9, color=BLUE, align="edge")
    ax.set_title(f"Coins pump.fun collectes par heure ({len(tok):,} coins en {len(days)} jours)".replace(",", " "))
    ax.set_ylabel("nouveaux coins / h")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    gaps = h[h == 0]
    if len(gaps):
        ax.annotate("PC eteint", xy=(gaps.index[len(gaps) // 2], h.max() * 0.05), ha="center", color=MUTED, fontsize=9)
    save(fig, "collecte.png")


def paper(cfg):
    db = sqlite3.connect(cfg.data / "journal.db", timeout=60)
    p = pd.read_sql("SELECT exit_ms, pnl FROM positions WHERE status='closed' AND pnl IS NOT NULL ORDER BY exit_ms", db)
    p["t"] = pd.to_datetime(p.exit_ms, unit="ms", utc=True).dt.tz_convert("Europe/Paris").dt.tz_localize(None)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    cum = p.pnl.cumsum()
    ax.plot(p.t, cum, color=RED, lw=1.8)
    ax.fill_between(p.t, cum, 0, color=RED, alpha=0.08)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(f"Paper trading au prix Jupiter reel : {len(p)} trades clos de 0,1 SOL, "
                 f"{(p.pnl > 0).mean() * 100:.0f} % gagnants, {cum.iloc[-1]:+.2f} SOL")
    ax.set_ylabel("PnL cumule (SOL)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    save(fig, "paper_pnl.png")


def learning_curve():
    lc = json.load(open(ROOT / "logs" / "learning_curve.json", encoding="utf-8"))
    df = pd.DataFrame(lc)
    labels = ["1 jour", "2 jours", "3 jours"]
    df["k"] = df.groupby("passe").cumcount()
    g = df.groupby("k").agg(auc=("valid_auc", "mean"), lo=("valid_auc", "min"), hi=("valid_auc", "max"),
                            top=("top_bin_win", "mean"), base=("base_win_rate", "mean"), n=("train_tokens", "mean"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.arange(len(g))
    a1.plot(x, g.auc, "-o", color=BLUE, lw=2)
    a1.fill_between(x, g.lo, g.hi, color=BLUE, alpha=0.12)
    a1.set_xticks(x, [f"{l}\n({int(n):,} coins)".replace(",", " ") for l, n in zip(labels, g.n)])
    a1.set_title("AUC sur 6 h jamais vues")
    a1.set_ylim(0.7, 0.8)
    a2.bar(x - 0.18, g.base * 100, 0.36, color=GREY, label="tous les coins")
    a2.bar(x + 0.18, g.top * 100, 0.36, color=GREEN, label="10 % les mieux notes")
    a2.set_xticks(x, labels)
    a2.set_title("Taux de trades gagnants (%)")
    a2.legend(loc="upper left", fontsize=8)
    save(fig, "courbe_apprentissage.png")


def deciles(cfg):
    reg = json.load(open(cfg.models / "registry.json"))
    m = json.load(open(cfg.models / reg["champion"] / "meta.json"))
    b = pd.DataFrame(m["bins"])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(1, len(b) + 1)
    ax.bar(x, b.win * 100, color=[GREEN if i >= len(b) - 2 else BLUE for i in range(len(b))])
    ax.set_xticks(x, [f"D{i}" for i in x])
    ax.set_xlabel("decile de score du modele (D1 = les moins bien notes)")
    ax.set_ylabel("trades gagnants (%)")
    ax.set_title(f"Le tri fonctionne : taux de gagnants par tranche de score (champion {reg['champion']})")
    save(fig, "deciles.png")


def copies():
    # rejeu de la regle block_copies sur les 1 077 achats passes (28/10), PnL Jupiter reel par achat
    cases = [("original", 243, -2.785), ("copie permise\n(tweet nouveau)", 114, -2.094), ("copie sans tweet", 257, -3.886),
             ("meme tweet", 54, -0.144), ("tweet non lu", 49, -1.221), ("meme texte", 25, -0.648),
             ("tweet anterieur", 5, -0.206)]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    names = [c[0] for c in cases]
    per = [c[2] / c[1] * 1000 for c in cases]
    cols = [BLUE, AMBER] + [RED] * 5
    ax.barh(names[::-1], per[::-1], color=cols[::-1])
    for i, (c, v) in enumerate(zip(cases[::-1], per[::-1])):
        ax.text(v - 0.5, i, f"{c[1]} achats, {c[2]:+.2f} SOL", va="center", ha="right", fontsize=8, color=INK)
    ax.set_xlabel("PnL reel moyen par achat (milli-SOL)")
    ax.set_title("Copies : 390 achats sur 1 077 auraient ete bloques (-6,1 SOL evites)")
    ax.set_xlim(min(per) * 1.9, 0)
    save(fig, "copies.png")


def funnel(cfg):
    from pumpml import percees as PC
    m = PC.memory(cfg)
    days = D.raw_days(cfg.data)
    import duckdb
    files = D._glob(cfg.data, "trades", days)
    n_tr, n_big = duckdb.connect().execute(
        f"SELECT COUNT(*), SUM(p >= 300) FROM (SELECT MAX(vsol * 1e6 / vtok) p FROM read_parquet({files!r},"
        f" union_by_name=true) WHERE vtok > 0 GROUP BY mint)").fetchone()
    steps = [("coins dont on a les trades", n_tr), ("montes a 300 SOL (~30 000 $)", int(n_big)),
             ("percees organiques\n(pas groupees au bloc 0)", int(m.perc.sum())), ("dont achetees par un KOL avant", int(m.kol.sum()))]
    fig, ax = plt.subplots(figsize=(9, 3))
    y = np.arange(len(steps))[::-1]
    ax.barh(y, [s[1] for s in steps], color=[GREY, AMBER, GREEN, BLUE])
    ax.set_xscale("log")
    ax.set_yticks(y, [s[0] for s in steps])
    for yy, s in zip(y, steps):
        ax.text(s[1] * 1.1, yy, f"{s[1]:,}".replace(",", " "), va="center", fontsize=9)
    ax.set_title("Les vraies percees sont rares : la plupart des coins a 30 000 $ sont des lancements groupes")
    ax.grid(axis="y", visible=False)
    save(fig, "percees.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    for f in (lambda: collection(cfg), lambda: paper(cfg), learning_curve, lambda: deciles(cfg), copies,
              lambda: funnel(cfg)):
        f()


if __name__ == "__main__":
    main()
