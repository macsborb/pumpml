"""Annotations des coins importes cette nuit (narrative, viralite, categorie) -> /api/feedback.

Chaque entree : ticker (tel qu'importe) -> (verdict, categories, explication).
Les categories des 👍 deviennent des filtres de la bibliotheque de narratives.
"""
from __future__ import annotations

import json
import sqlite3
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NOTES = {
    # --- grands viraux (deja viraux AVANT le coin) ---
    "Pnut": (1, "animal,news du jour,tweet viral",
             "Ecureuil saisi et euthanasie par les autorites de New York (nov. 2024) : indignation nationale, "
             "reprise par Elon Musk et la presse. Histoire emotionnelle deja virale avant le coin."),
    "MOODENG": (1, "animal,culture internet",
                "Bebe hippopotame pygmee du zoo Khao Kheow, deja viral sur TikTok et Instagram (videos, fan art) "
                "avant le coin. Animal mignon + phenomene social existant."),
    "CHILLGUY": (1, "meme / humour,culture internet",
                 "'Just a chill guy' : meme du chien dessine de Phillip Banks, deja tendance massive sur TikTok "
                 "(#chillguy) avant le coin. Meme deja connu du grand public."),
    "GOAT": (1, "ia / tech,communaute / culte",
             "Premier meme cree par l'IA Truth Terminal, financee par Marc Andreessen : narrative 'agent IA' "
             "inedite + culte autour du bot sur X."),
    "Fartcoin": (1, "meme / humour,ia / tech,communaute / culte",
                 "Idee lancee par Truth Terminal (blague pipi-caca), humour absurde assume, communaute tres "
                 "soudee ('fart' culture). Meme simple, universel, porte par la narrative IA."),
    # --- coins recents qui ont perce (tradables sur la courbe) ---
    "swordcat": (1, "animal,meme / humour", "Chat avec une epee : meta 'wif' (dogwifhat) + chat, image forte et simple."),
    "DOPE": (1, "meme / humour,crypto meta", "Fusion de deux memes historiques (Doge + Pepe) : nostalgie crypto, lisible en une seconde."),
    "CATTLEAXE": (1, "animal,meme / humour",
                  "Chaton portant une hache plus lourde qu'un chevalier : image absurde et mignonne, personnage avec un compte X."),
    "SI276": (1, "ia / tech,celebrite / influenceur",
              "Projet lie au detenteur du record mondial de QI (YoungHoon Kim) : autorite d'une personnalite + theme intelligence."),
    "GTA": (1, "culture internet,news du jour", "Vague GTA (attente de GTA VI) : jeu tres populaire, nom provocateur et memorable."),
    "Chop": (1, "culture internet,animal", "Chop, le chien de GTA : personnage connu de millions de joueurs, surfe sur la vague GTA."),
    "███": (1, "meme / humour,culture internet", "Ticker censure ███ : curiosite et humour 'interdit', format inedit."),
    "NIO": (1, "meme / humour,culture internet", "Expression deja virale sur le 'normie Twitter' : phrase culte reprise en coin."),
    "PRESIDENT": (1, "politique", "Narrative presidentielle 2028 : sujet politique americain recurrent, nom simple."),
    # --- narratif seul (non achetables sur la courbe : migres des la creation) ---
    "CATE": (1, "animal,meme / humour", "Catecoin : chat, meme simple ; a tenu plus de 90 jours (communaute durable)."),
    "BTC": (1, "meme / humour", "Ticker 'BTC' detourne (Big Titty Coin) : blague provocante qui joue sur le Bitcoin."),
    "Musk": (1, "celebrite / influenceur", "Elon Musk : celebrite la plus liee aux memecoins, nom direct."),
    "REMUS": (1, "meme / humour,communaute / culte", "'Mascotte officielle du FOMO' avec date annoncee : mascotte + evenement."),
    "Dacity": (1, "meme / humour", "Coin a forte traction (plusieurs millions de market cap), lance hors courbe."),
    "COTE": (1, "meme / humour", "Coin a forte traction avec compte X dedie, lance hors courbe."),
    "FLOP": (1, "meme / humour", "Nom autoderisoire (Flop Labs) : humour assume."),
    # --- a eviter ---
    "Kimchi": (-1, "rug,communaute / culte",
               "Monte a ~15 M$ en moins d'une heure grace au repost du cofondateur de pump.fun, puis -98 % des que "
               "le repost a ete retire (23/10/2026). Pompe artificielle par influence, gains capte par les premiers wallets."),
    "UER": (-1, "arnaque / faux fonds", "Faux 'fonds' americain (energie) : createur qui achete toute la courbe a la creation, valorisation artificielle."),
    "UDTF": (-1, "arnaque / faux fonds", "Faux 'fonds de dividendes' : meme schema, nom financier serieux pour inspirer confiance."),
    "DOAS": (-1, "arnaque / faux fonds", "Faux actif petrolier : meme famille de lancements 'fonds' a valorisation artificielle."),
    "USOE": (-1, "arnaque / faux fonds", "Faux 'United States Oil Exchange' : meme famille."),
    "XTO": (-1, "arnaque / faux fonds", "Promet de 'payer en $USO' (petrole) : promesse de rendement, meme famille de faux fonds."),
    "USDF": (-1, "arnaque / faux fonds", "Faux 'United States Dividend Fund' : meme famille (relances sous plusieurs adresses)."),
    # --- cycle de 2 h 53 ---
    "TIT": (1, "politique", "'Trust in Trump' : slogan patriotique pro-Trump, compte X dedie ; narrative politique simple."),
    "Bow": (1, "celebrite / influenceur,communaute / culte",
            "'Save Bow' : les frais du coin sont reverses a un createur X (@ayakatz) via UsePaid -> communaute qui soutient une personnalite."),
    # --- cycle de 6 h 53 ---
    "T1K0": (1, "ia / tech,communaute / culte",
             "'Experience de survie IA 24/7' : un robot echoue, suivi en direct -> narrative agent IA + feuilleton a suivre."),
    "GTA 6": (1, "culture internet,news du jour", "Vague GTA (attente de GTA VI) : jeu navigateur 'GTA Online 6', compte X dedie."),
    "GIGATITS": (1, "meme / humour,culture internet", "Suite de la vague 'Giga Titty Ass' (GTA) : meme provocateur decline."),
    "NVIDIA": (-1, "arnaque / usurpation",
               "Reprend la description et le compte X OFFICIELS de NVIDIA : usurpation d'une grande marque, aucun lien reel."),
    "PMUP": (1, "ia / tech,celebrite / influenceur",
             "Lance sur un tweet de Shaw (@shawmakesmagic, createur d'ElizaOS / ai16z) : developpeur IA tres suivi."),
    "TFOMO": (1, "meme / humour,communaute / culte", "'FOMO Teacher' : personnage qui joue sur la peur de rater, compte X dedie."),
}
# Annotations par adresse (prioritaires) : quand plusieurs coins partagent un ticker.
MINT_NOTES = {
    "7JEfFdwmXf": (1, "politique,culture internet",
                   "'Grand Trump Agency' : Trump (personnalite demesuree) x univers GTA, deux vagues du moment combinees."),
}


def main():
    db = sqlite3.connect(ROOT / "data" / "journal.db")
    rows = db.execute("SELECT mint, TRIM(symbol) FROM examples WHERE status IN ('ok','narratif')").fetchall()
    by_sym = {}
    for mint, sym in rows:
        by_sym.setdefault(sym, []).append(mint)
    done = 0
    for sym, (verdict, tags, comment) in NOTES.items():
        for mint in by_sym.get(sym, []):
            req = urllib.request.Request("http://127.0.0.1:8787/api/feedback", data=json.dumps(
                {"mint": mint, "verdict": verdict, "tags": tags, "comment": comment}).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=20).read()
            done += 1
    for prefix, (verdict, tags, comment) in MINT_NOTES.items():
        for mint, _ in rows:
            if mint.startswith(prefix):
                req = urllib.request.Request("http://127.0.0.1:8787/api/feedback", data=json.dumps(
                    {"mint": mint, "verdict": verdict, "tags": tags, "comment": comment}).encode(),
                    headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=20).read()
                done += 1
    missing = [s for s in NOTES if s not in by_sym]
    print(f"{done} exemples annotes ; tickers absents : {missing}")


if __name__ == "__main__":
    main()
