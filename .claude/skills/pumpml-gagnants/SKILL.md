---
name: pumpml-gagnants
description: Methode pour analyser les gros gagnants pump.fun legitimes de la periode (migration naturelle, plus haut >= 100 000 $), rechercher leur narrative et l'enseigner au modele PumpML en les annotant. A utiliser pour la tache planifiee toutes les 4 h et chaque fois que l'utilisateur demande d'analyser des coins qui ont marche.
---

# Analyse des gros gagnants

But : que le modele comprenne POURQUOI une narrative a marche, pour reconnaitre la prochaine. Chaque annotation
devient un exemple de la bibliotheque de narratives (ressemblance texte / image, categories).

## 1. Le rapport
`.venv\Scripts\python scripts\winners_report.py --hours 6` (5 a 15 min : il met d'abord a jour la memoire avec
`scripts\import_fame.py`). Il ecrit `rapports\gagnants\<date>.md` et `.json` : migrations de la periode,
montees naturelles, lancements groupes, et les gros gagnants legitimes pas encore annotes.
Legitime = courbe remplie en plus d'1 min ET pas achetee au bloc 0 par un groupe de wallets ET au moins
100 transactions sur la courbe ; plus haut >= 100 000 $ sur bougies de 15 min ET encore >= 30 % du plus haut
1 h apres (regle de tenue, `fame.label`). Les vidages ecartes sont listes en tete du rapport. Une vraie montee
prend en general quelques minutes, avec beaucoup d'acheteurs differents ; une seule grosse bougie ou tout est
achete dans les memes secondes n'est pas legitime.

## 2. Recherche (par coin, les 15 plus gros au maximum)
- DexScreener (lien du rapport) : graphique, liens, age, volume.
- Liens du coin : site (ouvre-le : vrai projet, page vide, copie ?), compte X (recherche web si x.com ne
  s'ouvre pas), Telegram.
- Recherche web sur le nom, le ticker, le sujet et la date : d'ou vient l'idee ? actu du jour, tweet viral
  (de qui ?), video TikTok / Instagram, celebrite, animal, IA, jeu video, culture internet, reprise d'un ancien
  memecoin celebre avec un twist.
- Timing : l'evenement date de combien de temps avant le lancement ? (les narratives d'actu valent surtout
  dans les 2 a 6 premieres heures).
- Verification : faux fonds, usurpation de marque ou de personne, pompe d'influenceur payee puis lachee,
  communaute fabriquee, reponse a un tweet viral qui ajoute une info fausse = a eviter, meme si le prix a monte.
- Lien a ce qui a deja marche : copie, clin d'oeil drole que toute la communaute comprend, ou idee neuve.
- Ne presente jamais une supposition comme un fait. Si tu n'as pas pu verifier, ecris-le.

## 3. Annotation
`.venv\Scripts\python scripts\annotate_winner.py <mint> <1|-1> "<categories>" "<explication>"`
- `1` : narrative a reproduire. `-1` : a eviter (montee artificielle, arnaque, usurpation, buzz mensonger).
- Categories (virgules) : animal, meme / humour, culture internet, news du jour, tweet viral, ia / tech,
  celebrite / influenceur, politique, communaute / culte, crypto meta, sport, jeu video, clin d oeil,
  arnaque / faux fonds, arnaque / usurpation, rug. Une categorie nouvelle et precise est acceptee.
- Explication (2 a 4 phrases, faits verifies) : source de la narrative ; pourquoi elle a accroche (emotion,
  humour, timing, audience existante) ; lien nom / image / tweet / site ; ce que l'IA doit retenir.
- L'annotation est enregistree comme ecrite apres le resultat (pas de fuite du futur) et gardee dans
  `rapports\gagnants\annotations.jsonl`.

## 4. Synthese
`rapports\gagnants\<meme date>-analyse.md` : chiffres de la periode ; pour chaque coin verdict, categories,
explication et sources (URLs) ; 3 a 5 lignes de tendances (themes qui reviennent, clins d'oeil, ce qui separe
vrais et faux gagnants). Rapport vide = synthese avec les chiffres seulement.

## 5. Erreurs du modele et percees ratees (a chaque passage)
`.venv\Scripts\python scripts\errors_report.py --hours 24` ecrit `rapports\erreurs\<date>.md` : ses pires achats
en papier (PnL Jupiter reel, score, seuil, 5 raisons principales du modele) et les coins qui ont PERCE (~30 000 $
ou migres, hors lancements groupes) qu'il n'a pas achetes, avec ce qu'il en pensait et si un KOL avait achete.
- Pour chaque pire achat (5 au plus) : lis le nom, le texte, le tweet, l'image, le site. Pourquoi le modele y a cru
  (ses raisons) et pourquoi c'etait mauvais (faux projet, narrative vide, copie, compte X inconnu...) ? Si la
  lecon est claire : `annotate_winner.py <mint> -1 "<categories>" "<pourquoi il s'est trompe>"`.
- Pour chaque percee ratee (5 au plus) : meme recherche qu'en section 2. Pourquoi elle a marche, et pourquoi pas
  d'autres de la meme veine ? Les KOL l'avaient-ils achetee avant ? Si la narrative est verifiee :
  `annotate_winner.py <mint> 1 "<categories>" "<pourquoi elle a perce>"`.
- Cherche les erreurs qui se repetent (une raison du modele qui pousse toujours a tort, un type de coin achete en
  serie) et ecris-les en 3 a 5 lignes dans `rapports\erreurs\<meme date>-lecons.md` ; ajoute les pistes de fond
  (signal mal appris, regle a ajouter) dans la section **Idees** de `rapports\experiences\JOURNAL.md`.
- N'annote que ce que tu as compris et verifie : une mauvaise annotation apprend le faux au modele.

## Limites
Ne modifie pas le code et ne relance pas les services pendant cette tache (si un script plante : correction
simple et sure seulement, sinon note-le). Pas d'achat, pas de publication, aucune cle affichee.
