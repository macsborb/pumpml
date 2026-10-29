---
name: pumpml-experience
description: Protocole pour toute experience sur le modele PumpML (nouveau signal, reglage, objectif, filtre, donnees externes) — cadrer, tester sans fuite du futur, juger en SOL sur des heures jamais vues, consigner dans le journal. A utiliser avant de modifier model.py, search.py, features.py, fame.py, protos.py ou la config [training] / [strategy], et chaque fois qu'on compare des modeles ou qu'on annonce un resultat.
---

# Experience sur le modele PumpML

Methode inspiree des Probabl Skills (frame / build / evaluate / review / backlog), adaptee a notre pipeline
sur mesure : flux temps reel, regle d'achat « premier point d'analyse au-dessus du seuil », jugement en SOL.
Le journal est `rapports/experiences/JOURNAL.md`.

## 1. Cadrer avant de coder (decisions verrouillees)
Les decisions de la section « Decisions verrouillees » du journal s'appliquent a toute experience. Ne les change
que si l'utilisateur le demande, et note le changement dans le journal :
- **Ce qu'on predit** : resultat d'un achat de 0,1 SOL a un point d'analyse, avec la strategie de sortie de
  `[strategy]` (TP, SL, sortie a 2 min, frais 1,25 %/sens + 0,0005 SOL/tx).
- **Signaux autorises** : `training.feature_groups` (narrative seule depuis le 26/10).
- **Decoupage** : temporel par coin. fit < es < validation (heures les plus recentes). Les periodes de
  jugement (folds) sont dans le train. La validation n'est JAMAIS utilisee pour choisir quoi que ce soit.
- **Mesure qui compte** : PnL en SOL sur la validation avec la vraie regle d'achat, compare a « acheter tout »
  (par coin) et au champion sur les memes heures. AUC et taux de gagnants = diagnostic, pas verdict.
- **Garde-fous de promotion** : `min_valid_trades`, PnL > 0, meilleur que le champion.

## 2. Construire sans fuite du futur
Toute nouvelle feature doit n'utiliser que ce qui etait connu a `decision_ms + latence` (heure murale pour les
metadonnees et X, heure on-chain pour les trades). Points a verifier systematiquement :
- resultats d'autres coins : seulement ceux resolus avant l'instant de decision (`resolve_ms`, `known_ms`) ;
- annotations faites apres coup : connues a partir de leur date d'ecriture (`after_outcome`) ;
- memoire (fame) : `known_ms` = moment ou le succes etait connu, pas la date du coin ;
- une feature qui mesure seulement le temps qui passe (compteur cumule, nombre de gagnants connus) est une
  fausse bonne idee : elle apprend la date, pas le coin (cas `fame_known`, retire le 26/10).
Ajoute un test dans `tests/test_core.py` pour chaque nouvelle feature dependant de l'historique (sans fuite,
cas limite, valeur attendue). Incremente `FEATURE_VERSION` dans features.py si une feature calculee dans
`dataset.token_row` change (le cache se reconstruit).

## 3. Evaluer
- Comparaison A/B : memes donnees, meme validation, meme nombre d'essais (voir le script de comparaison
  narrative / complet du 26/10 : entrainer deux fois en changeant une seule chose). Garder `search_trials`
  identique des deux cotes.
- Plus on essaie de reglages, plus on trouve des gagnants par chance : regarder l'ecart es -> validation ;
  un essai doit tenir sur les 3 periodes (folds).
- **Moins de ~30 achats en validation = pas de conclusion.** Le dire explicitement. Regarder plusieurs modeles
  successifs (registre) plutot qu'un seul.
- Regarder les tranches de score (`bins` dans meta.json) : le tri doit etre monotone ET l'argent doit suivre.
- Pour expliquer un modele : contributions (`Predictor.booster.contrib`) + effet reel par tranche de valeur
  (taux de gagnants), sur les deux jours de donnees.

## 4. Relire et consigner
Apres chaque experience, ajoute une ligne a **Historique** du journal :
`date | intention | ce qui a change | resultat (validation, achats, vs tout acheter, vs champion) | decision`.
Puis une ligne par piste a **Idees** (question, pourquoi maintenant, ce qui changerait, source), et deplace dans
**Backlog** celles qu'on decide de faire. Ne declare jamais un gagnant sans resultat sur la validation.

## 5. Mise en production
- Tests (`pytest -q tests`) + import de tous les modules + `node --check` si le dashboard change.
- Commit (voir CLAUDE.md), relance, puis verification dans les logs : `train` « entrainement ... »,
  `paper` « modele charge : <id> », pas de `Traceback`.
- Le paper trader prend le champion, sinon le dernier modele (non promu). Verifier lequel est charge.
- Rapporter a l'utilisateur en francais simple : resultat, nombre d'achats, ce que ca veut dire, prudence.

## Outils utiles
- Recherche profonde : `.venv\Scripts\python -m pumpml deepsearch` (prend le verrou de l'apprentissage).
- Registre : `models/registry.json` ; detail d'un modele : `models/<id>/meta.json` (metrics, params, trials,
  importance, bins, bad_buys, missed).
- Si les Probabl Skills sont installees : leur methode (frame / review / backlog) s'applique ; leurs outils
  (skore, skrub) ne s'appliquent pas tels quels a ce pipeline (pas sklearn, jugement en SOL).
