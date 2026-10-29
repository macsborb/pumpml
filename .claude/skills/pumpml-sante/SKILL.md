---
name: pumpml-sante
description: Controle de sante complet du systeme PumpML (collecte, credits Helius, lectures X, migrations, paper trader, modele charge, entrainement, imports, disque) et reparations sures. A utiliser au debut de chaque intervention, apres chaque relance, et quand l'utilisateur demande si tout tourne ou signale un probleme.
---

# Sante du systeme PumpML

Projet : `J:\Projet_de_code\leprojetdunevie`. Python : `.venv\Scripts\python`. Logs : `logs\`.
Ne jamais afficher de cle. Toute reparation risquee ou couteuse (credits, argent) se demande a l'utilisateur.

## Checklist (dans cet ordre)
1. **Processus** : `Get-CimInstance Win32_Process -Filter "Name like 'python%'" | ? { $_.CommandLine -match 'pumpml' }`
   -> `supervise`, `collect`, `paper`, `dashboard` (chacun en double : lanceur venv + python). Pause ?
   fichier `data\PAUSE`. Tache Windows « PumpML ».
2. **Superviseur** : `logs\supervise.log` — un service qui redemarre en boucle (« s'est arrete (code 1) ») =
   plantage au demarrage : lancer le module a la main pour voir l'erreur
   (`.venv\Scripts\python -c "from pumpml.collector import Collector; ..."`), corriger, relancer.
3. **Collecteur** : derniere ligne `stats` de `logs\collect.log` (une par minute) :
   - `tokens_new` ~30-50/min, `tokens_suivis`, `tokens_sans_lien`, `tokens_spam`, `trades` ;
   - `credits_jour` vs `budget_jour` (300 000 ; `tokens_hors_budget` > 0 = plus de nouveaux coins suivis) ;
   - cout : `credits_ws`, `kb_curve` / `n_curve` (robots de volume si n_curve >> trades), `rpc_*` ;
   - `social_ok` / `social_fail` (X ; 402 = plus de credits twitterapi), `meta_ok` / `meta_fail` ;
   - `migrations_*`, `pools`, `swaps`, `amm_file` (file d'attente des migrations ; doit rester basse).
   Attention : une ligne `stats` peut dater d'avant une relance ; verifier l'heure.
4. **Migrations** : comparer `migrations_pumpportal` et `pools` sur une heure ; un ecart fort = traitement
   bloque. Onglet « Apres migration » du dashboard.
5. **Lectures X** : dans `data\hot.db` table `social`, les 5 dernieres minutes : provider `twitterapi` et `ok`.
6. **Paper trader** : `logs\paper.log` — « modele charge : <id> » (champion sinon dernier modele), pas de
   « observation seulement » inattendu, pas de « boucle paper » (erreur). Heartbeat `paper` dans `hot.db`.
7. **Entrainement** : `logs\learn.log` / `logs\deepsearch.log` — derniere ligne « entrainement <id> » chaque
   heure ; `models\registry.json` (champion, derniers resultats). Le verrou `learn` empeche deux entrainements.
8. **Imports** : `logs\import_fame.log` (memoire des gagnants ; progression « plus hauts : n / N »),
   `logs\winners_report.log`. Un import qui n'avance plus (meme ligne depuis > 20 min) = limite de debit
   GeckoTerminal ou verrou SQLite.
9. **Disque** : `Get-PSDrive C,J`. Les donnees sont sur J:. Si C: est plein, prevenir l'utilisateur (ses jeux,
   caches, hiberfil.sys) — ne rien supprimer chez lui. Les fichiers temporaires des services vont dans `data\tmp`.
10. **Dashboard** : http://127.0.0.1:8787 repond (`/api/state` : `sol_usd`, heartbeats).

## Reparations sures
- Un seul service : tuer son processus `pumpml <service>`, le superviseur le relance (backoff).
- Tout : `scripts\pause.ps1` puis `scripts\resume.ps1` (tue aussi `deepsearch`, pas les `scripts\*.py`).
- Avant toute relance apres modification : import de tous les modules + `pytest -q tests` (voir CLAUDE.md).
- Budget Helius depasse : ne pas relever sans accord ; si accorde, `helius_budget_override` (une date).

## Rapport
Termine par un etat en quelques lignes : ce qui tourne, chiffres cles (coins/min, credits du jour, lectures X,
migrations/h, modele charge, dernier entrainement), problemes trouves, ce qui a ete repare, ce qui attend
l'utilisateur.
