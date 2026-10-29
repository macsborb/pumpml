# PumpML : règles du projet

IA qui apprend à reconnaître les bonnes **narratives** de memecoins pump.fun (Solana) et les trade **en papier** (prix Jupiter réels). Tourne 24 h/24 sur le PC de l'utilisateur (RTX 3070, 32 Go). Tout se fait et s'écrit en **français**.

## Règles absolues
- **Jamais d'achat réel.** Paper trading uniquement, jusqu'à preuve d'un vrai avantage et décision explicite de l'utilisateur.
- **Jamais de clé dans git, dans le chat ou dans un log.** Les clés sont dans `.env` (Helius, Jupiter, twitterapi.io) ; l'utilisateur la remplit lui-même. `logutil` masque `api-key=`.
- **Prix du paper trading = cotations Jupiter réelles**, jamais un calcul de courbe ou de pool.
- **Budget Helius : 300 000 crédits / jour** (plan Developer, 10 M/mois). Une hausse ponctuelle passe par `helius_budget_override` (une date) et seulement avec l'accord de l'utilisateur.
- **twitterapi.io** : plan Starter (pris le 26/10), utilisé sans plafond ; point sur la consommation vers le 08/11.

## Stratégie voulue par l'utilisateur
- Le modèle décide sur la **narrative seulement** (`training.feature_groups`) : tweet (frais, < 2 min), image, nom, liens nom ↔ tweet ↔ site, mémoire des gros gagnants et des flops, copies / clins d'œil, créateur. **Pas** de prix, holders, flux de trades ni heure dans les poids.
- Règles d'exclusion (pas des poids) : lancement groupé (> 25 % de la courbe au bloc 0), tweet de plus de 2 min, réponse « parasite », coin sans Twitter ni site, coin noyé sous les robots de volume, **copie** d'un coin déjà sorti (même image, lien X, ou nom + ticker) sauf tweet nouveau publié après l'original et différent de son tweet (`block_copies`, veto après le score du modèle).
- Coins de 7 min maximum sur la courbe ; après migration : observation seulement (onglet « Après migration »).
- « Gros gagnant » = migration naturelle (courbe remplie en plus d'1 min, pas achetée au bloc 0, au moins 100 transactions) **et** plus haut ≥ 100 000 $ sur bougies de 15 min **et** tenue (encore ≥ 30 % du plus haut 1 h après). Sinon c'est un vidage, compté comme flop (`[fame]`, `fame.label`). Les petits x2 ne comptent pas.

## Architecture (voir README.md)
`pumpml/` : `collector` (PumpPortal + Helius coin par coin, `tokstream`, `amm` pour PumpSwap), `paper` (décisions + positions Jupiter), `model` + `search` (Optuna, LightGBM / XGBoost CUDA, ensemble), `fame` (mémoire des gagnants / flops, `data/fame.db`), `protos` (tes exemples), `dashboard` (http://127.0.0.1:8787), `supervisor` (tâche Windows « PumpML »). Données : `data/raw` (Parquet), `data/hot.db`, `data/journal.db`, `data/emb.db`, `data/fame.db`.

## Façon de travailler
- **Relancer** : `scripts\pause.ps1` puis `scripts\resume.ps1` (PowerShell). Pour un seul service, tuer son processus `pumpml <service>` : le superviseur le relance. `pause.ps1` tue aussi `deepsearch` (mais pas les scripts `scripts\*.py`).
- **Avant toute relance** : `.venv\Scripts\python -c "import pumpml.collector, pumpml.paper, pumpml.dashboard, pumpml.model, pumpml.amm, pumpml.fame, pumpml.search"` et `.venv\Scripts\python -m pytest -q tests`. Les tests n'importent pas tout : une erreur de syntaxe dans `collector.py` a déjà coupé la collecte 27 min. Si le collecteur change, l'instancier aussi (`Collector(Config())`) : une nouvelle table Parquet sans schéma dans `storage.SCHEMAS` l'a fait planter au démarrage (27/10).
- **Après une relance** : vérifier les logs (`logs\collect.log` « stats », `logs\paper.log` « modele charge », `logs\supervise.log`) — pas seulement les tests.
- **Modifier un fichier** : Edit / Write, ou un script Python dans le scratchpad qui fait des `replace` avec `assert count == 1`. **Jamais** `Set-Content` PowerShell sur `config.toml` (BOM → tomllib plante). Éviter les heredocs bash avec des apostrophes ou des `$` : écrire le script dans un fichier.
- **JS du dashboard** : pas d'apostrophe dans une chaîne entre `'…'` (`l'actu` casse tout le fichier) ; vérifier avec `node --check web/ai.js` et le script inline de `web/index.html`.
- **SQLite partagé** : toujours `timeout=` (plusieurs processus écrivent dans `fame.db`, `journal.db`).
- **Commits** : `git -c user.name=PumpML -c user.email=109520260+macsborb@users.noreply.github.com commit …`, message en français, attribution Claude en fin de message. Un commit par changement cohérent.
- **Expériences de modèle** : suivre la skill `pumpml-experience` (verdict sur des heures jamais vues, comparer à « tout acheter », méfiance sous ~30 achats).
- **Santé du système** : skill `pumpml-sante`. **Analyse des gros gagnants** : skill `pumpml-gagnants` (tâche planifiée toutes les 4 h).

## Pièges déjà rencontrés
- Le WS public Solana perd ~2/3 des messages et retarde : on utilise l'heure on-chain et Helius coin par coin.
- PumpPortal rate ~58 % des créations : abonnement au compte d'autorité de mint (100 %).
- Les migrations se détectent via le compte `39azUYFW…` (présent dans chaque `MigrateV2`) ; leur traitement doit être parallèle.
- Les robots de volume (jusqu'à 12 500 tx/min sur un coin, souvent échouées) étaient facturés par Helius : coupure au-delà de 1 000/min.
- `Registry.latest()` et le paper trader doivent reconnaître les modèles en ensemble (`meta.json` / `members`) et narratifs (sans `age_s`).
- Les compteurs de crédits des scripts d'import ne sont pas vus par le garde-fou du collecteur.
- GeckoTerminal (gratuit) limite à ~20-30 requêtes/min : réessayer le même coin après une pause, ne jamais le sauter.
