# Journal des experiences PumpML

Voir la skill `pumpml-experience` pour le protocole.

## Decisions verrouillees
| Decision | Valeur | Depuis |
|---|---|---|
| Ce qu'on predit | Resultat d'un achat de 0,1 SOL a un point d'analyse (5 s a 7 min), sortie TP +50 % / SL -40 % / vente a 2 min si < +10 % / 30 min max | 26/10 |
| Frais | 1,25 % par sens (pump.fun) + 0,0005 SOL par transaction ; paper au prix Jupiter | 25/10 |
| Signaux autorises | Narrative seule : narrative, tweet, meta, gagnants, copies, createur | 26/10 |
| Exclusions (regles) | Lancement groupe > 25 % au bloc 0, tweet > 2 min, reponse parasite, sans Twitter ni site, robots de volume | 26/10 |
| Decoupage | Temporel par coin : fit / es (+ 2 periodes) / validation = 6 dernieres heures | 26/10 |
| Mesure | PnL SOL validation, regle d'achat reelle, vs acheter tout et vs champion | 25/10 |
| Promotion | >= 10 achats en validation, PnL > 0, meilleur que le champion | 25/10 |

## Historique
| Date | Intention | Changement | Resultat (validation) | Decision |
|---|---|---|---|---|
| 26/10 matin | Sorties | TP 100 -> 50 %, vente a 2 min si rien | -9 % de pertes, positions 3-5 min | Adopte |
| 26/10 matin | Objectif | binaire / pondere / regression | Aucun ne bat acheter tout par achat | pnl_weighted garde |
| 26/10 13 h | Bundles | Exclusion > 25 % au bloc 0 | 42 % des achats IA etaient des bundles, -0,021 SOL/trade en direct | Adopte |
| 26/10 14 h 35 | Recherche profonde | 400 essais, 1 periode | es +0,55 -> validation -0,13 SOL / 9 achats | Sur-ajustement : passage a 3 periodes |
| 26/10 14 h 53 | Recherche profonde | 400 essais, 3 periodes, ensemble 5 | -0,20 SOL / 26 achats (tout acheter -6,67 / 792) | Garde (plus fiable) |
| 26/10 17 h 46 | Narrative seule vs complet | feature_groups | narrative -0,13 / 11 ; complet -0,36 / 23 (tout acheter -11,5 / 1321) | Narrative seule adoptee |
| 26/10 22 h 04 | Horaire narrative | XGBoost, poids meta x2,6 createur x2,2 tweet x0,4 | +0,044 SOL / 12 achats, ancien champion -0,40 | Premier champion (prudence : 12 achats) |
| 27/10 14 h | Memoire des gagnants | Regle de tenue (plus haut 15 min, >= 30 % 1 h apres, >= 100 tx) vs plus haut journalier | tenue -0,099 SOL / 8 achats, AUC 0,68 ; ancienne -0,171 / 25, AUC 0,74 (tout acheter -7,4 / -7,6) ; 101 vrais gagnants sur 494 | Pas de conclusion (< 30 achats). Tenue gardee : c est la definition voulue |
| 27/10 16 h | Signaux KOL | feature_groups + kol (10 signaux kol_*, flair sans fuite) | avec -0,118 SOL / 20 achats, AUC 0,76 ; sans -0,109 / 17, AUC 0,73 (tout acheter -8,8 / -8,2). Signaux kol jamais utilises : KOL avant decision sur 1,1 % des lignes (125 coins) ; 16 % gagnants vs 2,4 % mais -0,0089 SOL/achat vs -0,0061 | Pas active. On collecte, on refait le test quand on aura ~500 coins avec KOL |
| 27/10 17 h | Sortie sur coins KOL | Achat au 1er point apres le 1er KOL (129 coins), 10 sorties rejouees sur les vrais trades | Toutes perdent. Actuelle -0,0099/achat ; sans vente 2 min -0,0115 ; sans TP -0,0217 ; meilleure : vente 2 min + TP 100 % -0,0089 (dans le bruit). 36 % des coins KOL touchent le SL -40 % apres notre achat. Coins sans KOL : -0,015/achat | Rejete : garder la vente a 2 min. Les KOL vendent sur nous |
| 27/10 21 h | Predire les achats des KOL | Cible : un KOL (ou un gros nom) achete dans les 10 min apres la decision, aucun KOL encore entre ; LightGBM, test 6 dernieres heures | Borne haute (si on savait) : acheter avant un gros nom = 70 % gagnants, +0,029 SOL/achat. Prediction : narrative seule AUC 0,55 (tout KOL) / 0,65 (gros nom), 0 % de precision en haut du classement ; + flux AUC 0,75 (porte par la market cap) mais -0,0127 SOL/achat vs -0,0103 tout acheter. 382 / 92 cas positifs en train | Pas encore predictible. Refaire avec ~2 semaines (top 50 Kolscan ajoute le 27/10) |
| 27/10 23 h | Courbe d apprentissage | Meme test (6 dernieres heures, ~1 400 coins, tout acheter -14,4 SOL), entrainement sur 1 / 2 / 3 jours, 2 passes | AUC moyenne 0,745 -> 0,754 -> 0,769 ; gain moyen -0,142 -> +0,004 -> -0,014 SOL (3 a 16 achats) ; gagnants du meilleur dixieme 4,8 % -> 7,0 % -> 6,5 % | Le modele progresse avec les donnees (surtout de 1 a 2 jours) : continuer a collecter |
| 28/10 21 h | Memoire des coins perces (~30 000 $ ou migres) | Groupe percees (perc_*, 7 signaux narratifs, 44 percees organiques sur 10 276 coins suivis) ajoute ou non, 60 essais chacun | Sans : -0,049 SOL / 4 achats, AUC 0,779 ; avec : -0,063 / 7 achats, AUC 0,784 ; tout acheter -13,8 / -17,0 (la validation a grandi entre les deux runs) ; aucun signal perc_ dans les 40 plus importants | Pas de conclusion (moins de 30 achats) : groupe garde inactif, a retester quand ~150 percees (vers le 08/11) |

## Idees
| Question | Statut | Experience | Source |
|---|---|---|---|
| Le poids du tweet remonte-t-il avec une journee complete de lectures X ? | ouverte | horaire 27/10 | trou de credits twitterapi 26/10 7 h 30-13 h |
| La memoire des gagnants (90 jours en cours d'import) ameliore-t-elle la validation ? | ouverte | comparer avant / apres import | fame.db |
| Une cible « gros decollage » (x2 avant SL) en plus du PnL ? | ouverte | - | rapport du 26/10 |
| Rachat sur repli (dip-buy) : 13 combinaisons positives sur 1326 | a cote | - | analyse du 26/10 (flux, hors narrative) |
| Le signal proto_mauvais_texte est appris a l'envers (28/10 : achats a 0,95-0,97 de ressemblance a tes mauvais exemples, contribution +0,10 a +0,23, 28 perdants sur 30) : les annotations -1 visent des montees artificielles dont le prix a monte. Retirer les protos -1 des poids, ou les garder en regle d'exclusion ? | ouverte | A/B sans protos negatifs | errors_report 28/10 |
| Les signaux de ressemblance aux flops (`flop_img_sim`, `flop_text_sim`) poussent aussi a l'achat (13 des 15 pires achats du 28/10, jusqu'a +0,39) et `fame_vs_flop` vaut ~+0,65 sur les perdants : meme inversion que les protos. Tester sans le groupe flop, ou avec contrainte monotone (ressemblance flop ne peut que baisser le score) ? | ouverte | A/B groupe flop retire / monotone | errors_report 28/10 20 h |
| Signal « le tweet lie parle-t-il du coin ? » (nom / ticker / CA dans le texte du tweet) : NYSE (@solana « soon »), LOFIGIRL, BIBICOIN pointaient des tweets de gros comptes sans rapport | ouverte | - | lecons erreurs 28/10 20 h |
| `cat_crypto_meta` pousse a l'achat des coins vides (+0,10 a +0,22 sur 7 des 15 pires achats du 29/10 : texte « NOM \| NOM » ou buzzwords) alors que les vrais produits qui tiennent ont une description precise. Regle d'exclusion « description vide ou = nom » pour les coins crypto meta ? | ouverte | - | lecons erreurs 29/10 01 h 32 |
| Deux coins sur le meme tweet (Thousand yard stare ×2, 29/10) tous deux achetes ; `block_copies` ne vise que les copies d'un coin deja sorti, donc c'est la regle telle que voulue. Veto aussi sur un 2e coin pointant le meme tweet qu'un coin deja achete ? | ouverte | - | lecons erreurs 29/10 01 h 32 |
| `block_copies` ne normalise pas l'alphabet : « Мerge » (M cyrillique U+041C) imitait le launchpad Merge (compte @mergelaunch qui epingle un autre mint) et a ete achete le 29/10. Normaliser les homoglyphes (NFKC + table cyrillique / grec → latin) avant de comparer nom + ticker ? | ouverte | - | lecons erreurs 29/10 04 h 03 |
| Compte X lie inexistant (404 : OpenPad, Faceoff, Grokiverse, Blockora, achetes 28-29/10) : signal ou exclusion « le compte X existe » (une lecture fxtwitter / twitterapi par coin candidat) ? Le 29/10 08 h : Optionpad et SKETCH aussi (404), et Cadence / Playtest avaient un compte cree le jour meme ou la veille (6-7 abonnes) : ajouter l'age et les abonnes du compte X lie ? Attention, le profil fxtwitter de @CadenceCredits a renvoye 200 puis 404 a quelques secondes d'intervalle : verifier un 404 par une 2e source (twitterapi.io) avant d'en faire une regle. 29/10 12 h : Coin Casino, compte cree ~45 min avant le coin (7 abonnes), meme 404 puis 200 sur fxtwitter ; l'age du compte se lit sans appel dans l'id (snowflake). | ouverte | - | lecons erreurs 29/10 04 h 03, 08 h 01, 12 h 01 |
| Signaux createur : sur les 30 derniers achats clos, 19 ont un portefeuille createur neuf (`creator_n_1h` = 0, `creator_prev_tokens` = 0) et `creator_n_1h` / `creator_n_6h` sont dans les raisons positives (+0,17 a +0,19 sur Cadence, Playtest) ; 5 autres viennent d'usines a coins (42 a 151 coins / h : BundleBug, Phoenix, Thousand yard stare, DAG Perps, Memeliquid). Comparer ces taux a la base de tous les coins vus avant de conclure ; exclure les createurs > ~20 coins / h ? 29/10 12 h : 3e cas (Coin Casino, `creator_n_1h` +0,17, `creator_n_6h` +0,16, premieres raisons d'achat). | ouverte | - | lecons erreurs 29/10 08 h 01, 12 h 01 |
| Les coins KOL perdent quand meme : sortie differente ? | rejetee 27/10 (10 sorties testees, toutes negatives) | journal 27/10 17 h | test KOL 27/10 |

## Backlog
| # | Piste | Source |
|---|---|---|
| B1 | Comparer chaque nuit le champion aux modeles horaires sur 24 h glissantes (plus d'achats) | 26/10 |
| B2 | Annotations des gros gagnants toutes les 4 h (skill pumpml-gagnants) : mesurer l'effet sur proto_bon_net | 26/10 |
| B3 | Gagnant = plus haut >= 100 k$ ET tenue (ex. market cap 1 h apres le plus haut >= 30 % du plus haut) ; sinon « vidage » (4 des 8 gagnants du 27/10 retombes a ~2 k$ en < 30 min) | 27/10 |
| B4 | Plus haut sur bougies 15 min plutot que journalieres (meche GARFIELD 106 k$ vs 53 k$) ; nombre minimal d'acheteurs sur la courbe (STONK : 7 tx) | 27/10 |
| B5 | Mesurer l'effet sur la memoire : la verification complete classe 61 / 70 « gagnants » en lancement groupe | 27/10 |
