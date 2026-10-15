AXPIndexerNG — audit du goulet vectoriel, 15 octobre 2026

L’audit et les mesures historiques ci-dessous ont été exécutés sur le dépôt fourni au commit 61480d6. Pour créer la PR, la branche a ensuite été rebasée sur main au commit ba40c44 ; les traces hiérarchiques, le cache d’identité, la progression et l’annulation de cette version sont préservés. Les chiffres historiques ne sont pas présentés comme un benchmark de la version rebasée. Aucun AGENTS.md n’est présent dans le dépôt, /workspace ou à la racine. Le commit ba40c44 n’était pas présent lors des premières mesures ; il a été récupéré lors de la préparation de la PR. Les 9 096 ms vectorielles et les 1 776 ms de reconstruction des compteurs restent des références utilisateur, pas des mesures reproduites ici. La base /workspace/axp-data/cloud.db est vide. Toutes les mesures ci-dessous utilisent un fixture synthétique ; aucun résultat ne qualifie les performances ou la pertinence du corpus utilisateur.

Le blocage des compteurs est démontré dans le code : configure_search_reader appelait search_reader_diagnostics ; sans entrée en cache pour cette connexion, cette fonction exécutait count(*) sur chunks et chunk_vectors et un comptage DISTINCT sur chunks_fts_idx. Le cache était propre à chaque connexion, donc chaque nouveau reader répétait le travail. Ce coût faisait partie de reader_open_ms, avant toute recherche.

La correction remplace ce cache par des échantillons partagés entre readers et pools d’un même processus, identifiés par le chemin et l’identité du fichier. Une seule reconstruction par catalogue peut fonctionner à la fois, avec sa propre connexion readonly et un snapshot WAL commun aux compteurs. L’absence de snapshot renvoie immédiatement des compteurs nuls et le statut unavailable, puis programme le calcul. Les erreurs de création du worker ou de calcul ne rendent pas la recherche indisponible. Les readers de recherche ne sont jamais prêtés au worker. Aucun comptage ne reste dans le chemin synchrone de recherche.

La fraîcheur est explicitement celle d’un échantillon : sampled avant cinq minutes, aged ensuite, avec date de prélèvement, âge, portée et statut de recalcul. Elle ne certifie pas que le catalogue n’a pas changé depuis le prélèvement. Aucun commit de statut, d’identité ou de métadonnées n’invalide systématiquement les compteurs. Après cinq minutes, la prochaine consultation programme un recalcul, sans attendre. Le cache est borné à huit entrées inactives, en préservant les travaux actifs. Un nouveau processus repart sans snapshot. refresh=True est réservé à un diagnostic explicite hors transaction appelante ; un travail déjà actif est réutilisé sans attente.

L’API expose séparément search_data.status, identity_cache et les champs catalog_snapshot_*. La version rebasée conserve le cache de résultats d’identité de ba40c44, indexé par incarnation/révision des métadonnées pertinentes, requête et portée. L’API expose son hit/miss/bypass, sa génération, sa raison d’invalidation et l’âge de l’entrée utilisée ou publiée, indépendamment de l’âge des compteurs. Les générations pertinentes restent mises à jour transactionnellement ; aucun compteur ne sert à invalider ce cache. Au moment des mesures historiques sur 61480d6, ce cache n’existait pas encore. L’UX l’indique, affiche « Compteurs indisponibles » si nécessaire et ne confond pas cette absence avec l’indisponibilité de la recherche.

Le SQL global réel est conservé dans COMBINED_SQL, shared/axp_core/vectors.py. Le schéma crée vec0(embedding float[dimension] distance_metric=cosine). La dimension est fournie par le modèle, et non déduite du nombre de résultats. Sur le fixture : sqlite-vec 0.1.9, SQLite 3.53.1, dimension 384, 467 255 vecteurs float32, soit 717 703 680 octets utiles (684,46 Mio), hors métadonnées et surcoûts SQLite. La dimension du corpus utilisateur reste à relever avec l’outil livré.

La recherche globale utilise embedding MATCH ? AND k=?, k=100 par défaut, borné à 500, puis ORDER BY distance. Les jointures internes sont v.rowid=c.id, c.document_id=d.id et d.source_id=s.id. Il n’y a pas de filtre métier global sur enabled, status, ingestion_mode ou source. Leur inclusion historique est préservée. Les restrictions documentaires de search_documents restent appliquées à c.document_id IN (...) avant le calcul ciblé vec_distance_cosine ; ce chemin n’a pas été réécrit.

Le diagnostic développeur explicite developer_diagnostics et scripts/benchmark_vectors.py relèvent la déclaration réelle, les index, la version de l’extension, le top-k, les filtres, les SQL et leurs EXPLAIN QUERY PLAN. Aucun EXPLAIN ni comptage n’est exécuté à chaque recherche normale. Le plan constaté est :

```text
KNN seul :
SCAN chunk_vectors VIRTUAL TABLE INDEX 0:3{___}___
USE TEMP B-TREE FOR ORDER BY

Requête combinée :
SCAN v VIRTUAL TABLE INDEX 0:3{___}___
SEARCH c USING INTEGER PRIMARY KEY (rowid=?)
SEARCH d USING INTEGER PRIMARY KEY (rowid=?)
SEARCH s USING INTEGER PRIMARY KEY (rowid=?)
USE TEMP B-TREE FOR ORDER BY

Enrichissement groupé :
SEARCH c USING INTEGER PRIMARY KEY (rowid=?)
SEARCH d USING INTEGER PRIMARY KEY (rowid=?)
SEARCH s USING INTEGER PRIMARY KEY (rowid=?)
```

Les index ordinaires pertinents sont les clés primaires de chunks/documents/sources, UNIQUE(document_id,chunk_no), documents_source et les index uniques des chemins. Aucun index ANN n’est présent ou introduit. La contrainte native k borne les candidats transmis aux jointures ; elle ne borne pas le nombre de vecteurs examinés par la recherche exacte. Le nombre effectivement visité n’est pas exposé par cette instrumentation et reste null. Les tests vérifient notamment qu’un vecteur orphelin consomme encore sa place dans le top-k et que les jointures ne remplissent pas ensuite artificiellement cette place.

Les variantes de mesure sont : KNN ne renvoyant que chunk_id/distance ; une requête IN groupée sur ces identifiants ; et la requête combinée actuelle. Chaque paire partage un snapshot et l’ordre de passage alterne. L’enrichissement restitue les distances natives et l’ordre exact, y compris les égalités. search_split conserve aussi une transaction appelante et utilise un savepoint pour préserver le snapshot entre ses deux étapes. Les suppressions et modifications concurrentes ne mélangent pas anciennes distances et nouvelles métadonnées. Il n’y a pas de N+1.

Les durées execute/fetchall sont instrumentées séparément pour combined, knn et enrichment, avec des dictionnaires locaux à la requête. execute peut couvrir les lectures, le moteur, l’extension, les jointures et les attentes ; ce n’est pas une mesure de calcul vectoriel pur. Les observations CPU et read_bytes sont celles du processus, pas des statistiques de hits/misses du cache SQLite.

Sur sept répétitions après la première paire, les médianes en ms sont :

| Requête synthétique | Combinée | KNN seul | Enrichissement | Deux étapes |
| --- | ---: | ---: | ---: | ---: |
| DOMINO | 398,21 | 395,86 | 0,490 | 396,27 |
| reactor pressure | 412,95 | 415,74 | 0,477 | 416,22 |
| DOMINO pressure | 418,31 | 422,29 | 0,473 | 422,73 |
| DOMINO storage pressure | 401,66 | 402,00 | 0,486 | 402,47 |
| unobtainium calibration | 402,90 | 397,61 | 0,501 | 398,05 |

Les différences entre combinée et deux étapes changent de signe et restent dans une dispersion plus large. La séparation n’apporte pas de gain établi. La production garde donc la requête combinée. Le KNN seul domine ici, avec environ 390–420 ms de CPU processus et zéro octet de lecture physique rapporté pour ces appels, premier appel compris. Le cache OS était déjà alimenté par la construction du fixture ; aucun cache disque n’a été vidé. Ce résultat soutient une limite CPU sur ce fixture chaud, sans expliquer les neuf secondes du poste utilisateur ni exclure I/O et attentes sur ce poste.

Les readers utilisent toujours un cache SQLite de 64 Mio, mmap de 256 Mio quand disponible, temp_store=MEMORY et busy_timeout=5000. Les réglages sont relevés dans le JSON ; le cache configuré n’est pas une mesure de hits/misses. La mémoire utile du fixture dépasse mmap et le cache SQLite, mais les pages peuvent être dans le cache OS. Aucun dimensionnement mémoire supplémentaire n’est imposé sans mesure sur le poste. Le daemon emploie une autre connexion WAL et peut concurrencer les recherches pour CPU, stockage et mémoire pendant ses batches. Aucun daemon réel n’était actif ici. Sa concurrence réelle n’est donc pas qualifiée ; les tests de commits concurrents ne remplacent pas ce benchmark.

La comparaison d’ouverture utilise huit pools successifs, une connexion neuve par pool, même fichier et mêmes réglages. L’ancienne fonction est chargée depuis git show HEAD:shared/axp_core/database.py. Le calcul de fond est laissé finir hors mesure avant l’essai suivant. Le premier appel inclut un contexte différent des répétitions ; ce n’est pas un froid disque.

| Ouverture du reader | Avant | Après |
| --- | ---: | ---: |
| Premier appel de l’expérience, ms | 213,36 | 1,63 |
| Médiane des sept autres connexions neuves, ms | 113,70 | 0,71 |
| Min–max des sept autres, ms | 112,91–118,10 | 0,66–1,36 |

Le benchmark Search existant a également été exécuté avant/après sur le même fixture, les cinq mêmes requêtes et trois répétitions. Il mesure axp_client.search.search sur connexion directe, hors ouverture du pool et HTTP. Les noms cold_connection/warm dans ses données signifient connexion neuve/réutilisée ; pas froid/chaud du disque. Médianes avant → après en ms :

| Scénario | Connexion neuve | Connexion réutilisée |
| --- | ---: | ---: |
| Nom | 565,38 → 454,26 | 419,81 → 456,48 |
| Hybride | 495,99 → 481,41 | 388,88 → 447,35 |
| Identité | 527,75 → 436,50 | 405,50 → 456,95 |
| Plusieurs passages | 523,16 → 442,75 | 405,67 → 443,49 |
| Absent | 538,69 → 468,86 | 421,51 → 451,46 |

Ces séries courtes et dispersées ne démontrent pas de gain global stable, particulièrement sur connexion réutilisée. Le classement de tête et les décisions RAG sont conservés : document 1 pour les quatre scénarios avec réponse, abstention RAG pour la question absente. Les embeddings sont artificiels : la recherche Search peut encore retourner des archives sur la question absente ; ce fixture ne mesure donc pas un rappel sémantique réel. Les 40 paires SQL conservent tous les champs, les distances et l’ordre, pas seulement le document de tête.

Le navigateur conserve son horloge performance.now, distincte de http_search_total_ms. L’observation du premier résultat se fait après insertion du classement final et deux requestAnimationFrame, soit une occasion de rendu ; elle ne certifie pas les pixels visibles à l’écran. Une liste vide n’a pas de temps de premier résultat. Les requêtes successives annulent le fetch précédent et ignorent ses réponses, erreurs et callbacks de rendu tardifs. Les mesures sont disponibles via axp-search-measured et window.axpSearchMeasurement.

Chromium headless 154, serveur redémarré, même fixture : premier appel DOMINO, 613,09 ms serveur et 626,20 ms navigateur. L’ouverture du reader vaut 88,23 ms : connexion 83,21 ms, réglages 0,09 ms, consultation/planification des compteurs 4,93 ms. Elle inclut notamment le chargement initial de l’extension et de ses imports ; le protocole ne permet pas d’en isoler davantage la cause. Cela diffère de l’expérience avec extension déjà chargée ci-dessus. Les nouveaux chronomètres empêchent d’attribuer tout reader_open_ms aux compteurs.

| Requête | Serveur médian, ms | Premier résultat navigateur médian, ms | Min–max navigateur | IQR navigateur |
| --- | ---: | ---: | --- | ---: |
| DOMINO | 400,64 | 414,50 | 398,10–461,20 | 9,00 |
| reactor pressure | 415,16 | 430,50 | 397,60–464,40 | 33,30 |
| DOMINO pressure | 409,15 | 418,50 | 397,20–431,50 | 23,85 |
| DOMINO storage pressure | 401,66 | 414,00 | 412,00–480,90 | 6,50 |
| unobtainium calibration | 405,42 | 416,10 | 414,30–447,90 | 17,35 |

Chaque ligne contient sept répétitions, distinctes du premier tour conservé dans le JSON. Il n’existe pas de mesure navigateur avant correction sur ce fixture. Ces valeurs ne se comparent pas aux 13 890 ms du navigateur utilisateur.

Le lexical s’exécute déjà avant le vectoriel. L’audit synthétique trouve le document final dans les candidats lexicaux des quatre scénarios avec réponse et aucun candidat lexical pour la question absente. Les sondes lexicales isolées sont prises après la recherche complète, sur cache chaud ; leur chevauchement avec le classement final n’est pas une annotation de pertinence. Faute de validation réelle, aucun affichage progressif n’est activé et le classement hybride final est conservé.

La validation Python initiale donnait 452 succès et cinq échecs. Le commit initial 61480d6 a été extrait et testé séparément : 435 succès et les mêmes cinq échecs, dans test_alpha3_hardening, test_document_coverage, test_e2e, test_multisource_scan et test_scanner, concernant l’ingestion/les chemins Windows sous Linux. Les 17 cas initialement ajoutés couvrent parité complète et bornes, sources désactivées, vecteurs orphelins, suppression/modification, enrichissement groupé, transactions, commits concurrents, absence de compteurs via HTTP, cache partagé, recalcul unique, erreur du worker, âge, isolation des catalogues et des chronomètres. Les trois tests Node couvrent le rendu sans compteurs, les horloges séparées, les réponses/erreurs tardives et les callbacks de rendu obsolètes. Ruff et git diff --check passent.

Les données brutes sont dans docs/benchmarks/vector-bottleneck-vectors.json, vector-bottleneck-browser.json et vector-bottleneck-readers.json. Le résumé avant/après, sa dispersion et les échantillons Search sont dans vector-bottleneck-summary.json. Aucun binaire, modèle ou base n’est livré.

Pour reproduire sur un catalogue existant sans le modifier :

```sh
python scripts/benchmark_vectors.py --db CHEMIN_CATALOGUE --output work/vectors-idle.json --repeats 7 --top-k 100 --full-search --query "REQUETE_1" --query "REQUETE_2" --environment-label "daemon au repos"
node scripts/benchmark_search_browser.mjs --url http://127.0.0.1:8765 --output work/browser-idle.json --repeats 7 --queries-json work/queries.json --chrome CHEMIN_CHROMIUM
```

work/queries.json doit être une liste JSON des mêmes chaînes. Les commandes doivent être exécutées successivement, sans lancer deux benchmarks simultanément. Le benchmark vectoriel charge l’embedder du catalogue ; --vectors-json accepte aussi une liste d’objets query/vector pour isoler SQLite, sans --full-search. Il produit les plans uniquement parce que ce diagnostic a été déclenché explicitement. Le script navigateur nécessite Node avec WebSocket natif (validé avec Node 24) et Chromium ; il ne démarre pas le serveur. --no-sandbox a été nécessaire dans le conteneur de validation, pas dans l’exemple standard.

Relancer les mêmes commandes avec les mêmes requêtes et le même snapshot de données, d’abord daemon au repos puis durant un scan normal, sans réindexer pour ce benchmark. Utiliser un autre nom de sortie et --environment-label pour chaque condition. L’outil vectoriel observe aussi les PID/CPU/lectures des processus axp_daemon présents, sans les arrêter ni les démarrer. Enregistrer séparément les commits éventuels : une transaction garantit un snapshot commun aux variantes d’une paire, pas l’identité du catalogue entre deux sessions. Sur Linux, pidstat et iostat peuvent compléter CPU/I/O/attentes ; sur Windows, le Moniteur de ressources ou WPA est nécessaire. Les statistiques fines de cache SQLite ne sont pas exposées par le connecteur Python standard utilisé ici.

Pour recréer uniquement le fixture synthétique :

```sh
python scripts/benchmark_recovery.py --db work/vector-fixture.db --output work/recovery.json --repeats 3
python scripts/benchmark_vectors.py --db work/vector-fixture.db --synthetic --full-search --output work/vectors.json --repeats 7 --environment-label "fixture synthétique, daemon absent"
python -m pytest -q tests
node --test tests/search_frontend.test.mjs
ruff check shared client scripts tests
```

Le premier script est réservé à un chemin de fixture dédié, jamais au catalogue utilisateur. Pour comparer l’ouverture avant/après, exécuter scripts/benchmark_search.py sur chaque checkout avec le même catalogue ; son champ db_acquire_ms inclut l’ouverture ; l’API /api/search?q=...&explain=1 expose les sous-chronomètres reader_*. Le gain local sur les compteurs est établi ; la cause précise des neuf secondes sur le poste, l’effet du daemon réel, le froid disque et la pertinence des requêtes utilisateur restent à mesurer avec ces outils. Rien n’établit à ce stade la nécessité d’un autre moteur ou d’un index approximatif.

Validation après rebase pour la PR : 465 tests Python réussis, avec les cinq mêmes échecs préexistants. Le checkout ba40c44 testé séparément donne 448 succès et ces mêmes cinq échecs. Les 66 tests ciblés (y compris les diagnostics de ba40c44) et les trois tests Node passent. Le test HTTP vérifie aussi qu’un cache d’identité chaud et daté reste utilisable alors que les compteurs sont indisponibles, sans mélange de traces entre les deux requêtes.
Un contrôle Chromium réel après rebase a également passé les cinq requêtes du fixture (premier tour puis une répétition), avec l’interface de progression et le cache d’identité préservés. Ce contrôle de fonctionnement ne constitue pas une nouvelle comparaison statistique avant/après.
