# alpha4 — Diagnostic de Recherche et audit de latence

Base examinée : `61480d6` (v0.3.7-alpha4 observée dans la demande). Aucun `AGENTS.md` trouvé dans l’espace de travail. Modifications exclusivement textuelles.

## Causes établies et corrections

Le préfiltre d’identité ajoutait **tous les documents contenant un caractère non ASCII**, y compris les accents ordinaires déjà gérés par `unicode61`. Le matcher Python normalisait ensuite titre, nom, chemins et collection pour chaque candidat, même pour une requête sans résultat. Sur la fixture accentuée de 2 000 documents, cela sélectionnait les 2 000 documents. Le profilage confirme la prédominance de ces normalisations. Cela établit un goulet reproductible ; le corpus utilisateur n’est pas disponible pour attribuer précisément ses 13–17 secondes à chaque routine.

Un index FTS additionnel `document_identity_unicode_fts` contient la représentation NFKD/casefold des métadonnées Unicode. Le préfiltre interroge cet index avec les mêmes préfixes, puis conserve le matcher et son classement historiques. Les accents, ligatures, caractères pleine chasse, pluriels, collections et identifiants gardent leur traitement. L’ancien parcours reste disponible sur un catalogue non migré. Les normalisations pures courtes sont mémorisées avec des valeurs immuables, dans trois LRU de 2 048 entrées maximum et 512 caractères par entrée ; les longs contenus contournent ces caches.

L’index est construit une fois par une connexion écrivain, par lots de 256 documents, sous verrou SQLite, puis publié transactionnellement. Les triggers assurent les insertions, mises à jour et suppressions. La fonction SQL `axp_identity_compat` est enregistrée par `axp_core.database.connect` sur chaque écrivain : **redémarrer daemon et client avec la même version**, et conserver ce point d’entrée pour les écrivains. Aucun changement de version du schéma principal, des passages ou des vecteurs.

Un petit cache des résolutions finales contient au maximum 128 clés et 48 IDs par clé, avec une limite de 8 192 caractères par clé. Sa génération est un jeton d’incarnation de la base + une révision transactionnelle. Elle change sur ajout/suppression de documents/sources, changement de nom, titre, chemins, source, extension, date documentaire, label ou chemin de source. Les écritures de progression, statistiques de scan, date d’indexation et réécritures de valeurs identiques ne l’invalident pas. La clé inclut base, génération, question, scope et limite. La lecture de génération et le calcul partagent un snapshot SQLite. Les reconstructions d’une même clé sont partagées entre threads ; 16 reconstructions simultanées maximum sont enregistrées. Une transaction appelante non terminée contourne ce cache. Échec/annulation libèrent l’attente sans publier de résultat partiel.

L’identité ne lit pas les fichiers sources. Dans Recherche, l’hydratation des extraits est une requête bornée à huit documents ; dans le routage RAG, les vérifications supplémentaires sont bornées à trois documents. Aucun N+1 sur le catalogue constaté. Les plans réellement capturés montrent `SEARCH d USING INTEGER PRIMARY KEY`, `SEARCH s USING INTEGER PRIMARY KEY` et, pour le premier passage, l’index couvrant `(document_id, chunk_no)`. Les préfiltres FTS utilisent leurs index virtuels `M5`/`M1`.

## Périmètres et progression

Le traceur utilise `perf_counter_ns` et des `ContextVar`. Il expose IDs, parent, nom métier/routine, début relatif, inclusif, propre, appels, maximum, volumes, statut et détails cache. La durée propre soustrait l’**union** des intervalles enfants, y compris les chevauchements. Les boucles sont agrégées ; aucun événement par vecteur/passage n’est stocké. Limites : 256 lignes et 128 spans actifs par trace, 16 compteurs/détails par ligne, troncature annoncée. Le registre conserve au plus 32 requêtes, dont huit actives ; les traces terminées expirent après 120 secondes.

Recherche affiche le panneau ouvert par défaut, sa préférence persistée, l’étape active et sa durée, un arbre dépliable, les trois coûts propres dominants, les informations de cache/snapshot et la copie JSON, avec zone texte de secours si le presse-papiers est indisponible. Le JSON inclut aussi métriques historiques et compteurs du catalogue. Les étapes ignorées affichent « — » avec leur motif.

`GET /api/search/progress?request_id=…` est interrogé toutes les 250 ms. `ThreadingHTTPServer` traite ce canal indépendamment du thread de recherche et du pool des readers. Les tests vérifient la progression pendant une recherche bloquée, puis l’annulation, y compris d’un SQL long. L’interruption SQLite utilise son progress handler ; un appel natif non interruptible doit revenir avant que l’annulation puisse être constatée. Les réponses/progressions anciennes sont rejetées côté navigateur par un jeton de génération. La panne du canal de progression laisse la réponse finale utilisable.

Les horloges navigateur et serveur restent séparées. Le serveur mesure le handler après lecture de l’ID de corrélation : validation, attente/ouverture du reader, récupération, présentation et premier encodage des résultats. Il exclut accept/backlog socket, dernier encodage de l’enveloppe de diagnostic et écriture socket. L’attente avant handler est inconnue (`null`), jamais présentée comme zéro. Le navigateur mesure séparément préparation avant HTTP, fetch/décodage JSON et construction DOM. Deux frames donnent une borne après une possibilité de peinture pour le premier résultat/état vide ; cette méthode est incluse dans le JSON. Le résidu HTTP moins serveur n’est pas présenté comme une mesure de réseau pur. Les résultats sont insérés avant la construction du diagnostic final.

Les anciens chronomètres restent disponibles et ne sont **pas additifs** :

- `db_acquire_ms` contient attente et ouverture ; `search_total_ms` exclut cette acquisition.
- `presentation_ms` inclut classement et drilldown ; `drilldown_vector_ms` peut inclure un embedding alternatif également compté dans `query_embedding_ms`.
- `metadata_identity_ms` de Recherche contient résolution + hydratation bornée ; celui du coordinateur RAG contient résolution + vérifications d’identité.
- `http_search_total_ms` s’arrête avant l’encodage des résultats ; le nouveau périmètre serveur inclut cet encodage.

`catalog_snapshot_stale=True` signifie exactement que `PRAGMA data_version` ou `total_changes` diffère du snapshot des **compteurs de diagnostic du reader**. Toute écriture peut le déclencher. Il n’atteste ni un catalogue d’identité périmé ni une reconstruction à chaque requête. Les compteurs restent ceux de l’ouverture/du dernier rafraîchissement explicite ; les requêtes interactives ne les recomptent pas. La génération d’identité est indépendante.

## Vecteurs et RAG

Le KNN global était déjà un `embedding MATCH … AND k=…` natif dans `sqlite-vec`, avec jointures sur les clés primaires et tri SQL du top-k. Le drilldown utilise déjà `vec_distance_cosine` natif sur les documents sélectionnés. Il n’y a pas de décodage Python des vecteurs du corpus à supprimer. Les nouveaux spans séparent sérialisation du vecteur de requête, exécution SQL, lecture des lignes, conversion et tri local. Le scan/calcul/top-k interne de `sqlite-vec` reste un périmètre natif indivisible ; le nombre de vecteurs globaux effectivement visités est inconnu. Les lignes rendues ne sont pas présentées comme ce nombre. Aucun changement arbitraire d’architecture ni gain vectoriel revendiqué sur les 3,9–8,6 secondes utilisateur.

Le même traceur couvre la récupération RAG et est joint à ses timings. `expansion_reason` décrit le besoin d’expansion ; `expanded` décrit une expansion effectivement réalisée. À profondeur zéro, des preuves faibles peuvent déclencher une expansion selon les tests existants. Les budgets restent 8 s primaire / 12 s total : une identité dépassant ces budgets peut expliquer un motif renseigné sans appel global ni expansion. Le champ additionnel `expansion_status` distingue notamment budget épuisé, scope strict et absence de nouveaux candidats. Budgets, règles de preuve et abstention sont conservés. Une suppression concurrente entre routage et hydratation est désormais tolérée.

## Mesures synthétiques et validation

La base montée dans cet environnement contient zéro document. Fixture commune avant/après : 2 000 documents, 6 000 passages/vecteurs, chemins accentués, embeddings déterministes de test à trois dimensions. Ce corpus accentué cible le goulet ; ce n’est pas un échantillon du corpus utilisateur. Premier appel = premier appel du processus/reader, **sans purge du cache disque système**. Temps ci-dessous : acquisition + recherche, après activation de la trace dans la version corrigée, hors export JSON.

| Requête | Avant, appels 1 / 2 / 3 (ms) | Après, appels 1 / 2 / 3 (ms) |
|---|---:|---:|
| domino sequence | 352,5 / 360,7 / 349,4 | 15,1 / 6,1 / 5,9 |
| densité MTBE | 370,2 / 383,2 / 347,6 | 33,4 / 6,5 / 6,9 |
| zzqv_unobtainium (vide) | 358,8 / 377,5 / 366,0 | 6,3 / 5,5 / 5,4 |

Identité au premier appel : respectivement 325,5 / 344,7 / 338,9 ms avant, 1,3 / 0,6 / 0,4 ms après. Les neuf listes ordonnées de documents **et leurs scores** sont identiques, avec `[1]`, `[2,1]` et `[]`. Un appel isolé plus lent figure dans les mesures brutes ; trois répétitions ne constituent pas une estimation statistique robuste.

Avec un écrivain synthétique mettant à jour le statut des sources toutes les 20 ms, médianes sur trois appels : 353,3 → 17,0 ms ; 392,7 → 6,6 ms ; 349,7 → 5,7 ms. Le flag des compteurs devient ancien tandis que la résolution d’identité reste réutilisable. Il ne s’agit pas du daemon réel indexant le corpus utilisateur. Le backfill Unicode + ouverture écrivain a pris 68,2 ms sur une copie de cette fixture ; ce coût ponctuel est hors temps de recherche.

Surcoût, 40 paires alternant trace désactivée/active : médianes 5,516 / 6,263 ms, soit +0,747 ms ; export JSON séparé 0,203 ms médian. Cela mesure l’activation du traceur sur le code corrigé, hors transport HTTP et rendu ; aucune extrapolation au poste utilisateur.

Validation : Ruff, compilation Python et contrôle du patch ; tests ciblés Recherche/RAG/HTTP : 69 réussis. Chromium réel : progression avant réponse, arbre, copie JSON de secours, ancien résultat ignoré, panne du canal, annulation et préférence persistée. Suite complète finale : **448 tests réussis, 5 échecs préexistants reproduits sur le commit d’origine**. Les cinq échecs Linux liés aux chemins Windows ont aussi été reproduits sur `61480d6` sans modifications : `test_alpha3_hardening`, `test_document_coverage`, `test_e2e`, `test_multisource_scan`, `test_scanner` (un cas chacun). Validation Windows et mesures du corpus utilisateur restent à faire. Version/commit proviennent du `BUILD_INFO.json` packagé ; un checkout sans cette métadonnée affiche honnêtement `dev` / commit inconnu.

## Reproduction

Dans un shell avec les dépendances du projet et depuis sa racine :

```bash
mkdir -p work/alpha4-before
git archive 61480d6 | tar -x -C work/alpha4-before
python scripts/benchmark_search_diagnostics.py --code-root work/alpha4-before --synthetic work/comparison.db > work/before.jsonl
python scripts/benchmark_search_diagnostics.py --synthetic work/comparison.db --trace --overhead-pairs 40 > work/after.jsonl
python scripts/benchmark_search_diagnostics.py --code-root work/alpha4-before --synthetic work/comparison.db --writer-activity > work/before-writer.jsonl
python scripts/benchmark_search_diagnostics.py --synthetic work/comparison.db --trace --writer-activity > work/after-writer.jsonl
# Index réel, lecture seule ; migration préalable par le daemon à jour :
python scripts/benchmark_search_diagnostics.py --db /chemin/index.db --trace --repeat 3
ruff check .
pytest -q
# Playwright et Chromium disponibles ; CHROMIUM_PATH est facultatif :
node tests/browser_search_diagnostics.mjs
```

Les [mesures brutes et les plans SQL de la fixture](benchmarks/alpha4-search-diagnostics-synthetic.json) accompagnent la livraison. Après redémarrage de la nouvelle version, lancer les trois requêtes dans Recherche, développer le panneau et copier le JSON. Refaire les essais daemon au repos puis actif sur le même index utilisateur avant de conclure sur les temps réels.
