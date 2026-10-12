Corrections AXPIndexerNG — audit du 12 octobre 2026

Le code a été audité sur le commit 7a93dee. Aucun AGENTS.md n’est présent dans le dépôt ou son répertoire parent. La BDD cloud fournie est vide : les mesures suivantes sont exclusivement synthétiques, sous Linux, et ne qualifient pas le poste Windows ni les modèles Intel SYCL.

Causes confirmées dans le code

- `metadata_identity_ms` couvre `resolve_identity_documents` puis la matérialisation d’un premier chunk par document. La matérialisation utilise déjà l’index UNIQUE(document_id,chunk_no). Le résolveur effectuait un SELECT de toutes les métadonnées des documents, puis normalisation Unicode, extraction des termes/identifiants et classement Python pour chaque document, à chaque requête. Il ne lisait pas de vecteurs ni de corps dans ce scan. Les timings du poste ne permettent pas de confirmer que cette seule cause explique exactement ses 46–49 secondes.
- Le heartbeat était déjà produit par un thread indépendant toutes les deux secondes, avec écriture atomique et retries bornés. Le watchdog actuel conserve un seuil de 90 secondes et veto toute relance si le verrou OS est occupé. Il évite donc les doublons, mais ne récupère pas le détenteur du verrou réellement bloqué. Le cycle à 61 secondes signalé dans les logs ne correspond pas à ce code : version installée, contention de publication, blocage du GIL et origine des arrêts restent à vérifier sur Windows.
- Le contrôle relisait une commande persistante ancienne à chaque nouvelle instance ; un ancien stop pouvait être rejoué. De plus `stop_daemon(intentional=False)` appelait `send_control('stop')`, qui réécrivait systématiquement l’état désiré à stopped.
- Le coordinateur RAG excluait l’expansion si `search_depth` était non nul. Sa lecture ciblée examine déjà tous les passages du document ; un seul chunk dans les logs peut donc refléter un document réellement indexé avec un seul fragment, une sélection ultérieure, ou des preuves faibles. L’index utilisateur manque pour les départager.
- Backend vectoriel : sqlite-vec, vec0, cosine, KNN exhaustive. La recherche globale calcule la distance sur le corpus natif, puis hydrate au plus k résultats (k≤500). La requête ciblée utilise l’index chunks(document_id,chunk_no) et des accès rowid au vecteur. Elle décodait les blobs et recalculait normes et produits scalaires en Python. Les diagnostics faisaient également count(*) sur vec0 et un scan de l’index FTS en ligne.

Modifications et compatibilité

- Index FTS5 additionnel `document_identity_fts`, uniquement sur les métadonnées documentaires, rempli une fois dans une transaction. Triggers d’ajout, mise à jour et suppression, y compris les suppressions en cascade. Création relançable ; version de schéma et signature vectorielle inchangées. Aucun VACUUM, purge, suppression de BDD, changement de modèle ou réindexation des chunks/vecteurs. Les lecteurs anciens sans cet index gardent le chemin exhaustif jusqu’à la prochaine connexion normale en écriture.
- Candidats non plafonnés avant le classement historique ; préfixes pour les variantes singulier/pluriel et composés. Les identifiants avec séparateurs utilisent une sélection SQL conservatrice. Un marqueur FTS conserve les documents Unicode dans le surensemble de candidats pour les cas où unicode61 ne reproduit pas NFKD. Les recherches très larges, les identifiants et les corpus majoritairement Unicode peuvent donc encore être plus coûteux ; aucune pertinence n’est sacrifiée pour masquer ce coût.
- Calcul des distances ciblées en code natif `vec_distance_cosine`, sans décodage Python. Les vecteurs nuls gardent la distance 1 historique. Un vecteur de question est réutilisé pour le drilldown Search et le chemin global RAG. Les diagnostics de cardinalité sont des snapshots au démarrage du lecteur, explicitement marqués périmés après changement ; ils ne rescannent plus le corpus à chaque requête. Aucun serveur vectoriel ni ANN ajouté.
- Instance UUID, PID, création du processus et chemin BDD dans l’état ; horloge monotone pour la fraîcheur et les checkpoints worker. Les commandes obsolètes ou ciblant une autre instance sont ignorées. Chaque contrôle journalise expéditeur, raison et instance.
- Watchdog : démarrage explicite de 180 s, heartbeat inchangé à 90 s, progression indépendante du worker (30 min sans checkpoint), challenge de santé de 5 s hors du thread Tk, réponse du worker dans un fichier distinct de l’état. Un worker répondant, un changement d’instance, un arrêt intentionnel ou une identité non vérifiable interdit la terminaison. Arrêt propre puis terminaison seulement du PID authentifié ; verrou OS revérifié avant relance. Backoff 60/120/240 s, au plus trois tentatives en quinze minutes. Les tâches planifiées sont authentifiées aussi par executable, cwd, module et argument BDD. Un traitement natif dépassant 30 min sans checkpoint demeure une limite à qualifier sur le poste.
- Expansion RAG bornée : tous les passages du document sont déjà examinés au primaire ; en cas de faiblesse, au plus une requête de contenu alternative par document (max. trois), puis au plus une recherche hybride globale, avec au plus trois nouveaux documents. Un seed hybride existant est réutilisé. `search_depth=1` demande réellement cette exploration. Les seuils d’answerability, citations et abstention ne sont pas abaissés. Budget de 8 s pour l’étape locale et 12 s pour démarrer les étapes d’expansion ; ces budgets sont vérifiés entre opérations et ne constituent pas une interruption dure des appels natifs SQLite/embedding.
- Mesures : attente du pool de lecteurs, ouverture, acquisition totale, embedding, FTS, vecteurs, identité, expansion locale/globale, contexte, chargement modèle, prefill natif, premier contenu généré, premier contenu visible, decode et total. La voie de génération rejette busy au lieu de mettre en file. L’inférence réutilise déjà le serveur actif via ensure_loaded ; cette politique est conservée.
- Les deltas vides, rôles et métadonnées SSE ne définissent aucun premier token. Premier contenu généré (raisonnement compris) et premier texte visible sont distincts. Le mode non streaming ne prétend plus mesurer un TTFT. L’UI nomme la mesure « First visible text ». Un premier texte mesuré avant le prefill natif est marqué `ttft_prefill_inconsistent` ; sans les événements originaux, les 38 ms SmolLM3 ne peuvent pas être expliqués de façon confirmée. Le TTFT n’est plus utilisé pour inventer une durée de prefill manquante.

Mesures réellement exécutées

Corpus : 23 363 documents, 467 255 chunks et vecteurs, dimension 384, environ 788 Mo de fichier SQLite. Embeddings déterministes artificiels ; aucune génération LLM. Trois répétitions par scénario, avec nouvelle connexion puis connexion réutilisée. Le cache du système n’a pas été vidé : « nouvelle connexion » n’est pas un démarrage à froid du disque. Les p95 ne sont pas publiés pour n=3. L’index métadonnées occupe 1 523 712 octets de pages SQLite ; remplissage mesuré en 0,082 s sur ce corpus.

Toutes les durées ci-dessous sont des médianes en millisecondes, avant → après.

| Scénario | Lecteur | Identité | Search totale | Retrieval RAG |
| --- | --- | --- | --- | --- |
| Nom de document | nouvelle connexion | 2256.19 → 0.59 | 2771.8 → 518.7 | 2180.2 → 4.8 |
| Nom de document | chaud | 2324.34 → 0.39 | 2858.6 → 422.2 | 2307.0 → 4.8 |
| Hybride | nouvelle connexion | 2377.87 → 0.44 | 2884.6 → 557.3 | 2803.5 → 408.5 |
| Hybride | chaud | 2426.98 → 0.22 | 2937.5 → 432.9 | 2821.1 → 417.3 |
| Question sur identité | nouvelle connexion | 2478.59 → 0.65 | 2987.4 → 544.3 | 2337.4 → 5.1 |
| Question sur identité | chaud | 2383.62 → 0.66 | 2910.8 → 455.6 | 2454.9 → 6.1 |
| Plusieurs passages | nouvelle connexion | 2375.45 → 0.73 | 2889.3 → 540.1 | 2403.7 → 5.2 |
| Plusieurs passages | chaud | 2474.70 → 0.50 | 3009.7 → 430.1 | 2479.9 → 5.7 |
| Sans réponse | nouvelle connexion | 2344.81 → 0.41 | 2849.0 → 535.7 | 3498.3 → 433.5 |
| Sans réponse | chaud | 2405.31 → 0.27 | 2926.9 → 411.5 | 3465.9 → 434.4 |

Les durées détaillées par étape sont conservées dans `docs/benchmarks/october-recovery-summary.json`. Les quatre scénarios avec réponse gardent leurs décisions positives ; le scénario absent s’abstient à toutes les répétitions. Le scénario plusieurs passages contient les faits pression et stockage dans des passages séparés du même document ; un test distinct vérifie l’expansion vers un second document. Il ne s’agit pas d’une évaluation de réponses générées ou de citations sur le corpus utilisateur. Les objectifs identité <1 s et recherche chaude <3 s sont atteints sur ce fixture sélectif seulement.

Validation et limites

17 nouveaux tests de régression couvrent migration/CRUD/Unicode, plan vectoriel et vecteurs nuls, snapshots, heartbeat indépendant, contrôle ciblé, horloge, backoff, identité avant terminaison, réponse de santé et changement d’instance, preuves faibles, search_depth, expansion multidocument, abstention et SSE/TTFT. Ruff et git diff --check passent. La suite complète avant modification donne 418 succès et cinq échecs ; après modification, 435 succès et les mêmes cinq échecs. Les tests HTTP passent avec accès loopback. Les échecs préexistants concernent test_alpha3_hardening, test_document_coverage, test_e2e, test_multisource_scan et test_scanner, liés aux chemins Windows/ingestion sous Linux ; ils sont hors du périmètre de cette correction. Les qualifications Windows, l’Intel GPU, les timings de modèles et la BDD réelle ne sont pas disponibles.

Actions sur Windows

Déployer la branche, fermer proprement les anciennes instances puis lancer normalement le daemon/tray. La première connexion en écriture ajoutera automatiquement l’index de métadonnées. Conserver la BDD, les modèles et les sources ; ne pas demander de réindexation intégrale. Vérifier dans les logs la version déployée, les UUID/PID, les expéditeurs des stop, les résultats des challenges et la progression pendant un fichier long. Contrôler ensuite Search par nom, question factuelle, plusieurs passages et question absente sur la BDD réelle, avec les mêmes répétitions et les durées par étape. Les tâches planifiées et les seuils de traitement long doivent être qualifiés sur le poste.

Reproduction synthétique :

```sh
python scripts/benchmark_recovery.py --db work/representative.db --output work/recovery.json --repeats 3
```

Utiliser exclusivement un fichier dédié au fixture. Pour la référence avant modification, lancer le script dans le checkout 7a93dee ; pour l’après, ouvrir une fois le fixture via `axp_core.database.connect` en écriture afin d’installer l’index additif, puis lancer le script sur la branche corrigée.
