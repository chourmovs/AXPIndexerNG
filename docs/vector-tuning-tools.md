# Mesurer avant de régler la recherche

Ces outils mesurent le moteur existant ; ils ne modifient ni le classement de production ni le catalogue. Utiliser l'environnement Python du projet avec ses dépendances runtime, notamment sqlite-vec et psutil. Exécuter depuis la racine du dépôt.

## 1. Baseline et vecteurs reproductibles

Choisir 10 à 30 requêtes représentatives : titres exacts, questions sémantiques, requêtes rares, absences de réponse. Répéter `--query` pour chacune. Le modèle est chargé d'après la signature du catalogue. L'embedding se fait avant les mesures SQL.

```sh
python scripts/benchmark_vectors.py --db /chemin/catalog.db --output work/baseline.json --save-vectors work/query-vectors.json --repeats 20 --top-k 100 --full-search --query "pression réacteur" --query "stockage" --environment-label "daemon au repos"
```

Le fichier de vecteurs doit être nouveau. Le JSON contient les SQL, EXPLAIN QUERY PLAN, version sqlite-vec, dimension, volumétrie, réglages effectifs, temps execute/fetchall, CPU et lectures physiques du processus. Les variantes combinée et séparée doivent produire les mêmes résultats complets ; une divergence arrête le benchmark. Les statistiques excluent le premier couple de chaque requête.

`--full-search` ajoute les temps FTS, scoring, identité et présentation. Cette recherche utilise un reader déjà échauffé. Le top-k du benchmark concerne seulement les variantes vectorielles isolées ; il ne règle pas le top-k interne de la recherche complète.

## 2. Matrice SQLite / top-k

```sh
python scripts/benchmark_vector_tuning.py --db /chemin/catalog.db --vectors-json work/query-vectors.json --output-dir work/tuning-idle --cache-mib 64 128 --mmap-mib 0 256 1024 --top-k 50 100 200 --repeats 20 --rounds 2 --seed 42 --environment-label "daemon au repos"
```

Le répertoire de sortie doit être nouveau et distinct de l'arborescence contenant le catalogue. La matrice ci-dessus comporte 18 configurations par tour ; commencer avec une seule requête et moins de valeurs si les appels coûtent plusieurs secondes. Un tour effectue, pour chaque requête et configuration, `repeats + 1` couples combinée/séparée, puis relève les diagnostics. Les diagnostics effectuent eux-mêmes une recherche et un comptage.

Fichiers produits après chaque configuration terminée :

- `raw.json` : mesures et diagnostics complets, réglages demandés et effectifs ;
- `stages.csv` : une ligne par tour, configuration, requête et étape ;
- `report.md` : comparaison p50/p95 lisible.

Chaque configuration utilise une nouvelle connexion read-only ; cache_size et mmap_size restent locaux à cette connexion. L'ordre des configurations est mélangé avec une graine reproductible ; les variantes alternent dans chaque paire. Les configurations ultérieures bénéficient potentiellement du cache OS. Aucun vidage de cache disque n'est effectué. `mmap_size` peut être plafonné par SQLite : lire la valeur effective du JSON.

Le contrôle `same_k_order_and_distances_unchanged` compare les identifiants ordonnés et distances au premier échantillon du même top-k et de la même requête. Une différence peut aussi venir d'une écriture concurrente. Les paires partagent un snapshot ; toute la campagne ne partage pas un snapshot unique. Ce contrôle ne mesure pas la pertinence et n'est pas un recall ANN.

## 3. Latence HTTP et concurrence

Démarrer le serveur normalement, puis écrire dans `work/queries.json` une liste JSON de chaînes correspondant aux requêtes précédentes.

```sh
python scripts/benchmark_search_load.py --url http://127.0.0.1:8765 --queries-json work/queries.json --output work/http-idle.json --concurrency 1 2 4 --repeats 20 --warmups 1 --timeout 120
```

La sonde appelle `/api/search?q=...&explain=1`. Elle conserve les temps internes renvoyés, la latence client p50/p95, le débit réussi et chaque erreur. Les warmups séquentiels sont séparés. La concurrence est celle d'un client en boucle fermée : aucun taux d'arrivée imposé, aucune modification de l'ordonnancement interne FTS/vecteurs. Les niveaux sont exécutés successivement ; les caches restent chauds. Le fichier de sortie doit être nouveau.

Pour le rendu réel dans Chromium, utiliser l'outil déjà présent :

```sh
node scripts/benchmark_search_browser.mjs --url http://127.0.0.1:8765 --output work/browser-idle.json --repeats 7 --queries-json work/queries.json --chrome /chemin/chromium
```

## 4. Décider

Comparer d'abord à top-k identique, sur le même catalogue et avec le daemon dans le même état. Refaire ensuite la campagne pendant un scan normal, avec de nouveaux noms de sortie. Ne pas exécuter plusieurs benchmarks simultanément.

- KNN lent, enrichissement rapide : calcul/parcours vectoriel dominant ; un essai ANN peut devenir pertinent.
- Combinée nettement plus lente que KNN + enrichissement : examiner le plan des jointures.
- mmap/cache plus rapides sur plusieurs tours et requêtes : retenir le réglage seulement après validation HTTP sous charge ; le budget cache se multiplie par le nombre de readers.
- p50 correct mais p95 dégradé en concurrence : examiner contention et activité du daemon.
- top-k réduit : valider la pertinence du classement hybride final avant tout changement de production.

Le p95 utilise le rang supérieur observé ; sur moins de 20 répétitions il est peu informatif. Ces outils n'implémentent pas d'index ANN ni de benchmark de recall annoté. Les relevés de processus ne sont pas des compteurs de défauts de cache SQLite. Un nouveau reader n'est pas une mesure de disque froid.
