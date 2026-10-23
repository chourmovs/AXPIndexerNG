# PR-B — USearch HNSW qualification (read-only)

Ce banc compare un index USearch HNSW à un calcul exhaustif cosinus FP32
sur **exactement le même sous-ensemble** de vecteurs existants.

La base originale est ouverte exclusivement en mode lecture. Les embeddings
ne sont ni recalculés ni remplacés. Aucun backend ANN n'est activé en production.

## Installation optionnelle

```powershell
python -m pip install -r requirements-ann-benchmark.txt
```

## Mesures 50k / 100k / corpus complet

```powershell
python scripts/benchmark_ann.py --db "C:\chemin\axpindex.db" --max-vectors 50000 --output "C:\temp\ann-50k.json"
python scripts/benchmark_ann.py --db "C:\chemin\axpindex.db" --max-vectors 100000 --output "C:\temp\ann-100k.json"
python scripts/benchmark_ann.py --db "C:\chemin\axpindex.db" --max-vectors 9999999 --output "C:\temp\ann-full.json"
```

Pour comparer HNSW: exécuter les mêmes suites avec
`--connectivity 16/24/32`, `--expansion-add 64/128/256`,
`--expansion-search 32/64/128/256` et `--dtype f32/f16`.
Chaque invocation construit un index indépendant, sans altérer le catalogue.

Pour éviter que le rappel soit biaisé par les requêtes constituées de vecteurs
déjà indexés, préférer `--queries-json` avec le format JSON produit par
`scripts/benchmark_vectors.py --save-vectors` (le modèle de requête doit
correspondre à la signature de la BDD).

**Précautions de mesure :** séparer le premier run du reste, documenter le daemon,
la RAM, les délais d'I/O, et répéter les recherches avec le même jeu de queries.
Les quantiles avec 20 requêtes sont indicatifs, pas une garantie de p95.
Les performances d'un index sur les 50k premières lignes ne représenteraient
pas nécessairement le corpus : le sampler de PR-B répartit les points sur toutes
les positions triées par rowid.

## Critères de décision

- Recall@20 de recherche ANN vs exhaustive au moins 98 % pour les
  embeddings des requêtes métier.
- Mesure p50/p95, RSS, temps de construction, taille du sidecar,
  validation de persistance et comparaison exhaustive sqlite-vec (corpus complet).
- Pas de migration ni de changement de route tant que la qualité
  documentaire (notamment identité produit chimique) n'a pas été qualifiée.

**Hors de cette PR :** SQLite Vec1 IVFADC, FAISS, filtres par Skill dans
l'ANN, crash recovery/rollover atomique, réindexation, integration en runtime.
Ces points nécessitent un second niveau de qualification avant la PR-C.
