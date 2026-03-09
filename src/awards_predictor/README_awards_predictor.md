# awards_predictor (src)

Package “production” du projet **NBA Awards Predictor**.

Objectif : exécuter l’inférence (prédire une saison future / en cours) à partir :
- d’un historique **1996–2025** déjà buildé en `data/processed/...`
- d’un snapshot *target* (fetch + build) en `data/target/<year>/asof_<date>/...`
- de modèles entraînés et exportés dans `data/experiments/...`

> Aucun `pip install -e .` requis : les scripts bootstrappent automatiquement le dossier `src/`
> dans le `PYTHONPATH`. Le projet est **directement runnable après clonage GitHub**.

---

## 1) Pré-requis data

### Historique (déjà construit)
Attendu :
```
data/processed/players/final/all_years_enriched.parquet
```

### Target snapshot (saison à prédire)
Créé via *fetch* + *build* :
```
data/target/<year>/asof_YYYY-MM-DD/raw/...
data/target/<year>/asof_YYYY-MM-DD/build/players/final/all_years_with_bio.parquet
```

---

## 2) Pipeline target season (fetch + build)

Depuis la racine du repo :

```bash
# Fetch snapshot (Basketball-Reference + NBA bios)
python -m scripts.fetch.run_target_season --year 2026

# Build snapshot (regular + percentiles + final + merge bio)
python -m scripts.build.run_target_season --year 2026
```

---

## 3) Prédire une saison (inference)

### Modèles baseline (logistic regression)
Runs attendus dans :
```
data/experiments/logreg_baseline/<award>/<run_id>/
```

Commande :
```bash
python src/awards_predictor/predict/predict_season.py   --year 2026   --models-root data/experiments/logreg_baseline   --model-family baseline
```

---

### Modèles tree / boosting (XGBoost, etc.)
Runs attendus dans :
```
data/experiments/tree_models/xgb/<award>/<run_id>/
```

Commande :
```bash
python src/awards_predictor/predict/predict_season.py   --year 2026   --models-root data/experiments/tree_models/xgb   --model-family tree
```

---

## 4) Outputs

Toutes les prédictions sont exportées dans le snapshot target :

```
data/target/2026/asof_YYYY-MM-DD/predictions/
```

Fichiers générés :
- `<award>_top10.csv` : top-K candidats par trophée
- `all_awards_top10.parquet` : scores complets tous joueurs / awards
- `prediction_meta.json` : traçabilité (snapshot, date, modèles, features)

---

## 5) Hypothèses & garanties

- **Pas de look-ahead bias** (features calculées à date *as-of*).
- Les features `prev_*` pour la saison target sont calculées à partir :
  - historique 1996–2025
  - + snapshot target partiel
- Chemins **repo-relatifs**, compatibles GitHub / CI / reviewers.

---

## 6) Organisation interne (src)

```
src/awards_predictor/
├── config.py
├── io/
│   └── paths.py
├── features/
│   ├── columns.py
│   ├── eligibility.py
│   └── build_matrix.py
├── predict/
│   └── predict_season.py
└── README.md
```

---

## Auteur

**Luc Renaud**  
Data & AI Engineer — NBA Analytics  
Projet académique & personnel (ML appliqué au sport)
