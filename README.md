
# NBA Awards Predictor — MVP/MIP/6MOTY/ROTY (Starter)

> Projet Data/IA pour prédire les trophées NBA et le champion à partir de stats partielles.
> **Version initiale incluse** : pipeline MVP (features percentiles + RandomForest).

## 🎯 Objectifs
- Construire un pipeline reproductible pour prédire les récompenses (MVP d'abord).
- Utiliser des **stats avancées** et des **percentiles** pour comparer les joueurs à la ligue.
- Supporter la saison en cours (stats **partielles**).

## 🧱 Architecture
```
nba-awards-predictor/
├── data/
│   ├── raw/                # Données sources (CSV simulés inclus)
│   └── processed/          # Données featurisées
├── notebooks/              # EDA / prototypes
├── scripts/                # CLI : fetch/build/train/predict
├── src/awards_predictor/
│   ├── data/               # Collecte et IO
│   ├── features/           # Feature engineering (percentiles, z-scores, etc.)
│   ├── models/             # Entraînement / persistance
│   ├── evaluation/         # Métriques / évaluation
│   └── viz/                # Dashboard (Streamlit placeholder)
├── tests/                  # Pytest
├── models/                 # Modèles sauvegardés (.pkl)
├── .pre-commit-config.yaml # Qualité
├── requirements.txt
└── README.md
```

## ⚙️ Installation rapide
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pre-commit install  # optionnel
```

## 🚀 Démarrage rapide (avec les CSV inclus)
```bash
# 1) Construire les features à partir des CSV d'exemple
python scripts/build_features.py --season 2024 --input data/raw/sample_players_2024_partial.csv --teams data/raw/sample_teams_2024_partial.csv --out data/processed/mvp_features_2024.parquet

# 2) Entraîner un modèle MVP (RandomForest)
python scripts/train_mvp.py --features data/processed/mvp_features_2024.parquet --out models/mvp_random_forest.pkl --metrics models/mvp_metrics.json

# 3) Prédire le classement MVP actuel
python scripts/predict_mvp.py --features data/processed/mvp_features_2024.parquet --model models/mvp_random_forest.pkl --topk 10 --out data/processed/mvp_predictions_2024.csv
```

## 🧪 Données d'exemple
- `data/raw/sample_players_2024_partial.csv` : ~20 joueurs avec stats partielles simulées (PTS, AST, REB, TS%, USG%, WS, BPM, VORP, minutes, etc.).
- `data/raw/sample_teams_2024_partial.csv` : win% et ratings basiques par équipe.

> Vous pouvez remplacer ces CSV par vos exports (Basketball-Reference ou `nba_api`).

## 🧠 MVP : Features (extrait)
- Percentiles ligue : `pts_pctile`, `ast_pctile`, `reb_pctile`, `ts_pctile`, …
- Intensité : `usg_pct`, minutes, GP
- Impact : `ws`, `bpm`, `vorp`, `team_win_pct`
- Contexte : position, âge (placeholder), starter/bench (placeholder)

## 📏 Évaluation
- Binaire "MVP vs autres" sur historique (à brancher lorsque vos labels sont disponibles).
- Sur dataset partiel, on fait un **ranking** des probabilités pour obtenir un **Top 10 MVP**.
- Métriques exportées : AUC, LogLoss (si labels), importance des features.

## 🗺️ Roadmap (12 semaines)
- **S1–S2** : collecte automatique (`nba_api`) + EDA.
- **S3–S4** : engineering percentiles & positions (z-scores par poste).
- **S5–S6** : modèles MVP (RF/XGBoost) + cross-val + SHAP.
- **S7–S8** : MIP (diff N vs N-1), 6MOTY (bench), ROTY (rookies).
- **S9–S10** : prédiction Playoffs (simulateur séries / Elo simple).
- **S11–S12** : Streamlit dashboard + packaging (Docker) + README final.

## 🧭 Commandes utiles (Windows / PowerShell)
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\build_features.py --help
python scripts\train_mvp.py --help
python scripts\predict_mvp.py --help
```

## 📚 Bonnes pratiques
- Données immuables dans `data/raw/`, dérivées dans `data/processed/`.
- Scripts **idempotents**, logs clairs, erreurs explicites.
- Tests unitaires pour les transformations clés (percentiles, agrégations).

---

**Crédit & licence** : Projet éducatif. Données NBA © sources respectives.
