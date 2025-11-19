# 🏀 NBA Awards Predictor — MVP / MIP / 6MOTY / ROTY (Machine Learning Project)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Security](https://img.shields.io/badge/CodeSql-enabled-brightgreen)
![Dependabot](https://img.shields.io/badge/Dependabot-enabled-brightgreen)
![Secret%20Scanning](https://img.shields.io/badge/Secret%20Scanning-active-blue)
![Push%20Protection](https://img.shields.io/badge/Push%20Protection-enabled-purple)
![Contributions](https://img.shields.io/badge/Contributions-welcome-orange)

> **Projet IA & Data Science** visant à prédire les trophées NBA à partir de statistiques avancées et partielles de saison.  
> Basé sur un pipeline complet de feature engineering (percentiles, z-scores, impact metrics) et un apprentissage supervisé.
---

## 🎯 Objectifs
- Créer un **pipeline reproductible** pour prédire les récompenses NBA (MVP, MIP, 6MOTY, ROTY).  
- Exploiter des **statistiques avancées** pour évaluer la performance réelle des joueurs.  
- Démontrer des compétences en **Data Science appliquée, Machine Learning et CI/CD**.  
- Supporter la **saison en cours** via des données partielles.
---

## 🧱 Architecture du projet

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
├── .github/workflows/      # CI/CD — tests automatiques GitHub Actions
├── requirements.txt
└── README.md
```
---

## ⚙️ Installation rapide

```bash
# 1) Créer un environnement virtuel
python -m venv .venv

# 2) Activer l'environnement
# Windows :
.venv\Scripts\activate
# macOS/Linux :
source .venv/bin/activate

# 3) Installer les dépendances
pip install -r requirements.txt

# (Optionnel) Installer les hooks qualité
pre-commit install
```
---

## 🚀 Démarrage rapide (avec données incluses)

```bash
# 1) Construire les features à partir des CSV d'exemple
python scripts/build_features.py \
  --season 2024 \
  --input data/raw/sample_players_2024_partial.csv \
  --teams data/raw/sample_teams_2024_partial.csv \
  --out data/processed/mvp_features_2024.parquet

# 2) Entraîner un modèle MVP (baseline)
python scripts/train_mvp.py \
  --features data/processed/mvp_features_2024.parquet \
  --out models/mvp_random_forest.pkl \
  --metrics models/mvp_metrics.json

# 3) Prédire le classement MVP actuel
python scripts/predict_mvp.py \
  --features data/processed/mvp_features_2024.parquet \
  --model models/mvp_random_forest.pkl \
  --topk 10 \
  --out data/processed/mvp_predictions_2024.csv
```
---

## 🧠 Feature Engineering
- **Percentiles Ligue** : `pts_pctile`, `ast_pctile`, `reb_pctile`, `ts_pctile`, etc.  
- **Z-Scores par position** pour comparer les profils de joueurs équivalents.  
- **Impact metrics** : combinaisons statistiques (`TS%`, `USG%`, `BPM`, `WS`, `VORP`).  
- **Features contextuelles** : minutes, rôle (starter/bench), pourcentage de victoires de l’équipe.
---

## 🧩 Approche de modélisation

- Modèles ML supervisés : Random Forest, Gradient Boosting, ExtraTrees.
- Pipeline de ranking pour produire un classement type votants médias.
- Validation croisée, analyse d’importance (SHAP prévu).
- Entraînement reproductible via scripts CLI.
---

## 🧪 Évaluation & CI/CD
- Métriques : AUC, F1, LogLoss, Spearman Rank Corr.
- Tests unitaires via Pytest.
  - CI automatisée via GitHub Actions (tests + sécurité).
- Sécurité GitHub activée :
  - CodeQL
  - Dependabot
  - Secret Scanning
  - Push Protection
---

## 📊 Exemple de résultats (mock)

| Joueur | Équipe | Position | Score MVP |
|--------|---------|-----------|------------|
| Nikola Jokic | DEN | C | 0.92 |
| Luka Doncic | DAL | PG | 0.89 |
| Jayson Tatum | BOS | SF | 0.86 |
---

## 👨‍💻 Auteur

**Luc Renaud**  
Master 1 — Ingénierie Data & IA (ECE Paris)
Passionné de NBA, ML, et Data Science appliquée au sport
[lucR8](https://github.com/lucR8)
---

## 🧩 Licence

Ce projet est distribué sous licence **MIT**.  
Voir le fichier [LICENSE](./LICENSE) pour plus d'informations.
---

