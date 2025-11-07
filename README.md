# 🏀 NBA Awards Predictor — MVP / MIP / 6MOTY / ROTY (Machine Learning Project)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Build](https://img.shields.io/github/actions/workflow/status/lucR8/nba-awards-predictor/tests.yml?label=Tests)
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
# Créer un environnement virtuel
python -m venv .venv && source .venv/bin/activate  # Windows : .venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# (Optionnel) Configurer la qualité de code
pre-commit install
```

---

## 🚀 Démarrage rapide (avec données incluses)

```bash
# 1) Construire les features à partir des CSV d'exemple
python scripts/build_features.py --season 2024 --input data/raw/sample_players_2024_partial.csv --teams data/raw/sample_teams_2024_partial.csv --out data/processed/mvp_features_2024.parquet

# 2) Entraîner un modèle MVP (baseline)
python scripts/train_mvp.py --features data/processed/mvp_features_2024.parquet --out models/mvp_random_forest.pkl --metrics models/mvp_metrics.json

# 3) Prédire le classement MVP actuel
python scripts/predict_mvp.py --features data/processed/mvp_features_2024.parquet --model models/mvp_random_forest.pkl --topk 10 --out data/processed/mvp_predictions_2024.csv
```

---

## 🧠 Feature Engineering
- **Percentiles Ligue** : `pts_pctile`, `ast_pctile`, `reb_pctile`, `ts_pctile`, etc.  
- **Z-Scores par position** pour comparer les profils de joueurs équivalents.  
- **Impact metrics** : combinaisons statistiques (`TS%`, `USG%`, `BPM`, `WS`, `VORP`).  
- **Features contextuelles** : minutes, rôle (starter/bench), pourcentage de victoires de l’équipe.

---

## 🧩 Approche de modélisation

- Utilisation d’approches **supervisées de classification et de ranking** pour prédire la probabilité d’obtention de trophée.  
- Itérations prévues :
  - Sélection automatique de features (mutual information, SHAP).  
  - Comparaison de plusieurs familles de modèles (forêts aléatoires, boosting, réseaux légers).  
  - Validation croisée et ajustement de l’importance des stats par position.  
- Les labels sont simulés dans cette version, et seront remplacés par les **récompenses officielles** dès leur publication.

---

## 🧪 Évaluation & CI/CD
- Métriques : AUC, LogLoss, F1, Spearman Rank Corr (selon les labels disponibles).  
- Tests unitaires `pytest` exécutés automatiquement via **GitHub Actions** à chaque push.  
- Statut CI : ![Build](https://img.shields.io/github/actions/workflow/status/lucR8/nba-awards-predictor/tests.yml?label=Tests)

---

## 📊 Exemple de résultats (mock)

| Joueur | Équipe | Position | Score MVP |
|--------|---------|-----------|------------|
| Nikola Jokic | DEN | C | 0.92 |
| Luka Doncic | DAL | PG | 0.89 |
| Jayson Tatum | BOS | SF | 0.86 |

---

## 🗺️ Roadmap (12 semaines)

| Phase | Période | Objectif principal |
|-------|----------|--------------------|
| **S1–S2** | Collecte via `nba_api` + EDA | ✅ |
| **S3–S4** | Feature engineering avancé | 🔄 |
| **S5–S6** | Entraînement et ranking multi-modèles | 🔜 |
| **S7–S8** | Ajout MIP / 6MOTY / ROTY | 🔜 |
| **S9–S10** | Simulation Playoffs / Elo | 🔜 |
| **S11–S12** | Streamlit Dashboard + Docker | 🔜 |

---

## 👨‍💻 Auteur

**Luc Renaud**  
🎓 Master 1 — Ingénierie Data & IA (ECE Paris)  
🏀 Passionné de NBA, Machine Learning et Data Science appliquée au sport  
📫 [lucR8](https://github.com/lucR8)

---

## 🧩 Licence

Ce projet est distribué sous licence **MIT**.  
Voir le fichier [LICENSE](./LICENSE) pour plus d'informations.
