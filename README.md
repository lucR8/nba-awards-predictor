# 🏀 NBA Awards Predictor — MVP / DPOY / ROY / SMOY / MIP  
### Data Engineering & Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-brightgreen)
![Reproducibility](https://img.shields.io/badge/Reproducibility-Guaranteed-blue)
![Data%20Engineering](https://img.shields.io/badge/Data%20Engineering-Core-orange)
![ML](https://img.shields.io/badge/Machine%20Learning-Ranking-purple)

> **End-to-end Data Engineering & Machine Learning project** focused on predicting NBA individual awards  
> (MVP, Defensive Player of the Year, Rookie of the Year, Sixth Man, Most Improved Player)  
> using historical multi-source data, robust entity resolution, and season-aware ranking models.

---

## 🎯 Project goals

The main objective of this project is to design a **robust, reproducible and auditable pipeline** able to:

- Consolidate **historical NBA player data (1996 → 2025)** from heterogeneous sources
- Solve a **multi-source entity resolution problem** without a shared unique identifier
- Train and evaluate **season-aware ranking models** for multiple NBA awards
- Analyze **model limitations**, leakage risks, and narrative-driven failure cases
- Prepare a **future-season prediction workflow** (no labels available)

This project deliberately focuses on **real-world constraints**:
partial data, temporal consistency, imbalanced targets, and award-specific business rules.

---

## 📦 Data sources

### Primary source
- **Basketball-Reference**
  - Regular season & playoffs
  - Tables:
    - `per_game`, `totals`, `per_36`, `per_100`
    - `advanced`, `shooting`, `adjusted_shooting`
  - **Intra-season percentiles** computed to ensure inter-era comparability

### External data
- Player biographical data (age, height, weight, country, draft, position)
- Advanced impact metrics:
  - **RAPTOR**
  - **LEBRON**
  - **MAMBA**

⚠️ No universal player ID exists across sources →  
**safe name normalization + fuzzy matching + temporal guards** are used.

---

## 🏗️ Data engineering pipeline

The full build pipeline is **CLI-driven, idempotent, and fully logged**:

```bash
python -m scripts.build.players.run_all --start 1996 --end 2025
```

### Pipeline steps
1. **Basketball-Reference build**
   - Regular season + playoffs
   - Percentile computation (season-level)
2. **Bio merge**
   - Safe joins with coverage reports  
   - Outputs:
     - `all_years_with_bio.parquet`
     - `bio_merge_report.xlsx`
3. **External metrics merge**
   - Temporal validation (no look-ahead)
   - Outputs:
     - `all_years_enriched.parquet`
     - `metric_merge_report.xlsx`
4. **Assertions, logs, and audit artifacts** at each critical step

The pipeline guarantees:
- No season leakage
- Full traceability
- Reproducible builds

---

## 📊 Notebooks & analysis workflow

The project is structured around **sequential, responsibility-driven notebooks**:

| Notebook | Purpose |
|--------|--------|
| `01_exploration.ipynb` | Exploratory analysis (read-only) |
| `02_build_df_clean.ipynb` | Label creation, feature derivation, audits |
| `03_build_feature_matrices.ipynb` | Numerical matrices (era-aware) |
| `04_award_datasets.ipynb` | Award-specific datasets & eligibility |
| `05_modeling_logreg_baseline.ipynb` | Logistic ranking baseline |
| `06_modeling_tree_models.ipynb` | Tree-based models (GBDT) |
| `07_error_analysis_audit.ipynb` | Cross-award audit & failure analysis |

All splits are **time-aware** and evaluated at the **season level**.

---

## 🧠 Award-specific logic

Each award is modeled with **distinct business rules**:

- **ROY**: rookies only, no year-over-year features
- **SMOY**: bench role constraint (games started vs played)
- **DPOY**: defensive-focused metrics, known observability limits
- **MIP**: year-over-year dynamics (high variance)
- **MVP**:
  - full population
  - extreme class imbalance
  - strong narrative component

---

## 🧪 Evaluation methodology

The task is framed as a **ranking problem**, not pure classification.

Main metrics:
- **MRR (Mean Reciprocal Rank)**
- **Top-K hit rate** (Top-1, Top-3, Top-5, Top-10)
- Winner **median / worst rank** across seasons

This evaluation reflects real voting dynamics:
> *“How high does the true winner appear in the ranked candidate list?”*

---

## 🔍 Model comparison & audit

A dedicated audit compares **baseline vs tree-based models**:

Key findings:
- Tree models improve performance on **MVP and DPOY**
- No benefit (or degradation) on **SMOY and MIP**
- **ROY** shows split-dependent sensitivity (small cohorts)
- Near-perfect MVP scores require **cautious interpretation**

Worst-case season analysis highlights:
- narrative-driven winners
- defensive impact under-captured by boxscore data
- structural limits of purely statistical modeling

---

## 📁 Project structure

```
nba-awards-predictor/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       └── reports/
├── notebooks/
├── scripts/
│   └── build/
├── src/
│   └── awards_predictor/
├── models/
├── tests/
├── .github/workflows/
└── README.md
```

---

## 📤 Outputs

The pipeline produces:
- Clean, enriched Parquet datasets
- Excel audit reports for merges
- CSV summaries for:
  - global metrics per award
  - winner ranking behavior
- Fully reproducible experiment artifacts

---

## 🚧 Known limitations

- Awards with strong **narrative components** cannot be fully captured statistically
- Defensive impact remains partially observable
- Tree models require careful auditing to avoid overconfidence
- No media / voting data integrated (by design)

These limitations are explicitly analyzed and documented.

---

## 🧠 Future work

- Dedicated **future-season prediction script** (no labels)
- Feature ablation studies (stats-only vs advanced-only)
- Optional integration of contextual signals (team success, standings)
- Visualization layer (candidate dashboards)

---

## 👨‍💻 Author

**Luc Renaud**  
Data Engineering & Machine Learning  
NBA analytics enthusiast  

GitHub: https://github.com/lucR8

---

## 📄 License

This project is released under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.
