# 🏀 NBA Awards Predictor  
### End-to-End Data Engineering & Machine Learning Pipeline

This repository contains a **fully reproducible, script-driven pipeline** to predict NBA individual awards  
(**MVP, DPOY, ROY, SMOY, MIP**) using historical multi-source data and season-aware ranking models.

The project is designed at **master / research level**, with strong emphasis on:
- data engineering realism,
- temporal consistency,
- auditability and leakage prevention,
- clear separation between exploration and production pipelines.

---

## 🎯 Project overview

The goal of this project is to:

- Consolidate NBA player data from **1996 to 2025**
- Solve a **multi-source entity resolution** problem (no shared player ID)
- Build **award-specific datasets** with explicit domain constraints
- Train **season-aware ranking models**
- Evaluate models with ranking-oriented metrics
- Support **future-season inference** (no labels available)

This project intentionally mirrors real-world constraints:
- strong label imbalance (especially MVP),
- partial observability (defensive impact),
- narrative-driven outcomes,
- evolving statistical definitions across eras.

---

## 📦 Data sources

- **Basketball-Reference**
  - Regular season, playoffs
  - Season-level percentiles
- External impact metrics:
  - RAPTOR
  - LEBRON
  - MAMBA
- Player biographical attributes (age, height, weight, draft, country)

⚠️ No universal player identifier exists across sources.  
Robust name normalization, fuzzy matching, and temporal guards are applied throughout the pipeline.

---

## 🏗️ How to run the full pipeline (recommended)

The entire pipeline can be executed from **a single entry point**: `run_all.py`.

This script orchestrates:
1. data fetching,
2. dataset construction,
3. model training,
4. leakage checks,
5. season prediction,
6. offline evaluation,
7. figure generation.

### ▶️ Full run (default)

```bash
python run_all.py --year 2026 --evaluate --plot
```

This will:
- fetch and build the target season (2026),
- train models on historical data,
- run leakage checks,
- generate Top-K predictions per award,
- compute offline evaluation metrics,
- generate plots for the report.

### 🔧 Common options

```bash
# Skip data fetching (if already done)
python run_all.py --skip-fetch --skip-build

# Train only (no prediction)
python run_all.py --skip-predict

# Prediction only (models already trained)
python run_all.py --skip-fetch --skip-build --skip-train

# Control validation / test horizon
python run_all.py --val-years 2 --test-years 3

# Strict leakage detection
python run_all.py --leakage-strict
```

### 📂 Outputs

- Predictions:
  ```text
  data/target/<year>/asof_*/predictions/
  ```

- Evaluation artifacts:
  ```text
  reports/model_eval/
  ├── metrics_by_award.csv
  └── plots/
  ```

---

## 🧠 Modeling strategy

The task is framed as a **ranking problem**, not pure classification.

Each award has explicit eligibility rules:

- **MVP**: full population, extreme imbalance
- **DPOY**: defensive metrics, partial observability
- **ROY**: rookies only, no lagged features
- **SMOY**: bench role constraint
- **MIP**: year-over-year dynamics (`prev_*`, `delta_*` features)

Models are trained with **strict temporal splits** and evaluated on their ability to rank the true winner.

---

## 🧪 Evaluation methodology

Primary metrics:
- Mean Reciprocal Rank (MRR)
- Top-K accuracy (Top-1, Top-5)
- Mean winner rank

These metrics directly reflect real voting dynamics:
> *How high does the true winner appear in the ranked candidate list?*

Learning curves, hyperparameter sweeps, and feature ablation studies are available for deeper analysis.

---

## 📓 Understanding the notebooks (optional)

Notebooks are provided **only for exploration and diagnostics**.

They are **not required** to run the pipeline.

Typical notebook flow:
1. `01_exploration.ipynb`  
   → raw data inspection, distributions, missingness
2. `02_build_df_clean.ipynb`  
   → dataset assembly sanity checks
3. `03_build_feature_matrices.ipynb`  
   → feature engineering validation
4. `04_award_datasets.ipynb`  
   → award-specific eligibility inspection
5. `05–07_modeling_*.ipynb`  
   → exploratory modeling and error analysis

⚠️ All production logic lives in **Python scripts under `src/` and `scripts/`**.  
Notebooks should be seen as **documentation and analysis support**, not execution units.

---

## 📁 Project structure

```
nba-awards-predictor/
├── data/                 # raw / processed / target datasets
├── scripts/              # pipeline orchestration scripts
├── src/awards_predictor/ # core Python package
├── notebooks/            # exploratory analysis only
├── tests/                # unit tests
└── reports/              # generated artifacts (not versioned)
```

---

## 🚧 Known limitations

- Media narratives and voter bias are not explicitly modeled
- Defensive impact remains partially observable
- Near-perfect scores do not imply causal understanding

These limitations are discussed and quantified during evaluation.

---

## 👨‍💻 Author

**Luc Renaud**  
Data Engineering & Machine Learning  
ECE Paris

---

## 📄 License

MIT License — see [LICENSE](./LICENSE)
