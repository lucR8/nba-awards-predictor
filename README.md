# 🏀 NBA Awards Predictor
### End-to-End Data Engineering & Machine Learning Pipeline

This repository provides a **fully script-driven, reproducible pipeline** to predict NBA individual awards
(**MVP, DPOY, ROY, SMOY, MIP**) using historical data and season-aware ranking models.

The project is designed to be usable in **two modes**:
1. **Research / full pipeline mode** (fetch → build → train → evaluate)
2. **User / inference-only mode** using **pretrained models** (no historical data required)

---

## 🎯 Project goals

- Aggregate NBA player data (1996–2025)
- Resolve multi-source entity matching (no shared player ID)
- Build award-specific datasets with strict eligibility rules
- Train season-aware ranking models
- Evaluate models with ranking-oriented metrics
- Support **future-season inference** without labels

---

## 🚀 Quick start (recommended for GitHub users)

If you only want to **run predictions for a target season** (e.g. 2026) **without downloading or training on historical data**, use the **pretrained models**.

### 1️⃣ Export pretrained models (project maintainer)

This step is already done for release **v1**, but documented here for transparency.

```bash
python scripts/export_pretrained_models.py \
  --prediction-meta data/target/2026/asof_2026-01-26/predictions/prediction_meta.json \
  --out models/pretrained/v1 \
  --overwrite
```

This command:
- reads which models were actually **used for prediction** (from `prediction_meta.json`),
- copies the corresponding trained models,
- writes a portable manifest:

```text
models/pretrained/v1/models_manifest.json
```

This manifest is the **single source of truth** for inference.

---

### 2️⃣ Run predictions using pretrained models (users)

Once pretrained models are available, users can run:

```bash
python run_all.py \
  --year 2026 \
  --skip-fetch \
  --skip-build \
  --skip-train \
  --pretrained models/pretrained/v1/models_manifest.json
```

✅ This will:
- skip historical data fetching and training,
- load pretrained models only,
- build the minimal target-season dataset,
- generate Top-K predictions for each award.

### 📂 Outputs

```text
data/target/2026/asof_*/predictions/
├── mvp_top5.csv
├── dpoy_top5.csv
├── smoy_top5.csv
├── roy_top5.csv
├── mip_top5.csv
├── all_awards_top5.parquet
└── prediction_meta.json
```

The `prediction_meta.json` file records:
- season metadata,
- which model was used per award,
- feature set,
- confidence score (Top-1 probability).

---

## 🧠 Modeling strategy

The problem is framed as a **ranking task**, not a simple classification.

Each award has explicit constraints:

- **MVP**: extreme imbalance, full population
- **DPOY**: defensive metrics, partial observability
- **ROY**: rookies only, no lagged features
- **SMOY**: bench-only constraint
- **MIP**: year-over-year dynamics (`prev_*`, `delta_*`)

Models are evaluated on:
- Mean Reciprocal Rank (MRR)
- Top-1 / Top-5 accuracy
- Mean winner rank

---

## 🧪 Leakage prevention

The pipeline includes **mandatory leakage checks** before any prediction:

- name-based detection (e.g. `winner`, `rank`, `vote`),
- correlation-based detection,
- award-specific guards.

This ensures that no label-derived information can influence predictions.

---

## 📊 Offline evaluation (optional)

For research or reporting purposes:

```bash
python run_all.py --evaluate --plot
```

Generates:

```text
reports/model_eval/
├── metrics_by_award.csv
└── plots/
```

---

## 📁 Repository structure

```text
nba-awards-predictor/
├── models/
│   └── pretrained/v1/        # ✅ versioned pretrained models
├── data/                     # raw / processed / target
├── src/awards_predictor/     # core package
├── scripts/                  # orchestration utilities
├── notebooks/                # analysis only
├── tests/
└── reports/                  # generated artifacts
```

---

## 🚧 Known limitations

- Media narrative and voter bias are not explicitly modeled
- Defensive impact remains partially observable
- Near-perfect metrics ≠ causal understanding

These points are discussed in the accompanying report.

---

## 👨‍💻 Author

**Luc Renaud**  
Data Engineering & Machine Learning  
ECE Paris

---

## 📄 License

MIT License — see `LICENSE`

