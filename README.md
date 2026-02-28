# NBA Awards Predictor

### End-to-End Data Engineering and Machine Learning Pipeline

This repository contains a fully script-driven and reproducible pipeline to predict NBA individual awards (MVP, DPOY, ROY, SMOY, MIP) using historical data and season-aware ranking models.

The project supports two usage modes:

1. Full pipeline mode (fetch → build → train → evaluate)
2. Inference-only mode using pretrained models (no historical training data required)

---

## Project Overview

The pipeline is designed to:

* aggregate NBA player data from multiple sources across seasons,
* resolve player identity matching when no shared identifier is available,
* build award-specific datasets with dedicated eligibility rules,
* train season-aware ranking models,
* evaluate models using ranking-based metrics,
* run future-season predictions even when labels are not yet available.

---

## Quick Start

If you only want to generate predictions for a target season (for example, 2026) without downloading historical data or training models, use the pretrained models.

### Export pretrained models

This step is already completed for release `v1`, but is documented here for reproducibility.

```bash
python scripts/export_pretrained_models.py \
  --prediction-meta data/target/2026/asof_2026-01-26/predictions/prediction_meta.json \
  --out models/pretrained/v1 \
  --overwrite
```

This command:

* reads the models used for prediction from `prediction_meta.json`,
* copies the relevant trained models,
* writes a portable manifest file:

```text
models/pretrained/v1/models_manifest.json
```

This manifest is used as the reference entry point for inference.

---

### Run predictions with pretrained models

Once pretrained models are available, users can run:

```bash
python run_all.py \
  --year 2026 \
  --skip-fetch \
  --skip-build \
  --skip-train \
  --pretrained models/pretrained/v1/models_manifest.json
```

This will:

* skip historical data fetching and training,
* load pretrained models only,
* build the target-season dataset,
* generate Top-K predictions for each award.

### Output files

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

* season metadata,
* the selected model for each award,
* the feature set used,
* a confidence score based on Top-1 probability.

---

## Modeling Approach

The problem is treated as a ranking task rather than a standard classification task.

Each award has its own constraints:

* **MVP**: extreme class imbalance across the full population
* **DPOY**: defensive metrics with partial observability
* **ROY**: rookie-only population with no lagged features
* **SMOY**: bench-player eligibility constraint
* **MIP**: year-over-year progression using `prev_*` and `delta_*` features

Models are evaluated with:

* Mean Reciprocal Rank (MRR)
* Top-1 accuracy
* Top-5 accuracy
* Mean winner rank

---

## Leakage Prevention

The pipeline includes mandatory leakage checks before prediction:

* name-based checks (for terms such as `winner`, `rank`, or `vote`)
* correlation-based checks
* award-specific guards

These checks are designed to prevent label-derived information from influencing predictions.

---

## Offline Evaluation

For research or reporting purposes, the following command runs evaluation and plotting:

```bash
python run_all.py --evaluate --plot
```

This generates:

```text
reports/model_eval/
├── metrics_by_award.csv
└── plots/
```

---

## Repository Structure

```text
nba-awards-predictor/
├── models/
│   └── pretrained/v1/        # versioned pretrained models
├── data/                     # raw, processed, and target data
├── src/awards_predictor/     # core package
├── scripts/                  # orchestration utilities
├── notebooks/                # analysis notebooks
├── tests/
└── reports/                  # generated outputs
```

---

## Limitations

* Media narrative and voter bias are not explicitly modeled
* Defensive impact remains only partially observable
* Strong offline metrics do not imply causal understanding

These limitations are discussed in the accompanying report.

---

## Author

**Luc Renaud**
Data Engineering and Machine Learning
ECE Paris

---

## License

MIT License. See `LICENSE` for details.
