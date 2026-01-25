# Contributing to NBA Awards Predictor

Thank you for your interest in contributing to **NBA Awards Predictor**.

This repository is primarily developed as an academic and research-oriented project,
but external contributions are welcome as long as they follow the guidelines below.

---

## 🚀 Contribution workflow

1. Fork the repository
2. Create a dedicated branch:
```bash
git checkout -b feature/short-description
```

3. Implement your changes
4. Commit using clear and conventional messages:
```
feat: add season-aware ranking metric
fix: prevent temporal leakage in feature matrix
refactor: simplify eligibility rules for SMOY
```

5. Open a Pull Request targeting the `main` branch.

---

## 🧪 Tests

Before submitting a Pull Request, make sure all tests pass:
```bash
pytest -q
```

If you add new functionality, please include at least one minimal test under `tests/`.

---

## 🧹 Code quality guidelines

- Respect the existing project structure (`src/awards_predictor/`)
- Prefer explicit and readable code over clever shortcuts
- Avoid hidden state or implicit data leakage
- All modeling logic must be **season-aware**
- Do **not** scrape Basketball-Reference within this repository

Optional but recommended:
```bash
flake8 src
```

---

## 📦 Project structure

```
src/awards_predictor/
    data/          # Data loading, validation, eligibility rules
    features/     # Feature engineering & matrix construction
    models/       # Model definitions
    evaluation/   # Metrics, ranking logic, audits
    plot/         # Reproducible figures
```

Raw datasets must **never** be committed.

---

## 📬 Questions

Please open a GitHub Issue for any question or suggestion.
