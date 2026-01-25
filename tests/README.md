# 🧪 Tests — NBA Awards Predictor

This directory contains **lightweight, deterministic tests** designed to validate
the **critical invariants** of the NBA Awards Predictor pipeline.

The goal of these tests is **not** to achieve high coverage,
but to guarantee that the most important scientific and engineering assumptions
remain valid over time.

---

## 🎯 Testing philosophy

This project follows a **research-oriented testing strategy**:

- ✅ Test *invariants*, not performance
- ✅ No dependency on real NBA datasets
- ✅ No model training inside tests
- ✅ Fully deterministic and fast (< 1s)

This makes the test suite:
- stable across environments,
- suitable for CI,
- defensible in an academic context.

---

## 📂 Test files

### `test_percentiles.py`
Validates feature engineering primitives:
- percentile computation
- group-wise z-score normalization

These functions are widely reused and must remain mathematically correct.

---

### `test_invariants.py`
Validates **core pipeline assumptions**:

1. **Leakage prevention**
   - obvious label-like columns are detected and rejected

2. **Ranking metrics**
   - winner rank
   - Mean Reciprocal Rank (MRR)
   - Hit@K behavior

3. **Eligibility rules**
   - SMOY: bench role filtering
   - MIP: exclusion of rookies and low-minute players

These tests ensure that domain logic and evaluation framing remain consistent.

---

## ▶️ How to run the tests

### 1️⃣ Install dependencies
From the repository root:

```bash
pip install -r requirements.txt
```

### 2️⃣ Run all tests

```bash
pytest -q
```

Expected output:
```text
X passed in <1s
```

---

## 🔁 Continuous Integration (CI)

Tests are automatically executed via GitHub Actions on:
- every push to `main`
- every pull request targeting `main`

See:
```text
.github/workflows/tests.yml
```

---

## 🚫 What is intentionally NOT tested

- End-to-end training pipelines
- Exact AUC / accuracy values
- Model convergence
- Dataset completeness

These aspects are:
- data-dependent,
- non-deterministic,
- evaluated offline via scripts and reports instead.

---

## ✅ Summary

If these tests pass, you can be confident that:
- no obvious data leakage was introduced,
- ranking metrics behave as expected,
- award eligibility rules remain valid.

This provides a **strong safety net** for a research-grade ML pipeline.
