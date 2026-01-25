from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class ModelSpec:
    """Simple spec to build a model consistently across awards."""
    name: str
    builder: Callable[[], object]
    family: str


def _logreg_l2() -> LogisticRegression:
    return LogisticRegression(
        solver="lbfgs",
        max_iter=4000,
        n_jobs=None,
        class_weight="balanced",
    )


def _logreg_elasticnet() -> LogisticRegression:
    # Interprétable + robustesse aux corrélations
    return LogisticRegression(
        solver="saga",
        penalty="elasticnet",
        l1_ratio=0.2,
        max_iter=8000,
        class_weight="balanced",
    )


def _linear_svm_calibrated() -> CalibratedClassifierCV:
    # Très bon sur features tabulaires “densément informatives”
    base = LinearSVC(class_weight="balanced")
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def _hgb() -> HistGradientBoostingClassifier:
    # Boosting sklearn (pas besoin de xgboost/lightgbm)
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
    )


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=0,
    )


def get_model_specs() -> Dict[str, ModelSpec]:
    """
    Return a model grid (small & defendable).
    You can expand later, but keep this stable for the report.
    """
    specs = [
        ModelSpec("logreg_l2", _logreg_l2, family="linear"),
        ModelSpec("logreg_elasticnet", _logreg_elasticnet, family="linear"),
        ModelSpec("linear_svm_calibrated", _linear_svm_calibrated, family="linear"),
        ModelSpec("hgb", _hgb, family="tree"),
        ModelSpec("rf", _rf, family="tree"),
    ]
    return {s.name: s for s in specs}
