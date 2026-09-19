"""Tabular heads used in the manuscript (random seed 42)."""
from __future__ import annotations

from typing import Any


def make_classifier(name: str, *, n_class: int = 2, n_jobs: int = 1) -> Any:
    name = (name or "rf").lower()
    if name in ("rf", "random_forest"):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=n_jobs,
        )
    if name in ("et", "extratrees", "extra_trees"):
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=n_jobs,
        )
    if name in ("hgb", "hist"):
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=200,
            random_state=42,
            class_weight="balanced",
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        params = dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=n_jobs,
            tree_method="hist",
            eval_metric="logloss",
        )
        if n_class > 2:
            params.update(objective="multi:softprob", num_class=n_class)
        return XGBClassifier(**params)
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        params = dict(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=n_jobs,
            class_weight="balanced",
            verbose=-1,
        )
        if n_class > 2:
            params["objective"] = "multiclass"
            params["num_class"] = n_class
        return LGBMClassifier(**params)
    raise ValueError(f"unknown classifier: {name}")
