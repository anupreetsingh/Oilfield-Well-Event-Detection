"""scikit-learn well-event classifiers. Both handle the NaN features left by dropped sensors."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


def make_random_forest(params: dict, seed: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 200),
        min_samples_leaf=params.get("min_samples_leaf", 10),
        max_features=params.get("max_features", "sqrt"),
        max_samples=params.get("max_samples", 0.3),
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def make_gradient_boosting(params: dict, seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=params.get("learning_rate", 0.08),
        max_iter=params.get("max_iter", 300),
        max_leaf_nodes=params.get("max_leaf_nodes", 63),
        min_samples_leaf=params.get("min_samples_leaf", 40),
        l2_regularization=params.get("l2_regularization", 1.0),
        class_weight="balanced",
        early_stopping=False,
        random_state=seed,
    )


SKLEARN_MODELS = {"rf": make_random_forest, "hgb": make_gradient_boosting}
