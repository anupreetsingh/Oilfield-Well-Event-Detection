"""Window-level classification metrics on held-out wells."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from oilfield.threew.schema import CLASS_NAMES

LABELS = list(range(len(CLASS_NAMES)))


def full_proba(model, X, n_classes: int = len(CLASS_NAMES)) -> np.ndarray:
    """predict_proba padded to every class, for folds whose training data lacks one."""
    proba = model.predict_proba(X)
    out = np.zeros((len(X), n_classes), dtype=np.float32)
    out[:, model.classes_.astype(int)] = proba
    return out


def classification_summary(y_true: np.ndarray, proba: np.ndarray, folds: np.ndarray | None = None) -> dict:
    y_pred = proba.argmax(axis=1)
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    summary = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "n_windows": int(len(y_true)),
        "per_class_f1": {name: float(f) for name, f in zip(CLASS_NAMES, per_class)},
        "support": {name: int((y_true == i).sum()) for i, name in enumerate(CLASS_NAMES)},
    }
    if folds is not None:
        fold_scores = []
        for k in np.unique(folds):
            m = folds == k
            present = sorted(set(y_true[m]))
            fold_scores.append(f1_score(y_true[m], y_pred[m], labels=present, average="macro", zero_division=0))
        summary["fold_macro_f1"] = [float(s) for s in fold_scores]
    return summary


def confusion_frame(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, proba.argmax(axis=1), labels=LABELS)
    return pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
