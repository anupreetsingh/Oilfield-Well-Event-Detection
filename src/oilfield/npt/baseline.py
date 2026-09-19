"""TF-IDF + logistic regression baseline for NPT cause tagging."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from oilfield.npt.labeling import CAUSES


def make_tfidf(params: dict) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, params.get("ngram_max", 2)), min_df=params.get("min_df", 2),
                                  sublinear_tf=True, strip_accents="unicode")),
        ("clf", LogisticRegression(C=params.get("C", 4.0), class_weight="balanced", max_iter=2000)),
    ])


def tfidf_proba(model: Pipeline, texts: list[str]) -> np.ndarray:
    """Class probabilities over every cause, including ones absent from the training split."""
    out = np.zeros((len(texts), len(CAUSES)), dtype=np.float32)
    out[:, model.classes_.astype(int)] = model.predict_proba(texts)
    return out
