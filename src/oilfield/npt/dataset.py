"""NPT remark examples and the well-grouped train / validation / test split."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from oilfield.npt.labeling import CAUSES, apply_silver_labels

LABEL_TO_ID = {c: i for i, c in enumerate(CAUSES)}


def normalize_remark(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def build_examples(activities: pd.DataFrame, gold: pd.DataFrame | None = None) -> pd.DataFrame:
    """NPT activities with a remark, a silver label, and the label used for training/eval.

    Hand-reviewed causes replace silver ones wherever they exist (``label_source = gold``).
    """
    npt = activities[activities["is_npt"] & activities["comments"].notna()].copy()
    npt["text"] = npt["comments"].map(normalize_remark)
    npt = apply_silver_labels(npt[npt["text"].str.len() > 0])
    npt["label"], npt["label_source"] = npt["silver_cause"], "silver"
    if gold is not None:
        g = npt["activity_id"].map(gold.set_index("activity_id")["cause"])
        npt.loc[g.notna(), "label"] = g[g.notna()]
        npt.loc[g.notna(), "label_source"] = "gold"
    npt["y"] = npt["label"].map(LABEL_TO_ID)
    return npt.reset_index(drop=True)


def split_by_well(examples: pd.DataFrame, test_share: float, val_share: float, seed: int) -> pd.Series:
    """Assign whole wells to test, then val, until each holds its share of NPT examples."""
    counts = examples.groupby("well").size()
    wells = counts.index.to_numpy()
    np.random.default_rng(seed).shuffle(wells)
    total, n_test, n_val, assigned = counts.sum(), 0, 0, {}
    for w in wells:
        if n_test < test_share * total:
            assigned[w], n_test = "test", n_test + counts[w]
        elif n_val < val_share * total:
            assigned[w], n_val = "val", n_val + counts[w]
        else:
            assigned[w] = "train"
    if "train" not in assigned.values():
        raise ValueError(f"only {len(wells)} wells: too few to hold out test and validation wells")
    return examples["well"].map(assigned).rename("split")
