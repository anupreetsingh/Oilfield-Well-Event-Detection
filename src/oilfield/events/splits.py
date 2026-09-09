"""Cross-validation folds that hold out whole wells.

Windows from one well are strongly correlated (same gauges, same reservoir, often the same
event recurring), so a random split would leak. Folds are drawn over real wells, stratified by
each run's event type; simulated and hand-drawn runs belong to no well and are always used for
training, as the 3W authors recommend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from oilfield.threew.schema import model_label

TRAIN_ONLY = -1


def well_folds(instances: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> pd.Series:
    """Map instance_id -> fold index, or TRAIN_ONLY for runs without a real well."""
    folds = pd.Series(TRAIN_ONLY, index=instances["instance_id"].to_numpy(), name="fold")
    real = instances[instances["source"] == "WELL"]
    y = real["event_type"].map(model_label).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(splitter.split(np.zeros(len(real)), y, groups=real["well_id"])):
        folds[real["instance_id"].iloc[test_idx].to_numpy()] = k
    return folds


def stride_mask(windows: pd.DataFrame, stride: int) -> np.ndarray:
    """Keep every ``stride``-th window of each run (windows must be sorted by run and time)."""
    if stride <= 1:
        return np.ones(len(windows), dtype=bool)
    return (windows.groupby("instance_id", sort=False).cumcount() % stride == 0).to_numpy()
