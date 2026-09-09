import numpy as np
import pandas as pd

from oilfield.events.splits import TRAIN_ONLY, stride_mask, well_folds


def make_instances(n_wells=12, runs_per_well=6, n_sim=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_wells):
        for r in range(runs_per_well):
            rows.append({"instance_id": f"W{w}-{r}", "source": "WELL", "well_id": f"WELL-{w:05d}",
                         "event_type": int(rng.choice([0, 0, 2, 4, 7]))})
    for s in range(n_sim):
        rows.append({"instance_id": f"S{s}", "source": "SIMULATED", "well_id": None, "event_type": int(rng.choice([1, 5, 6]))})
    return pd.DataFrame(rows)


def test_each_well_lands_in_exactly_one_fold():
    inst = make_instances()
    folds = well_folds(inst, n_folds=4, seed=1)
    real = inst[inst.source == "WELL"].assign(fold=lambda d: d.instance_id.map(folds))
    assert real.groupby("well_id")["fold"].nunique().eq(1).all()
    assert set(real["fold"]) == {0, 1, 2, 3}


def test_synthetic_runs_are_train_only():
    inst = make_instances()
    folds = well_folds(inst, n_folds=4)
    sim = inst[inst.source == "SIMULATED"]["instance_id"]
    assert (folds[sim.to_numpy()] == TRAIN_ONLY).all()


def test_stride_keeps_every_kth_window_per_run():
    windows = pd.DataFrame({"instance_id": ["a"] * 7 + ["b"] * 4})
    assert stride_mask(windows, 3).tolist() == [True, False, False, True, False, False, True,
                                                True, False, False, True]
    assert stride_mask(windows, 1).all()
