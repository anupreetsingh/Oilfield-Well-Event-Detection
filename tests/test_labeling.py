import pandas as pd
import pytest

from oilfield.npt.dataset import build_examples, split_by_well
from oilfield.npt.labeling import label_activity


@pytest.mark.parametrize(
    "code, text, cause",
    [
        ("other", "Pipe stuck at 2350 m. Worked pipe with 60 t overpull, no go.", "stuck_pipe"),
        ("other", "Packed off while reaming, worked string free.", "stuck_pipe"),
        ("fish", "RIH with overshot.", "fishing"),
        ("other", "Jarred on fish at 3100 m.", "fishing"),
        ("other", "Observed 3 m3 gain in pit, shut in well.", "well_control"),
        ("other", "Total losses to formation, pumped LCM.", "well_control"),
        ("repair", "MWD lost signal, no pulses. POOH to change tool.", "downhole_tool_failure"),
        ("repair", "Repaired top drive IBOP hydraulic leak.", "rig_repair"),
        ("maintain", "Changed out saver sub.", "rig_repair"),
        ("repair", "Repaired heave compensator.", "rig_repair"),         # rig kit, not weather
        ("wait", "WOW, wind 45 knots.", "wait_on_weather"),
        ("other", "Waiting on weather, wave height 5 m.", "wait_on_weather"),
        ("wait", "Waited for supply boat with casing.", "wait_on_equipment"),
        ("other", "Waiting on orders from town.", "wait_on_equipment"),
        ("other", "Ran swell packers to 2800 m.", "other"),               # completion kit, not weather
        ("other", "Safety stand down after near miss.", "other"),
    ],
)
def test_rules(code, text, cause):
    assert label_activity(code, text)[0] == cause


def test_gold_labels_override_silver():
    acts = pd.DataFrame({
        "activity_id": ["A:1", "A:2", "A:3"],
        "well": ["W1"] * 3,
        "is_npt": [True, True, False],
        "code_minor": ["other", "repair", None],
        "comments": ["Pipe stuck at 100 m.", "Repaired pump.", "Drilled ahead."],
    })
    gold = pd.DataFrame({"activity_id": ["A:2"], "cause": ["downhole_tool_failure"]})
    ex = build_examples(acts, gold)
    assert ex["label"].tolist() == ["stuck_pipe", "downhole_tool_failure"]
    assert ex["label_source"].tolist() == ["silver", "gold"]


def test_split_keeps_wells_whole():
    ex = pd.DataFrame({"well": [f"W{i % 9}" for i in range(180)]})
    split = split_by_well(ex, test_share=0.25, val_share=0.15, seed=0)
    assert ex.assign(split=split).groupby("well")["split"].nunique().eq(1).all()
    assert set(split) == {"train", "val", "test"}
