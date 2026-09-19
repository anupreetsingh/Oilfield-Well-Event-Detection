import pytest

from oilfield.npt.baseline import make_tfidf, tfidf_proba
from oilfield.npt.dataset import build_examples
from oilfield.npt.labeling import CAUSES
from oilfield.npt.synthetic import generate
from oilfield.npt.witsml import parse_ddr_dir


@pytest.fixture(scope="module")
def examples(tmp_path_factory):
    out = tmp_path_factory.mktemp("ddr")
    generate(out, n_wellbores=4, n_days=30, seed=2)
    activities, _ = parse_ddr_dir(out)
    return build_examples(activities)


def test_tfidf_baseline_learns_synthetic_causes(examples):
    pipe = make_tfidf({"min_df": 1}).fit(examples["text"], examples["y"])
    proba = tfidf_proba(pipe, examples["text"].tolist())
    assert proba.shape == (len(examples), len(CAUSES))
    assert (proba.argmax(1) == examples["y"].to_numpy()).mean() > 0.9

