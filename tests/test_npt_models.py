import numpy as np
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


def test_distilbert_fine_tune_loop_runs_on_a_tiny_model(examples):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from oilfield.npt import distilbert

    words = sorted({w for t in examples["text"].str.lower() for w in t.replace(",", " ").replace(".", " ").split()})
    vocab = {tok: i for i, tok in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words])}
    tok = transformers.BertTokenizerFast(vocab=vocab)
    config = transformers.DistilBertConfig(vocab_size=len(vocab), n_layers=1, n_heads=2, dim=32, hidden_dim=64,
                                           num_labels=len(CAUSES))
    model = transformers.DistilBertForSequenceClassification(config)

    texts, y = examples["text"].tolist(), examples["y"].to_numpy()
    history = []
    model = distilbert.fine_tune(model, tok, texts, y, texts, y,
                                 {"epochs": 3, "batch_size": 16, "learning_rate": 1e-3, "max_length": 48},
                                 device=torch.device("cpu"), on_epoch=lambda e, m: history.append(m))
    proba = distilbert.predict_proba(model, tok, texts[:10], max_length=48)
    assert proba.shape == (10, len(CAUSES)) and np.allclose(proba.sum(1), 1, atol=1e-5)
    assert len(history) == 3 and history[-1]["train_loss"] < history[0]["train_loss"]
