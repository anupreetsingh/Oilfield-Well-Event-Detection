"""Tag every NPT activity with a predicted cause and write the npt_events table.

    python -m oilfield.npt.tag            # uses models/npt-cause and data/npt/activities.parquet
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd

from oilfield.config import load_config, parse_overrides, resolve_path
from oilfield.npt.baseline import tfidf_proba
from oilfield.npt.dataset import build_examples
from oilfield.npt.labeling import CAUSES, load_gold

log = logging.getLogger(__name__)


def predict_causes(texts: list[str], model_dir) -> np.ndarray:
    meta = json.loads((model_dir / "npt_model.json").read_text())
    if meta["kind"] == "tfidf":
        return tfidf_proba(joblib.load(model_dir / "tfidf.joblib"), texts)
    from oilfield.npt import distilbert

    model, tok = distilbert.load(model_dir)
    return distilbert.predict_proba(model, tok, texts, meta.get("max_length", 128))


def tag_npt(activities: pd.DataFrame, model_dir, gold: pd.DataFrame | None = None) -> pd.DataFrame:
    ex = build_examples(activities, gold)
    proba = predict_causes(ex["text"].tolist(), model_dir)
    ex["cause"] = [CAUSES[i] for i in proba.argmax(1)]
    ex["confidence"] = proba.max(1).round(3)
    # A reviewed label is the truth for that remark; keep the model's view alongside it.
    ex["predicted_cause"] = ex["cause"]
    reviewed = ex["label_source"] == "gold"
    ex.loc[reviewed, "cause"] = ex.loc[reviewed, "label"]
    cols = ["activity_id", "wellbore", "well", "report_date", "start_time", "end_time", "duration_h", "md_m",
            "code", "comments", "cause", "confidence", "predicted_cause", "silver_cause", "label_source"]
    return ex[cols]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/npt.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, parse_overrides(args.set))

    out = resolve_path(cfg["out_dir"])
    activities = pd.read_parquet(out / "activities.parquet")
    events = tag_npt(activities, resolve_path(cfg["model_dir"]), load_gold(resolve_path(cfg["gold_labels"])))
    events.to_parquet(out / "npt_events.parquet", index=False)
    log.info("tagged %d NPT activities, %.1f h:\n%s", len(events), events["duration_h"].sum(),
             events.groupby("cause")["duration_h"].sum().round(1).sort_values(ascending=False))


if __name__ == "__main__":
    main()
