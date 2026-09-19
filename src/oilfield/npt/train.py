"""Train NPT cause taggers on drilling-report remarks and compare them on held-out wells.

Flow:

    DDR XML -> activities (parsed once, cached as Parquet)
      -> NPT activities with remarks -> silver labels from codes + rules (gold labels override)
      -> well-grouped train / val / test split
      -> TF-IDF + logistic regression baseline, DistilBERT fine-tune (best epoch on val)
      -> macro-F1 on the test wells, MLflow runs, reports/npt_metrics.{json,md}
      -> the better model saved to models/npt-cause for tagging

Run locally:  python -m oilfield.npt.train --set raw_dir=data/raw/volve_synthetic
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from oilfield.config import REPO_ROOT, load_config, parse_overrides, resolve_path
from oilfield.npt.baseline import make_tfidf, tfidf_proba
from oilfield.npt.dataset import build_examples, split_by_well
from oilfield.npt.labeling import CAUSES, export_review_sample, load_gold
from oilfield.npt.witsml import parse_ddr_dir
from oilfield.spark import on_databricks

log = logging.getLogger(__name__)


def load_activities(cfg: dict, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = resolve_path(cfg["out_dir"])
    acts_path, reps_path = out / "activities.parquet", out / "daily_reports.parquet"
    if acts_path.exists() and reps_path.exists() and not refresh:
        return pd.read_parquet(acts_path), pd.read_parquet(reps_path)
    activities, reports = parse_ddr_dir(resolve_path(cfg["raw_dir"]))
    out.mkdir(parents=True, exist_ok=True)
    activities.to_parquet(acts_path, index=False)
    reports.to_parquet(reps_path, index=False)
    log.info("parsed %d activities (%d NPT) from %d reports", len(activities), activities["is_npt"].sum(), len(reports))
    return activities, reports


def score(y_true: np.ndarray, proba: np.ndarray) -> dict:
    y_pred = proba.argmax(1)
    labels = list(range(len(CAUSES)))
    present = sorted(set(y_true))
    report = classification_report(y_true, y_pred, labels=labels, target_names=CAUSES, output_dict=True, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "n": int(len(y_true)),
        "per_class": {c: {k: float(v) for k, v in report[c].items()} for c in CAUSES},
    }


def run(cfg: dict, models: list[str], refresh: bool = False, export_review: bool = False) -> dict:
    if not on_databricks() and not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(cfg["experiment"])

    activities, _ = load_activities(cfg, refresh)
    gold = load_gold(resolve_path(cfg["gold_labels"]))
    ex = build_examples(activities, gold)
    if export_review:
        export_review_sample(ex, resolve_path(cfg["review_sample"]))
        log.info("wrote review sample to %s", cfg["review_sample"])

    s = cfg["split"]
    ex["split"] = split_by_well(ex, s["test_share"], s["val_share"], s["seed"])
    tr, va, te = (ex[ex["split"] == k] for k in ("train", "val", "test"))
    label_source = "gold" if (te["label_source"] == "gold").all() else (
        "mixed" if (te["label_source"] == "gold").any() else "silver")
    data_summary = {
        "npt_examples": int(len(ex)),
        "by_split": ex["split"].value_counts().to_dict(),
        "wells_by_split": ex.groupby("split")["well"].nunique().to_dict(),
        "test_label_source": label_source,
        "label_counts": ex["label"].value_counts().to_dict(),
    }
    log.info("data: %s", json.dumps(data_summary))
    if label_source != "gold":
        log.warning("test labels are %s: scores measure agreement with the labelling rules, not with engineers", label_source)

    results, best = {}, (None, -1.0)
    model_dir = resolve_path(cfg["model_dir"])
    with mlflow.start_run(run_name="npt-cause-tagging"):
        mlflow.log_dict(data_summary, "data_summary.json")
        if "tfidf" in models:
            with mlflow.start_run(run_name="tfidf", nested=True):
                mlflow.log_params(cfg["tfidf"])
                pipe = make_tfidf(cfg["tfidf"]).fit(pd.concat([tr, va])["text"], pd.concat([tr, va])["y"])
                results["tfidf"] = score(te["y"].to_numpy(), tfidf_proba(pipe, te["text"].tolist()))
                mlflow.log_metric("test_macro_f1", results["tfidf"]["macro_f1"])
                mlflow.log_dict(results["tfidf"], "test_scores.json")
                if results["tfidf"]["macro_f1"] > best[1]:
                    best = ("tfidf", results["tfidf"]["macro_f1"])
                    model_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(pipe, model_dir / "tfidf.joblib")

        if "distilbert" in models:
            from oilfield.npt import distilbert

            params = cfg["distilbert"]
            with mlflow.start_run(run_name="distilbert", nested=True):
                mlflow.log_params(params)
                model, tok = distilbert.load_pretrained(params["model_name"])
                model = distilbert.fine_tune(
                    model, tok, tr["text"].tolist(), tr["y"].to_numpy(), va["text"].tolist(), va["y"].to_numpy(), params,
                    on_epoch=lambda e, m: mlflow.log_metrics(m, step=e),
                )
                proba = distilbert.predict_proba(model, tok, te["text"].tolist(), params["max_length"])
                results["distilbert"] = score(te["y"].to_numpy(), proba)
                mlflow.log_metric("test_macro_f1", results["distilbert"]["macro_f1"])
                mlflow.log_dict(results["distilbert"], "test_scores.json")
                if results["distilbert"]["macro_f1"] > best[1]:
                    best = ("distilbert", results["distilbert"]["macro_f1"])
                    distilbert.save(model, tok, model_dir, {"max_length": params["max_length"]})

        mlflow.set_tag("best_model", best[0])
        write_reports(results, data_summary, best[0], resolve_path(cfg["reports_dir"]))
    if best[0] == "tfidf":
        (model_dir / "npt_model.json").write_text(json.dumps({"kind": "tfidf", "causes": CAUSES}, indent=2))
    return results


def write_reports(results: dict, data_summary: dict, best: str | None, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "npt_metrics.json", "w") as f:
        json.dump({"data": data_summary, "best_model": best, "models": results}, f, indent=2)
    lines = [f"Test labels: **{data_summary['test_label_source']}**, {data_summary['by_split'].get('test', 0)} remarks "
             f"from {data_summary['wells_by_split'].get('test', 0)} held-out wells.", "",
             "| model | macro-F1 | accuracy |", "|---|---|---|"]
    lines += [f"| {m} | {r['macro_f1']:.3f} | {r['accuracy']:.3f} |" for m, r in results.items()]
    lines += ["", "| cause | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    for c in CAUSES:
        lines.append(f"| {c} | " + " | ".join(f"{r['per_class'][c]['f1-score']:.2f}" for r in results.values()) + " |")
    (out_dir / "npt_metrics.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/npt.yaml")
    parser.add_argument("--models", nargs="+", default=["tfidf", "distilbert"])
    parser.add_argument("--refresh", action="store_true", help="re-parse the XML even if a cached parse exists")
    parser.add_argument("--export-review", action="store_true", help="write a stratified sample for hand labelling")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. distilbert.epochs=2")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(load_config(args.config, parse_overrides(args.set)), args.models, args.refresh, args.export_review)


if __name__ == "__main__":
    main()
