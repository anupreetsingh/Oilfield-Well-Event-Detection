"""Train and compare well-event classifiers on held-out wells, tracked in MLflow.

Flow:

    gold_3w_windows (+ silver_3w_readings sequences for the CNN)
      -> folds over real wells; simulated runs always train
      -> per model and fold: fit on the training wells, predict the held-out wells' windows
      -> pooled out-of-fold macro-F1, per-class F1, confusion matrix   (window level)
      -> alerts replayed on the out-of-fold probabilities: detection rate, lead time ahead of
         steady state, false alerts per day of normal operation        (run level)
      -> MLflow: one parent run, one nested run per model; reports/event_metrics.{json,md}

Run locally:  python -m oilfield.events.train --models rf hgb cnn ensemble
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from oilfield.config import REPO_ROOT, load_config, parse_overrides, resolve_path
from oilfield.events.alerts import AlertRule, score_alerts, summarize_alerts
from oilfield.events.data import SequenceStore, feature_columns, load_instances, load_windows
from oilfield.events.metrics import classification_summary, confusion_frame, full_proba
from oilfield.events.models import SKLEARN_MODELS
from oilfield.events.splits import stride_mask, well_folds
from oilfield.spark import TableStore, get_spark, on_databricks
from oilfield.threew.pipeline import sensor_columns
from oilfield.threew.schema import CLASS_NAMES, N_CLASSES

log = logging.getLogger(__name__)


class Experiment:
    """Everything the models share: windows, folds, labels, and (lazily) CNN sequences."""

    def __init__(self, cfg: dict, spark=None):
        self.cfg = cfg
        self.pipe_cfg = load_config(cfg["pipeline_config"])
        self.sensors = sensor_columns(self.pipe_cfg)
        self.spark = spark or get_spark("oilfield-events")
        self.store = TableStore.from_config(self.spark, self.pipe_cfg["tables"])

        self.instances = load_instances(self.store)
        self.windows = load_windows(self.store, self.sensors)
        folds = well_folds(self.instances, cfg["cv"]["n_folds"], cfg["cv"]["seed"])
        self.windows["fold"] = self.windows["instance_id"].map(folds).astype(int)
        self.fold = self.windows["fold"].to_numpy()
        self.y = self.windows["label"].to_numpy()
        self.train_keep = stride_mask(self.windows, cfg.get("train_stride", 1))
        self.eval_mask = self.fold >= 0
        self.n_folds = cfg["cv"]["n_folds"]
        self._seqs = None

    @property
    def sequences(self) -> SequenceStore:
        if self._seqs is None:
            self._seqs = SequenceStore.from_store(self.store, self.sensors, self.pipe_cfg["cleaning"]["resample_seconds"])
            self.starts = self._seqs.window_starts(self.windows, self.pipe_cfg["windows"]["length_seconds"])
        return self._seqs

    def data_summary(self) -> dict:
        real = self.windows[self.eval_mask]
        return {
            "runs_kept": int(len(self.instances)),
            "runs_by_source": self.instances["source"].value_counts().to_dict(),
            "real_wells": int(self.instances.loc[self.instances["source"] == "WELL", "well_id"].nunique()),
            "windows": int(len(self.windows)),
            "eval_windows": int(len(real)),
            "eval_windows_by_class": {CLASS_NAMES[c]: int(n) for c, n in real["label"].value_counts().sort_index().items()},
            "wells_per_fold": {int(k): int(g["well_id"].nunique()) for k, g in real.groupby("fold")},
        }

    # -- per-model out-of-fold predictions ---------------------------------------------------

    def oof_sklearn(self, name: str, params: dict) -> np.ndarray:
        cols = feature_columns(self.sensors, params.get("features", "all"))
        X = self.windows[cols].to_numpy(dtype=np.float32)
        proba = np.full((len(X), N_CLASSES), np.nan, dtype=np.float32)
        for k in range(self.n_folds):
            tr, te = self.train_keep & (self.fold != k), self.fold == k
            started = time.time()
            model = SKLEARN_MODELS[name](params, seed=self.cfg["cv"]["seed"] + k)
            model.fit(X[tr], self.y[tr])
            proba[te] = full_proba(model, X[te])
            self._log_fold(name, k, proba, te, time.time() - started)
        return proba

    def oof_cnn(self, params: dict) -> np.ndarray:
        from oilfield.events.cnn import pick_device, predict_cnn, sensor_scale, train_cnn

        seqs, device = self.sequences, pick_device()
        length = self.pipe_cfg["windows"]["length_seconds"] // self.pipe_cfg["cleaning"]["resample_seconds"]
        std_cols = [f"{c}_std" for c in self.sensors]
        proba = np.full((len(self.windows), N_CLASSES), np.nan, dtype=np.float32)
        for k in range(self.n_folds):
            tr, te = self.train_keep & (self.fold != k), self.fold == k
            started = time.time()
            scale = sensor_scale(self.windows.loc[tr, std_cols].to_numpy(dtype=np.float32))
            model = train_cnn(
                seqs.values, self.starts[tr], self.y[tr], length=length, scale=scale, n_classes=N_CLASSES,
                params=params, seed=self.cfg["cv"]["seed"] + k, device=device,
                on_epoch=lambda epoch, loss, k=k: mlflow.log_metric(f"fold{k}_train_loss", loss, step=epoch),
            )
            proba[te] = predict_cnn(model, seqs.values, self.starts[te], length=length, scale=scale, device=device)
            self._log_fold("cnn", k, proba, te, time.time() - started)
        return proba

    def _log_fold(self, name: str, k: int, proba: np.ndarray, te: np.ndarray, seconds: float) -> None:
        y, pred = self.y[te], proba[te].argmax(1)
        present = sorted(set(y))
        f1 = f1_score(y, pred, labels=present, average="macro", zero_division=0)
        mlflow.log_metric("fold_macro_f1", f1, step=k)
        log.info("%s fold %d: macro-F1 %.3f over %d classes, accuracy %.3f (%.0fs)",
                 name, k, f1, len(present), (y == pred).mean(), seconds)

    # -- evaluation ----------------------------------------------------------------------------

    def evaluate(self, proba: np.ndarray) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        m = self.eval_mask
        summary = classification_summary(self.y[m], proba[m], self.fold[m])
        a = self.cfg["alerts"]
        scored = score_alerts(self.windows[m], proba[m], self.instances, AlertRule(a["threshold"], a["k"], a["n"]))
        summary["alerts"] = summarize_alerts(scored)
        return summary, confusion_frame(self.y[m], proba[m]), scored


def log_summary(summary: dict) -> None:
    mlflow.log_metric("macro_f1", summary["macro_f1"])
    mlflow.log_metric("accuracy", summary["accuracy"])
    mlflow.log_metric("fold_macro_f1_mean", float(np.mean(summary["fold_macro_f1"])))
    for name, f1 in summary["per_class_f1"].items():
        mlflow.log_metric(f"f1_{name}", f1)
    for key, value in summary["alerts"].items():
        if isinstance(value, (int, float)) and np.isfinite(value):
            mlflow.log_metric(f"alert_{key}", value)
    mlflow.log_dict(summary, "summary.json")


def write_reports(results: dict, data_summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "event_metrics.json", "w") as f:
        json.dump({"data": data_summary, "models": results}, f, indent=2, default=float)

    lines = [
        "| model | macro-F1 (pooled, held-out wells) | macro-F1 per fold | detection rate | median lead (min) | alerted before steady state | false alerts / normal day |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in results.items():
        a, folds = s["alerts"], s["fold_macro_f1"]
        lines.append(
            f"| {name} | {s['macro_f1']:.3f} | {np.mean(folds):.3f} ± {np.std(folds):.3f} | {a['detection_rate']:.2f} "
            f"| {a['median_lead_minutes']:.1f} | {a['share_alerted_before_steady_state']:.2f} | {a['false_alerts_per_normal_day']:.2f} |"
        )
    per_class = ["", "| class | support | " + " | ".join(results) + " |", "|---|---|" + "---|" * len(results)]
    first = next(iter(results.values()))
    for cls in CLASS_NAMES:
        per_class.append(f"| {cls} | {first['support'][cls]} | " + " | ".join(f"{s['per_class_f1'][cls]:.2f}" for s in results.values()) + " |")
    (out_dir / "event_metrics.md").write_text("\n".join(lines + per_class) + "\n")


def run(cfg: dict, models: list[str], spark=None) -> dict:
    if not on_databricks() and not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(cfg["experiment"])

    exp = Experiment(cfg, spark)
    data_summary = exp.data_summary()
    log.info("data: %s", json.dumps(data_summary, default=str))

    results, oof = {}, {}
    with mlflow.start_run(run_name="well-events-cv"):
        mlflow.log_params({"n_folds": exp.n_folds, "train_stride": cfg.get("train_stride", 1),
                           "alert_rule": json.dumps(cfg["alerts"]), "sensors": ",".join(exp.sensors)})
        mlflow.log_dict(data_summary, "data_summary.json")
        for name in models:
            params = cfg["models"].get(name, {})
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({k: v for k, v in params.items() if not isinstance(v, list)})
                if name in SKLEARN_MODELS:
                    oof[name] = exp.oof_sklearn(name, params)
                elif name == "cnn":
                    oof[name] = exp.oof_cnn(params)
                elif name == "ensemble":
                    members = [m for m in params["members"] if m in oof]
                    mlflow.log_param("members", ",".join(members))
                    oof[name] = np.mean([oof[m] for m in members], axis=0)
                else:
                    raise ValueError(f"unknown model {name!r}")

                summary, cm, scored = exp.evaluate(oof[name])
                log_summary(summary)
                mlflow.log_text(cm.to_csv(), "confusion_matrix.csv")
                mlflow.log_text(scored.to_csv(index=False), "alerts_by_run.csv")
                results[name] = summary
                log.info("%s: macro-F1 %.3f, median lead %.1f min, detection %.2f", name, summary["macro_f1"],
                         summary["alerts"]["median_lead_minutes"], summary["alerts"]["detection_rate"])

        out_dir = resolve_path(cfg["reports_dir"])
        write_reports(results, data_summary, out_dir)
        mlflow.log_artifact(str(out_dir / "event_metrics.md"))

    save_predictions(exp, oof, resolve_path(cfg["predictions_path"]))
    return results


def save_predictions(exp: Experiment, oof: dict[str, np.ndarray], path: Path) -> None:
    m = exp.eval_mask
    out = exp.windows.loc[m, ["instance_id", "well_id", "window_end", "label", "fold"]].reset_index(drop=True)
    for name, proba in oof.items():
        out[f"{name}_pred"] = proba[m].argmax(1)
        out[f"{name}_conf"] = proba[m].max(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/events.yaml")
    parser.add_argument("--models", nargs="+", default=["rf", "hgb", "cnn", "ensemble"])
    parser.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. cv.n_folds=3")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(load_config(args.config, parse_overrides(args.set)), args.models)


if __name__ == "__main__":
    main()
