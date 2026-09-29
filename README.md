# Oilfield Well-Event Detection & NPT Copilot

Predictive and generative AI on two public oilfield datasets:

- **Well events (Petrobras 3W).** A PySpark pipeline cleans 2,228 labelled offshore-well runs
  and windows them into Delta feature tables. scikit-learn models and a PyTorch 1D-CNN then
  classify normal operation plus 8 fault types (slugging, hydrates, scaling, ...). They are
  scored on wells they never saw and replayed as debounced alerts to measure how far ahead of
  steady state they warn.
- **Non-productive time (Equinor Volve).** Daily drilling report remarks are tagged with their
  NPT cause (stuck pipe, rig repair, waiting on weather, ...) by a fine-tuned DistilBERT,
  compared against a TF-IDF baseline. A LangChain agent on Claude then drafts each wellbore's
  daily NPT summary from the activity tables, citing the remarks it used. NPT hours by cause
  go to Power BI.

## How it fits together

### Well events: raw runs to alerts

```
dataset/<event>/<run>.parquet  (1 Hz, 27 variables, 2,228 runs)
  │  mapInPandas, one task per run file: no 1 Hz row is ever shuffled
  ▼
clean ── out-of-range (sentinels, overflow, dead gauges at 0 bar)
      ── frozen (a P/T value repeated ≥ 10 min on a real well = stuck gauge)
      ── sparse (sensor < 50% valid in the run → dropped for the run)
  │  resample to 10 s
  ▼
silver_3w_quality · silver_3w_instances · silver_3w_readings      (Delta)
  │  Spark: 60 s buckets of additive stats → range-window functions
  ▼
gold_3w_windows   one row per (run, minute): 10-min level/shape/drift + 1-h context features
  │  folds over real wells (simulated + hand-drawn runs only ever train)
  ▼
random forest (baseline) · gradient boosting · 1D-CNN on the 10 s sequence · ensemble
  │  out-of-fold probabilities on held-out wells, tracked in MLflow
  ▼
macro-F1 (window level)  →  alerts: k-of-n votes  →  lead time before steady state, false alerts/day
```

Each run's observation labels are `0` (normal), `k` (steady state of event `k`) or `100 + k`
(the transient leading into it). A window takes the label at its end, and transient windows
count as the fault itself. That is what lets an alert fire *before* the fault settles; the lead
time is measured against the run's first steady-state timestamp.

### NPT: drilling reports to a morning summary

```
Volve DDR XML (WITSML 1.4 drillReport)
  │  oilfield.npt.witsml: activities (time, depth, code, remark) + 24 h summary/forecast
  ▼
NPT activities (code "interruption -- …") ── silver cause labels: activity code + remark rules
  │                                      ── hand-reviewed labels (npt_gold.csv) override them
  │  split by well
  ▼
TF-IDF + logistic regression (baseline) · DistilBERT fine-tune (best epoch on validation wells)
  │  better model tags every NPT remark
  ▼
npt_events ──► Power BI star schema (fact_npt, fact_activity, dim_*), DAX measures
     │
     ▼
LangChain agent on Claude (claude-opus-5-5)
  tools: get_npt_events · get_npt_hours_by_cause · get_daily_report · search_remarks (BM25)
         · query_activity_tables (one read-only SELECT on a read-only connection)
  │  draft cites remarks as [F-12:20080214:03]; citations are checked against the IDs the
  │  tools actually returned, with one repair turn if any are unknown
  ▼
daily NPT summary per wellbore (Markdown + citation report)
```

## Results

### Well events: 3W v2.0.0, scored on held-out real wells

Cleaning keeps 2,114 of the 2,228 runs: 1,005 real runs from 39 wells, 1,089 simulated and 20
hand-drawn. The other 114 runs have fewer than two usable sensors. Windowing gives 1,140,047
one-minute windows. The 420,600 windows from real wells are each predicted by a model that
never saw their well (5 folds over wells). Full output: [reports/event_metrics.md](reports/event_metrics.md).

| model | macro-F1, pooled | macro-F1 per fold | fault runs alerted | false alerts / normal day |
|---|---|---|---|---|
| random forest, level stats (baseline) | 0.213 | 0.266 ± 0.051 | 13% | 0.5 |
| **gradient boosting, level + shape + drift + 1 h context** | **0.279** | 0.357 ± 0.099 | 29% | 10.7 |
| 1D-CNN on 10 s sequences | 0.143 | 0.227 ± 0.042 | 26% | 27.4 |
| ensemble (GB + CNN) | 0.218 | 0.317 ± 0.093 | 25% | 9.6 |

Alerts use p ≥ 0.6 in 3 of the last 5 windows. Per fault class for gradient boosting:

| class | held-out runs | alerted | median minutes after event start | median minutes before steady state |
|---|---|---|---|---|
| spurious DHSV closure | 22 | 95% | 5.2 | 86.2 |
| hydrate | 71 | 35% | 4.8 | 1,964.7 |
| scaling in PCK | 35 | 40% | 62.3 | - (runs end in transient) |
| flow instability | 303 | 26% | 3.0 | - (no transient label) |
| severe slugging, BSW increase, productivity loss, PCK restriction | 52 | 0% | - | - |

What these numbers say:

- Held-out wells are hard. Each of the four classes that score zero has real examples from only
  one to three wells. When those wells are held out, the model has mostly simulated runs for
  that class, and simulated runs don't transfer to real gauges. A random split over windows
  would hide this, because windows from the same well would land on both sides.
- Gradient boosting beats the level-only baseline by 6.6 macro-F1 points. Most of the gain is
  on scaling (F1 0.00 → 0.37) and flow instability (0.20 → 0.39).
- The CNN only sees mean-removed shapes, so it loses the level information hydrates and DHSV
  closures depend on, and averaging it in hurts the ensemble.
- At this threshold, gradient boosting raises about 11 false alerts per day of normal operation,
  too many to page anyone with. The alert rule needs tuning per class before use.

Quick ablations of the gradient-boosting model on the same folds (run from a scratch script,
not the CLI):

| variant | pooled macro-F1 |
|---|---|
| 10-minute features only | 0.274 |
| + 1 h context features | 0.275 (0.279 in the CLI run, other seeds) |
| without P-PDG (valid in every simulated run, usable in about a fifth of real runs) | 0.258 |
| without absolute levels (mean / min / max) | 0.223 |
| real wells only, no simulated runs | 0.251 |
| simulated runs at ¼ weight | 0.270 |

### NPT tagging and the copilot

Not yet measured on Volve. The reports need an Equinor data-village account and can't be
redistributed, so the repo has no Volve numbers. `make demo-npt` runs the whole chain on
synthetic reports. Its scores (TF-IDF macro-F1 0.92 on templated remarks) only show the code
works. They say nothing about real remarks.

To get real numbers: download the DDR XML, run `python -m oilfield.npt.train --export-review`,
hand-label the exported sample into `data/labels/npt_gold.csv`, and train again. The report
then compares TF-IDF and DistilBERT on reviewed labels from held-out wells
(`reports/npt_metrics.md`).

The copilot's tool loop and citation repair are tested with a scripted chat model. A live run
needs `ANTHROPIC_API_KEY`.

## Repository layout

```
src/oilfield/
  threew/    schema, ingest, cleaning (pandas, per run), windowing (Spark), pipeline CLI
  events/    folds over wells, scikit-learn models, 1D-CNN, alerts and lead time, training CLI (MLflow)
  npt/       WITSML parser, cause taxonomy and labelling rules, TF-IDF, DistilBERT, training and tagging CLIs,
             synthetic report generator
  copilot/   SQLite store, BM25 remark index, LangChain tools, agent, citation check, CLI
  bi/        Power BI star-schema export
notebooks/   Databricks notebooks (thin wrappers over the package), deployed by databricks.yml
configs/     threew.yaml, events.yaml, npt.yaml, copilot.yaml
powerbi/     DAX measures, Power Query loader, report layout
reports/     metrics written by the training CLIs
tests/       pytest (Spark tests need Java 17)
```

## Running it

Python 3.10+ and Java 17 for Spark.

```bash
make setup                      # .venv with every extra: spark, ml, nlp, agent, dev
make test

# Well events on the real 3W data (~1.7 GB download)
make data-3w                    # scripts/download_3w.sh
make pipeline                   # silver + gold Delta tables under data/delta
make train-events               # held-out-well CV -> reports/event_metrics.md, MLflow runs
make mlflow-ui

# NPT end to end on synthetic reports: no download, no API key
make demo-npt

# NPT on the real Volve reports
#   1. register at Equinor's Volve data village and download "Daily Drilling Report - XML Version"
#      into data/raw/volve/ddr (the licence does not allow redistributing it here)
#   2. python -m oilfield.npt.train --export-review    # also writes data/labels/npt_review.csv
#   3. review that sample, save it as data/labels/npt_gold.csv, and train again: evaluation then
#      uses the reviewed labels instead of the rules
make train-npt tag-npt copilot bi

# Daily summary with Claude (needs ANTHROPIC_API_KEY)
make summary WELLBORE=F-12 DATE=2008-02-14
make summary WELLBORE=F-12 DATE=2008-02-14 OFFLINE=--offline   # template, no LLM
```

### Databricks

`databricks bundle deploy -t dev` builds the wheel and deploys three jobs:

| job | tasks |
|---|---|
| `oilfield-well-events` | `01_build_3w_tables` → `02_train_event_models` |
| `oilfield-npt-models` | `03_train_npt_models` (single-node GPU) |
| `oilfield-npt-copilot` | `04_tag_npt` → `05_npt_daily_summaries` (scheduled 05:30, paused by default) |

Tables go to Unity Catalog (`main.oilfield` by default) and files to a UC volume. MLflow
experiments live under `/Shared/oilfield`. The copilot reads `ANTHROPIC_API_KEY` from the
`oilfield` secret scope and falls back to template summaries without it.

## Design notes

- **Held-out wells, not held-out windows.** Windows from one well share gauges, reservoir and
  often the same recurring event, so a random split leaks. Folds are drawn over the 39 real
  wells, stratified by event type. Several fault types have real examples from only one to three
  wells, so scores are low and fold-to-fold variance is large.
- **Cleaning where the data already is.** Each 3W run is a complete, time-ordered file, so
  masking and resampling run inside the Spark task that reads the file. Spark-native aggregation
  and window functions take over at 10 s resolution, where the data is 10× smaller. The OSS
  Parquet reader also lacks the Brotli codec 3W uses; reading with pyarrow inside `mapInPandas`
  sidesteps that.
- **Data quirks the pipeline handles.** Hand-drawn runs store pressure in bar, although
  `dataset.ini` says Pa, so units are detected per run. Their curves are piecewise flat by
  construction, so the stuck-gauge check runs only on real wells. Many real wells carry only the
  TPT gauges, so runs need two usable sensors, not three.
- **Silver NPT labels are a bootstrap.** Volve's activity codes say time was lost and sometimes
  roughly why (`repair`, `wait`, `fish`), but not stuck pipe versus other downhole trouble, or
  weather versus late equipment. Rules over code + remark give starting labels. Until reviewed
  labels exist, test scores measure agreement with those rules, and the training report says so.
- **Grounded summaries.** The agent can only read: every SQL tool goes through a single-SELECT
  check, a table allow-list, a row cap, and a read-only SQLite connection. Every activity ID a
  tool returns is recorded, and the summary may only cite those IDs.

## Data and licences

- [Petrobras 3W](https://github.com/petrobras/3W) v2.0.0, data files under
  [CC BY 4.0](https://github.com/petrobras/3W/blob/main/dataset/LICENSE-CC-BY). Downloaded by
  `scripts/download_3w.sh`; not stored in this repo.
- [Equinor Volve](https://www.equinor.com/energy/volve-data-sharing), Equinor Open Data Licence.
  Download it from Equinor. `oilfield.npt.synthetic` generates look-alike reports (wellbores
  `NO 15/9-S-*`) for tests and demos.
